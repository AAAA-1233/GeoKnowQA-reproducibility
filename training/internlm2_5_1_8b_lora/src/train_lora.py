#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from common import load_json, resolve_from_project, seed_everything, user_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/internlm2_5_1_8b_lora.json")
    p.add_argument("--resume", default="auto", help="auto, none, or an explicit checkpoint directory")
    p.add_argument("--output-dir", default=None, help="Optional override, useful for an isolated smoke test")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    cfg = load_json(resolve_from_project(args.config, project_dir))
    if args.train_batch_size is not None:
        cfg["per_device_train_batch_size"] = args.train_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing is not None:
        cfg["gradient_checkpointing"] = args.gradient_checkpointing
    data_dir = resolve_from_project(cfg["data_dir"], project_dir)
    output_dir = resolve_from_project(cfg["output_dir"], project_dir)
    if args.output_dir:
        output_dir = resolve_from_project(args.output_dir, project_dir)
    model_path = Path(cfg["model_name_or_path"])
    if not model_path.is_absolute() and (project_dir / model_path).exists():
        cfg["model_name_or_path"] = str(project_dir / model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"], use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
        },
    )
    if args.train_limit:
        raw["train"] = raw["train"].select(range(min(args.train_limit, len(raw["train"]))))
        raw["validation"] = raw["validation"].select(
            range(min(max(32, args.train_limit // 10), len(raw["validation"])))
        )
    max_length = int(cfg["max_length"])
    overflow_context = int(cfg.get("overflow_context_tokens", 128))

    def tokenize(batch):
        all_input_ids, all_attention_mask, all_labels, lengths = [], [], [], []
        for instruction, extra, output, system in zip(
            batch["instruction"], batch["input"], batch["output"], batch["system"]
        ):
            example = {"instruction": instruction, "input": extra, "system": system}
            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system.strip()})
            messages.append({"role": "user", "content": user_text(example)})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_conversation = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": output.strip()}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_conversation, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("The chat template does not preserve the prompt prefix.")
            # Includes InternLM's assistant terminator (<|im_end|>) instead of
            # incorrectly forcing the generic tokenizer EOS token.
            answer_ids = full_ids[len(prompt_ids) :]
            if len(prompt_ids) >= max_length:
                raise ValueError(
                    f"Prompt has {len(prompt_ids)} tokens, exceeding max_length={max_length}; "
                    "increase max_length rather than truncating source text."
                )

            # Never discard source/answer tokens. The first block contains the complete prompt;
            # later blocks carry answer context whose labels are masked, and supervise only new tokens.
            first_capacity = max_length - len(prompt_ids)
            first_answer = answer_ids[:first_capacity]
            input_ids = prompt_ids + first_answer
            all_input_ids.append(input_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_labels.append([-100] * len(prompt_ids) + first_answer)
            lengths.append(len(input_ids))

            consumed = len(first_answer)
            while consumed < len(answer_ids):
                context_start = max(0, consumed - overflow_context)
                context_ids = answer_ids[context_start:consumed]
                capacity = max_length - len(context_ids)
                new_ids = answer_ids[consumed : consumed + capacity]
                continuation = context_ids + new_ids
                all_input_ids.append(continuation)
                all_attention_mask.append([1] * len(continuation))
                all_labels.append([-100] * len(context_ids) + new_ids)
                lengths.append(len(continuation))
                consumed += len(new_ids)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "length": lengths,
        }

    tokenized = raw.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=min(8, os.cpu_count() or 1),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing once (cached for resumes)",
    )

    dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float16
    quantization_config = None
    if cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map={"": 0} if quantization_config is not None else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if cfg.get("use_liger_kernel", False):
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        apply_liger_kernel_to_qwen2(
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=False,
            model=model,
        )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        logging_first_step=True,
        save_strategy=cfg["save_strategy"],
        save_steps=int(cfg.get("save_steps", 500)),
        eval_strategy=cfg["eval_strategy"],
        save_only_model=False,
        bf16=bool(cfg.get("bf16", False)),
        fp16=not bool(cfg.get("bf16", False)),
        tf32=bool(cfg.get("tf32", True)),
        optim=cfg.get("optim", "adamw_torch_fused"),
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to="none",
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        load_best_model_at_end=False,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        use_liger_kernel=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    checkpoint = None
    if args.resume == "auto":
        checkpoint = get_last_checkpoint(str(output_dir))
    elif args.resume.lower() != "none":
        checkpoint = args.resume
    print(f"Resume checkpoint: {checkpoint or 'none'}")
    result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["train_samples"] = len(tokenized["train"])
    metrics["validation_samples"] = len(tokenized["validation"])
    metrics["effective_batch_size"] = (
        train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps
    )
    metrics["perplexity_from_train_loss"] = math.exp(min(20, metrics.get("train_loss", 20)))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
