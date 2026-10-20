#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/randeng_t5_784m_lora.json")
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    config_path = resolve(project_dir, args.config)
    cfg = load_json(config_path)
    if args.train_batch_size is not None:
        cfg["per_device_train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        cfg["per_device_eval_batch_size"] = args.eval_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing is not None:
        cfg["gradient_checkpointing"] = args.gradient_checkpointing

    model_path = resolve(project_dir, cfg["model_name_or_path"])
    data_dir = resolve(project_dir, cfg["data_dir"])
    output_dir = resolve(project_dir, args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "validation.jsonl"),
        },
    )
    if args.train_limit:
        raw["train"] = raw["train"].select(range(min(args.train_limit, len(raw["train"]))))
        validation_limit = min(max(64, args.train_limit // 10), len(raw["validation"]))
        raw["validation"] = raw["validation"].select(range(validation_limit))

    max_source = int(cfg["max_source_length"])
    max_target = int(cfg["max_target_length"])
    continuation_context = int(cfg["continuation_context_tokens"])
    native_template = bool(cfg.get("native_qa_template", False))
    target_with_sentinel = bool(cfg.get("target_with_answer_sentinel", False))
    answer_sentinel_ids = tokenizer("<extra_id_0>", add_special_tokens=False)["input_ids"]
    if target_with_sentinel and len(answer_sentinel_ids) != 1:
        raise ValueError("Tokenizer does not contain the required <extra_id_0> answer sentinel.")
    answer_sentinel_id = answer_sentinel_ids[0]

    def encode_batch(batch):
        sources: list[str] = []
        target_chunks: list[list[int]] = []
        lengths: list[int] = []
        source_truncations = 0
        continued_chunks = 0

        for instruction, extra, output, system in zip(
            batch["instruction"], batch["input"], batch["output"], batch["system"]
        ):
            question = str(instruction).strip()
            context_parts = [str(system).strip(), str(extra).strip()]
            context = "\n".join(part for part in context_parts if part)
            if native_template:
                # This is the original Randeng-T5 QA prompt convention documented
                # with the checkpoint: no extra whitespace around field markers.
                base_source = f"question:{question}"
                if context:
                    base_source += f"knowledge:{context}"
                base_source += "answer:<extra_id_0>"
            else:
                base_source = f"question:{question}"
                if context:
                    base_source += f" knowledge:{context}"
                base_source += " answer: <extra_id_0>"

            source_ids = tokenizer(base_source, add_special_tokens=True)["input_ids"]
            if len(source_ids) > max_source:
                source_truncations += 1
                raise ValueError(
                    f"Source has {len(source_ids)} tokens > max_source_length={max_source}; "
                    "increase the limit instead of truncating it."
                )
            answer_content_ids = tokenizer(str(output).strip(), add_special_tokens=True)["input_ids"]
            answer_prefix = [answer_sentinel_id] if target_with_sentinel else []
            answer_ids = answer_prefix + answer_content_ids
            if len(answer_ids) <= max_target:
                sources.append(base_source)
                target_chunks.append(answer_ids)
                lengths.append(len(source_ids) + len(answer_ids))
                continue

            # Preserve every target token. Later blocks are explicit continuation
            # examples conditioned on the preceding target context.
            start = 0
            content_chunk_size = max_target - len(answer_prefix)
            while start < len(answer_content_ids):
                chunk_content = answer_content_ids[start : start + content_chunk_size]
                chunk = answer_prefix + chunk_content
                if start == 0:
                    chunk_source = base_source
                else:
                    context_ids = answer_content_ids[max(0, start - continuation_context) : start]
                    prior = tokenizer.decode(context_ids, skip_special_tokens=True)
                    if native_template:
                        chunk_source = f"question:{question}knowledge:{prior}answer:<extra_id_0>"
                    else:
                        chunk_source = (
                            f"question:{question} continuation context:{prior} "
                            "answer: <extra_id_0>"
                        )
                    continued_chunks += 1
                chunk_source_ids = tokenizer(chunk_source, add_special_tokens=True)["input_ids"]
                if len(chunk_source_ids) > max_source:
                    raise ValueError(
                        f"Continuation source has {len(chunk_source_ids)} tokens > "
                        f"max_source_length={max_source}."
                    )
                sources.append(chunk_source)
                target_chunks.append(chunk)
                lengths.append(len(chunk_source_ids) + len(chunk))
                start += len(chunk_content)

        model_inputs = tokenizer(
            sources,
            max_length=max_source,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        model_inputs["labels"] = target_chunks
        model_inputs["length"] = lengths
        model_inputs["source_truncations"] = [source_truncations] * len(sources)
        model_inputs["continued_chunks"] = [continued_chunks] * len(sources)
        return model_inputs

    tokenized = raw.map(
        encode_batch,
        batched=True,
        batch_size=512,
        num_proc=min(8, os.cpu_count() or 1),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing T5 data without discarding source/target tokens",
    )

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
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
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        eval_strategy=cfg["eval_strategy"],
        prediction_loss_only=True,
        bf16=bool(cfg.get("bf16", True)),
        fp16=not bool(cfg.get("bf16", True)),
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
        max_steps=args.max_steps if args.max_steps is not None else -1,
        load_best_model_at_end=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
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
    print(f"Resume checkpoint: {checkpoint or 'none'}", flush=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics.update(
        {
            "raw_train_samples": len(raw["train"]),
            "tokenized_train_blocks": len(tokenized["train"]),
            "raw_validation_samples": len(raw["validation"]),
            "tokenized_validation_blocks": len(tokenized["validation"]),
            "effective_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps,
            "perplexity_from_train_loss": math.exp(
                min(20, metrics.get("train_loss", 20))
            ),
        }
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
