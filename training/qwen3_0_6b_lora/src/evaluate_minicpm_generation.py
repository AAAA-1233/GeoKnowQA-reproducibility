#!/usr/bin/env python3
"""Reference-compatible, non-thinking generation evaluation for MiniCPM5.

MiniCPM's tokenizer returns token_type_ids, while its Llama model's generate()
does not accept them.  All prompts, decoding, resuming, and metric computation
otherwise match the reference evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REFERENCE_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(REFERENCE_SRC))
from common import atomic_write_json, iter_jsonl, messages_without_answer, seed_everything
from reference_metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--repetition-penalty", type=float, default=1.15,
                   help="Token-level repetition gate; 1.0 disables it.")
    p.add_argument("--no-repeat-ngram-size", type=int, default=4,
                   help="Block any repeated n-gram of this size; 0 disables it.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Enable MiniCPM5's thinking mode. Disabled by default for concise QA.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    test_path, output_path = Path(args.test_file), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = list(iter_jsonl(test_path))
    if args.limit is not None:
        examples = examples[: args.limit]
    completed = {}
    if output_path.exists():
        completed = {row["id"]: row for row in iter_jsonl(output_path)}
    pending = [x for x in examples if x["id"] not in completed]
    print(f"test={len(examples)} completed={len(completed)} pending={len(pending)}")

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, device_map={"": 0},
        )
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        with open(output_path, "a", encoding="utf-8", buffering=1) as out:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                prompts = [tokenizer.apply_chat_template(
                    messages_without_answer(x), tokenize=False, add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                ) for x in batch]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True)
                encoded.pop("token_type_ids", None)
                encoded = encoded.to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                        num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        renormalize_logits=True,
                        eos_token_id=model.generation_config.eos_token_id,
                    )
                new_tokens = generated[:, encoded["input_ids"].shape[1]:]
                predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for example, prediction in zip(batch, predictions):
                    row = {"id": example["id"], "instruction": example["instruction"],
                           "reference": example["output"], "prediction": prediction.strip(),
                           "source_file": example["source_file"]}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed[row["id"]] = row
                print(f"generated {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    ordered = [completed[x["id"]] for x in examples]
    metrics = compute_metrics(ordered)
    metrics.update({"model": args.model, "adapter": args.adapter, "test_file": str(test_path),
                    "generation": {"max_new_tokens": args.max_new_tokens,
                                   "repetition_penalty": args.repetition_penalty,
                                   "no_repeat_ngram_size": args.no_repeat_ngram_size,
                                   "enable_thinking": args.enable_thinking,
                                   "do_sample": False}})
    metrics_path = output_path.with_suffix(".metrics.json")
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
