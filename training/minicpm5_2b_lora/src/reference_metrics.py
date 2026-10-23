#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import jieba
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import atomic_write_json, iter_jsonl, messages_without_answer, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Resumable generation and Chinese QA metrics.")
    p.add_argument("--model", default="base_models/InternLM2_5-1_8B-Chat")
    p.add_argument("--adapter", default=None)
    p.add_argument("--test-file", default="data/splits_seed42_no_watermark/test.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Debug only; omit for official full-test metrics.")
    return p.parse_args()


def tokens(text: str) -> list[str]:
    return [x for x in jieba.lcut(text.strip()) if x.strip()]


def f1_score(pred: list[str], ref: list[str]) -> float:
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_score_pair(pair: tuple[list[str], list[str]]) -> dict:
    hyp, ref = pair
    return Rouge().get_scores(" ".join(hyp) or "空", " ".join(ref) or "空")[0]


def compute_metrics(rows: list[dict]) -> dict:
    hypotheses = [tokens(r["prediction"]) for r in rows]
    references = [[tokens(r["reference"])] for r in rows]
    smooth = SmoothingFunction().method1
    bleu_weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    result = {
        name: corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smooth)
        for name, weights in bleu_weights.items()
    }
    workers = min(8, os.cpu_count() or 1)
    with Pool(workers) as pool:
        rouge_rows = list(
            pool.imap(rouge_score_pair, ((hyp, refs[0]) for hyp, refs in zip(hypotheses, references)), chunksize=32)
        )
    for key in ("rouge-1", "rouge-2", "rouge-l"):
        result[key] = sum(x[key]["f"] for x in rouge_rows) / len(rouge_rows)
    result["token_f1"] = sum(f1_score(h, r[0]) for h, r in zip(hypotheses, references)) / len(rows)
    result["exact_match"] = sum(
        r["prediction"].strip() == r["reference"].strip() for r in rows
    ) / len(rows)
    result["samples"] = len(rows)
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)
    project_dir = Path(__file__).resolve().parents[1]
    test_path = Path(args.test_file)
    if not test_path.is_absolute():
        test_path = project_dir / test_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.is_absolute() and (project_dir / model_path).exists():
        args.model = str(project_dir / model_path)

    examples = list(iter_jsonl(test_path))
    if args.limit is not None:
        examples = examples[: args.limit]
    completed: dict[str, dict] = {}
    if output_path.exists():
        for row in iter_jsonl(output_path):
            completed[row["id"]] = row
    pending = [x for x in examples if x["id"] not in completed]
    print(f"test={len(examples)} completed={len(completed)} pending={len(pending)}")

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=False, padding_side="left", trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": 0},
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        with open(output_path, "a", encoding="utf-8", buffering=1) as out:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        messages_without_answer(x), tokenize=False, add_generation_prompt=True
                    )
                    for x in batch
                ]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                    )
                new_tokens = generated[:, encoded["input_ids"].shape[1] :]
                predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for example, prediction in zip(batch, predictions):
                    row = {
                        "id": example["id"],
                        "instruction": example["instruction"],
                        "reference": example["output"],
                        "prediction": prediction.strip(),
                        "source_file": example["source_file"],
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed[row["id"]] = row
                print(f"generated {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)

    ordered = [completed[x["id"]] for x in examples]
    metrics = compute_metrics(ordered)
    metrics.update({"model": args.model, "adapter": args.adapter, "test_file": str(test_path)})
    metrics_path = output_path.with_suffix(".metrics.json")
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
