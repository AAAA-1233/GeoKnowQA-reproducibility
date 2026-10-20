#!/usr/bin/env python3
"""Resumable fixed-test evaluation for Randeng-T5 QA, with Chinese QA metrics."""
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T5 answers and evaluate the full fixed test set.")
    parser.add_argument("--model", default="base_models/Randeng-T5-784M-QA-Chinese")
    parser.add_argument("--adapter", default=None, help="Optional final LoRA adapter directory.")
    parser.add_argument("--test-file", default="data/splits_seed42/test.jsonl")
    parser.add_argument("--output", required=True, help="JSONL predictions; resumes safely if it already exists.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument("--limit", type=int, default=None, help="Debug only; omit for official metrics.")
    return parser.parse_args()


def resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_source(example: dict) -> str:
    question = str(example["instruction"]).strip()
    context = "\n".join(
        part for part in (str(example.get("system", "")).strip(), str(example.get("input", "")).strip()) if part
    )
    source = f"question:{question}"
    if context:
        source += f" knowledge:{context}"
    return source + " answer: <extra_id_0>"


def tokens(text: str) -> list[str]:
    return [part for part in jieba.lcut(text.strip()) if part.strip()]


def rouge_pair(pair: tuple[list[str], list[str]]) -> dict:
    prediction, reference = pair
    return Rouge().get_scores(" ".join(prediction) or "空", " ".join(reference) or "空")[0]


def token_f1(prediction: list[str], reference: list[str]) -> float:
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    if not prediction or not reference or not overlap:
        return 0.0
    precision, recall = overlap / len(prediction), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def distinct(tokens_per_answer: list[list[str]], n: int) -> float:
    ngrams = [tuple(answer[i : i + n]) for answer in tokens_per_answer for i in range(len(answer) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def compute_metrics(rows: list[dict]) -> dict:
    hypotheses = [tokens(row["prediction"]) for row in rows]
    references = [[tokens(row["reference"])] for row in rows]
    smoothing = SmoothingFunction().method1
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    result = {name: corpus_bleu(references, hypotheses, weights=value, smoothing_function=smoothing) for name, value in weights.items()}
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        rouge_rows = list(pool.imap(rouge_pair, ((hyp, ref[0]) for hyp, ref in zip(hypotheses, references)), chunksize=32))
    for name in ("rouge-1", "rouge-2", "rouge-l"):
        result[name] = sum(score[name]["f"] for score in rouge_rows) / len(rouge_rows)
    result["distinct_1"] = distinct(hypotheses, 1)
    result["distinct_2"] = distinct(hypotheses, 2)
    result["token_f1"] = sum(token_f1(hyp, ref[0]) for hyp, ref in zip(hypotheses, references)) / len(rows)
    result["exact_match"] = sum(row["prediction"].strip() == row["reference"].strip() for row in rows) / len(rows)
    result["samples"] = len(rows)
    return result


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    model_path = resolve(project_dir, args.model)
    test_path = resolve(project_dir, args.test_file)
    output_path = resolve(project_dir, args.output)
    adapter_path = resolve(project_dir, args.adapter) if args.adapter else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = list(iter_jsonl(test_path))
    if args.limit is not None:
        examples = examples[: args.limit]
    completed = {row["id"]: row for row in iter_jsonl(output_path)} if output_path.exists() else {}
    pending = [example for example in examples if example["id"] not in completed]
    print(f"test={len(examples)} completed={len(completed)} pending={len(pending)}", flush=True)

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.config.use_cache = True
        model.eval()

        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                sources = [make_source(example) for example in batch]
                encoded = tokenizer(sources, return_tensors="pt", padding=True, truncation=False)
                source_length = encoded["input_ids"].shape[1]
                if source_length > args.max_source_length:
                    raise ValueError(f"A source has {source_length} tokens, exceeding {args.max_source_length}; refusing truncation.")
                encoded = {name: value.to(model.device) for name, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        forced_eos_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                    )
                predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for example, prediction in zip(batch, predictions):
                    row = {
                        "id": example["id"],
                        "instruction": example["instruction"],
                        "reference": example["output"],
                        "prediction": prediction.strip(),
                        "source_file": example.get("source_file", ""),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed[row["id"]] = row
                print(f"generated {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)

    ordered = [completed[example["id"]] for example in examples]
    metrics = compute_metrics(ordered)
    metrics.update({"model": str(model_path), "adapter": str(adapter_path) if adapter_path else None, "test_file": str(test_path)})
    metrics_path = output_path.with_suffix(".metrics.json")
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
