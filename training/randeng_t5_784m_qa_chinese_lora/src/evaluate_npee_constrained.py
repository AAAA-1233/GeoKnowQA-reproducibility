#!/usr/bin/env python3
"""NPEE closed-form evaluation by forced candidate likelihood for T5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base_models/Randeng-T5-784M-QA-Chinese")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--objective", default="../../datasets/NPEEBench/test/objective.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_dir / path).resolve()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_source(example: dict) -> str:
    instruction = (
        "Choose the correct option. Output only its capital letter."
        if example["question_type"] == "choice"
        else "Judge the statement. Output only True or False."
    )
    return f"question:{instruction}\n{example['question']} answer: <extra_id_0>"


def candidate_log_likelihood(model, tokenizer, encoded: dict[str, torch.Tensor], candidate: str) -> torch.Tensor:
    labels = tokenizer([candidate] * encoded["input_ids"].shape[0], return_tensors="pt", padding=True, add_special_tokens=True)["input_ids"]
    labels = labels.to(model.device).masked_fill(labels.to(model.device) == tokenizer.pad_token_id, -100)
    with torch.inference_mode():
        logits = model(**encoded, labels=labels, use_cache=False).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * labels.ne(-100)).sum(dim=-1)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    model_path = resolve(project_dir, args.model)
    adapter_path = resolve(project_dir, args.adapter) if args.adapter else None
    output_dir = resolve(project_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "objective_constrained_predictions.jsonl"
    examples = [row for row in read_jsonl(resolve(project_dir, args.objective)) if row["question_type"] in {"choice", "tf"}]
    completed = {row["id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    pending = [row for row in examples if row["id"] not in completed]
    print(f"[{args.model_name}/constrained] total={len(examples)} completed={len(completed)} pending={len(pending)}", flush=True)

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for question_type, candidates in (("choice", ["A", "B", "C", "D", "E"]), ("tf", ["True", "False"])):
                group = [row for row in pending if row["question_type"] == question_type]
                for start in range(0, len(group), args.batch_size):
                    batch = group[start : start + args.batch_size]
                    encoded = tokenizer([make_source(row) for row in batch], return_tensors="pt", padding=True, truncation=False)
                    if encoded["input_ids"].shape[1] > 512:
                        raise ValueError("NPEE source exceeds 512 tokens; refusing truncation.")
                    encoded = {name: value.to(model.device) for name, value in encoded.items()}
                    scores = torch.stack([candidate_log_likelihood(model, tokenizer, encoded, candidate) for candidate in candidates], dim=1)
                    for example, index, row_scores in zip(batch, scores.argmax(dim=1).tolist(), scores.tolist()):
                        predicted = candidates[index]
                        row = {"id": example["id"], "question_type": question_type, "question": example["question"], "reference": example["answer"].strip(), "prediction": predicted, "candidate_scores": dict(zip(candidates, row_scores)), "correct": predicted == example["answer"].strip()}
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[row["id"]] = row
                    print(f"[{args.model_name}/constrained] {question_type} {min(start + len(batch), len(group))}/{len(group)}", flush=True)

    ordered = [completed[row["id"]] for row in examples]
    by_type = {}
    for question_type in ("choice", "tf"):
        rows = [row for row in ordered if row["question_type"] == question_type]
        correct = sum(row["correct"] for row in rows)
        by_type[question_type] = {"correct": correct, "samples": len(rows), "accuracy": correct / len(rows)}
    correct = sum(row["correct"] for row in ordered)
    result = {"method": "forced candidate likelihood scoring (not free generation)", "model_name": args.model_name, "model": str(model_path), "adapter": str(adapter_path) if adapter_path else None, "correct": correct, "samples": len(ordered), "accuracy": correct / len(ordered), "accuracy_by_question_type": by_type}
    write_json(output_dir / "objective_constrained_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
