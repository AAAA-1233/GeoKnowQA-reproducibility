#!/usr/bin/env python3
"""Evaluate the unmodified Randeng-T5 model on NPEE objective questions."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jieba
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_chinese import Rouge
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


EXPLICIT_CHOICE = re.compile(r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE|答案|选项)\s*(?:IS|为|是)?\s*[:：]?\s*[\(\[]?([A-E])\b", re.I)
LEADING_CHOICE = re.compile(r"^\s*[\(\[]?([A-E])(?:[\)\].:：\s]|$)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base_models/Randeng-T5-784M-QA-Chinese")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter; omit for the base model.")
    parser.add_argument("--model-name", default="randeng-t5-784m-base")
    parser.add_argument("--objective", default="../../datasets/NPEEBench/test/objective.jsonl")
    parser.add_argument("--output-dir", default="evaluations/npee/base")
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
    kind = example["question_type"]
    instruction = {
        "choice": "Choose the correct option. Output only its capital letter.",
        "tf": "Judge the statement. Output only True or False.",
        "completion": "Fill in the blank. Output only the required answer.",
    }[kind]
    return f"question:{instruction}\n{example['question']} answer: <extra_id_0>"


def extract_choice(text: str) -> str | None:
    normalized = re.sub(r"<extra_id_\d+>", "", text).strip().upper()
    for pattern in (EXPLICIT_CHOICE, LEADING_CHOICE):
        matched = pattern.search(normalized)
        if matched:
            return matched.group(1).upper()
    choices = re.findall(r"\b([A-E])\b", normalized)
    unique = list(dict.fromkeys(choice.upper() for choice in choices))
    return unique[0] if len(unique) == 1 else None


def extract_true_false(text: str) -> str | None:
    normalized = re.sub(r"<extra_id_\d+>", "", text).strip().lower()
    if re.search(r"^(?:the\s+(?:answer|statement)\s+is\s+)?(?:false|incorrect|wrong|no)\b", normalized):
        return "False"
    if re.search(r"^(?:the\s+(?:answer|statement)\s+is\s+)?(?:true|correct|right|yes)\b", normalized):
        return "True"
    return None


def metric_tokens(text: str) -> list[str]:
    return [piece for piece in jieba.lcut(text.lower()) if piece.strip()]


def completion_metrics(rows: list[dict]) -> dict:
    hypotheses = [metric_tokens(row["prediction"]) for row in rows]
    references = [[metric_tokens(row["reference"])] for row in rows]
    smoothing = SmoothingFunction().method1
    weights = {"bleu_1": (1, 0, 0, 0), "bleu_2": (.5, .5, 0, 0), "bleu_3": (1/3, 1/3, 1/3, 0), "bleu_4": (.25, .25, .25, .25)}
    result = {key: corpus_bleu(references, hypotheses, weights=value, smoothing_function=smoothing) for key, value in weights.items()}
    rouge_rows = [Rouge().get_scores(" ".join(hyp) or "空", " ".join(ref[0]) or "空")[0] for hyp, ref in zip(hypotheses, references)]
    for key in ("rouge-1", "rouge-2", "rouge-l"):
        result[key] = sum(item[key]["f"] for item in rouge_rows) / len(rouge_rows)
    result["samples"] = len(rows)
    return result


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    model_path, objective_path, output_dir = (resolve(project_dir, args.model), resolve(project_dir, args.objective), resolve(project_dir, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "objective_predictions.jsonl"
    examples = read_jsonl(objective_path)
    completed = {row["id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    pending = [row for row in examples if row["id"] not in completed]
    print(f"[randeng-t5-784m-base/objective] total={len(examples)} completed={len(completed)} pending={len(pending)}", flush=True)
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
        if args.adapter:
            adapter_path = resolve(project_dir, args.adapter)
            model = PeftModel.from_pretrained(model, adapter_path)
        model = model.eval()
        groups = (("choice", 32), ("completion", 256), ("tf", 16))
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for question_type, max_new_tokens in groups:
                group = [row for row in pending if row["question_type"] == question_type]
                for start in range(0, len(group), args.batch_size):
                    batch = group[start : start + args.batch_size]
                    encoded = tokenizer([make_source(row) for row in batch], return_tensors="pt", padding=True, truncation=False)
                    if encoded["input_ids"].shape[1] > 512:
                        raise ValueError("NPEE source exceeds 512 tokens; refusing truncation.")
                    encoded = {key: value.to(model.device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    for example, prediction in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                        row = {"id": example["id"], "category": "objective", "question_type": question_type, "question_type_zh": example.get("question_type_zh", ""), "question": example["question"], "reference": example["answer"], "prediction": prediction.strip()}
                        if question_type == "choice":
                            row["parsed_answer"] = extract_choice(row["prediction"])
                        elif question_type == "tf":
                            row["parsed_answer"] = extract_true_false(row["prediction"])
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[row["id"]] = row
                    print(f"[randeng-t5-784m-base/objective] generated {min(start + len(batch), len(group))}/{len(group)} (type={question_type})", flush=True)
    # T5 occasionally exposes its decoder sentinel in the decoded text. Re-parse
    # every saved prediction so a resumed run and a fresh run use identical rules.
    for row in completed.values():
        if row["question_type"] == "choice":
            row["parsed_answer"] = extract_choice(row["prediction"])
        elif row["question_type"] == "tf":
            row["parsed_answer"] = extract_true_false(row["prediction"])
    ordered = [completed[row["id"]] for row in examples]
    closed = [row for row in ordered if row["question_type"] in {"choice", "tf"}]
    by_type = {}
    for question_type in ("choice", "tf"):
        rows = [row for row in closed if row["question_type"] == question_type]
        correct = sum(row.get("parsed_answer") == row["reference"].strip() for row in rows)
        valid = sum(row.get("parsed_answer") is not None for row in rows)
        by_type[question_type] = {"accuracy": correct / len(rows), "correct": correct, "samples": len(rows), "valid_answer_rate": valid / len(rows)}
    correct = sum(row.get("parsed_answer") == row["reference"].strip() for row in closed)
    valid = sum(row.get("parsed_answer") is not None for row in closed)
    adapter_path = resolve(project_dir, args.adapter) if args.adapter else None
    result = {"model_name": args.model_name, "model": str(model_path), "adapter": str(adapter_path) if adapter_path else None, "accuracy": correct / len(closed), "accuracy_scope": "choice + true/false; completion is reported with text-overlap metrics", "correct": correct, "closed_form_samples": len(closed), "valid_answer_rate": valid / len(closed), "invalid_answers": len(closed) - valid, "samples": len(ordered), "accuracy_by_question_type": by_type, "completion_metrics": completion_metrics([row for row in ordered if row["question_type"] == "completion"])}
    write_json(output_dir / "objective_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
