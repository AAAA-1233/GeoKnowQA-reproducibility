#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[\u4e00-\u9fff]")
EXPLICIT_CHOICE_RE = re.compile(
    r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE|答案|选项)\s*(?:IS|为|是)?\s*[:：]?\s*[\(\[]?([A-E])\b",
    re.IGNORECASE,
)
LEADING_CHOICE_RE = re.compile(r"^\s*[\(\[]?([A-E])(?:[\)\].:：\s]|$)", re.IGNORECASE)
STANDALONE_CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one fine-tuned model on both NPEE subsets.")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="Omit for the pre-fine-tuning base model")
    p.add_argument("--model-name", required=True)
    p.add_argument("--subjective", default="datasets/NPEEBench/test/subjective.jsonl")
    p.add_argument("--objective", default="datasets/NPEEBench/test/objective.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--objective-first", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Smoke test only")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def metric_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(pred: list[str], ref: list[str]) -> float:
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def text_overlap_metrics(rows: list[dict]) -> dict:
    hypotheses = [metric_tokens(x["prediction"]) for x in rows]
    references = [[metric_tokens(x["reference"])] for x in rows]
    smooth = SmoothingFunction().method1
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    result = {
        name: corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth)
        for name, w in weights.items()
    }
    rouge_rows = [
        Rouge().get_scores(" ".join(hyp) or "empty", " ".join(refs[0]) or "empty")[0]
        for hyp, refs in zip(hypotheses, references)
    ]
    for key in ("rouge-1", "rouge-2", "rouge-l"):
        result[key] = sum(x[key]["f"] for x in rouge_rows) / len(rouge_rows)
    result["token_f1"] = sum(token_f1(h, r[0]) for h, r in zip(hypotheses, references)) / len(rows)
    result["exact_match"] = sum(
        metric_tokens(x["prediction"]) == metric_tokens(x["reference"]) for x in rows
    ) / len(rows)
    result["samples"] = len(rows)
    return result


def subjective_metrics(rows: list[dict]) -> dict:
    result = text_overlap_metrics(rows)
    result["metrics_by_question_type"] = {
        question_type: text_overlap_metrics([x for x in rows if x["question_type"] == question_type])
        for question_type in ("noun", "qa", "discussion")
    }
    return result


def extract_choice(text: str) -> str | None:
    normalized = text.strip().upper()
    for pattern in (EXPLICIT_CHOICE_RE, LEADING_CHOICE_RE):
        match = pattern.search(normalized)
        if match:
            return match.group(1).upper()
    choices = STANDALONE_CHOICE_RE.findall(normalized)
    unique = list(dict.fromkeys(x.upper() for x in choices))
    return unique[0] if len(unique) == 1 else None


def extract_true_false(text: str) -> str | None:
    normalized = text.strip().lower()
    false_match = re.search(r"^(?:the\s+(?:answer|statement)\s+is\s+)?(?:false|incorrect|wrong|no)\b", normalized)
    true_match = re.search(r"^(?:the\s+(?:answer|statement)\s+is\s+)?(?:true|correct|right|yes)\b", normalized)
    if false_match:
        return "False"
    if true_match:
        return "True"
    return None


def objective_metrics(rows: list[dict]) -> dict:
    closed_rows = [x for x in rows if x["question_type"] in {"choice", "tf"}]
    completion_rows = [x for x in rows if x["question_type"] == "completion"]
    by_type = {}
    for question_type in ("choice", "tf"):
        subset = [x for x in closed_rows if x["question_type"] == question_type]
        correct = sum(x.get("parsed_answer") == x["reference"].strip() for x in subset)
        valid = sum(x.get("parsed_answer") is not None for x in subset)
        by_type[question_type] = {
            "accuracy": correct / len(subset), "correct": correct, "samples": len(subset),
            "valid_answer_rate": valid / len(subset),
        }
    closed_correct = sum(x.get("parsed_answer") == x["reference"].strip() for x in closed_rows)
    closed_valid = sum(x.get("parsed_answer") is not None for x in closed_rows)
    completion_result = text_overlap_metrics(completion_rows)
    strict_exact = sum(
        metric_tokens(x["prediction"]) == metric_tokens(x["reference"]) for x in rows
    )
    return {
        "accuracy": closed_correct / len(closed_rows),
        "accuracy_scope": "choice + true/false; completion is reported with text-overlap metrics",
        "correct": closed_correct,
        "closed_form_samples": len(closed_rows),
        "valid_answer_rate": closed_valid / len(closed_rows),
        "invalid_answers": len(closed_rows) - closed_valid,
        "samples": len(rows),
        "strict_exact_match_all_objective": strict_exact / len(rows),
        "accuracy_by_question_type": by_type,
        "completion_metrics": completion_result,
    }


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def make_messages(example: dict, category: str) -> list[dict]:
    if category == "objective" and example["question_type"] == "choice":
        system = (
            "You are a rigorous geography exam assistant. Select the correct option and output only its "
            "capital letter, such as A, B, or C. Do not explain."
        )
    elif category == "objective" and example["question_type"] == "tf":
        system = (
            "You are a rigorous geography exam assistant. Judge the statement and output only True or False. "
            "Do not explain."
        )
    elif category == "objective":
        system = (
            "You are a rigorous geography exam assistant. Fill in the blank with the required term or terms "
            "in English. Output only the answer and do not explain."
        )
    elif example["question_type"] == "discussion":
        system = (
            "You are a rigorous geography question-answering assistant. Write a comprehensive, well-structured "
            "answer in English and cover all major points required by the question."
        )
    elif example["question_type"] == "noun":
        system = (
            "You are a rigorous geography question-answering assistant. Give a precise definition in English."
        )
    else:
        system = (
            "You are a rigorous geography question-answering assistant. Answer the question directly and "
            "concisely in English."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": example["question"]}]


def evaluate_subset(
    *, model, tokenizer, examples: list[dict], category: str, output_path: Path, batch_size: int,
    max_new_tokens: int, model_name: str, base_model_path: Path, adapter_path: Path,
) -> dict:
    completed = load_completed(output_path)
    pending = [x for x in examples if x["id"] not in completed]
    print(f"[{model_name}/{category}] total={len(examples)} completed={len(completed)} pending={len(pending)}", flush=True)
    if pending:
        with output_path.open("a", encoding="utf-8", buffering=1) as out:
            groups: list[tuple[list[dict], int, int]] = [(pending, batch_size, max_new_tokens)]
            if category == "subjective":
                generation_limits = {"noun": 384, "qa": 1024, "discussion": 1024}
                groups = []
                for question_type in ("noun", "qa", "discussion"):
                    group = [x for x in pending if x["question_type"] == question_type]
                    if group:
                        divisor = 1 if question_type == "noun" else 2 if question_type == "qa" else 4
                        groups.append((group, max(1, batch_size // divisor), generation_limits[question_type]))
            elif category == "objective":
                generation_limits = {"choice": 32, "completion": 256, "tf": 16}
                groups = []
                for question_type in ("choice", "completion", "tf"):
                    group = [x for x in pending if x["question_type"] == question_type]
                    if group:
                        divisor = 2 if question_type == "completion" else 1
                        groups.append((group, max(1, batch_size // divisor), generation_limits[question_type]))
            done_total = 0
            for group, group_batch_size, group_max_new_tokens in groups:
                for start in range(0, len(group), group_batch_size):
                    batch = group[start : start + group_batch_size]
                    prompts = [
                        tokenizer.apply_chat_template(
                            make_messages(x, category), tokenize=False, add_generation_prompt=True
                        )
                        for x in batch
                    ]
                    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=group_max_new_tokens,
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
                            "category": category,
                            "question_type": example["question_type"],
                            "question_type_zh": example.get("question_type_zh", ""),
                            "question": example["question"],
                            "reference": example["answer"],
                            "prediction": prediction.strip(),
                        }
                        if category == "objective" and example["question_type"] == "choice":
                            row["parsed_answer"] = extract_choice(row["prediction"])
                        elif category == "objective" and example["question_type"] == "tf":
                            row["parsed_answer"] = extract_true_false(row["prediction"])
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        completed[row["id"]] = row
                    done_total += len(batch)
                    print(
                        f"[{model_name}/{category}] generated {done_total}/{len(pending)} "
                        f"(type={batch[0]['question_type']}, max_new_tokens={group_max_new_tokens})",
                        flush=True,
                    )
    ordered = [completed[x["id"]] for x in examples]
    metrics = subjective_metrics(ordered) if category == "subjective" else objective_metrics(ordered)
    metrics.update({
        "model_name": model_name,
        "model": str(base_model_path),
        "adapter": str(adapter_path) if adapter_path is not None else None,
    })
    write_json(output_path.with_name(f"{category}_metrics.json"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    model_path = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    adapter_path = None
    if args.adapter:
        adapter_path = (root / args.adapter).resolve() if not Path(args.adapter).is_absolute() else Path(args.adapter)
    subjective_path = (root / args.subjective).resolve()
    objective_path = (root / args.objective).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjective = read_jsonl(subjective_path)
    objective = read_jsonl(objective_path)
    if args.limit is not None:
        subjective = subjective[: args.limit]
        objective = objective[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False if "InternLM" in model_path.name else True,
        padding_side="left", trust_remote_code=True,
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
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.eval()

    def run_subjective() -> dict:
        return evaluate_subset(
            model=model, tokenizer=tokenizer, examples=subjective, category="subjective",
            output_path=output_dir / "subjective_predictions.jsonl", batch_size=args.batch_size,
            max_new_tokens=384, model_name=args.model_name,
            base_model_path=model_path, adapter_path=adapter_path,
        )

    def run_objective() -> dict:
        return evaluate_subset(
            model=model, tokenizer=tokenizer, examples=objective, category="objective",
            output_path=output_dir / "objective_predictions.jsonl", batch_size=args.batch_size,
            max_new_tokens=32, model_name=args.model_name,
            base_model_path=model_path, adapter_path=adapter_path,
        )

    if args.objective_first:
        objective_result = run_objective()
        subjective_result = run_subjective()
    else:
        subjective_result = run_subjective()
        objective_result = run_objective()
    write_json(output_dir / "metrics.json", {"subjective": subjective_result, "objective": objective_result})


if __name__ == "__main__":
    main()
