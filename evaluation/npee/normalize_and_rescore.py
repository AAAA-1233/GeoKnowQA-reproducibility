#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_npee import objective_metrics, subjective_metrics, write_json  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--slow-tokenizer", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    result_dir = root / args.result_dir
    model_path = (root / args.model).resolve()
    adapter_path = (root / args.adapter).resolve() if args.adapter else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=not args.slow_tokenizer, trust_remote_code=True
    )

    subjective_path = result_dir / "subjective_predictions.jsonl"
    rows = read_jsonl(subjective_path)
    changed = 0
    for row in rows:
        if row["question_type"] != "discussion":
            continue
        token_ids = tokenizer(row["prediction"], add_special_tokens=False).input_ids
        if len(token_ids) > args.max_tokens:
            row["prediction"] = tokenizer.decode(
                token_ids[: args.max_tokens], skip_special_tokens=True
            ).strip()
            changed += 1
    tmp = subjective_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(subjective_path)

    subjective = subjective_metrics(rows)
    objective = objective_metrics(read_jsonl(result_dir / "objective_predictions.jsonl"))
    metadata = {
        "model_name": args.model_name,
        "model": str(model_path),
        "adapter": str(adapter_path) if adapter_path else None,
    }
    subjective.update(metadata)
    objective.update(metadata)
    write_json(result_dir / "subjective_metrics.json", subjective)
    write_json(result_dir / "objective_metrics.json", objective)
    write_json(result_dir / "metrics.json", {"subjective": subjective, "objective": objective})
    print(json.dumps({"model": args.model_name, "truncated_to_1024": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
