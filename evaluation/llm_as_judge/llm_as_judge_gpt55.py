#!/usr/bin/env python3
"""Score prediction JSONL records with GPT-5.5 using the project judge prompt."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


SCORE_RE = re.compile(r"^(?:10|[0-9])\.[0-9]$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with question/reference/prediction fields.")
    parser.add_argument("--output", required=True, help="JSONL path for judge scores; resumes by id.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate records and prompts without calling the API.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_text(row: dict, names: tuple[str, ...]) -> str:
    for name in names:
        if row.get(name) is not None:
            return str(row[name])
    raise KeyError(f"Missing all supported fields: {names}")


def main() -> None:
    args = parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    prompt_template = (Path(__file__).with_name("prompt.txt")).read_text(encoding="utf-8")
    rows = read_jsonl(input_path)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    completed = {str(row["id"]): row for row in read_jsonl(output_path)} if output_path.exists() else {}
    pending = [row for row in rows if str(row["id"]) not in completed]
    print(f"total={len(rows)} completed={len(completed)} pending={len(pending)}", flush=True)

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required unless --dry-run is used.")
        from openai import OpenAI
        client = OpenAI()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for index, row in enumerate(pending, start=1):
            question = first_text(row, ("question", "instruction"))
            reference = first_text(row, ("reference_answer", "reference", "output"))
            prediction = first_text(row, ("model_answer", "prediction"))
            prompt = prompt_template.format(
                question=question, reference_answer=reference, model_answer=prediction
            )
            result = {
                "id": str(row["id"]),
                "question": question,
                "reference_answer": reference,
                "model_answer": prediction,
                "judge_model": args.model,
            }
            if args.dry_run:
                result["status"] = "dry_run"
            else:
                response = client.responses.create(
                    model=args.model,
                    input=prompt,
                    max_output_tokens=8,
                    store=False,
                )
                raw = response.output_text.strip()
                result["raw_judge_output"] = raw
                if SCORE_RE.fullmatch(raw):
                    result["score"] = float(raw)
                    result["status"] = "ok"
                else:
                    result["status"] = "invalid_output"
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"processed {index}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
