#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "geobenchmark_npee.json"
OUT = ROOT / "test"
OBJECTIVE_TYPES = ("choice", "completion", "tf")
SUBJECTIVE_TYPES = ("noun", "qa", "discussion")
TYPE_NAMES = {
    "noun": "名词解释",
    "choice": "选择题",
    "completion": "填空题",
    "tf": "判断题",
    "qa": "简答题",
    "discussion": "论述题",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def flatten(raw: dict, types: tuple[str, ...], category: str) -> list[dict]:
    rows: list[dict] = []
    for question_type in types:
        section = raw[question_type]
        questions = section["question"]
        answers = section["answer"]
        if len(questions) != len(answers):
            raise ValueError(f"Mismatched question/answer count for {question_type}")
        for index, (question, answer) in enumerate(zip(questions, answers)):
            rows.append(
                {
                    "id": f"npee-{question_type}-{index + 1:04d}",
                    "category": category,
                    "question_type": question_type,
                    "question_type_zh": TYPE_NAMES[question_type],
                    "source_index": index,
                    "question": question,
                    "answer": answer,
                }
            )
    return rows


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    expected_types = set(OBJECTIVE_TYPES + SUBJECTIVE_TYPES)
    if set(raw) != expected_types:
        raise ValueError(f"Unexpected question types: {sorted(raw)}")

    OUT.mkdir(parents=True, exist_ok=True)
    objective = flatten(raw, OBJECTIVE_TYPES, "objective")
    subjective = flatten(raw, SUBJECTIVE_TYPES, "subjective")
    write_json(OUT / "objective.json", objective)
    write_jsonl(OUT / "objective.jsonl", objective)
    write_json(OUT / "subjective.json", subjective)
    write_jsonl(OUT / "subjective.jsonl", subjective)

    all_rows = objective + subjective
    manifest = {
        "dataset": "GeoBenchmark-NPEE",
        "source_repository": "https://huggingface.co/datasets/daven3/geobench",
        "source_file": "geobenchmark_npee.json",
        "source_sha256": sha256(RAW),
        "classification": {
            "objective": list(OBJECTIVE_TYPES),
            "subjective": list(SUBJECTIVE_TYPES),
        },
        "counts": {
            "total": len(all_rows),
            "objective": len(objective),
            "subjective": len(subjective),
            "by_type": dict(Counter(row["question_type"] for row in all_rows)),
        },
        "outputs_sha256": {
            name: sha256(OUT / name)
            for name in (
                "objective.json",
                "objective.jsonl",
                "subjective.json",
                "subjective.jsonl",
            )
        },
    }
    write_json(ROOT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
