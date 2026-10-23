#!/usr/bin/env python3
"""Export Qwen3 pre/post predictions on the exact MiniCPM5-2B 872-item set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/kg/AAA/实验补齐")
SOURCE = ROOT / "evaluations/minicpm5-2b/t5_overlap_exports/selected_872_before_after_matched_questions.jsonl"
BEFORE = ROOT / "evaluations/qwen3-0.6b/before/predictions_nothink_512.jsonl"
AFTER = ROOT / "evaluations/qwen3-0.6b/after/predictions_nothink_512.jsonl"
OUT = ROOT / "evaluations/qwen3-0.6b/2b_matched_872"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    selection = read_jsonl(SOURCE)
    if len(selection) != 872 or len({row["id"] for row in selection}) != 872:
        raise RuntimeError("The MiniCPM5-2B selection must contain 872 unique IDs.")
    before = {row["id"]: row for row in read_jsonl(BEFORE)}
    after = {row["id"]: row for row in read_jsonl(AFTER)}
    combined, before_out, after_out = [], [], []
    for rank, source in enumerate(selection, 1):
        item_id = source["id"]
        if item_id not in before or item_id not in after:
            raise RuntimeError(f"Missing Qwen3 prediction for {item_id}")
        b, a = before[item_id], after[item_id]
        for field in ("instruction", "reference", "source_file"):
            if b[field] != a[field]:
                raise RuntimeError(f"Pre/post metadata mismatch for {item_id}: {field}")
        # The reference selection is the authoritative identity of this comparison set.
        if b["instruction"] != source["question"] or b["reference"] != source["reference_answer"]:
            raise RuntimeError(f"2B selection mismatch for {item_id}")
        common = {"rank": rank, "id": item_id, "instruction": b["instruction"],
                  "reference": b["reference"], "source_file": b["source_file"]}
        before_out.append({**common, "prediction": b["prediction"]})
        after_out.append({**common, "prediction": a["prediction"]})
        combined.append({**common, "selection_source": source["selection_source"],
                         "before_prediction": b["prediction"], "after_prediction": a["prediction"]})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("before_872_matched_questions.json", before_out),
                       ("after_872_matched_questions.json", after_out)):
        (OUT / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "selected_872_before_after_matched_questions.jsonl").open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids = "\n".join(row["id"] for row in selection).encode()
    manifest = {"items": 872, "selection_reference": str(SOURCE),
                "id_order_sha256": hashlib.sha256(ids).hexdigest(),
                "selection_counts": {k: sum(r["selection_source"] == k for r in selection)
                                     for k in sorted({r["selection_source"] for r in selection})},
                "before": str(BEFORE), "after": str(AFTER)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
