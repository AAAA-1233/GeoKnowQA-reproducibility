#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from common import atomic_write_json


INSTRUCTION_PREFIX_RE = re.compile(r"^\s*\u6df1\u5ea6\u89e3\u8bfb\s*[：:]\s*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean GeoQA and create one immutable split shared by every model.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def canonical_hash(row: dict) -> str:
    payload = json.dumps(
        {k: row.get(k, "") for k in ("instruction", "input", "output", "system")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to replace fixed split: {manifest_path}. Use --overwrite explicitly.")

    rows: list[dict] = []
    field_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for source_path in sorted(input_dir.glob("*.json"), key=lambda p: p.name):
        with open(source_path, "r", encoding="utf-8") as f:
            source_rows = json.load(f)
        if not isinstance(source_rows, list):
            raise ValueError(f"Expected a JSON array: {source_path}")
        for source_index, raw in enumerate(source_rows):
            if not isinstance(raw, dict):
                raise ValueError(f"Expected object at {source_path}:{source_index}")
            row = {k: str(raw.get(k, "")) for k in ("instruction", "input", "output", "system")}
            row["instruction"] = INSTRUCTION_PREFIX_RE.sub("", row["instruction"], count=1).strip()
            if not row["instruction"].strip() or not row["output"].strip():
                raise ValueError(f"Empty instruction/output at {source_path}:{source_index}")
            row["id"] = hashlib.sha256(f"{source_path.name}:{source_index}".encode()).hexdigest()[:20]
            row["source_file"] = source_path.name
            row["source_index"] = source_index
            row["content_sha256"] = canonical_hash(row)
            field_counts.update(raw.keys())
            source_counts[source_path.name] += 1
            rows.append(row)

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    n_total = len(indices)
    n_train = round(n_total * 0.90)
    n_validation = round(n_total * 0.05)
    split_indices = {
        "train": indices[:n_train],
        "validation": indices[n_train : n_train + n_validation],
        "test": indices[n_train + n_validation :],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    for split, selected in split_indices.items():
        path = output_dir / f"{split}.jsonl"
        digest = hashlib.sha256()
        with open(path, "w", encoding="utf-8") as f:
            for i in selected:
                line = json.dumps(rows[i], ensure_ascii=False, separators=(",", ":")) + "\n"
                f.write(line)
                digest.update(line.encode("utf-8"))
        split_hashes[split] = digest.hexdigest()

    manifest = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "seed": args.seed,
        "algorithm": "sort filenames; preserve source order; random.Random(seed).shuffle(global indices); rounded 90/5/5 boundaries",
        "counts": {k: len(v) for k, v in split_indices.items()} | {"total": n_total},
        "ratios": {k: len(v) / n_total for k, v in split_indices.items()},
        "sha256": split_hashes,
        "source_counts": dict(source_counts),
        "observed_fields": dict(field_counts),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
