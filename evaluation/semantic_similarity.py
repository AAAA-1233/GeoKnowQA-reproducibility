#!/usr/bin/env python3
"""Semantic-similarity audit for instruction-answer records."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for source_path in sorted(Path(args.input_dir).glob("*.json")):
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        for source_index, row in enumerate(source_rows):
            instruction = re.sub(r"^\s*\u6df1\u5ea6\u89e3\u8bfb\s*[：:]\s*", "", str(row.get("instruction", "")), count=1).strip()
            text = "\n".join((instruction, str(row.get("input", "")).strip(), str(row.get("output", "")).strip()))
            rows.append({"id": f"{source_path.name}:{source_index}", "text": text})

    encoder = SentenceTransformer(args.model)
    embeddings = encoder.encode(
        [row["text"] for row in rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    scores, indices = index.search(embeddings, min(args.neighbors + 1, len(rows)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id_a", "id_b", "cosine_similarity"))
        writer.writeheader()
        emitted: set[tuple[str, str]] = set()
        for row_index, row in enumerate(rows):
            for score, neighbor in zip(scores[row_index, 1:], indices[row_index, 1:]):
                if score < args.threshold:
                    continue
                pair = tuple(sorted((row["id"], rows[int(neighbor)]["id"])))
                if pair not in emitted:
                    emitted.add(pair)
                    writer.writerow({"id_a": pair[0], "id_b": pair[1], "cosine_similarity": f"{float(score):.6f}"})
    print(f"records={len(rows)} similar_pairs={len(emitted)} output={output}")


if __name__ == "__main__":
    main()
