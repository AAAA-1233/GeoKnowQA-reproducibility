#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.before, encoding="utf-8") as f:
        before = json.load(f)
    with open(args.after, encoding="utf-8") as f:
        after = json.load(f)
    names = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge-1", "rouge-2", "rouge-l", "token_f1", "exact_match"]
    comparison = {
        "before": before,
        "after": after,
        "delta": {name: after[name] - before[name] for name in names},
    }
    comparison["improved"] = comparison["delta"]["rouge-l"] > 0 and comparison["delta"]["bleu_4"] > 0
    atomic_write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

