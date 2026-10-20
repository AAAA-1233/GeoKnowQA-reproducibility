#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen2.5-1.5b", "internlm2.5-1.8b", "qwen2.5-7b"]


def flatten_numeric(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value


long_rows = []
by_model = {}
for model in MODELS:
    by_model[model] = {}
    for stage, folder in (("before", ROOT / "results" / "base" / model), ("after", ROOT / "results" / model)):
        metrics = json.load((folder / "metrics.json").open(encoding="utf-8"))
        flat = {}
        flatten_numeric("", metrics, flat)
        row = {"model": model, "stage": stage, **flat}
        long_rows.append(row)
        by_model[model][stage] = flat

wide_rows = []
for model in MODELS:
    before, after = by_model[model]["before"], by_model[model]["after"]
    row = {"model": model}
    for metric in sorted(set(before) | set(after)):
        if metric.endswith("samples") or metric.endswith("correct") or metric.endswith("invalid_answers"):
            continue
        b, a = before.get(metric), after.get(metric)
        if b is None or a is None:
            continue
        row[f"{metric}__before"] = b
        row[f"{metric}__after"] = a
        row[f"{metric}__delta"] = a - b
    wide_rows.append(row)

results_dir = ROOT / "results"
pd.DataFrame(long_rows).to_csv(results_dir / "before_after_all_metrics.csv", index=False)
pd.DataFrame(wide_rows).to_csv(results_dir / "before_after_comparison.csv", index=False)
with (results_dir / "before_after_comparison.json").open("w", encoding="utf-8") as f:
    json.dump(wide_rows, f, ensure_ascii=False, indent=2)
    f.write("\n")

notes = pd.DataFrame([
    ["名词解释/简答/论述", "BLEU-1/2/3/4、ROUGE-1/2/L、Token-F1、Exact Match"],
    ["选择题", "Accuracy、有效答案率"],
    ["判断题", "True/False Accuracy、有效答案率"],
    ["填空题", "BLEU-1/2/3/4、ROUGE-1/2/L、Token-F1、Exact Match"],
], columns=["题型", "评分标准"])
with pd.ExcelWriter(results_dir / "NPEE微调前后对比总表.xlsx", engine="openpyxl") as writer:
    pd.DataFrame(wide_rows).to_excel(writer, sheet_name="前后差值", index=False)
    pd.DataFrame(long_rows).to_excel(writer, sheet_name="完整指标", index=False)
    notes.to_excel(writer, sheet_name="评分标准", index=False)
print(results_dir / "NPEE微调前后对比总表.xlsx")
