#!/usr/bin/env bash
# Matched pre/post SFT generation evaluation for Qwen3-0.6B.
set -euo pipefail

ROOT="/home/kg/AAA/实验补齐"
PYTHON="/opt/anaconda3/envs/geogpt3.10/bin/python"
MODEL="$ROOT/base_models/Qwen3-0.6B"
ADAPTER="$ROOT/outputs/qwen3-0.6b-lora/final_adapter"
TEST="$ROOT/data/splits_seed42_no_watermark/test.jsonl"
OUTDIR="$ROOT/evaluations/qwen3-0.6b"
EVALUATOR="$ROOT/evaluate_minicpm_generation.py"
mkdir -p "$OUTDIR/before" "$OUTDIR/after"

COMMON=(--model "$MODEL" --test-file "$TEST" --batch-size 32
        --max-new-tokens 512 --repetition-penalty 1.15 --no-repeat-ngram-size 4)
cd "$ROOT"
"$PYTHON" "$EVALUATOR" "${COMMON[@]}" \
  --output "$OUTDIR/before/predictions_nothink_512.jsonl"
"$PYTHON" add_distinct_metrics.py \
  --predictions "$OUTDIR/before/predictions_nothink_512.jsonl" \
  --metrics "$OUTDIR/before/predictions_nothink_512.metrics.json"

"$PYTHON" "$EVALUATOR" "${COMMON[@]}" --adapter "$ADAPTER" \
  --output "$OUTDIR/after/predictions_nothink_512.jsonl"
"$PYTHON" add_distinct_metrics.py \
  --predictions "$OUTDIR/after/predictions_nothink_512.jsonl" \
  --metrics "$OUTDIR/after/predictions_nothink_512.metrics.json"
"$PYTHON" compare_all_metrics.py \
  --before "$OUTDIR/before/predictions_nothink_512.metrics.json" \
  --after "$OUTDIR/after/predictions_nothink_512.metrics.json" \
  --output "$OUTDIR/comparison_nothink_512.json"
