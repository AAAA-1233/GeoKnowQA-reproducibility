#!/usr/bin/env bash
# Evaluate the base and the LoRA-adapted MiniCPM5-2B under the same concise-QA policy.
set -euo pipefail

ROOT="/home/kg/AAA/实验补齐"
PYTHON="/opt/anaconda3/envs/geogpt3.10/bin/python"
MODEL="$ROOT/base_models/MiniCPM5-2B"
ADAPTER="$ROOT/outputs/minicpm5-2b-lora/final_adapter"
TEST_FILE="$ROOT/data/splits_seed42_no_watermark/test.jsonl"
EVALUATOR="$ROOT/evaluate_minicpm_generation.py"

COMMON=(--model "$MODEL" --test-file "$TEST_FILE" --batch-size 128
        --max-new-tokens 512 --repetition-penalty 1.15 --no-repeat-ngram-size 4)

cd "$ROOT"
"$PYTHON" "$EVALUATOR" "${COMMON[@]}" \
  --output evaluations/minicpm5-2b/before/predictions_nothink_512.jsonl

"$PYTHON" "$EVALUATOR" "${COMMON[@]}" --adapter "$ADAPTER" \
  --output evaluations/minicpm5-2b/after/predictions_nothink_512.jsonl
