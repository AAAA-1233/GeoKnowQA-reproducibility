#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

if [[ ! -f data/splits_seed42/manifest.json ]]; then
  python ../shared/prepare_data.py \
    --input-dir "../../datasets/GeoQA-train" \
    --output-dir data/splits_seed42 \
    --seed 42
fi

python src/evaluate_generation.py \
  --model base_models/Qwen2.5-7B-Instruct \
  --load-in-4bit \
  --output evaluations/before/predictions.jsonl \
  --batch-size "$EVAL_BATCH_SIZE"

python src/train_lora.py \
  --config configs/qwen2_5_7b_qlora.json \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACC_STEPS" \
  --resume auto

python src/evaluate_generation.py \
  --model base_models/Qwen2.5-7B-Instruct \
  --load-in-4bit \
  --adapter outputs/qwen2.5-7b-qlora/final_adapter \
  --output evaluations/after/predictions.jsonl \
  --batch-size "$EVAL_BATCH_SIZE"

python src/compare_metrics.py \
  --before evaluations/before/predictions.metrics.json \
  --after evaluations/after/predictions.metrics.json \
  --output evaluations/comparison.json
