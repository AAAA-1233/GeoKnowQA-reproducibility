#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f data/splits_seed42/manifest.json ]]; then
  python ../shared/prepare_data.py \
    --input-dir "../../datasets/GeoQA-train" \
    --output-dir data/splits_seed42 \
    --seed 42
fi

python src/evaluate_generation.py \
  --output evaluations/before/predictions.jsonl \
  --batch-size 64 \
  --max-new-tokens 256 \
  --no-repeat-ngram-size 4 \
  --repetition-penalty 1.10

python src/train_lora.py \
  --config configs/randeng_t5_784m_lora.json \
  --resume auto

python src/evaluate_generation.py \
  --adapter outputs/randeng-t5-784m-lora/final_adapter \
  --output evaluations/after/predictions.jsonl \
  --batch-size 64 \
  --max-new-tokens 256 \
  --no-repeat-ngram-size 4 \
  --repetition-penalty 1.10

python src/compare_metrics.py \
  --before evaluations/before/predictions.metrics.json \
  --after evaluations/after/predictions.metrics.json \
  --output evaluations/comparison.json
