#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$ROOT/data/splits_seed42_no_watermark/manifest.json" ]]; then
  python "$ROOT/../shared/prepare_data.py" \
    --input-dir "$ROOT/../../datasets/GeoKnowQA-train" \
    --output-dir "$ROOT/data/splits_seed42_no_watermark" --seed 42
fi
python "$ROOT/src/train_lora.py" --config "$ROOT/configs/minicpm5_2b_lora.json" --resume auto
