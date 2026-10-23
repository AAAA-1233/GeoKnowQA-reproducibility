#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAINER="$ROOT/src/train_lora.py"
CONFIG="$ROOT/configs/qwen3_0_6b_lora.json"
cd "$ROOT"
if [[ ! -f "$ROOT/data/splits_seed42_no_watermark/manifest.json" ]]; then
  python "$ROOT/../shared/prepare_data.py" \
    --input-dir "$ROOT/../../datasets/GeoKnowQA-train" \
    --output-dir "$ROOT/data/splits_seed42_no_watermark" --seed 42
fi
python "$TRAINER" --config "$CONFIG" --resume auto
