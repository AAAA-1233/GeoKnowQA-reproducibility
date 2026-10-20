#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "${BASH_SOURCE[0]}")"
python src/train_lora.py \
  --config configs/qwen2_5_7b_qlora.json \
  --output-dir outputs/smoke-qwen2.5-7b-qlora \
  --resume none \
  --train-limit 512 \
  --max-steps 5
