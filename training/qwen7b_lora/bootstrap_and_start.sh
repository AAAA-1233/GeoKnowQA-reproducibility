#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[$(date '+%F %T')] waiting for model download"
while ! grep -q 'MODEL_DOWNLOAD_COMPLETE' download.log 2>/dev/null; do
  if ! tmux has-session -t geoqa_qwen7b_download 2>/dev/null; then
    echo "model download stopped before completion" >&2
    exit 1
  fi
  sleep 30
done

python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

root = Path("base_models/Qwen2.5-7B-Instruct")
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 4, shards
for shard in shards:
    with safe_open(root / shard, framework="pt", device="cpu") as f:
        assert list(f.keys()), shard
print("model shards validated:", shards)
PY

selected_train_batch=""
selected_grad_acc=""
for train_batch in 8 4 2 1; do
  grad_acc=$((64 / train_batch))
  smoke_dir="outputs/smoke-b${train_batch}-ga${grad_acc}"
  echo "[$(date '+%F %T')] training smoke test: batch=$train_batch grad_acc=$grad_acc"
  if python src/train_lora.py \
      --config configs/qwen2_5_7b_qlora.json \
      --output-dir "$smoke_dir" \
      --resume none --train-limit 512 --max-steps 3 \
      --train-batch-size "$train_batch" \
      --gradient-accumulation-steps "$grad_acc"; then
    selected_train_batch="$train_batch"
    selected_grad_acc="$grad_acc"
    break
  fi
done
[[ -n "$selected_train_batch" ]] || { echo "all training smoke tests failed" >&2; exit 1; }

selected_eval_batch=""
for eval_batch in 16 8 4 2 1; do
  smoke_output="evaluations/smoke/base-b${eval_batch}.jsonl"
  echo "[$(date '+%F %T')] generation smoke test: batch=$eval_batch"
  if python src/evaluate_generation.py \
      --model base_models/Qwen2.5-7B-Instruct --load-in-4bit \
      --output "$smoke_output" --batch-size "$eval_batch" --limit 32; then
    selected_eval_batch="$eval_batch"
    break
  fi
done
[[ -n "$selected_eval_batch" ]] || { echo "all generation smoke tests failed" >&2; exit 1; }

{
  echo "TRAIN_BATCH_SIZE=$selected_train_batch"
  echo "GRAD_ACC_STEPS=$selected_grad_acc"
  echo "EVAL_BATCH_SIZE=$selected_eval_batch"
  echo "EFFECTIVE_BATCH_SIZE=64"
} > selected_runtime.env

echo "[$(date '+%F %T')] smoke tests passed; starting full pipeline"
TRAIN_BATCH_SIZE="$selected_train_batch" \
GRAD_ACC_STEPS="$selected_grad_acc" \
EVAL_BATCH_SIZE="$selected_eval_batch" \
  bash run_pipeline.sh > pipeline.log 2>&1
