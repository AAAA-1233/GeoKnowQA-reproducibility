#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
target=11211

echo "[$(date '+%F %T')] guard started; waiting for $target baseline rows and metrics"
while true; do
  rows=$(wc -l < evaluations/before/predictions.jsonl 2>/dev/null || echo 0)
  if (( rows >= target )) && [[ -s evaluations/before/predictions.metrics.json ]]; then
    break
  fi
  echo "[$(date '+%F %T')] baseline rows=$rows/$target"
  sleep 30
done

echo "[$(date '+%F %T')] baseline and metrics complete"

# The sequential pipeline normally reaps this process itself. Give it time to exit;
# terminate only a genuine leftover baseline process before training is allowed.
for _ in {1..30}; do
  mapfile -t eval_pids < <(
    ps -eo pid=,args= | awk \
      '/python src\/evaluate_generation.py/ && /evaluations\/before\/predictions.jsonl/ {print $1}'
  )
  ((${#eval_pids[@]} == 0)) && break
  sleep 2
done
if ((${#eval_pids[@]})); then
  echo "[$(date '+%F %T')] terminating leftover baseline PIDs: ${eval_pids[*]}"
  kill -TERM "${eval_pids[@]}"
fi

for _ in {1..90}; do
  train_pid=$(
    ps -eo pid=,args= | awk \
      '/python src\/train_lora.py/ && /qwen2_5_7b_qlora.json/ {print $1; exit}'
  )
  if [[ -n "$train_pid" ]]; then
    echo "[$(date '+%F %T')] training confirmed, PID=$train_pid"
    exit 0
  fi
  sleep 2
done

echo "[$(date '+%F %T')] ERROR: baseline finished but training did not start" >&2
exit 1
