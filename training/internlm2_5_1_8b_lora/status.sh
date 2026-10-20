#!/usr/bin/env bash
set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== pipeline ==="
tmux has-session -t internlm18 2>/dev/null && echo "running (tmux: internlm18)" || echo "not running"
echo "=== GPU ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo "=== predictions ==="
for f in evaluations/before/predictions.jsonl evaluations/after/predictions.jsonl; do
  if [[ -f "$f" ]]; then printf '%s: %s / 8723\n' "$f" "$(wc -l < "$f")"; fi
done
echo "=== latest checkpoint ==="
find outputs/internlm2.5-1.8b-lora -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1
echo "=== recent log ==="
tail -20 pipeline.log 2>/dev/null || true
