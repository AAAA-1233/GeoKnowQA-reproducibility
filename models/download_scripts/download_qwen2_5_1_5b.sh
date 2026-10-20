#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir "$root/training/qwen1_5b_lora/base_models/Qwen2.5-1.5B-Instruct"
