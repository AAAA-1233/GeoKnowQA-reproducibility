#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
huggingface-cli download internlm/internlm2_5-1_8b-chat \
  --local-dir "$root/training/internlm2_5_1_8b_lora/base_models/InternLM2_5-1_8B-Chat"
