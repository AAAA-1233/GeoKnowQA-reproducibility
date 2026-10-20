#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
huggingface-cli download IDEA-CCNL/Randeng-T5-784M-QA-Chinese \
  --local-dir "$root/training/randeng_t5_784m_qa_chinese_lora/base_models/Randeng-T5-784M-QA-Chinese"
