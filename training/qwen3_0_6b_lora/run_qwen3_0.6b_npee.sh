#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/kg/AAA/实验补齐"
PYTHON="/opt/anaconda3/envs/geogpt3.10/bin/python"
MODEL="$ROOT/base_models/Qwen3-0.6B"
ADAPTER="$ROOT/outputs/qwen3-0.6b-lora/final_adapter"
DATA='/media/kg/WD_BLACK/qa paper relate/geobenchmark-NPEE/test/objective.jsonl'
OUT="$ROOT/evaluations/qwen3-0.6b/npee"
mkdir -p "$OUT"
cd "$ROOT"
"$PYTHON" evaluate_npee_objective.py --model "$MODEL" --data "$DATA" --output "$OUT/before.jsonl" --batch-size 16 --disable-thinking
"$PYTHON" evaluate_npee_objective.py --model "$MODEL" --adapter "$ADAPTER" --data "$DATA" --output "$OUT/after.jsonl" --batch-size 16 --disable-thinking
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('/home/kg/AAA/实验补齐/evaluations/qwen3-0.6b/npee')
before=json.loads((root/'before.summary.json').read_text())
after=json.loads((root/'after.summary.json').read_text())
(root/'comparison.json').write_text(json.dumps({'before':before,'after':after,'delta_accuracy':after['accuracy']-before['accuracy']},ensure_ascii=False,indent=2)+'\n')
PY
