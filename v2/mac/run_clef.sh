#!/bin/bash
# Mac queue 2 (MLX GPU): Clef-flash 8-bit, official + 30 VALID sets, first 200 items (cap per Davut, deviation), dataset-sequential (chunk 600 > items). Resumable.
cd ~/uq-budget-v2; export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export CLEF_DIR="$HOME/Library/Application Support/ClipboardGuard/models/dfa0993decb4f8507a0eae01afd1b2d33a4bb734"
"$HOME/Library/Application Support/ClipboardGuard/venv/bin/python" -B src/score_clef.py --items ${CLEF_N:-200} --sets valid --chunk 600 --ds ${CLEF_DS:-agnews,bbc2025,sst2,massive,dbpedia}
rc=$?; echo "CLEF_EXIT rc=$rc $(date)"   # 2026-10-29: the old unconditional CLEF_QUEUE_DONE marker was also printed after the process was killed
exit $rc
