#!/bin/bash
# Added by the finishing worker (deviation, see report): the full NLI BBC-2025 cell (1200 items, all strings) will not be reached by queue A
# within the time budget, so a 600-item / VALID-strings subset is scored to a separate path (does not interfere with queue A's file check).
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_nli.py cross-encoder/nli-deberta-v3-small bbc2025 --items 600 --strings valid --dtype fp32 --bs 64 --threads 2 --out raw/subset/bbc2025__nli__nli-deberta-v3-small.npz 2>&1 | grep -E "DONE|rror"
echo NLI_BBC_SUB_DONE
