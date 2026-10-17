#!/bin/bash
# Added by the finishing worker (2026-10-17 ~15:30 Istanbul): runs the cheap post-queue steps of run_after.sh early,
# because the scoring queues will not finish within the time budget. Same commands/arguments as run_after.sh / run_repeat.sh.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
THREADS=4 ./run_repeat.sh
for m in Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct; do THREADS=4 python src/minkprob.py $m 500 2>&1 | grep -E "DONE|rror| agnews | bbc2025 "; done
echo EARLY_AFTER_DONE
