#!/bin/bash
# Finishing worker: same commands as run_repeat.sh (+ minkprob from run_after.sh), but each step is skipped if its output already exists (resume after a restart).
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2; mkdir -p raw/repeat
T=${THREADS:-4}
lines=$(grep -E "python src/score_" run_repeat.sh | sed 's/^ *//')
for r in 0 1 2 3; do
  echo "$lines" | grep -F '$r' | while read -r line; do
    l=${line//\$r/$r}; out=$(echo "$l" | grep -oE "raw/repeat/[a-z0-9_]+\.npz"); [ -f "$out" ] || eval "$l"
  done
done
echo "$lines" | grep -vF '$r' | while read -r l; do out=$(echo "$l" | grep -oE "raw/repeat/[a-z0-9_]+\.npz"); [ -f "$out" ] || eval "$l"; done
echo REPEAT_DONE
for m in Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct; do f=results/contamination/minkprob__$(basename $m).json; [ -f $f ] || THREADS=4 python src/minkprob.py $m 500 2>&1 | grep -E "DONE|rror| agnews | bbc2025 "; done
echo EARLY_AFTER_DONE
