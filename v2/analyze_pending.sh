#!/bin/bash
# Analyse every raw score file that has no result yet (main analysis + sensitivity variants for bi-encoders/LLMs).
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/api_build.py --cond main 2>&1 | grep -E "BUILT|rror"   # T5 API cells (exploratory): assemble finished JSONL caches
export OMP_NUM_THREADS=${T:-2} OPENBLAS_NUM_THREADS=${T:-2}
for f in raw/*__nli__*.npz raw/*__bienc__*.npz raw/*__llm__*.npz raw/*__api__*.npz; do
  [ -f "$f" ] || continue
  b=$(basename $f .npz); [ -f results/cells/$b.json ] || python src/analyze.py $f 2>&1 | grep -E "DONE|Error|rror"
  case $b in *__bienc__*) for tau in 0.02 0.1; do [ -f results/cells/${b}_tau$tau.json ] || python src/analyze.py $f --tau $tau --tag _tau$tau --quick 2>&1 | grep -E "DONE|rror"; done;; esac
  case $b in *__llm__*) [ -f results/cells/${b}_rawll.json ] || python src/analyze.py $f --llm-mode raw --tag _rawll --quick 2>&1 | grep -E "DONE|rror";; esac
done
