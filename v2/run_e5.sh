#!/bin/bash
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
T=1 ./analyze_pending.sh
for ds in agnews bbc2025 sst2 dbpedia massive; do [ -f results/cells/${ds}__sup__e5-base-v2.json ] || python src/sup.py $ds e5-base-v2 2>&1 | grep -E "DONE|rror"; done
echo E5_DONE
