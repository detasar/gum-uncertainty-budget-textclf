#!/bin/bash
# gpt-4.1-nano queue C (added to parallelise; per-cache file locks prevent two processes scoring the same condition)
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py nano dbpedia main --items 1000 --sets core --threads 16
python src/score_api.py nano bbc2025 ord1 --items 600 --sets valid --threads 16
python src/score_api.py nano bbc2025 ord2 --items 600 --sets valid --threads 16
echo NANO_C_DONE
