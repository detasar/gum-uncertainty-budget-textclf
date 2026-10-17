#!/bin/bash
# gpt-4.1-nano queue A (resumable: rerun to continue; finished (item,set) pairs are skipped)
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py nano agnews main --items 1200 --sets core --threads 16
python src/score_api.py nano bbc2025 main --items 1200 --sets core --threads 16
python src/score_api.py nano agnews ord1 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews ord2 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews rt1 --items 200 --sets valid --threads 16
echo NANO_A_DONE
