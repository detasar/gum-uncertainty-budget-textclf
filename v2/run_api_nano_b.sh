#!/bin/bash
# gpt-4.1-nano queue B (resumable)
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py nano sst2 main --items 1200 --sets core --threads 16
python src/score_api.py nano massive main --items 1200 --sets core --threads 16
python src/score_api.py nano dbpedia main --items 1000 --sets core --threads 16
python src/score_api.py nano bbc2025 ord1 --items 600 --sets valid --threads 16
python src/score_api.py nano bbc2025 ord2 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews rt2 --items 200 --sets valid --threads 16
echo NANO_B_DONE
