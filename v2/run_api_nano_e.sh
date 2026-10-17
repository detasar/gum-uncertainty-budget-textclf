#!/bin/bash
# gpt-4.1-nano queue E: MASSIVE main (queue B skips it via the cache lock while E holds it), then AG News rt2
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py nano massive main --items 1200 --sets core --threads 10
python src/score_api.py nano agnews rt2 --items 200 --sets valid --threads 10
echo NANO_E_DONE
