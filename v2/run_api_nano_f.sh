#!/bin/bash
# gpt-4.1-nano queue F: AG News order/retest conditions started early (queue D finds them locked or done)
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
sleep 30
python src/score_api.py nano agnews ord1 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews rt1 --items 200 --sets valid --threads 16
python src/score_api.py nano agnews ord2 --items 600 --sets valid --threads 16
echo NANO_F_DONE
