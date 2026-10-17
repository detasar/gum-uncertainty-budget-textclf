#!/bin/bash
# Luna pilot: 100 AG News items x (official + 30 valid sets); 1 question/request vs 31 questions/request, each run twice.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py luna agnews pilot_q1 --items 100 --sets valid --qpr 1 --threads 16
python src/score_api.py luna agnews pilot_q31 --items 100 --sets valid --qpr 31 --threads 16
python src/score_api.py luna agnews pilot_q1b --items 100 --sets valid --qpr 1 --threads 16
python src/score_api.py luna agnews pilot_q31b --items 100 --sets valid --qpr 31 --threads 16
python src/score_api.py luna agnews pilot_q8 --items 100 --sets valid --qpr 8 --threads 16
echo PILOT_DONE
