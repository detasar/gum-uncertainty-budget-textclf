#!/bin/bash
# gpt-4.1-nano queue D: starts after queue C finishes; takes AG News order/retest conditions (queue A skips them via the cache lock or finds them done)
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
until grep -q NANO_C_DONE logs/api/nano_c.log; do sleep 30; done
python src/score_api.py nano agnews ord1 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews ord2 --items 600 --sets valid --threads 16
python src/score_api.py nano agnews rt1 --items 200 --sets valid --threads 16
echo NANO_D_DONE
