#!/bin/bash
# gpt-6-luna queue (resumable). Pilot showed bitwise-identical probabilities for 1, 8, 31 questions/request -> multi-question requests.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
python src/score_api.py luna agnews main --items 1200 --sets main --qpr 60 --threads 12
python src/score_api.py luna bbc2025 main --items 1200 --sets main --qpr 60 --threads 12
python src/score_api.py luna sst2 main --items 1200 --sets core --qpr 60 --threads 12
python src/score_api.py luna massive main --items 1200 --sets core --qpr 60 --threads 12
python src/score_api.py luna dbpedia main --items 1000 --sets core --qpr 60 --threads 12
for ds in agnews bbc2025; do for o in ord1 ord2; do python src/score_api.py luna $ds $o --items 600 --sets valid --qpr 31 --threads 12; done; done
for r in rt1 rt2; do python src/score_api.py luna agnews $r --items 200 --sets valid --qpr 31 --threads 12; done
echo LUNA_DONE
