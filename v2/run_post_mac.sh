#!/bin/bash
# Post-processing after Mac outputs are copied into raw/ (resumable; safe to rerun).
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
./analyze_pending.sh
for f in raw/decision__dummy raw/*__decision__*.npz; do [ -f "$f" ] || continue; b=$(basename $f .npz)   # T7 Clef: 200-item reduced design
  [ -f results/cells/${b}_sub200.json ] || UQ_TEST_SIZE=100 python src/analyze.py $f --tag _sub200 2>&1 | grep -E "DONE|rror"; done
python src/aggregate.py > /dev/null && python src/aggregate.py --with-api > /dev/null && python src/aggregate.py --det > /dev/null && echo AGG_OK
python src/repeat_analysis.py > logs/post_repeat.log 2>&1 && echo REP_OK
python src/report_tables.py > logs/post_tables.log 2>&1 && echo TAB_OK
python src/api_tables.py > logs/post_api_tables.log 2>&1 && echo APITAB_OK
python src/figures.py > logs/post_fig.log 2>&1 && echo FIG_OK
python src/api_figures.py > logs/post_apifig.log 2>&1 && echo APIFIG_OK
python src/mac_tables.py > /dev/null && echo MACTAB_OK
echo POST_DONE
