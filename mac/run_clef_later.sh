#!/bin/bash
# Clef jobs run alone on the GPU (concurrent MLX + llama.cpp Metal jobs stalled each other).
cd "$(dirname "$0")"; export PYTHONDONTWRITEBYTECODE=1
PY="${CLEF_PY:-python3}"
while pgrep -f run_queue2.sh >/dev/null; do sleep 10; done
date; "$PY" run_repeat_clef.py; "$PY" run_clef_calx.py; date; echo CLEF LATER DONE
