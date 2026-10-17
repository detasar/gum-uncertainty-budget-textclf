#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f run_all.sh >/dev/null; do sleep 20; done
PYTHONDONTWRITEBYTECODE=1 "${CLEF_PY:-python3}" run_clef_calx.py
date; echo AFTER DONE
