#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "run_clef_later.sh|run_queue2.sh" >/dev/null; do sleep 10; done
PYTHONDONTWRITEBYTECODE=1 "${CLEF_PY:-python3}" run_nameswap.py; echo NAMESWAP DONE
