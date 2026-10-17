#!/bin/bash
# Phase 1: wait for the running qwen3.5:9b main job; Phase 2: Clef jobs || 9B repeatability; Phase 3: 27B main + repeatability
cd "$(dirname "$0")"; export PYTHONDONTWRITEBYTECODE=1
PY="${CLEF_PY:-python3}"
while pgrep -f "run_ollama.py qwen3.5" >/dev/null; do sleep 10; done
date; echo PHASE2
( "$PY" run_clef.py cross; "$PY" run_clef_calx.py; "$PY" run_repeat_clef.py ) > out/clef_phase2.log 2>&1 &
CP=$!
python3 run_repeat_llm.py qwen3.5:9b w0_official 150 t0 4
python3 run_repeat_llm.py qwen3.5:9b w0_official 150 default 4
python3 run_repeat_llm.py qwen3.5:9b w2_topic 50 t0 4
/usr/local/bin/ollama stop qwen3.5:9b
wait $CP
date; echo PHASE3
python3 run_ollama.py qwen38-q5-fast:latest w0_official 1
python3 run_repeat_llm.py qwen38-q5-fast:latest w0_official 50 t0 4
python3 run_repeat_llm.py qwen38-q5-fast:latest w0_official 50 default 4
/usr/local/bin/ollama stop qwen38-q5-fast:latest
date; echo ALL DONE
