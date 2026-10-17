#!/bin/bash
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="${CLEF_PY:-python3}"
date; "$PY" run_clef.py
date; python3 run_ollama.py qwen3.5:9b w0_official,w1_news,w2_topic,w3_short 1
/usr/local/bin/ollama stop qwen3.5:9b 2>/dev/null
date; python3 run_ollama.py qwen38-q5-fast:latest w0_official 1
/usr/local/bin/ollama stop qwen38-q5-fast:latest 2>/dev/null
date; "$PY" run_clef.py cross
date; echo ALL DONE
