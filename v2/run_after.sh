#!/bin/bash
# Runs after the scoring queues: contamination proxy (after queue B) and repeatability (after queue A); then Qwen2.5-1.5B on AG/BBC subsets if both are done.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
until grep -q QUEUE_B_DONE logs/queue_b.log; do sleep 60; done
for m in Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct; do THREADS=4 python src/minkprob.py $m 500 2>&1 | grep -E "DONE|rror| agnews | bbc2025 "; done
until grep -q QUEUE_A_DONE logs/queue_a.log; do sleep 60; done
THREADS=4 ./run_repeat.sh
for ds in agnews bbc2025; do python src/score_llm.py Qwen/Qwen2.5-1.5B-Instruct $ds --items 600 --strings valid --dtype fp32 --threads 8 2>&1 | grep -E "DONE|rror"; done
THREADS=8 python src/minkprob.py Qwen/Qwen2.5-1.5B-Instruct 300 2>&1 | grep -E "DONE|rror"
echo AFTER_DONE
