#!/bin/bash
# Queue B: small LLMs by label log-likelihood (fp32). 4 threads.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
for m in Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct; do
  for ds in agnews bbc2025 sst2 massive dbpedia; do
    n=1200; st=all; [ $ds = dbpedia ] && n=1000; case $ds in sst2|dbpedia|massive) st=valid;; esac
    f=raw/${ds}__llm__$(basename $m).npz; [ -f $f ] || python src/score_llm.py $m $ds --items $n --strings $st --dtype fp32 --threads 4 2>&1 | grep -E "DONE|Error|rror"
  done
done
echo QUEUE_B_DONE
