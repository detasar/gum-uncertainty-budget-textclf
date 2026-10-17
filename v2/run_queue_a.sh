#!/bin/bash
# Queue A: bi-encoders (full pools), then NLI cross-encoders (first 1200 items; DBpedia 1000). 4 threads.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
for m in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 intfloat/e5-base-v2; do
  for ds in agnews bbc2025 sst2 dbpedia massive; do
    f=raw/${ds}__bienc__$(basename $m).npz; [ -f $f ] || python src/score_bienc.py $m $ds --threads 4 2>&1 | grep -E "DONE|Error|rror" 
  done
done
for m in cross-encoder/nli-deberta-v3-small MoritzLaurer/ModernBERT-base-zeroshot-v2.0; do
  for ds in agnews bbc2025 sst2 dbpedia massive; do
    n=1200; st=all; [ $ds = dbpedia ] && n=1000; case $ds in sst2|dbpedia|massive) st=valid;; esac
    f=raw/${ds}__nli__$(basename $m).npz; [ -f $f ] || python src/score_nli.py $m $ds --items $n --strings $st --dtype fp32 --bs 64 --threads 4 2>&1 | grep -E "DONE|Error|rror"
  done
done
echo QUEUE_A_DONE
