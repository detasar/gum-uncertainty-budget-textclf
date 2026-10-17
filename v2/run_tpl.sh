#!/bin/bash
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2
export OMP_NUM_THREADS=1
for m in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 intfloat/e5-base-v2; do for ds in agnews bbc2025 sst2 dbpedia massive; do
  b=$(basename $m)
  [ -f raw/templates/${ds}__bienc__${b}__tpl1.npz ] || python src/score_bienc.py $m $ds --threads 1 --strings valid --template "{}" --out raw/templates/${ds}__bienc__${b}__tpl1.npz 2>&1 | grep -E "DONE|rror"
  [ -f raw/templates/${ds}__bienc__${b}__tpl2.npz ] || python src/score_bienc.py $m $ds --threads 1 --strings valid --template "Category: {}." --out raw/templates/${ds}__bienc__${b}__tpl2.npz 2>&1 | grep -E "DONE|rror"
done; done
python src/template_factor.py
echo TPL_DONE
