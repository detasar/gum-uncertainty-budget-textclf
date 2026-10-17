#!/bin/bash
# Mac queue 1 (torch/MPS): venv setup, cross-platform check (AG deberta, full cell), remaining NLI cells, then GLiNER2.5-Decide cells.
# Resumable: rerun to continue (finished npz/jsonl items are skipped; NLI checkpoints every ~2 min).
cd ~/uq-budget-v2; export PYTHONDONTWRITEBYTECODE=1 HF_HOME=~/uq-budget-v2/hf TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p raw raw_xplat logs
if [ ! -x venv/bin/python ]; then
  ~/.local/bin/python3.12 -m venv venv && venv/bin/pip install -q torch==2.14.1 transformers==5.20.1 numpy==2.5.4 sentencepiece protobuf 2>&1 | tail -3
fi
PY=venv/bin/python; $PY -c "import torch,transformers,numpy;print('ENV',torch.__version__,transformers.__version__,numpy.__version__,'mps',torch.backends.mps.is_available())"
t() { local s=$(date +%s); "$@"; echo "TIME $(( $(date +%s)-s ))s :: $*"; }
f=raw_xplat/agnews__nli__nli-deberta-v3-small.npz
[ -f $f ] || t $PY src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 1200 --strings all --dtype fp32 --bs 64 --out $f 2>&1 | grep -vE "^[0-9]+/" 
for m in cross-encoder/nli-deberta-v3-small MoritzLaurer/ModernBERT-base-zeroshot-v2.0; do
  for ds in ${NLI_DS:-dbpedia massive agnews bbc2025 sst2}; do
    n=1200; st=all; [ $ds = dbpedia ] && n=1000; case $ds in sst2|dbpedia|massive) st=valid;; esac
    f=raw/${ds}__nli__$(basename $m).npz
    [ $m = cross-encoder/nli-deberta-v3-small ] && case $ds in agnews|bbc2025|sst2) continue;; esac   # finished on the box
    [ -f $f ] || t $PY src/score_nli.py $m $ds --items $n --strings $st --dtype fp32 --bs 64 2>&1 | grep -vE "^[0-9]+/"
  done
done
echo NLI_QUEUE_DONE
GP="$HOME/Library/Application Support/ClipboardGuard/venv-gliner/bin/python"; GM="$HOME/Library/Application Support/ClipboardGuard/models/gliner-5a7adf72a23b4d311abae6ce050d7f0012bb3416"
for ds in agnews bbc2025 sst2 dbpedia massive; do
  n=1200; [ $ds = dbpedia ] && n=1000
  t "$GP" -B src/score_gliner.py "$GM" $ds --items $n --sets core --bs 32 2>&1 | grep -E "DONE|TIME|rror|Traceback|^  File" 
done
echo TORCH_QUEUE_DONE
