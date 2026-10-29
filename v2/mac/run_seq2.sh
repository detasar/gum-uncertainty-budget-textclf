#!/bin/bash
# 2026-10-29 resume driver (strictly sequential, one model job at a time; run under caffeinate): GLiNER2.5-Decide x5 (first 600 items, GLINER_N), then Clef-flash x5 (first 200 items).
# Each GLiNER cell is retried once; caches are resumable (finished items are skipped).
cd ~/uq-budget-v2; export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0
GP="$HOME/Library/Application Support/ClipboardGuard/venv-gliner/bin/python"; GM="$HOME/Library/Application Support/ClipboardGuard/models/gliner-5a7adf72a23b4d311abae6ce050d7f0012bb3416"
echo "=== seq2 start $(date)"
for ds in ${GL_DS:-agnews bbc2025 sst2 dbpedia massive}; do
  n=1200; [ $ds = dbpedia ] && n=1000
  for try in 1 2; do s=$(date +%s); "$GP" -B src/score_gliner.py "$GM" $ds --items $n --sets core --bs 32 >> logs/gliner.log 2>&1; rc=$?
    echo "GLINER $ds try$try rc=$rc $(( $(date +%s)-s ))s rows=$(wc -l < raw/schema_cache/gliner/$ds.jsonl) $(date)"; [ $rc = 0 ] && break; done
done
echo "GLINER_QUEUE_END $(date)"
[ "${SKIP_CLEF:-0}" = 1 ] || { ./run_clef.sh >> logs/clef.log 2>&1; echo "CLEF rc=$? $(date)"; }
echo "SEQ2_DONE $(date)"
