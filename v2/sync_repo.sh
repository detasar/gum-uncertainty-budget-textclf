#!/bin/bash
# Copy v2 into the repo clone (repo/v2). BBC-2025 article texts are not redistributed: only links/dates/labels + text hashes.
set -e
SRC=/workspace/x-master/uq-budget/v2; DST=/workspace/x-master/uq-budget/repo/v2
mkdir -p $DST
rm -rf $DST && mkdir -p $DST && (cd $SRC && tar --exclude='data/bbc2025.json' --exclude='__pycache__' --exclude='refs/*.html' --exclude='refs/gum.pdf' --exclude='refs/gum.txt' --exclude='raw/api_cache' -cf - .) | (cd $DST && tar -xf -)
# API response caches (JSONL, ~260 MB) are shipped compressed; unpack with: tar -xzf raw/api_cache.tar.gz -C raw
(cd $SRC/raw && tar -czf $DST/raw/api_cache.tar.gz api_cache)
/workspace/.venv-uq2/bin/python - <<'PY'
import json, hashlib
d = json.load(open("/workspace/x-master/uq-budget/v2/data/bbc2025.json"))
pub = {k: v for k, v in d.items() if k not in ("pool_text", "train_text")}
pub["pool_text_sha256"] = [hashlib.sha256(t.encode()).hexdigest() for t in d["pool_text"]]
pub["note"] = "Texts not redistributed. Rebuild with src/prep_data.py bbc (downloads RealTimeData/bbc_news_alltime from Hugging Face) and check the hashes."
json.dump(pub, open("/workspace/x-master/uq-budget/repo/v2/data/bbc2025_public.json", "w"))
PY
