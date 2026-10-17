#!/bin/bash
# Repeatability (4 identical fresh-process runs) and intermediate-precision conditions, AG News first N items, official + v1 strings.
source /workspace/.venv-uq2/bin/activate; cd /workspace/x-master/uq-budget/v2; mkdir -p raw/repeat
T=${THREADS:-4}
for r in 0 1 2 3; do
  python src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 200 --strings v1 --dtype fp32 --bs 64 --threads $T --out raw/repeat/nli_rep$r.npz 2>&1 | grep DONE
  python src/score_bienc.py BAAI/bge-small-en-v1.5 agnews --items 200 --strings v1 --dtype fp32 --bs 64 --threads $T --out raw/repeat/bienc_rep$r.npz 2>&1 | grep DONE
  python src/score_llm.py Qwen/Qwen2.5-0.5B-Instruct agnews --items 100 --strings v1 --dtype fp32 --chunk 64 --threads $T --out raw/repeat/llm_rep$r.npz 2>&1 | grep DONE
done
python src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 200 --strings v1 --dtype fp32 --bs 1 --threads $T --out raw/repeat/nli_fp32_bs1.npz 2>&1 | grep DONE
python src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 200 --strings v1 --dtype fp32 --bs 64 --threads 1 --out raw/repeat/nli_fp32_t1.npz 2>&1 | grep DONE
python src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 200 --strings v1 --dtype bf16 --bs 64 --threads $T --out raw/repeat/nli_bf16_bs64.npz 2>&1 | grep DONE
python src/score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 200 --strings v1 --dtype bf16 --bs 1 --threads $T --out raw/repeat/nli_bf16_bs1.npz 2>&1 | grep DONE
python src/score_bienc.py BAAI/bge-small-en-v1.5 agnews --items 200 --strings v1 --dtype fp32 --bs 1 --threads $T --out raw/repeat/bienc_fp32_bs1.npz 2>&1 | grep DONE
python src/score_bienc.py BAAI/bge-small-en-v1.5 agnews --items 200 --strings v1 --dtype fp32 --bs 64 --threads 1 --out raw/repeat/bienc_fp32_t1.npz 2>&1 | grep DONE
python src/score_bienc.py BAAI/bge-small-en-v1.5 agnews --items 200 --strings v1 --dtype bf16 --bs 64 --threads $T --out raw/repeat/bienc_bf16_bs64.npz 2>&1 | grep DONE
python src/score_bienc.py BAAI/bge-small-en-v1.5 agnews --items 200 --strings v1 --dtype bf16 --bs 1 --threads $T --out raw/repeat/bienc_bf16_bs1.npz 2>&1 | grep DONE
python src/score_llm.py Qwen/Qwen2.5-0.5B-Instruct agnews --items 100 --strings v1 --dtype fp32 --chunk 1 --threads $T --out raw/repeat/llm_fp32_bs1.npz 2>&1 | grep DONE
python src/score_llm.py Qwen/Qwen2.5-0.5B-Instruct agnews --items 100 --strings v1 --dtype fp32 --chunk 64 --threads 1 --out raw/repeat/llm_fp32_t1.npz 2>&1 | grep DONE
python src/score_llm.py Qwen/Qwen2.5-0.5B-Instruct agnews --items 100 --strings v1 --dtype bf16 --chunk 64 --threads $T --out raw/repeat/llm_bf16_bs64.npz 2>&1 | grep DONE
python src/score_llm.py Qwen/Qwen2.5-0.5B-Instruct agnews --items 100 --strings v1 --dtype bf16 --chunk 1 --threads $T --out raw/repeat/llm_bf16_bs1.npz 2>&1 | grep DONE
echo REPEAT_DONE
