"""Sentence embeddings on the box CPU (fastembed / ONNX) for the supervised baselines."""
import json, sys, time, numpy as np
from fastembed import TextEmbedding
S = json.load(open("cache/supervised_set.json"))
for name, tag in [("sentence-transformers/all-MiniLM-L6-v2", "minilm"), ("BAAI/bge-small-en-v1.5", "bgesmall")]:
    t = time.time(); m = TextEmbedding(name, threads=6)
    E = lambda x: np.array(list(m.embed(x, batch_size=32)), dtype=np.float32)
    arrs = {"train": E(S["train_text"]), "cal": E(S["cal_text"])}
    for l, tt in S["test_text"].items(): arrs[f"test_{l}"] = E(tt)
    np.savez_compressed(f"cache/emb_{tag}.npz", **arrs); print(tag, time.time() - t, flush=True); del m
