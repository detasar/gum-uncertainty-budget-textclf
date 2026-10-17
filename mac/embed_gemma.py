"""embeddinggemma embeddings via local Ollama for the supervised (embedding + LR) baseline."""
import json, urllib.request, numpy as np
S = json.load(open("supervised_set.json"))
def embed(texts):
    out = []
    for i in range(0, len(texts), 64):
        body = {"model": "embeddinggemma:latest", "input": texts[i:i+64], "keep_alive": "1m"}
        req = urllib.request.Request("http://localhost:11434/api/embed", json.dumps(body).encode(), {"Content-Type": "application/json"})
        out += json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]
    return np.array(out, dtype=np.float32)
arrs = {"train": embed(S["train_text"]), "cal": embed(S["cal_text"])}
for l, t in S["test_text"].items():
    arrs[f"test_{l}"] = embed(t)
np.savez_compressed("out/emb_embeddinggemma.npz", **arrs)
print({k: v.shape for k, v in arrs.items()})
