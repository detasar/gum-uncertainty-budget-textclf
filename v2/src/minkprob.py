"""Contamination proxy: Min-K% Prob (Shi et al. 2023, arXiv:2310.16789), k = 20%, of pool texts under each LLM (plain LM, no chat template).
Also mean token log-prob and zlib-normalised log-likelihood. usage: minkprob.py <model> [n_items]"""
import json, os, sys, time, zlib
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); from core import load_ds, ROOT
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
m = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
tok = AutoTokenizer.from_pretrained(m); mod = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
out = {"model": m, "k": 0.2, "n_items": n, "datasets": {}}
for ds in ["agnews", "bbc2025", "dbpedia", "sst2", "massive"]:
    d = load_ds(ds); texts = d["pool_text"][:n]; ys = d["pool_y"][:n]; rows = []; t0 = time.time()
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
        if tok.bos_token_id is not None and (ids.shape[1] == 0 or ids[0, 0] != tok.bos_token_id): ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)
        with torch.inference_mode(): lg = torch.log_softmax(mod(input_ids=ids).logits[0, :-1].float(), -1)
        lp = lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]].numpy()
        if len(lp) < 3: continue
        kk = max(1, int(round(0.2 * len(lp)))); mink = float(np.sort(lp)[:kk].mean())
        rows.append({"mink20": mink, "mean_lp": float(lp.mean()), "n_tok": int(len(lp)), "zlib_ratio": float(lp.sum() / len(zlib.compress(t.encode())))})
    out["datasets"][ds] = {"rows": rows, "y": ys[:len(rows)], "mean_mink20": float(np.mean([r["mink20"] for r in rows])), "mean_lp": float(np.mean([r["mean_lp"] for r in rows])),
                           "mean_ntok": float(np.mean([r["n_tok"] for r in rows]))}
    print(m, ds, round(out["datasets"][ds]["mean_mink20"], 3), round(out["datasets"][ds]["mean_lp"], 3), f"{time.time()-t0:.0f}s", flush=True)
os.makedirs(f"{ROOT}/results/contamination", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/contamination/minkprob__{m.split('/')[-1]}.json", "w"))
print("DONE", m)
