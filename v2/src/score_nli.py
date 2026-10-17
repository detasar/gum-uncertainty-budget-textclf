"""Zero-shot NLI cross-encoder scoring: logits for every (pool text, label string) pair.
usage: score_nli.py <model_id> <ds> [template] [--items N] [--strings official] [--out path] [--dtype bf16|fp32] [--bs 64] [--threads 8]"""
import argparse, json, os, time, sys, platform
import numpy as np, torch, transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__)); from strings import label_strings, load_ds
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ds"); ap.add_argument("--template", default="This text is about {}.")
ap.add_argument("--items", type=int, default=0); ap.add_argument("--strings", default="all"); ap.add_argument("--out", default=None)
ap.add_argument("--dtype", default="bf16"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--maxlen", type=int, default=256)
a = ap.parse_args(); torch.set_num_threads(a.threads)
d = load_ds(a.ds); S, L = label_strings(a.ds, "valid" if a.strings == "valid" else "all")
if a.strings == "official": S = [r["text"] for v in L["variants"] for r in v if "G0_official" in r["generators"]]
texts = d["pool_text"][: a.items or None]
tok = AutoTokenizer.from_pretrained(a.model); dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
mod = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=dt).eval()
hyps = [a.template.format(s) for s in S]
pairs = [(i, j) for i in range(len(texts)) for j in range(len(S))]
# sort by premise length for efficient padding (deterministic order)
lens = np.array([len(t) for t in texts]); order = sorted(range(len(pairs)), key=lambda k: (lens[pairs[k][0]], pairs[k][0], pairs[k][1]))
nlab = mod.config.num_labels; out = np.zeros((len(texts), len(S), nlab), np.float32); t0 = time.time()
with torch.inference_mode():
    for b in range(0, len(order), a.bs):
        idx = [pairs[k] for k in order[b:b + a.bs]]
        enc = tok([texts[i] for i, _ in idx], [hyps[j] for _, j in idx], return_tensors="pt", padding=True, truncation="only_first", max_length=a.maxlen)
        lg = mod(**enc).logits.float().numpy()
        for (i, j), l in zip(idx, lg): out[i, j] = l
        if (b // a.bs) % 200 == 0: print(f"{b}/{len(order)} {time.time()-t0:.0f}s", flush=True)
ent = [k for k, v in mod.config.id2label.items() if v.lower().startswith("entail")][0]
tag = a.model.split("/")[-1]
path = a.out or os.path.join(os.path.dirname(__file__), "..", "raw", f"{a.ds}__nli__{tag}.npz")
np.savez_compressed(path, logits=out, strings=np.array(S, dtype=object), entail_index=ent, id2label=json.dumps(mod.config.id2label),
                    meta=json.dumps({"model": a.model, "revision": getattr(mod.config, "_commit_hash", None), "template": a.template, "dtype": a.dtype, "bs": a.bs,
                                     "threads": a.threads, "torch": torch.__version__, "transformers": transformers.__version__, "cpu": platform.processor(),
                                     "n_items": len(texts), "seconds": time.time() - t0}))
print("DONE", path, f"{time.time()-t0:.0f}s", flush=True)
