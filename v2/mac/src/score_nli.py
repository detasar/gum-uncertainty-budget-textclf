"""Zero-shot NLI cross-encoder scoring on Apple MPS (port of v2 src/score_nli.py; identical algorithm, pair order, batch composition and npz format).
Differences: --device (default mps), tensors moved to the device, resumable checkpoints (<out>.ckpt.npz, every ~120 s), meta gains device/platform keys.
usage: score_nli.py <model_id> <ds> [--template ..] [--items N] [--strings all|valid|official] [--out path] [--dtype fp32] [--bs 64] [--device mps]"""
import argparse, json, os, time, sys, platform
import numpy as np, torch, transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__)); from strings import label_strings, load_ds
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ds"); ap.add_argument("--template", default="This text is about {}.")
ap.add_argument("--items", type=int, default=0); ap.add_argument("--strings", default="all"); ap.add_argument("--out", default=None)
ap.add_argument("--dtype", default="fp32"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--maxlen", type=int, default=256)
ap.add_argument("--device", default="mps")
a = ap.parse_args(); torch.set_num_threads(a.threads); dev = torch.device(a.device)
d = load_ds(a.ds); S, L = label_strings(a.ds, a.strings if a.strings in ("valid", "v1") else "all")
if a.strings == "official": S = [r["text"] for v in L["variants"] for r in v if "G0_official" in r["generators"]]
texts = d["pool_text"][: a.items or None]
tok = AutoTokenizer.from_pretrained(a.model); dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
mod = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=dt).eval().to(dev)
hyps = [a.template.format(s) for s in S]
pairs = [(i, j) for i in range(len(texts)) for j in range(len(S))]
lens = np.array([len(t) for t in texts]); order = sorted(range(len(pairs)), key=lambda k: (lens[pairs[k][0]], pairs[k][0], pairs[k][1]))
nlab = mod.config.num_labels; out = np.zeros((len(texts), len(S), nlab), np.float32)
tag = a.model.split("/")[-1]
path = a.out or os.path.join(os.path.dirname(__file__), "..", "raw", f"{a.ds}__nli__{tag}.npz"); ck = path + ".ckpt.npz"
start = 0; prev = 0.0
if os.path.exists(ck):
    c = np.load(ck, allow_pickle=True)
    if c["out"].shape == out.shape and list(c["strings"]) == S: out = c["out"]; start = int(c["next"]); prev = float(c["seconds"]); print("RESUME at", start, flush=True)
t0 = time.time(); last = t0
with torch.inference_mode():
    for b in range(start, len(order), a.bs):
        idx = [pairs[k] for k in order[b:b + a.bs]]
        enc = tok([texts[i] for i, _ in idx], [hyps[j] for _, j in idx], return_tensors="pt", padding=True, truncation="only_first", max_length=a.maxlen).to(dev)
        lg = mod(**enc).logits.float().cpu().numpy()
        for (i, j), l in zip(idx, lg): out[i, j] = l
        if (b // a.bs) % 200 == 0: print(f"{b}/{len(order)} {prev + time.time()-t0:.0f}s", flush=True)
        if time.time() - last > 120:
            np.savez(ck + ".tmp.npz", out=out, next=b + a.bs, strings=np.array(S, dtype=object), seconds=prev + time.time() - t0); os.replace(ck + ".tmp.npz", ck); last = time.time()
ent = [k for k, v in mod.config.id2label.items() if v.lower().startswith("entail")][0]
secs = prev + time.time() - t0
np.savez_compressed(path, logits=out, strings=np.array(S, dtype=object), entail_index=ent, id2label=json.dumps(mod.config.id2label),
                    meta=json.dumps({"model": a.model, "revision": getattr(mod.config, "_commit_hash", None), "template": a.template, "dtype": a.dtype, "bs": a.bs,
                                     "threads": a.threads, "torch": torch.__version__, "transformers": transformers.__version__, "cpu": platform.processor(),
                                     "n_items": len(texts), "seconds": secs, "device": a.device, "platform": platform.platform(), "host": "mac-m5-32gb",
                                     "resumed": bool(start)}))
if os.path.exists(ck): os.remove(ck)
print("DONE", path, f"{secs:.0f}s", flush=True)
