"""Zero-shot bi-encoder scoring: cosine(text, template(label)) for every pool text and label string; also saves text embeddings
(reused by the supervised probes). usage: score_bienc.py <model_id> <ds> [--template ...]"""
import argparse, json, os, time, sys
import numpy as np, torch, transformers, sentence_transformers
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.dirname(__file__)); from strings import label_strings, load_ds
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ds"); ap.add_argument("--template", default="This text is about {}.")
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--out", default=None); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--items", type=int, default=0)
a = ap.parse_args(); torch.set_num_threads(a.threads)
d = load_ds(a.ds); S, L = label_strings(a.ds); texts = d["pool_text"][: a.items or None]
pre = "query: " if "e5" in a.model else ""
m = SentenceTransformer(a.model, device="cpu"); t0 = time.time()
E = m.encode([pre + t for t in texts], batch_size=a.bs, normalize_embeddings=True, convert_to_numpy=True)
H = m.encode([pre + a.template.format(s) for s in S], batch_size=a.bs, normalize_embeddings=True, convert_to_numpy=True)
Etr = m.encode([pre + t for t in d["train_text"]], batch_size=a.bs, normalize_embeddings=True, convert_to_numpy=True) if d["train_text"] else np.zeros((0, E.shape[1]), np.float32)
tag = a.model.split("/")[-1]
path = a.out or os.path.join(os.path.dirname(__file__), "..", "raw", f"{a.ds}__bienc__{tag}.npz")
np.savez_compressed(path, cos=(E @ H.T).astype(np.float32), strings=np.array(S, dtype=object), emb_pool=E.astype(np.float32), emb_train=Etr.astype(np.float32),
                    meta=json.dumps({"model": a.model, "template": a.template, "prefix": pre, "torch": torch.__version__, "st": sentence_transformers.__version__,
                                     "n_items": len(texts), "seconds": time.time() - t0}))
print("DONE", path, f"{time.time()-t0:.0f}s", flush=True)
