"""Assemble Mac JSONL caches of the T6 deterministic schema classifiers (exploratory, not preregistered) into npz score files (kind 'schema' for GLiNER2.5-Decide, kind 'decision' for Clef-flash):
raw/<ds>__schema__<model>.npz with the same per-set layout as the T5 API cells: P (N x n_sets x K float32, canonical class order),
set_names, set_labels (json), strings, lg (raw per-label logits/option logits), meta (json). P = softmax(lg) per set.
Only the leading block of items that has every requested set is kept (items are scored item-major), so partial runs give complete cells.
  clef   -> Clef-flash 8-bit (MLX), raw/schema_cache/clef/<ds>.jsonl
  gliner -> GLiNER2.5-Decide (gliner2 2.0.0, torch MPS), raw/schema_cache/gliner/<ds>.jsonl
usage: schema_build.py [--min-items 550]"""
import json, os, sys, glob, argparse
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); CACHE = f"{ROOT}/raw/schema_cache"
TAG = {"clef": "clef-flash-8bit", "gliner": "GLiNER2.5-Decide"}
KIND = {"clef": "decision", "gliner": "schema"}   # Clef = decision model (T7, with gpt-6-luna); GLiNER = encoder classifier (T6, deterministic-classifier group)
INFO = {"clef": {"model": "mlx-community/clef-flash-8bit (Cloudflare/clef-flash, 9B, 8-bit MLX; local snapshot dfa0993d)", "loader": "bundled clef_mlx.py (logits)",
                 "scheme": "one 'choice' question per wording set and item; option ids A.. in canonical class order, label string as option description; instruction 'What is the {facet} of this {genre}? Choose the best option.'; P = softmax(option logits)",
                 "hardware": "Apple M5 32 GB, MLX GPU", "sets": "official + 30 VALID"},
        "gliner": {"model": "fastino GLiNER2.5-Decide (local snapshot 5a7adf72; DeBERTa-v3-large encoder)", "loader": "gliner2 2.0.0 Classifier/ClassificationScorer",
                   "scheme": "one single-label classification task per wording set (labels = the set's K strings); raw classification-head logits per label; P = softmax(logits) (the library's activation for exclusive tasks, temperature 1). Document-level label score from the native classification head, not a span-score proxy.",
                   "hardware": "Apple M5 32 GB, torch 2.14.1 MPS fp32", "sets": "core (official, valid, surface, identity, nonsense, shuffled)"}}
def softmax(z): z = np.asarray(z, np.float64); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def build(model, ds, min_items):
    min_items = {"clef": 200}.get(model, min_items)
    path = f"{CACHE}/{model}/{ds}.jsonl"; recs = {}
    for l in open(path):
        try: r = json.loads(l); recs[r["i"]] = r
        except Exception: pass
    if not recs: return None
    sets = json.load(open(f"{ROOT}/data/wordsets_{ds}.json"))["sets"]; names0 = set(recs[min(recs)]["lg"])
    order = [s for s in sets if s["name"] in names0]; K = len(sets[0]["labels"])
    N = 0
    while N in recs and all(s["name"] in recs[N]["lg"] for s in order): N += 1
    if N < min_items: print(f"{model} {ds}: only {N} complete items (< {min_items}) - skipped"); return None
    LG = np.array([[recs[i]["lg"][s["name"]] for s in order] for i in range(N)], np.float64)
    P = softmax(LG).astype(np.float32); mism = sorted({x for i in range(N) for x in recs[i].get("name_mismatch", [])})
    strings = []
    for s in order:
        for l in s["labels"]:
            if l not in strings: strings.append(l)
    meta = dict(INFO[model], n_items=N, n_sets=len(order), eps=0.0, name_mismatch_sets=mism, items_in_cache=len(recs))
    out = f"{ROOT}/raw/{ds}__{KIND[model]}__{TAG[model]}.npz"
    np.savez_compressed(out, P=P, lg=LG.astype(np.float32), set_names=np.array([s["name"] for s in order], dtype=object),
                        set_labels=json.dumps([s["labels"] for s in order]), strings=np.array(strings, dtype=object), meta=json.dumps(meta))
    print("BUILT", out, f"N={N} sets={len(order)}"); return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--min-items", type=int, default=550); a = ap.parse_args()
    for f in sorted(glob.glob(f"{CACHE}/*/*.jsonl")):
        build(f.split("/")[-2], os.path.basename(f)[:-6], a.min_items)
