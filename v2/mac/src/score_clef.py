"""Clef-flash (mlx-community/clef-flash-8bit; local ClipboardGuard copy, read-only; bundled clef_mlx.py loader) as a zero-shot classifier.
One 'choice' question per WORDING SET and item (a fresh forward pass each; questions are never combined because they would share context).
Option ids A, B, C, ... in canonical class order with the set's label string as the option description (v1 convention);
instruction 'What is the {facet} of this {genre}? Choose the best option.' (same wording as the T5 API cells). Raw option logits are stored;
class probability = softmax(logits) (= clef_mlx.predict). Items are processed item-major in chunks round-robin over datasets, so whatever
finishes is a complete block of the first n items x all sets for every dataset. Output JSONL per dataset (resumable).
usage: CLEF_DIR=<dir> score_clef.py --items N --sets valid [--chunk 25] [--ds agnews,bbc2025,...]"""
import sys, json, os, time, argparse, platform
P = os.environ["CLEF_DIR"]; sys.dont_write_bytecode = True; sys.path.insert(0, P)
import clef_mlx, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("--items", type=int, default=600); ap.add_argument("--sets", default="valid"); ap.add_argument("--chunk", type=int, default=25)
ap.add_argument("--ds", default="agnews,bbc2025,sst2,dbpedia,massive"); a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = f"{ROOT}/data"; LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DS = a.ds.split(","); keep = ("official", "valid", "surface", "identity", "nonsense", "shuffled") if a.sets == "core" else ("official", "valid")
data = {ds: json.load(open(f"{D}/{ds}.json")) for ds in DS}
SETS = {ds: [s for s in json.load(open(f"{D}/wordsets_{ds}.json"))["sets"] if s["pop"] in keep] for ds in DS}
os.makedirs(f"{ROOT}/raw/schema_cache/clef", exist_ok=True); done = {}; F = {}
for ds in DS:
    p = f"{ROOT}/raw/schema_cache/clef/{ds}.jsonl"; done[ds] = set()
    if os.path.exists(p):
        for l in open(p):
            try: done[ds].add(json.loads(l)["i"])
            except Exception: pass
    F[ds] = open(p, "a")
m = clef_mlx.load(P); t0 = time.time(); n = 0
print("loaded", f"{time.time()-t0:.0f}s", mx.__version__ if hasattr(mx, "__version__") else "", platform.platform(), flush=True)
for c0 in range(0, a.items, a.chunk):
    for ds in DS:
        d = data[ds]; ins = f"What is the {d['facet']} of this {d['genre']}? Choose the best option."
        for i in range(c0, min(c0 + a.chunk, a.items, len(d["pool_text"]))):
            if i in done[ds]: continue
            rec = {"i": i, "lg": {}}
            for s in SETS[ds]:
                K = len(s["labels"])
                r = {"state": d["pool_text"][i], "questions": {"answer": {"type": "choice", "instructions": ins, "criteria": {LET[k]: s["labels"][k] for k in range(K)}}}}
                enc, lg = m.logits(r); ids = enc.questions[0].option_ids
                v = dict(zip(ids, lg[0].astype(mx.float32).tolist())); rec["lg"][s["name"]] = [v[LET[k]] for k in range(K)]
            F[ds].write(json.dumps(rec) + "\n"); F[ds].flush(); n += 1
            if n % 20 == 0: print(ds, i, n, f"{time.time()-t0:.0f}s", flush=True)
    print("chunk", c0 + a.chunk, f"{time.time()-t0:.0f}s", flush=True)
print("CLEF DONE", n, f"{time.time()-t0:.0f}s", flush=True)
