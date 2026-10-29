"""GLiNER2.5-Decide (local ClipboardGuard copy, read-only) as a zero-shot document classifier: one single-label classification task per
WORDING SET (labels = the set's K strings, canonical class order). Records the raw per-label logits of the library's classification head
(ClassificationScorer); class probability = softmax(logit) = the library's own 'exclusive task' activation (temperature 1).
Labels are part of the encoder input, so scores are per set (not per string). Output: JSONL, one line per item with logits for every set (resumable).
usage: score_gliner.py <model_dir> <ds> --items N --sets core|valid [--bs 32]"""
import argparse, json, os, sys, time, platform
import torch
ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("ds"); ap.add_argument("--items", type=int, required=True)
ap.add_argument("--sets", default="core"); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="mps"); a = ap.parse_args()
a.items = min(a.items, int(os.environ.get("GLINER_N", "600")))   # runtime cap (Davut: avoid long runs): first 600 items per dataset
from gliner2.classification.engine import Classifier
from gliner2.classification.schema import ClassificationSchema
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = f"{ROOT}/data"
d = json.load(open(f"{D}/{a.ds}.json")); sets = json.load(open(f"{D}/wordsets_{a.ds}.json"))["sets"]
keep = ("official", "valid", "surface", "identity", "nonsense", "shuffled") if a.sets == "core" else ("official", "valid")
sets = [s for s in sets if s["pop"] in keep]
os.makedirs(f"{ROOT}/raw/schema_cache/gliner", exist_ok=True); OUT = f"{ROOT}/raw/schema_cache/gliner/{a.ds}.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["i"])
        except Exception: pass
clf = Classifier.from_pretrained(a.model_dir, device=a.device, dtype=torch.float32).eval()
clf.to(device=a.device, dtype=torch.float32)   # 2026-10-29 fix: from_pretrained only records the device; weights stayed on CPU -> "Passed CPU tensor to MPS op" (17 Oct run)
assert str(next(clf.scorer.model.parameters()).device).startswith(a.device), "model not on requested device"
schemas = {s["name"]: ClassificationSchema().task("label", s["labels"], min_labels=1, max_labels=1) for s in sets}
texts = d["pool_text"][: a.items]; f = open(OUT, "a"); t0 = time.time(); n = 0
todo = [i for i in range(len(texts)) if i not in done]
for i in todo:
    names = [s["name"] for s in sets]
    sc = clf.scorer.batch_score([texts[i]] * len(names), [clf.compile_schema(schemas[nm]) for nm in names], batch_size=a.bs)
    rec = {"i": i, "lg": {}}
    for s, r in zip(sets, sc):
        t = r.tasks["label"]; names_ret = list(t)
        rec["lg"][s["name"]] = [t[l] for l in s["labels"]] if all(l in t for l in s["labels"]) else [t[x] for x in names_ret]
        if not all(l in t for l in s["labels"]): rec.setdefault("name_mismatch", []).append(s["name"])
    f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
    if n % 100 == 0: print(a.ds, i, f"{time.time()-t0:.0f}s", flush=True)
print("GLINER DONE", a.ds, len(todo), f"{time.time()-t0:.0f}s", torch.__version__, platform.platform(), flush=True)
