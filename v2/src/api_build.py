"""Assemble API JSONL caches into npz score files (kind 'api').
  main  -> raw/<ds>__api__<model>.npz              (consumed by analyze.py / analyze_pending.sh like any other cell)
  ordK  -> raw/apiorder/<ds>__api__<model>__ordK.npz
  rtK   -> raw/repeat/<ds>__api__<model>__rtK.npz
npz keys: P (N x n_sets x K float32, canonical class order, NaN = refusal/missing), set_names, set_labels (json), strings (unique label strings),
lp (nano only: letter logprobs), miss (nano only: letter absent from top-20), meta (json).
usage: api_build.py [--model luna|nano] [--ds ds] [--cond main] [--allow-partial]"""
import json, os, sys, glob, argparse, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from score_api import MODELS, CACHE, PREFIX, choose_sets, load, cost
from core import API_EPS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SETMODE = {"luna": {"agnews": "main", "bbc2025": "main"}, "nano": {}}
def build(model, ds, cond, allow_partial=False):
    path = f"{CACHE}/{model}/{ds}__{cond}.jsonl"
    if not os.path.exists(path): return None
    d, sets = load(ds); recs = [json.loads(l) for l in open(path) if l.strip()]
    names = []; 
    for r in recs:
        for nm in r["sets"]:
            if nm not in names: names.append(nm)
    order = [s["name"] for s in sets if s["name"] in names]; S = {s["name"]: s for s in sets}
    items = sorted({r["i"] for r in recs}); N = max(items) + 1; K = len(sets[0]["labels"]); j = {nm: t for t, nm in enumerate(order)}
    P = np.full((N, len(order), K), np.nan, np.float32); got = np.zeros((N, len(order)), bool); refused = 0
    LP = np.full((N, len(order), K), np.nan, np.float32) if model == "nano" else None; MS = np.zeros((N, len(order), K), np.int8) if model == "nano" else None
    usd = 0.0; tin = 0; tout = 0; mids = set(); fps = set(); ts = []
    for r in recs:
        usd += cost(model, r["usage"]); tin += r["usage"].get("input_tokens", r["usage"].get("prompt_tokens", 0)); tout += r["usage"].get("output_tokens", r["usage"].get("completion_tokens", 0))
        mids.add(r.get("model")); ts.append(r.get("ts", 0))
        if r.get("fp"): fps.add(r["fp"])
        for nm in r["sets"]:
            got[r["i"], j[nm]] = True
            if model == "luna":
                v = r["p"][nm]
                if v is None: refused += 1; continue
                P[r["i"], j[nm]] = v
            else:
                lp = np.array(r["lp"][nm], float); LP[r["i"], j[nm]] = lp; MS[r["i"], j[nm]] = r["miss"][nm]
                e = np.exp(lp - lp.max()); P[r["i"], j[nm]] = e / e.sum()
    complete = got.all(1); Nc = int(np.argmin(complete)) if not complete.all() else N
    if Nc < N and not allow_partial: print(f"{model} {ds} {cond}: incomplete ({complete.sum()}/{N} items complete) - skipped"); return None
    P = P[:Nc]; got = got[:Nc]
    meta = {"model": MODELS[model], "model_ids_returned": sorted(m for m in mids if m), "system_fingerprints": sorted(fps), "api": "decisions" if model == "luna" else "chat.completions",
            "scheme": ("Decisions API 'choice' question per wording set; choices = the set's K label strings; probabilities as returned (0.01 grid), floored at eps then renormalised in core.Scorer"
                       if model == "luna" else "chat completions, options as letters A..; class prob = softmax over the K letters' first-token logprobs (logsumexp of 'X' and ' X'); absent letters get the minimum top-20 logprob"),
            "prompt_prefix": PREFIX, "eps": API_EPS if model == "luna" else 0.0, "cond": cond, "n_items": Nc, "n_sets": len(order), "refusals": refused,
            "missing_letter_rate": float(MS[:Nc].mean()) if model == "nano" else None, "usd": usd, "input_tokens": tin, "output_tokens": tout,
            "first_request_utc": datetime.datetime.fromtimestamp(min(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "last_request_utc": datetime.datetime.fromtimestamp(max(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    tag = MODELS[model]
    if cond == "main": out = f"{ROOT}/raw/{ds}__api__{tag}.npz"
    elif cond.startswith("ord"): os.makedirs(f"{ROOT}/raw/apiorder", exist_ok=True); out = f"{ROOT}/raw/apiorder/{ds}__api__{tag}__{cond}.npz"
    elif cond.startswith("rt"): out = f"{ROOT}/raw/repeat/{ds}__api__{tag}__{cond}.npz"
    else: out = f"{ROOT}/raw/api_pilot/{ds}__api__{tag}__{cond}.npz"; os.makedirs(os.path.dirname(out), exist_ok=True)
    strings = []
    for nm in order:
        for l in S[nm]["labels"]:
            if l not in strings: strings.append(l)
    extra = {"lp": LP[:Nc], "miss": MS[:Nc]} if model == "nano" else {}
    np.savez_compressed(out, P=P, set_names=np.array(order, dtype=object), set_labels=json.dumps([S[nm]["labels"] for nm in order]),
                        strings=np.array(strings, dtype=object), meta=json.dumps(meta), **extra)
    print("BUILT", out, f"N={Nc} sets={len(order)} refusals={refused} usd={usd:.4f}"); return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--ds"); ap.add_argument("--cond"); ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    for f in sorted(glob.glob(f"{CACHE}/*/*.jsonl")):
        model = f.split("/")[-2]; ds, cond = os.path.basename(f)[:-6].split("__")
        if (a.model and model != a.model) or (a.ds and ds != a.ds) or (a.cond and cond != a.cond) or cond.startswith("smoke"): continue
        build(model, ds, cond, a.allow_partial)
