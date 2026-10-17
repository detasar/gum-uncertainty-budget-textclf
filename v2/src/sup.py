"""T4 supervised probes: logistic regression on frozen bi-encoder embeddings (wording-free reference instrument).
For each dataset x embedder: reference fit on the training split (BBC-2025: trained on AG News = real temporal/source shift),
20 training-bootstrap refits; Design A floor curves with the bootstrap refit as the crossed factor (sigma_train vs sigma_split).
usage: sup.py <ds> <embedder-tag>"""
import json, os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
from sklearn.linear_model import LogisticRegression
ds, tag = sys.argv[1], sys.argv[2]; t0 = time.time()
z = np.load(f"{RAW}/{ds}__bienc__{tag}.npz", allow_pickle=True); E = z["emb_pool"].astype(np.float64)
d = load_ds(ds); y = np.array(d["pool_y"]); M = len(y)
if ds == "bbc2025":
    za = np.load(f"{RAW}/agnews__bienc__{tag}.npz", allow_pickle=True); Etr = za["emb_train"].astype(np.float64); ytr = np.array(load_ds("agnews")["train_y"]); trained_on = "agnews"
else:
    Etr = z["emb_train"].astype(np.float64); ytr = np.array(d["train_y"]); trained_on = ds
fit = lambda X, yy: LogisticRegression(C=1.0, max_iter=3000).fit(X, yy)
ref = fit(Etr, ytr).predict_proba(E)
rng = np.random.default_rng(42); boots = []
for b in range(20):
    i = rng.integers(0, len(ytr), len(ytr))
    if len(np.unique(ytr[i])) < len(np.unique(ytr)): i = rng.integers(0, len(ytr), len(ytr))
    m = fit(Etr[i], ytr[i]); Pb = np.zeros((M, ref.shape[1])); Pb[:, m.classes_] = m.predict_proba(E); boots.append(np.clip(Pb, 1e-12, 1))
u = np.random.default_rng(99).uniform(size=M); grid = n_grid(M); nmax = grid[-1]
res = {"dataset": ds, "kind": "sup", "type": TYPES["sup"], "model": f"{tag}+LogReg", "trained_on": trained_on, "N": M, "n_train": int(len(ytr)), "n_grid": grid,
       "point": {"reference": point_metrics(ref, y)}, "point_boot_acc": [point_metrics(p, y)["accuracy"] for p in boots]}
SCr = {m: scores_all(ref, m, u) for m in ["lac", "aps", "raps"]}; SCb = {m: [scores_all(p, m, u) for p in boots] for m in ["lac", "aps"]}
res["floor"] = {}; Ystore = {}
for design in ["A", "B"]:
    for method in ["lac", "aps"]:
        for n in grid:
            cal, te = (splits_A if design == "A" else splits_B)(M, n)
            rr = [conformal_eval(s, y, cal, te, 0.1) for s in SCb[method]]
            blk = {}
            for met in ["set_size", "coverage"]:
                Y = np.stack([r[met] for r in rr]); v = vc_summary(Y, n_boot=1000, seed=n)
                blk[met] = {k: vv for k, vv in v.items() if not k.startswith("boot_")}; Ystore[f"{design}__{n}__{method}__{met}"] = Y
                rref = conformal_eval(SCr[method], y, cal, te, 0.1)[met]; blk[met]["ref_sd_split"] = float(np.std(rref, ddof=1)); blk[met]["ref_mean"] = float(np.mean(rref))
            res["floor"][f"{design}|{method}|trainboot|{n}"] = blk
res["grid"] = {}
for n in sorted({250, nmax}):
    cal, te = splits_A(M, n)
    for method in ["lac", "aps", "raps"]:
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            if alpha == 0.01 and n < 500: continue
            r = conformal_eval(SCr[method], y, cal, te, alpha)
            res["grid"][f"{method}|{alpha}|{n}"] = {m: {"mean": float(np.mean(r[m])), "sd_split": float(np.std(r[m], ddof=1))} for m in ["set_size", "coverage", "singleton", "empty", "worst_class_cov"]}
OUT = os.path.join(ROOT, "results", "cells"); os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/{ds}__sup__{tag}.json", "w"), default=float); np.savez_compressed(f"{OUT}/{ds}__sup__{tag}_Y.npz", **Ystore)
np.savez_compressed(f"{RAW}/{ds}__supprobs__{tag}.npz", ref=ref.astype(np.float32), boots=np.array(boots, np.float32))
print("DONE", ds, tag, f"acc={res['point']['reference']['accuracy']:.3f}", f"{time.time()-t0:.0f}s", flush=True)
