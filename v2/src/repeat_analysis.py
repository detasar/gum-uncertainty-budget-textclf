"""RQ5: repeatability (identical fresh-process runs) and intermediate precision (batch size, threads, dtype) on CPU.
Inputs raw/repeat/*.npz (AG News, official + v1 strings). Output results/repeatability.json"""
import json, os, sys, glob, itertools, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import softmax, load_ds, load_sets, qlevel, ROOT
from sklearn.metrics import cohen_kappa_score
y_all = np.array(load_ds("agnews")["pool_y"]); sets = [s for s in load_sets("agnews") if s["name"] in ("official",)]
V1 = {"official": ["World", "Sports", "Business", "Sci/Tech"], "v1_w0": ["World", "Sports", "Business", "Science/Technology"],
      "v1_w1": ["world news", "sports news", "business news", "science and technology news"],
      "v1_w2": ["international politics and conflicts", "athletes, teams and games", "companies, markets and the economy", "computers, the internet and scientific research"],
      "v1_w3": ["politics", "sport", "money", "tech"]}
def probs(path, labels):
    z = np.load(path, allow_pickle=True); S = list(z["strings"]); ix = [S.index(l) for l in labels]
    if "logits" in z: X = z["logits"][:, :, int(z["entail_index"])]
    elif "cos" in z: X = z["cos"] / 0.05
    else: X = z["raw"] - z["null"][None, :]
    return softmax(X[:, ix].astype(np.float64)), json.loads(str(z["meta"]))
def icc(Y):
    n, k = Y.shape; g = Y.mean(); SSR = k * ((Y.mean(1) - g) ** 2).sum(); SSC = n * ((Y.mean(0) - g) ** 2).sum(); SST = ((Y - g) ** 2).sum()
    SSE = max(SST - SSR - SSC, 0.0); MSR = SSR / (n - 1); MSC = SSC / (k - 1); MSE = SSE / ((n - 1) * (k - 1))
    d2 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n; d3 = MSR + (k - 1) * MSE
    return (float((MSR - MSE) / d2) if d2 > 0 else None, float((MSR - MSE) / d3) if d3 > 0 else None)
def fleiss(L, K):
    n, k = L.shape; cnt = np.stack([(L == c).sum(1) for c in range(K)], 1); P_i = ((cnt ** 2).sum(1) - k) / (k * (k - 1)); p = cnt.sum(0) / (n * k); Pe = (p ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else None
def sets_of(P, y):   # half calibration / half test, LAC alpha 0.1
    h = len(y) // 2; s = 1 - P[np.arange(h), y[:h]]; q = np.quantile(s, qlevel(h, 0.1), method="higher"); return (1 - P[h:]) <= q
out = {}
for inst in ["nli", "bienc", "llm"]:
    reps = sorted(glob.glob(f"{ROOT}/raw/repeat/{inst}_rep*.npz"))
    if len(reps) < 2: continue
    r = {"runs": len(reps)}
    for wname, lab in V1.items():
        try: Ps = [probs(p, lab)[0] for p in reps]
        except ValueError: continue
        N = Ps[0].shape[0]; y = y_all[:N]; A = np.stack(Ps)
        pt = A[:, np.arange(N), y].T; i2, i3 = icc(pt); Lb = A.argmax(2).T
        S = np.stack([sets_of(P, y) for P in Ps])
        r[wname] = {"items": N, "max_abs_dp": float(np.abs(A[:, None] - A[None]).max()), "icc21": i2, "icc31": i3, "fleiss_kappa": fleiss(Lb, 4),
                    "share_items_same_label_all_runs": float((Lb == Lb[:, :1]).all(1).mean()), "share_test_sets_changed": float((S != S[:1]).any(0).any(1).mean()),
                    "saturation_maxp_gt_099": float((A[0].max(1) > 0.99).mean())}
    base = reps[0]; conds = {}
    for p in sorted(glob.glob(f"{ROOT}/raw/repeat/{inst}_*.npz")):
        if "_rep" in p: continue
        c = os.path.basename(p)[len(inst) + 1:-4]; row = {}
        for wname, lab in V1.items():
            P0, m0 = probs(base, lab); P1, m1 = probs(p, lab); N = P0.shape[0]; y = y_all[:N]
            s0, s1 = sets_of(P0, y), sets_of(P1, y)
            row[wname] = {"max_abs_dp": float(np.abs(P0 - P1).max()), "mean_abs_dp": float(np.abs(P0 - P1).mean()), "argmax_flips": float((P0.argmax(1) != P1.argmax(1)).mean()),
                          "share_test_sets_changed": float((s0 != s1).any(1).mean()), "kappa_vs_rep0": float(cohen_kappa_score(P0.argmax(1), P1.argmax(1)))}
        row["meta"] = {k: m1.get(k) for k in ("dtype", "bs", "chunk", "threads")}; conds[c] = row
    r["conditions_vs_rep0"] = conds; r["rep0_meta"] = probs(base, V1["official"])[1]; out[inst] = r
json.dump(out, open(f"{ROOT}/results/repeatability.json", "w"), indent=1, default=float)
for inst, r in out.items():
    print(inst, "identical runs:", {w: (v["max_abs_dp"], v["share_test_sets_changed"]) for w, v in r.items() if isinstance(v, dict) and "max_abs_dp" in v})
    for c, row in r["conditions_vs_rep0"].items(): print("  ", c, {w: (round(v["max_abs_dp"], 6), round(v["share_test_sets_changed"], 3), round(v["argmax_flips"], 3)) for w, v in row.items() if w != "meta"})
