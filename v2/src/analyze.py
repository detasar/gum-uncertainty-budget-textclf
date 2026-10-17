"""Per (dataset, instrument) analysis. usage: analyze.py <raw npz> [--tau 0.05] [--llm-mode pmi|raw] [--tag suffix]
Writes results/cells/<ds>__<kind>__<model><tag>.json and results/cells/<...>_Y.npz (W x S matrices for figures)."""
import argparse, json, os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("raw"); ap.add_argument("--tau", type=float, default=TAU); ap.add_argument("--llm-mode", default="pmi")
ap.add_argument("--tag", default=""); ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); t0 = time.time()
sc = Scorer(a.raw, tau=a.tau, llm_mode=a.llm_mode); ds = sc.ds; d = load_ds(ds); y_all = np.array(d["pool_y"]); M = sc.N; y = y_all[:M]
sets = load_sets(ds); L = load_labels(ds); K = len(L["classes_official"])
OUT = os.path.join(ROOT, "results", "cells"); os.makedirs(OUT, exist_ok=True); name = f"{ds}__{sc.kind}__{sc.model}{a.tag}"
u = np.random.default_rng(99).uniform(size=M)
avail = [s for s in sets if sc.has(s["labels"])]
P = {s["name"]: sc.probs(s["labels"]) for s in avail}
pop = {}
for s in avail: pop.setdefault(s["pop"], []).append(s["name"])
res = {"dataset": ds, "kind": sc.kind, "type": TYPES[sc.kind], "model": sc.model, "N": M, "K": K, "tau": a.tau if sc.kind == "bienc" else None,
       "llm_mode": a.llm_mode if sc.kind == "llm" else None, "meta": sc.meta, "populations": {k: len(v) for k, v in pop.items()},
       "unavailable_sets": [s["name"] for s in sets if not sc.has(s["labels"])]}
# 1. point metrics per set (whole scored pool)
res["point"] = {k: point_metrics(v, y) for k, v in P.items()}
off = P["official"]; off_pred = off.argmax(1)
res["flip_vs_official"] = {k: float((v.argmax(1) != off_pred).mean()) for k, v in P.items()}
res["tv_vs_official"] = {k: float(0.5 * np.abs(v - off).sum(1).mean()) for k, v in P.items()}
SC = {m: {k: scores_all(v, m, u) for k, v in P.items()} for m in ["lac", "aps", "raps"]}
grid = n_grid(M); nmax = grid[-1]; res["n_grid"] = grid
METRICS = ["set_size", "coverage", "singleton", "empty", "worst_class_cov"]
def run_design(design, n, method, alpha, names):
    cal, te = (splits_A if design == "A" else splits_B)(M, n)
    out = {}
    for k in names: out[k] = conformal_eval(SC[method][k], y, cal, te, alpha)
    return out
Ystore = {}
def vc_block(design, n, method, alpha, popname, metrics=("set_size", "coverage")):
    names = pop.get(popname, [])
    if len(names) < 3: return None
    r = run_design(design, n, method, alpha, names); blk = {}
    for met in metrics:
        Y = np.stack([r[k][met] for k in names])            # W x S
        blk[met] = vc_summary(Y, n_boot=1000 if not a.quick else 100, seed=n)
        if popname == "valid" and method in ("lac", "aps") and alpha == 0.1: Ystore[f"{design}|{n}|{method}|{met}"] = Y
    return blk
# 2. floor curves (Design A and B), valid population, LAC & APS at alpha 0.10
res["floor"] = {}
for design in ["A", "B"]:
    for method in ["lac", "aps"]:
        for n in grid:
            for p_ in (["valid", "strict", "all"] if (design == "A" and method == "lac") else ["valid"]):
                b = vc_block(design, n, method, 0.1, p_)
                if b: res["floor"][f"{design}|{method}|{p_}|{n}"] = {m: {kk: vv for kk, vv in v.items() if not kk.startswith("boot_")} for m, v in b.items()}
                if b and p_ == "valid": res["floor"][f"{design}|{method}|{p_}|{n}"]["_boot_sd"] = {m: {"w": v["boot_sd_w"], "s": v["boot_sd_s"]} for m, v in b.items()}
    print(name, "floor", design, f"{time.time()-t0:.0f}s", flush=True)
# 3. full method x alpha grid at n=250 and nmax (Design A, valid population), all metrics
res["grid"] = {}
for n in sorted({250 if 250 in grid else grid[0], nmax}):
    for method in ["lac", "aps", "raps"]:
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            if alpha == 0.01 and n < 500: continue
            b = vc_block("A", n, method, alpha, "valid", metrics=METRICS)
            if b: res["grid"][f"{method}|{alpha}|{n}"] = {m: {kk: vv for kk, vv in v.items() if not kk.startswith("boot_")} for m, v in b.items()}
print(name, "grid", f"{time.time()-t0:.0f}s", flush=True)
# 4. per-set mean conformal metrics at n=250 (all available sets incl. controls), LAC/APS alpha .1
n250 = 250 if 250 in grid else grid[0]; res["per_set"] = {}
for method in ["lac", "aps"]:
    r = run_design("A", n250, method, 0.1, list(P.keys()))
    res["per_set"][method] = {k: {m: float(np.mean(v[m])) for m in METRICS} for k, v in r.items()}
# 5. W-subsampling (how the sigma_w estimate depends on the number of paraphrase sets), Design A, LAC, n=250 and nmax
res["w_subsample"] = {}
for n in sorted({n250, nmax}):
    key = f"A|{n}|lac|set_size"
    if key not in Ystore: continue
    Y = Ystore[key]; W = Y.shape[0]; rng = np.random.default_rng(5); rows = {}
    for w in [2, 3, 4, 8, 16, W]:
        if w > W: continue
        est = [];
        for _ in range(200 if w < W else 1):
            iw = rng.choice(W, w, replace=False); v = vc_mom(Y[iw]); est.append(math.sqrt(v["var_w"]))
        # v1-style estimator: population SD (ddof=0) of the W per-wording means
        v1 = [float(np.std(Y[rng.choice(W, w, replace=False)].mean(1), ddof=0)) for _ in range(200 if w < W else 1)]
        rows[w] = {"mean_sd_w": float(np.mean(est)), "p2.5": float(np.percentile(est, 2.5)), "p97.5": float(np.percentile(est, 97.5)),
                   "v1_popsd_mean": float(np.mean(v1)), "v1_popsd_p2.5": float(np.percentile(v1, 2.5)), "v1_popsd_p97.5": float(np.percentile(v1, 97.5))}
    res["w_subsample"][n] = rows
# 6. dose-response: replace ONE class's official label by each candidate variant (others official)
offl = next(s for s in sets if s["name"] == "official")["labels"]; dr = []
for c, vars_ in enumerate(L["variants"]):
    for r in vars_:
        if "G0_official" in r["generators"]: continue
        lab = list(offl); lab[c] = r["text"]
        if not sc.has(lab): continue
        Pv = sc.probs(lab); pr = Pv.argmax(1)
        dr.append({"class": c, "text": r["text"], "generators": r["generators"], "valid": r["valid"], "strict": r["strict"], "distance": r["distance"],
                   "cos": r["cos_official"], "entail_min": r["entail_min"], "flip": float((pr != off_pred).mean()),
                   "flip_in_class": float((pr[y == c] != off_pred[y == c]).mean()), "tv": float(0.5 * np.abs(Pv - off).sum(1).mean()),
                   "d_acc": float((pr == y).mean() - (off_pred == y).mean()), "d_acc_class": float((pr[y == c] == c).mean() - (off_pred[y == c] == c).mean())})
res["dose_response"] = dr
if len(dr) > 5:
    dist = np.array([x["distance"] for x in dr]); fl = np.array([x["flip"] for x in dr]); tv = np.array([x["tv"] for x in dr])
    rng = np.random.default_rng(3); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(dr), len(dr)); bs.append(spearmanr(dist[i], tv[i])[0])
    res["dose_response_summary"] = {"spearman_dist_flip": float(spearmanr(dist, fl)[0]), "spearman_dist_tv": float(spearmanr(dist, tv)[0]),
                                    "ci_spearman_dist_tv": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_variants": len(dr)}
# 7. configuration shift: calibrate with official wording, deploy with each valid set (exchangeability violated by design)
cal, te = splits_A(M, n250); xs = []
for k in pop.get("valid", []):
    ccal = SC["lac"]["official"]; ctest = SC["lac"][k]
    cs = ccal[cal, y[cal]]; q = np.quantile(cs, qlevel(n250, 0.1), axis=1, method="higher")
    st = ctest[te] <= q[:, None, None]; cov = np.take_along_axis(st, y[te][..., None], 2)[..., 0].mean(1)
    xs.append({"set": k, "coverage_mean": float(cov.mean()), "set_size_mean": float(st.sum(2).mean())})
res["config_shift_official_to_valid"] = xs
json.dump(res, open(f"{OUT}/{name}.json", "w"), default=float)
np.savez_compressed(f"{OUT}/{name}_Y.npz", **{k.replace("|", "__"): v for k, v in Ystore.items()}, valid_names=np.array(pop.get("valid", [])))
print("DONE", name, f"{time.time()-t0:.0f}s", flush=True)
