"""Aggregate all per-cell results into results/summary.json + CSV tables + preregistered RQ verdicts.
Groups (exploratory extension, 2026-10-17): default = preregistered instruments only (T1-T3, outputs unchanged: summary.json, *.csv);
--with-api = T1-T3 plus the T5 API-LLM cells (suffix _with_api); --api-only = T5 API-LLM cells only (suffix _api); --det = deterministic classifiers T1+T2+T6 (suffix _det); --schema-only = T6 only (suffix _schema). Shares are counted
over cells; types are never ranked against each other."""
import json, os, sys, glob, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr, f as fdist, beta
sys.path.insert(0, os.path.dirname(__file__)); from core import ROOT, TYPES, DATASETS, vc_mom
C = f"{ROOT}/results/cells"; OUT = f"{ROOT}/results"
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def cp_ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0; return [float(lo), float(hi)]
cells = {}
for f in sorted(glob.glob(f"{C}/*.json")):
    b = os.path.basename(f)[:-5]
    if b.count("__") != 2: continue
    cells[b] = json.load(open(f))
main = {k: v for k, v in cells.items() if not any(t in k for t in ("_tau", "_rawll", "_sub"))}
GROUP = "with_api" if "--with-api" in sys.argv else ("api" if "--api-only" in sys.argv else ("det" if "--det" in sys.argv else ("schema" if "--schema-only" in sys.argv else "prereg")))
# --det = deterministic-classifier group (T1 NLI, T2 bi-encoder, exploratory T6 schema classifiers); LLMs (T3/T5) are never in this group.
KINDS = {"prereg": ["nli", "bienc", "llm"], "with_api": ["nli", "bienc", "llm", "api"], "api": ["api"], "det": ["nli", "bienc", "schema"], "schema": ["schema"]}[GROUP]; SUF = "" if GROUP == "prereg" else f"_{GROUP}"
zs = {k: v for k, v in main.items() if v["kind"] in KINDS}
sup = {k: v for k, v in main.items() if v["kind"] == "sup"}
def Yof(k, key):
    p = f"{C}/{k}_Y.npz"
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); return z[key] if key in z else None
def ftest(Y):
    W, S = Y.shape; g = Y.mean(); rw = Y.mean(1); cs = Y.mean(0)
    msw = S * ((rw - g) ** 2).sum() / (W - 1); res = Y - rw[:, None] - cs[None, :] + g; mse = (res ** 2).sum() / ((W - 1) * (S - 1))
    F = msw / mse if mse > 0 else np.inf; return float(F), float(fdist.sf(F, W - 1, (W - 1) * (S - 1))) if np.isfinite(F) else 0.0
rows = []
for k, v in zs.items():
    n250 = 250 if 250 in v["n_grid"] else v["n_grid"][0]; nmax = v["n_grid"][-1]
    for met in ["set_size", "coverage"]:
        for n in v["n_grid"]:
            key = f"A|lac|valid|{n}"
            if key not in v["floor"]: continue
            s = v["floor"][key][met]
            rows.append({"cell": k, "dataset": v["dataset"], "kind": v["kind"], "model": v["model"], "metric": met, "n": n, "mean": s["mean"], "sd_w": s["sd_w"], "ci_sd_w_lo": s["ci_sd_w"][0],
                         "ci_sd_w_hi": s["ci_sd_w"][1], "sd_s": s["sd_s"], "ci_sd_s_lo": s["ci_sd_s"][0], "ci_sd_s_hi": s["ci_sd_s"][1], "sd_e": s["sd_e"], "share_w": s["share_w"],
                         "ci_share_w_lo": s["ci_share_w"][0], "ci_share_w_hi": s["ci_share_w"][1], "share_s": s["share_s"], "share_e": s["share_e"], "ratio_w_s": s["ratio_w_s"]})
VC = pd.DataFrame(rows); VC.to_csv(f"{OUT}/variance_components_valid_lac_a010{SUF}.csv", index=False)
summary = {"group": GROUP, "kinds": KINDS, "n_cells_zero_shot": len(zs), "n_cells_supervised": len(sup), "cells": sorted(zs), "verdicts": {}}
# ---------------- RQ1
r1 = []
for k, v in zs.items():
    n = 250 if 250 in v["n_grid"] else v["n_grid"][0]; s = v["floor"][f"A|lac|valid|{n}"]["set_size"]; Y = Yof(k, f"A__{n}__lac__set_size")
    F, p = ftest(Y) if Y is not None else (np.nan, np.nan)
    r1.append({"cell": k, "kind": v["kind"], "dataset": v["dataset"], "model": v["model"], "n": n, "sd_w": s["sd_w"], "ci_sd_w": s["ci_sd_w"], "sd_s": s["sd_s"], "ratio": s["ratio_w_s"],
               "ci_ratio": s["ci_ratio_w_s"], "share_w": s["share_w"], "ci_share_w": s["ci_share_w"], "F": F, "p": p})
if r1:
    adj = holm([x["p"] for x in r1])
    for x, a in zip(r1, adj): x["p_holm"] = float(a)
    h1a = np.mean([x["ci_sd_w"][0] > 0 and x["p_holm"] < 0.05 for x in r1]); h1b = np.mean([1 / 3 <= x["ratio"] <= 3 for x in r1])
    lt = np.mean([x["ratio"] < 1 / 3 for x in r1]); gt = np.mean([x["ratio"] > 3 for x in r1])
    summary["verdicts"]["RQ1"] = {"cells": r1, "share_cells_sd_w_ci_excludes_0_and_holm_sig": float(h1a), "share_cells_ratio_in_[1/3,3]": float(h1b),
        "share_cells_ratio_lt_1/3": float(lt), "share_cells_ratio_gt_3": float(gt),
        "H1a": "supported" if h1a >= 0.8 else ("refuted" if h1a <= 0.5 else "not supported (between thresholds)"),
        "H1b": "supported" if h1b >= 0.5 else ("refuted: sampling negligible (ratio>3 in majority)" if gt > 0.5 else ("refuted: wording negligible" if lt > 0.5 else "not supported"))}
    for kind in KINDS:
        sub = [x for x in r1 if x["kind"] == kind]
        if sub: summary["verdicts"]["RQ1"][f"by_type_{kind}"] = {"median_ratio": float(np.median([x["ratio"] for x in sub])), "median_share_w": float(np.median([x["share_w"] for x in sub])),
                                                                 "share_in_range": float(np.mean([1 / 3 <= x["ratio"] <= 3 for x in sub])), "n": len(sub)}
# ---------------- RQ2
r2 = []
for k, v in zs.items():
    ns = [n for n in v["n_grid"] if f"A|lac|valid|{n}" in v["floor"]]
    sdw = np.array([v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_w"] for n in ns]); sds = np.array([v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_s"] for n in ns]); ns = np.array(ns)
    ms = ns >= 100; mw = ns >= 250
    slope = lambda x, yv: float(np.polyfit(np.log(x), np.log(np.maximum(yv, 1e-9)), 1)[0]) if len(x) >= 2 else np.nan
    sl_s = slope(ns[ms], sds[ms]); sl_w = slope(ns[mw], sdw[mw])
    bw = np.array([v["floor"][f"A|lac|valid|{n}"]["_boot_sd"]["set_size"]["w"] for n in ns]); bs_ = np.array([v["floor"][f"A|lac|valid|{n}"]["_boot_sd"]["set_size"]["s"] for n in ns])
    bsl_w = [slope(ns[mw], bw[mw, b]) for b in range(bw.shape[1])]; bsl_s = [slope(ns[ms], bs_[ms, b]) for b in range(bs_.shape[1])]
    last = v["floor"][f"A|lac|valid|{ns[-1]}"]["set_size"]
    ws = v.get("w_subsample", {}).get(str(ns[-1])) or v.get("w_subsample", {}).get(ns[-1])
    w4 = ws["4"]["mean_sd_w"] / ws[str(max(int(x) for x in ws))]["mean_sd_w"] if ws and "4" in ws and ws[str(max(int(x) for x in ws))]["mean_sd_w"] > 0 else np.nan
    nstar = (last["sd_s"] / last["sd_w"]) ** 2 * ns[-1] if last["sd_w"] > 0 else np.inf   # extrapolated crossing with slope -0.5
    r2.append({"cell": k, "kind": v["kind"], "dataset": v["dataset"], "model": v["model"], "slope_sd_split_n100": sl_s, "ci_slope_split": [float(np.nanpercentile(bsl_s, 2.5)), float(np.nanpercentile(bsl_s, 97.5))],
               "slope_sd_w_n250": sl_w, "ci_slope_w": [float(np.nanpercentile(bsl_w, 2.5)), float(np.nanpercentile(bsl_w, 97.5))], "n_max": int(ns[-1]), "sd_w_nmax": last["sd_w"],
               "ci_sd_w_nmax": last["ci_sd_w"], "sd_s_nmax": last["sd_s"], "ratio_W4_over_Wmax": float(w4), "n_star_crossing": float(nstar),
               "sd_w_n_min": float(sdw[0]), "sd_s_n_min": float(sds[0])})
if r2:
    a = np.mean([-0.65 <= x["slope_sd_split_n100"] <= -0.35 for x in r2]); b_ = np.mean([x["slope_sd_w_n250"] > -0.15 and x["ci_sd_w_nmax"][0] > 0 for x in r2])
    dec = np.mean([x["slope_sd_w_n250"] <= -0.35 for x in r2]); zero = np.mean([x["ci_sd_w_nmax"][0] <= 0 for x in r2])
    c_ = np.mean([0.8 <= x["ratio_W4_over_Wmax"] <= 1.25 for x in r2 if np.isfinite(x["ratio_W4_over_Wmax"])])
    summary["verdicts"]["RQ2"] = {"cells": r2, "share_H2a_split_slope_in_range": float(a), "share_H2b_flat_wording_floor": float(b_), "share_wording_declining": float(dec),
        "share_wording_ci_includes_0_at_nmax": float(zero), "share_H2c_W4_estimate_within_0.8_1.25": float(c_),
        "H2a": "supported" if a >= 2 / 3 else "not supported", "H2b": "supported" if b_ >= 2 / 3 else ("refuted" if (dec > 0.5 or zero > 0.5) else "not supported"),
        "H2c": "supported" if c_ >= 2 / 3 else "not supported"}
# ---------------- RQ3
r3 = []
for k, v in zs.items():
    N = v["N"]; fl = v["flip_vs_official"]
    surf = [fl[s] for s in fl if s.startswith("surface_")]; val = [fl[s] for s in fl if s.startswith("valid_")]
    kk = int(round(max(surf) * N)) if surf else 0
    row = {"cell": k, "kind": v["kind"], "dataset": v["dataset"], "model": v["model"], "surface_flip_mean": float(np.mean(surf)) if surf else np.nan,
           "surface_flip_max": float(max(surf)) if surf else np.nan, "surface_flip_max_ci": cp_ci(kk, N) if surf else None, "valid_flip_mean": float(np.mean(val)) if val else np.nan}
    row["surface_over_valid"] = row["surface_flip_mean"] / row["valid_flip_mean"] if val and row["valid_flip_mean"] > 0 else np.nan
    n = 250 if 250 in v["n_grid"] else v["n_grid"][0]
    if f"A|lac|all|{n}" in v["floor"]:
        row["sd_w_valid"] = v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_w"]; row["sd_w_all"] = v["floor"][f"A|lac|all|{n}"]["set_size"]["sd_w"]
        row["sd_w_strict"] = v["floor"][f"A|lac|strict|{n}"]["set_size"]["sd_w"] if f"A|lac|strict|{n}" in v["floor"] else np.nan
    r3.append(row)
dose = {}
for kind in KINDS:
    pts = [(x["distance"], x["tv"], x["flip"], x["valid"], v["dataset"]) for k, v in zs.items() if v["kind"] == kind for x in v.get("dose_response", [])]
    if len(pts) < 10: continue
    d_ = np.array([p[0] for p in pts]); tv = np.array([p[1] for p in pts]); fl = np.array([p[2] for p in pts]); rng = np.random.default_rng(0)
    bsr = [spearmanr(d_[i], tv[i])[0] for i in [rng.integers(0, len(pts), len(pts)) for _ in range(1000)]]
    vm = np.array([p[3] for p in pts])
    dose[kind] = {"n_variant_model_pairs": len(pts), "spearman_dist_tv": float(spearmanr(d_, tv)[0]), "ci": [float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))],
                  "spearman_dist_flip": float(spearmanr(d_, fl)[0]), "mean_tv_valid": float(tv[vm].mean()), "mean_tv_invalid": float(tv[~vm].mean()) if (~vm).any() else None,
                  "by_dataset": {ds: float(spearmanr(d_[[p[4] == ds for p in pts]], tv[[p[4] == ds for p in pts]])[0]) for ds in DATASETS if sum(p[4] == ds for p in pts) > 5}}
if r3:
    h3a = {kind: any((x["surface_flip_max_ci"] or [0])[0] > 0 for x in r3 if x["kind"] == kind) for kind in KINDS}
    ratio_major = np.nanmean([x["surface_over_valid"] >= 0.5 for x in r3 if np.isfinite(x["surface_over_valid"])])
    both = [x for x in r3 if "sd_w_all" in x]; h3c = np.mean([x["sd_w_valid"] < x["sd_w_all"] for x in both]) if both else np.nan
    summary["verdicts"]["RQ3"] = {"cells": r3, "dose_response": dose, "H3a_by_type": h3a, "share_cells_surface_ge_half_valid": float(ratio_major), "share_H3c_valid_lt_all": float(h3c),
        "H3b": {k: ("supported" if d["ci"][0] > 0 else "not supported") for k, d in dose.items()},
        "H3c": "supported" if h3c >= 2 / 3 else "not supported", "definitional_reading_refuted_by_surface_control": bool(ratio_major > 0.5)}
# ---------------- RQ4 (AG vs BBC)
r4 = []
for k, v in zs.items():
    if v["dataset"] != "agnews": continue
    kb = k.replace("agnews__", "bbc2025__")
    if kb not in zs: continue
    b = zs[kb]; n = 250
    vals = lambda c: [c["point"][s]["accuracy"] for s in c["point"] if s.startswith("valid_")]
    r4.append({"kind": v["kind"], "model": v["model"], "acc_official_ag": v["point"]["official"]["accuracy"], "acc_official_bbc": b["point"]["official"]["accuracy"],
               "acc_valid_mean_ag": float(np.mean(vals(v))), "acc_valid_mean_bbc": float(np.mean(vals(b))),
               "sd_w_ag": v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_w"], "ci_sd_w_ag": v["floor"][f"A|lac|valid|{n}"]["set_size"]["ci_sd_w"],
               "sd_w_bbc": b["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_w"], "ci_sd_w_bbc": b["floor"][f"A|lac|valid|{n}"]["set_size"]["ci_sd_w"],
               "share_maxp_gt_099_ag": v["point"]["official"]["share_maxp_gt_099"], "share_maxp_gt_099_bbc": b["point"]["official"]["share_maxp_gt_099"]})
mk = {}
for f in glob.glob(f"{OUT}/contamination/minkprob__*.json"):
    z = json.load(open(f)); m = z["model"].split("/")[-1]; mk[m] = {}
    for ds, dd in z["datasets"].items():
        x = np.array([r["mink20"] for r in dd["rows"]]); rng = np.random.default_rng(0); bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
        mk[m][ds] = {"mean": float(x.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "mean_lp": dd["mean_lp"], "n": len(x)}
summary["verdicts"]["RQ4"] = {"pairs": r4, "minkprob": mk,
    "H4b_acc_lower_on_bbc_share": float(np.mean([x["acc_valid_mean_bbc"] < x["acc_valid_mean_ag"] for x in r4])) if r4 else None,
    "H4b_sdw_larger_on_bbc_share": float(np.mean([x["sd_w_bbc"] > x["sd_w_ag"] for x in r4])) if r4 else None,
    "H4a": {m: (d["agnews"]["mean"] - d["bbc2025"]["mean"]) for m, d in mk.items() if "agnews" in d and "bbc2025" in d}}
# ---------------- supervised reference
sr = []
for k, v in sup.items():
    nmax = v["n_grid"][-1]; f250 = v["floor"]["A|lac|trainboot|250"]["set_size"]; fm = v["floor"][f"A|lac|trainboot|{nmax}"]["set_size"]
    sr.append({"cell": k, "dataset": v["dataset"], "model": v["model"], "trained_on": v["trained_on"], "acc": v["point"]["reference"]["accuracy"], "sd_train_250": f250["sd_w"],
               "ci_sd_train_250": f250["ci_sd_w"], "sd_split_250": f250["sd_s"], "share_train_250": f250["share_w"], "sd_train_nmax": fm["sd_w"], "sd_split_nmax": fm["sd_s"], "n_max": nmax})
summary["supervised"] = sr
# ---------------- grid summaries (coverage validity across methods/alpha)
g = []
for k, v in zs.items():
    for key, blk in v.get("grid", {}).items():
        m, a, n = key.split("|"); g.append({"cell": k, "kind": v["kind"], "dataset": v["dataset"], "model": v["model"], "method": m, "alpha": float(a), "n": int(n),
               **{f"{met}_mean": blk[met]["mean"] for met in blk}, **{f"{met}_share_w": blk[met]["share_w"] for met in blk}, **{f"{met}_sd_w": blk[met]["sd_w"] for met in blk},
               **{f"{met}_sd_s": blk[met]["sd_s"] for met in blk}})
G = pd.DataFrame(g); G.to_csv(f"{OUT}/grid_methods_alpha{SUF}.csv", index=False)
if len(G):
    summary["coverage_check"] = {"max_abs_mean_coverage_minus_target": float((G["coverage_mean"] - (1 - G["alpha"])).abs().max()),
                                 "by_method": G.assign(dev=G["coverage_mean"] - (1 - G["alpha"])).groupby("method")["dev"].describe().to_dict()}
    summary["share_w_by_method_alpha_n250"] = G[G.n == 250].groupby(["kind", "method", "alpha"])["set_size_share_w"].median().reset_index().to_dict("records")
# point metrics table
pts = []
for k, v in zs.items():
    vacc = [v["point"][s]["accuracy"] for s in v["point"] if s.startswith("valid_")]
    pts.append({"cell": k, "kind": v["kind"], "dataset": v["dataset"], "model": v["model"], "N": v["N"], "acc_official": v["point"]["official"]["accuracy"],
                "acc_valid_mean": float(np.mean(vacc)), "acc_valid_min": float(np.min(vacc)), "acc_valid_max": float(np.max(vacc)), "ece_official": v["point"]["official"]["ece"],
                "maxp_gt_099_official": v["point"]["official"]["share_maxp_gt_099"],
                "acc_nonsense": float(np.mean([v["point"][s]["accuracy"] for s in v["point"] if s.startswith("N")])) if any(s.startswith("N") for s in v["point"]) else np.nan,
                "acc_shuffled": float(np.mean([v["point"][s]["accuracy"] for s in v["point"] if s.startswith("shuffled")])) if any(s.startswith("shuffled") for s in v["point"]) else np.nan,
                "identity_flip": v["flip_vs_official"].get("identity", np.nan)})
pd.DataFrame(pts).to_csv(f"{OUT}/point_metrics{SUF}.csv", index=False); summary["point"] = pts
json.dump(summary, open(f"{OUT}/summary{SUF}.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("cells", "pairs")} for k, v in summary["verdicts"].items()}, indent=1, default=str)[:6000])
