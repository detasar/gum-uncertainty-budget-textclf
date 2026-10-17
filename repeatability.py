"""Test-retest repeatability (metrology sense) of each measurement setup.
Inputs: cache/mac/rep_clef.jsonl, cache/mac/rep_llm_*.jsonl (identical requests repeated R>=4 times).
Outputs: repeatability.json, saturation.png"""
import json, os, glob, itertools, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
Z = json.load(open(f"{C}/zs_set.json")); YCAL = np.array(Z["cal_y"]); YTE = np.array(Z["test_y"])
ALPHA = 0.10

def softmax(z):
    z = np.asarray(z, float); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def norm(P):
    P = np.asarray(P, float); P = np.where(P.sum(-1, keepdims=True) > 0, P, 0.25); return P / P.sum(-1, keepdims=True)

def icc(Y):
    """Y: items x runs. Returns ICC(2,1), ICC(3,1) (Shrout & Fleiss 1979). None if undefined (no variance at all)."""
    n, k = Y.shape; g = Y.mean()
    SSR = k * ((Y.mean(1) - g) ** 2).sum(); SSC = n * ((Y.mean(0) - g) ** 2).sum(); SST = ((Y - g) ** 2).sum()
    SSE = max(SST - SSR - SSC, 0.0); MSR = SSR / (n - 1); MSC = SSC / (k - 1); MSE = SSE / ((n - 1) * (k - 1))
    d2 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n; d3 = MSR + (k - 1) * MSE
    return (float((MSR - MSE) / d2) if d2 > 0 else None, float((MSR - MSE) / d3) if d3 > 0 else None)

def fleiss(L, K=4):
    """L: items x runs labels."""
    n, k = L.shape; cnt = np.stack([(L == c).sum(1) for c in range(K)], 1)
    P_i = ((cnt ** 2).sum(1) - k) / (k * (k - 1)); Pbar = P_i.mean(); p_j = cnt.sum(0) / (n * k); Pe = (p_j ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else None

def conformal(cal_p, cal_y, te_p, te_y):
    s = 1 - cal_p[np.arange(len(cal_y)), cal_y]; n = len(s); lvl = min(1, math.ceil((n + 1) * (1 - ALPHA)) / n)
    q = np.quantile(s, lvl, method="higher"); sets = (1 - te_p) <= q
    return sets, float(sets.sum(1).mean()), float(sets[np.arange(len(te_y)), te_y].mean())

def js(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1); m = (p + q) / 2
    return 0.5 * (p * np.log2(p / m)).sum(-1) + 0.5 * (q * np.log2(q / m)).sum(-1)

def analyse(cal, te, labels_cal=None, labels_te=None):
    """cal/te: runs x items x K probability arrays. labels_*: runs x items chosen labels (if None, argmax)."""
    R = cal.shape[0]; P = np.concatenate([cal, te], 1)                 # runs x N x K
    y = np.concatenate([YCAL[:cal.shape[1]], YTE[:te.shape[1]]])
    N = P.shape[1]
    sd_item = P.std(0, ddof=1)                                         # N x K
    maxdiff = float(np.abs(P[:, None] - P[None, :]).max())
    ptrue = P[:, np.arange(N), y].T                                    # N x R
    i2, i3 = icc(ptrue)
    d = ptrue[:, 1] - ptrue[:, 0]
    pairs = list(itertools.combinations(range(R), 2))
    tv = [float(0.5 * np.abs(P[a] - P[b]).sum(-1).mean()) for a, b in pairs]
    jsd = [float(js(P[a], P[b]).mean()) for a, b in pairs]
    L = (np.concatenate([labels_cal, labels_te], 1) if labels_cal is not None else P.argmax(-1)).T   # N x R
    kap = [float(cohen_kappa_score(L[:, a], L[:, b])) if len(set(L[:, a]) | set(L[:, b])) > 1 else None for a, b in pairs]
    agree_all = float((L == L[:, :1]).all(1).mean())
    conf = [conformal(cal[r], YCAL[:cal.shape[1]], te[r], YTE[:te.shape[1]]) for r in range(R)]
    S = np.stack([c[0] for c in conf]); set_changed = float((S != S[:1]).any(0).any(1).mean())
    mx = P.max(-1)
    return {"runs": R, "items": N, "max_abs_dp_any_pair": maxdiff, "mean_item_sd_p": float(sd_item.mean()), "max_item_sd_p": float(sd_item.max()),
            "icc21_p_true": i2, "icc31_p_true": i3,
            "bland_altman_run1_vs_run0_p_true": {"mean_diff": float(d.mean()), "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_high": float(d.mean() + 1.96 * d.std(ddof=1))},
            "mean_pairwise_total_variation": float(np.mean(tv)), "mean_pairwise_js_bits": float(np.mean(jsd)),
            "label_source": "chosen answer" if labels_cal is not None else "argmax",
            "cohen_kappa_pairwise_mean": (float(np.mean([k for k in kap if k is not None])) if any(k is not None for k in kap) else None),
            "cohen_kappa_pairwise_range": ([float(min(k for k in kap if k is not None)), float(max(k for k in kap if k is not None))] if any(k is not None for k in kap) else None),
            "fleiss_kappa": fleiss(L), "pct_items_identical_label_all_runs": agree_all,
            "set_size_per_run": [c[1] for c in conf], "coverage_per_run": [c[2] for c in conf],
            "sd_set_size_across_runs": float(np.std([c[1] for c in conf], ddof=1)), "sd_coverage_across_runs": float(np.std([c[2] for c in conf], ddof=1)),
            "pct_test_items_whose_conformal_set_changes": set_changed,
            "saturation": {"pct_max_p_gt_0.99": float((mx > 0.99).mean()), "pct_entries_outside_0.01_0.99": float(((P < 0.01) | (P > 0.99)).mean()),
                           "median_max_p": float(np.median(mx))}}, mx.ravel()

OUT = {}; HIST = {}
# Clef
f = f"{C}/mac/rep_clef.jsonl"
if os.path.exists(f):
    D = {}
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        D[(r["w"], r["run"], r["split"], r["i"])] = softmax(r["logits"])
    for w in sorted({k[0] for k in D}):
        nc = 1 + max(k[3] for k in D if k[0] == w and k[2] == "cal")
        runs = [r for r in sorted({k[1] for k in D if k[0] == w}) if all((w, r, s, i) in D for s in ("cal", "test") for i in range(nc))]
        try:
            cal = np.stack([np.stack([D[(w, r, "cal", i)] for i in range(nc)]) for r in runs])
            te = np.stack([np.stack([D[(w, r, "test", i)] for i in range(nc)]) for r in runs])
        except KeyError: continue
        if len(runs) < 2: continue
        OUT[f"clef-flash-8bit | {w} | deterministic"], h = analyse(cal, te); HIST.setdefault("clef-flash-8bit", h)
# LLMs
for f in sorted(glob.glob(f"{C}/mac/rep_llm_*.jsonl")):
    tag = os.path.basename(f)[8:-6]; D = {}
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        D[(r["w"], r["mode"], r["run"], r["split"], r["i"])] = (norm(r["p"]), "ABCD".find(r["answer"].strip()[:1]) if r["answer"].strip()[:1] in "ABCD" else -1)
    for w, mode in sorted({(k[0], k[1]) for k in D}):
        nc = 1 + max(k[4] for k in D if k[:2] == (w, mode) and k[3] == "cal")
        runs = [r for r in sorted({k[2] for k in D if k[:2] == (w, mode)}) if all((w, mode, r, s, i) in D for s in ("cal", "test") for i in range(nc))]
        try:
            g = lambda s, j: np.stack([np.stack([D[(w, mode, r, s, i)][j] for i in range(nc)]) for r in runs])
            cal, te, lc, lt = g("cal", 0), g("test", 0), g("cal", 1), g("test", 1)
        except KeyError: continue
        if len(runs) < 2: continue
        res, h = analyse(cal, te, lc, lt); res["argmax_vs_chosen_mismatch_rate"] = float((np.concatenate([lc, lt], 1) != np.concatenate([cal, te], 1).argmax(-1)).mean())
        OUT[f"{tag} | {w} | {mode}"] = res
        if mode == "t0": HIST.setdefault(tag, h)
json.dump(OUT, open(f"{HERE}/repeatability.json", "w"), indent=2)
for k, v in OUT.items():
    print(f"{k:50s} R={v['runs']} N={v['items']} max|dp|={v['max_abs_dp_any_pair']:.3g} sd_item={v['mean_item_sd_p']:.3g} ICC3={v['icc31_p_true']} "
          f"TV={v['mean_pairwise_total_variation']:.3g} kappa={v['cohen_kappa_pairwise_mean']} fleiss={v['fleiss_kappa']} sd_set={v['sd_set_size_across_runs']:.3g} "
          f"set_changes={v['pct_test_items_whose_conformal_set_changes']:.3g} sat={v['saturation']}")
if HIST:
    fig, axes = plt.subplots(1, len(HIST), figsize=(4 * len(HIST), 3.2), squeeze=False)
    for ax, (k, h) in zip(axes[0], HIST.items()):
        ax.hist(h, bins=np.linspace(0.25, 1, 31), color="#2c7fb8"); ax.set_title(k, fontsize=9); ax.set_xlabel("max class probability (one run)")
        ax.text(0.03, 0.9, f"p>0.99: {100*(h>0.99).mean():.0f}%", transform=ax.transAxes, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{HERE}/saturation.png", dpi=150); plt.close()

# cross-session check: main run (separate process, ~1 h earlier) vs repeatability run 0, same requests
def cross_session(main_path, rep_path, main_key, rep_filter, main_cfg, val):
    if not (os.path.exists(main_path) and os.path.exists(rep_path)): return None
    M = {}
    for line in open(main_path):
        try: r = json.loads(line)
        except Exception: continue
        if r["cfg"] == main_cfg and r.get("rep", 0) == 0: M[(r["split"], r["i"])] = np.array(r[main_key], float)
    diffs = []
    for line in open(rep_path):
        try: r = json.loads(line)
        except Exception: continue
        if rep_filter(r) and r["run"] == 0 and (r["split"], r["i"]) in M:
            diffs.append(np.abs(val(np.array(r[main_key], float)) - val(M[(r["split"], r["i"])])).max())
    return {"n_items": len(diffs), "max_abs_dp": float(max(diffs)) if diffs else None}
CS = {"clef-flash-8bit w0|q0 (probabilities)": cross_session(f"{C}/mac/clef_logits.jsonl", f"{C}/mac/rep_clef.jsonl", "logits",
                                                             lambda r: r["w"] == "w0_official", "w0_official|q0_topic", softmax),
      "qwen3.5 w0 t0 (A-D probabilities)": cross_session(f"{C}/mac/llm_qwen3.5.jsonl", f"{C}/mac/rep_llm_qwen3.5.jsonl", "p",
                                                       lambda r: r["w"] == "w0_official" and r["mode"] == "t0", "w0_official", norm)}
OUT["_cross_session_main_vs_rep_run0"] = CS
json.dump(OUT, open(f"{HERE}/repeatability.json", "w"), indent=2)
print("cross-session:", CS)
