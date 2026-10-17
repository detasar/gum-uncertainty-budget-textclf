"""All v2 figures (matplotlib, Okabe-Ito colour-blind-safe palette). Comparisons are WITHIN instrument type only."""
import json, os, sys, glob, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(__file__)); from core import ROOT, TYPES, DATASETS
C = f"{ROOT}/results/cells"; F = f"{ROOT}/figures"; os.makedirs(F, exist_ok=True)
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
COMP = {"wording": OI["verm"], "split": OI["blue"], "resid": "#BBBBBB", "train": OI["green"]}
MARK = ["o", "s", "^", "D", "v"]; LS = ["-", "--", ":", "-."]
DSN = {"agnews": "AG News", "bbc2025": "BBC 2024-25", "sst2": "SST-2", "dbpedia": "DBpedia-14", "massive": "MASSIVE-18"}
KN = {"nli": "T1 NLI cross-encoders", "bienc": "T2 bi-encoders", "llm": "T3 small LLMs (PMI)", "sup": "T4 supervised probes"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 220,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
FOOT = "Compare within an instrument type only; types differ in output and information used (not a leaderboard)."
def save(fig, name, foot=True):
    if foot: fig.text(0.01, 0.003, FOOT, fontsize=7, color="#555555", ha="left", va="bottom")
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight"); plt.close(fig); print("saved", name)
cells = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{C}/*.json")) if os.path.basename(f)[:-5].count("__") == 2}
main = {k: v for k, v in cells.items() if not any(t in k for t in ("_tau", "_rawll", "_sub"))}
zs = {k: v for k, v in main.items() if v["kind"] in ("nli", "bienc", "llm")}; sup = {k: v for k, v in main.items() if v["kind"] == "sup"}
S = json.load(open(f"{ROOT}/results/summary.json")) if os.path.exists(f"{ROOT}/results/summary.json") else {}
def models_of(kind): return sorted({v["model"] for v in zs.values() if v["kind"] == kind}) if kind != "sup" else sorted({v["model"] for v in sup.values()})
def short(m): return m.replace("-zeroshot-v2.0", "-zs").replace("nli-deberta-v3-small", "nli-deberta-v3-sm").replace("-Instruct", "").replace("all-MiniLM-L6-v2", "MiniLM-L6").replace("+LogReg", "+LR")
def n250(v): return 250 if 250 in v["n_grid"] else v["n_grid"][0]
ZK = [k for k in ["nli", "bienc", "llm"] if models_of(k)]   # zero-shot types that have at least one cell
from matplotlib.ticker import NullFormatter

# ---------------------------------------------------------------- fig01 variance shares at n=250
def fig01():
    kinds = [k for k in ["nli", "bienc", "llm", "sup"] if (models_of(k))]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 5.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        rows = []
        pool = zs if kind != "sup" else sup
        for ds in DATASETS:
            for m in models_of(kind):
                k = f"{ds}__{kind}__{m.replace('+LogReg', '')}" if kind == "sup" else f"{ds}__{kind}__{m}"
                if k not in pool: continue
                v = pool[k]; key = f"A|lac|valid|{n250(v)}" if kind != "sup" else "A|lac|trainboot|250"
                s = v["floor"][key]["set_size"]; rows.append((f"{DSN[ds]} · {short(m)}", s["share_w"], s["share_s"], s["share_e"], s["ci_share_w"]))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, w, s_, e, ci) in zip(y, rows):
            ax.barh(yi, w, color=COMP["train" if kind == "sup" else "wording"], height=0.75); ax.barh(yi, s_, left=w, color=COMP["split"], height=0.75)
            ax.barh(yi, e, left=w + s_, color=COMP["resid"], height=0.75); ax.errorbar(w, yi, xerr=[[w - ci[0]], [ci[1] - w]], fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.set_xlim(0, 1); ax.set_title(KN[kind]); ax.set_xlabel("share of variance of mean set size")
    h = [Patch(color=COMP["wording"], label="label wording (valid paraphrase sets, random)"), Patch(color=COMP["train"], label="training-sample bootstrap (T4 only)"),
         Patch(color=COMP["split"], label="calibration/test split sampling"), Patch(color=COMP["resid"], label="wording × split interaction + residual")]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Variance shares of mean conformal set size at n_cal = 250 (LAC, α = 0.10, test = 300); bars: 95% two-way bootstrap CI of the first component", y=1.07, fontsize=10)
    fig.tight_layout(); save(fig, "fig01_variance_shares_n250")

# ---------------------------------------------------------------- fig02 n-scaling with bands
def fig02(metric="set_size", design="A", method="lac", name="fig02_nscaling_setsize"):
    kinds = [k for k in ["nli", "bienc", "llm", "sup"] if models_of(k)]
    fig, axes = plt.subplots(len(kinds), len(DATASETS), figsize=(16, 3.0 * len(kinds)), sharex=True, squeeze=False)
    for i, kind in enumerate(kinds):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for mi, m in enumerate(models_of(kind)):
                k = f"{ds}__{kind}__{m.replace('+LogReg', '')}"; pool = sup if kind == "sup" else zs
                if k not in pool: continue
                v = pool[k]; pop = "trainboot" if kind == "sup" else "valid"
                ns = [n for n in v["n_grid"] if f"{design}|{method}|{pop}|{n}" in v["floor"]]
                if not ns: continue
                bl = [v["floor"][f"{design}|{method}|{pop}|{n}"][metric] for n in ns]
                sw = np.array([b["sd_w"] for b in bl]); ss = np.array([b["sd_s"] for b in bl])
                ax.plot(ns, ss, color=COMP["split"], ls=LS[mi % 4], marker=MARK[mi], ms=3, lw=1.2)
                ax.fill_between(ns, [b["ci_sd_s"][0] for b in bl], [b["ci_sd_s"][1] for b in bl], color=COMP["split"], alpha=0.12, lw=0)
                ax.plot(ns, sw, color=COMP["train" if kind == "sup" else "wording"], ls=LS[mi % 4], marker=MARK[mi], ms=3, lw=1.2)
                ax.fill_between(ns, np.maximum([b["ci_sd_w"][0] for b in bl], 1e-4), [b["ci_sd_w"][1] for b in bl], color=COMP["train" if kind == "sup" else "wording"], alpha=0.12, lw=0)
            xs = np.array([50, 2300]); ax.plot(xs, 0.6 * (xs / 50) ** -0.5, color="#777777", lw=0.8, ls=(0, (1, 2)))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-4, 2)
            if i == 0: ax.set_title(DSN[ds])
            if j == 0: ax.set_ylabel(f"{KN[kind]}\nSD of {metric.replace('_', ' ')}")
            if i == len(kinds) - 1: ax.set_xlabel("calibration size n")
    h = [Line2D([], [], color=COMP["wording"], label="σ wording (valid paraphrases)"), Line2D([], [], color=COMP["train"], label="σ training bootstrap (T4)"),
         Line2D([], [], color=COMP["split"], label="σ split sampling"), Line2D([], [], color="#777777", ls=(0, (1, 2)), label="slope −1/2 reference")]
    h += [Line2D([], [], color="black", ls=LS[i % 4], marker=MARK[i], ms=3, label=f"model {i+1} of type (see legend table)") for i in range(3)]
    fig.legend(handles=h, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"Scaling with calibration size (design {design}, {method.upper()}, α = 0.10); bands = 95% two-way bootstrap CI", y=1.05)
    txt = " | ".join(f"{KN[k].split()[0]}: " + ", ".join(f"{i+1}={short(m)}" for i, m in enumerate(models_of(k))) for k in kinds)
    fig.text(0.01, -0.02, "Models: " + txt, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, name)

# ---------------------------------------------------------------- fig03 normalised floor curves
def fig03():
    fig, axes = plt.subplots(1, len(ZK), figsize=(4.4 * len(ZK), 4), squeeze=False); axes = axes[0]
    for ax, kind in zip(axes, ZK):
        for k, v in zs.items():
            if v["kind"] != kind: continue
            ns = [n for n in v["n_grid"] if f"A|lac|valid|{n}" in v["floor"]]; ref = v["floor"][f"A|lac|valid|{n250(v)}"]["set_size"]
            sw = [v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_w"] / max(ref["sd_w"], 1e-9) for n in ns]; ss = [v["floor"][f"A|lac|valid|{n}"]["set_size"]["sd_s"] / max(ref["sd_s"], 1e-9) for n in ns]
            ax.plot(ns, sw, color=COMP["wording"], alpha=0.6, lw=1); ax.plot(ns, ss, color=COMP["split"], alpha=0.6, lw=1)
        xs = np.array([50, 2300]); ax.plot(xs, (xs / 250) ** -0.5, color="#777", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(KN[kind]); ax.set_xlabel("calibration size n"); ax.axhline(1, color="#ccc", lw=0.6)
    axes[0].set_ylabel("SD relative to its value at n = 250")
    h = [Line2D([], [], color=COMP["wording"], label="σ wording (one line per dataset × model)"), Line2D([], [], color=COMP["split"], label="σ split sampling"), Line2D([], [], color="#777", ls=":", label="n^(−1/2)")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("The 'floor': sampling SD falls ~n^(−1/2); the between-wording SD stays flat (it is a systematic difference between instruments)", y=1.09)
    fig.tight_layout(); save(fig, "fig03_floor_normalised")

# ---------------------------------------------------------------- fig04 dose-response
def fig04():
    fig, axes = plt.subplots(1, len(ZK), figsize=(4.4 * len(ZK), 4.2), sharey=True, squeeze=False); axes = axes[0]
    gcol = {"G1_surface": OI["black"], "G2_wn_syn": OI["sky"], "G3_wn_hyper": OI["blue"], "G4_keywords": OI["orange"], "G5_definition": OI["green"], "G6_llm1": OI["purple"], "G7_llm2": OI["purple"]}
    for ax, kind in zip(axes, ZK):
        for k, v in zs.items():
            if v["kind"] != kind: continue
            for x in v.get("dose_response", []):
                g = next((g for g in x["generators"] if g in gcol), "v1"); c = gcol.get(g, OI["verm"])
                ax.scatter(x["distance"], max(x["tv"], 1e-4), s=12, color=c, marker="o" if x["valid"] else "x", alpha=0.6, lw=0.8)
        d = S.get("verdicts", {}).get("RQ3", {}).get("dose_response", {}).get(kind)
        if d: ax.text(0.02, 0.97, f"Spearman ρ(distance, TV) = {d['spearman_dist_tv']:.2f}\n95% CI [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}], n = {d['n_variant_model_pairs']}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_yscale("log"); ax.set_title(KN[kind]); ax.set_xlabel("semantic distance to official name (1 − judge cosine)")
    axes[0].set_ylabel("prediction shift: mean total-variation distance\n(only one class's label replaced)")
    h = [Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in [("surface (case)", OI["black"]), ("WordNet synonym", OI["sky"]), ("WordNet hypernym", OI["blue"]), ("χ² keywords", OI["orange"]),
         ("definition", OI["green"]), ("LLM paraphrase", OI["purple"]), ("v1 hand-picked", OI["verm"])]] + [Line2D([], [], marker="x", ls="", color="k", label="invalid (judge/audit)")]
    fig.legend(handles=h, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Dose-response: the further a label variant is in meaning, the more predictions move", y=1.09); fig.tight_layout(); save(fig, "fig04_dose_response")

# ---------------------------------------------------------------- fig05 method x alpha grids
def fig05():
    G = pd.read_csv(f"{ROOT}/results/grid_methods_alpha.csv"); G = G[G.n == 250]
    fig, axes = plt.subplots(2, len(ZK), figsize=(5 * len(ZK), 8), squeeze=False)
    for j, kind in enumerate(ZK):
        g = G[G.kind == kind]
        if g.empty: continue
        g = g.assign(row=g.dataset.map(DSN) + " · " + g.model.map(short), col=g.method.str.upper() + " α=" + g.alpha.astype(str))
        for i, (val, title, cmap, vmin, vmax) in enumerate([("set_size_share_w", "share of variance from wording (set size)", "Oranges", 0, 1),
                                                              ("coverage_mean", "mean coverage (target 1−α)", "Blues", 0.75, 1.0)]):
            pv = g.pivot_table(index="row", columns="col", values=val); ax = axes[i, j]
            im = ax.imshow(pv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=7)
            for (a, b), z in np.ndenumerate(pv.values):
                if np.isfinite(z): ax.text(b, a, f"{z:.2f}", ha="center", va="center", fontsize=5.5, color="black")
            ax.set_title(f"{KN[kind]}: {title}", fontsize=9); fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Method × α grid at n_cal = 250 (valid wording population, design A)", y=1.0); fig.tight_layout(); save(fig, "fig05_method_alpha_grid")

# ---------------------------------------------------------------- fig06 coverage / set size vs alpha per method
def fig06():
    G = pd.read_csv(f"{ROOT}/results/grid_methods_alpha.csv"); G = G[G.n == 250]
    fig, axes = plt.subplots(2, len(ZK), figsize=(4.7 * len(ZK), 7), squeeze=False)
    mc = {"lac": OI["blue"], "aps": OI["orange"], "raps": OI["green"]}
    for j, kind in enumerate(ZK):
        g = G[G.kind == kind]
        for (cell, method), gg in g.groupby(["cell", "method"]):
            gg = gg.sort_values("alpha")
            axes[0, j].plot(gg.alpha, gg.coverage_mean - (1 - gg.alpha), color=mc[method], alpha=0.5, marker="o", ms=2, lw=0.8)
            axes[1, j].plot(gg.alpha, gg.set_size_mean, color=mc[method], alpha=0.5, marker="o", ms=2, lw=0.8)
        axes[0, j].axhline(0, color="k", lw=0.6); axes[0, j].set_title(KN[kind]); axes[1, j].set_xlabel("α"); axes[1, j].set_yscale("log")
    axes[0, 0].set_ylabel("mean coverage − (1 − α)"); axes[1, 0].set_ylabel("mean set size (averaged over wordings & splits)")
    fig.legend(handles=[Line2D([], [], color=c, label=m.upper()) for m, c in mc.items()], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Coverage is valid on average for all methods; efficiency differs by method (one line per dataset × model)", y=1.05); fig.tight_layout(); save(fig, "fig06_coverage_setsize_vs_alpha")

# ---------------------------------------------------------------- fig07 repeatability
def fig07():
    p = f"{ROOT}/results/repeatability.json"
    if not os.path.exists(p): return
    R = json.load(open(p)); rows, cols = [], []
    for inst in ["nli", "bienc", "llm"]:
        if inst not in R: continue
        r = R[inst]; rows.append((inst, "4 identical runs (fresh processes)", r["official"]["max_abs_dp"], r["official"]["share_test_sets_changed"]))
        for c, row in r["conditions_vs_rep0"].items(): rows.append((inst, c.replace("_", " "), row["official"]["max_abs_dp"], row["official"]["share_test_sets_changed"]))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.2))
    vals = np.array([math.log10(max(r[2], 1e-12)) for r in rows]); y = np.arange(len(rows))[::-1]
    col = [OI["green"] if r[2] == 0 else (OI["orange"] if r[2] < 1e-3 else OI["verm"]) for r in rows]
    ax.barh(y, vals + 12.5, left=-12.5, color=col)
    for yi, r in zip(y, rows): ax.text(0.3, yi, f"max|Δp| = {r[2]:.2g}; conformal sets changed: {100*r[3]:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([f"{KN[r[0]].split()[0]} · {r[1]}" for r in rows], fontsize=7); ax.set_xlim(-12.5, 7); ax.set_xlabel("log10 max |Δp| vs run 0 (−12 = exactly 0)")
    ax.set_title("Repeatability (identical runs) vs intermediate precision (batch size, threads, dtype), AG News, official labels")
    fig.tight_layout(); save(fig, "fig07_repeatability")

# ---------------------------------------------------------------- fig08 heatmaps model x paraphrase set
def fig08(metric="accuracy"):
    for ds in DATASETS:
        rows, labels = [], []
        for kind in ["nli", "bienc", "llm"]:
            for m in models_of(kind):
                k = f"{ds}__{kind}__{m}"
                if k not in zs: continue
                v = zs[k]; names = ["official"] + [f"valid_{i:02d}" for i in range(30)] + ["N1_nonsense", "shuffled_0"]
                rows.append([v["point"].get(n, {}).get(metric, np.nan) if metric == "accuracy" else v["per_set"]["lac"].get(n, {}).get("set_size", np.nan) for n in names]); labels.append(f"{kind} · {short(m)}")
        if not rows: continue
        A = np.array(rows, float); fig, ax = plt.subplots(figsize=(12, 0.38 * len(rows) + 1.5))
        im = ax.imshow(A, aspect="auto", cmap="viridis"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(["official"] + [str(i) for i in range(30)] + ["nonsense", "shuffled"], fontsize=6, rotation=90)
        prev = None
        for i, l in enumerate(labels):
            if prev and l.split(" · ")[0] != prev: ax.axhline(i - 0.5, color="white", lw=2)
            prev = l.split(" · ")[0]
        ax.axvline(0.5, color="white", lw=1.5); ax.axvline(30.5, color="white", lw=1.5); fig.colorbar(im, ax=ax, fraction=0.02, label=metric if metric == "accuracy" else "mean set size (LAC, n=250)")
        ax.set_title(f"{DSN[ds]}: {metric if metric=='accuracy' else 'mean set size'} per model (rows; white lines separate types) × wording set (columns: official, 30 random valid sets, controls)")
        fig.tight_layout(); save(fig, f"fig08_heatmap_{metric}_{ds}")

# ---------------------------------------------------------------- fig09 W-subsampling
def fig09():
    fig, axes = plt.subplots(1, len(ZK), figsize=(4.4 * len(ZK), 3.8), squeeze=False); axes = axes[0]
    for ax, kind in zip(axes, ZK):
        for k, v in zs.items():
            if v["kind"] != kind: continue
            ws = v["w_subsample"].get(str(n250(v)))
            if not ws: continue
            Wm = max(int(x) for x in ws); ref = ws[str(Wm)]["mean_sd_w"]
            if ref <= 0: continue
            W = sorted(int(x) for x in ws)
            ax.plot(W, [ws[str(w)]["mean_sd_w"] / ref for w in W], color=COMP["wording"], alpha=0.5, lw=1)
            ax.fill_between(W, [ws[str(w)]["p2.5"] / ref for w in W], [ws[str(w)]["p97.5"] / ref for w in W], color=COMP["wording"], alpha=0.04)
            ax.plot(W, [ws[str(w)]["v1_popsd_mean"] / ref for w in W], color=OI["black"], alpha=0.4, lw=0.8, ls="--")
        ax.axhline(1, color="#888", lw=0.6); ax.set_xscale("log"); ax.set_xticks([2, 3, 4, 8, 16, 30]); ax.set_xticklabels(["2", "3", "4", "8", "16", "30"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(KN[kind]); ax.set_xlabel("number of wording sets W"); ax.set_ylim(0, 2.5)
    axes[0].set_ylabel("σ_wording estimate / estimate at W = 30")
    h = [Line2D([], [], color=COMP["wording"], label="random-effects estimate: mean; band = 95% range over random W-subsets"), Line2D([], [], color="k", ls="--", label="v1-style population SD of W means (biased low at small W)")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04)); fig.suptitle("How much can 4 paraphrase sets tell you? (n = 250)", y=1.09)
    fig.tight_layout(); save(fig, "fig09_W_subsampling")

# ---------------------------------------------------------------- fig10 controls
def fig10():
    P = pd.DataFrame(S.get("point", []))
    if P.empty: return
    fig, axes = plt.subplots(1, len(ZK), figsize=(4.7 * len(ZK), 4.2), sharey=True, squeeze=False); axes = axes[0]
    for ax, kind in zip(axes, ZK):
        g = P[P.kind == kind].reset_index(drop=True)
        x = np.arange(len(g))
        ax.vlines(x, g.acc_valid_min, g.acc_valid_max, color=COMP["wording"], lw=3, alpha=0.5, label="range over 30 valid sets")
        ax.scatter(x, g.acc_official, color="k", s=18, zorder=3, label="official names")
        ax.scatter(x, g.acc_nonsense, color=OI["grey"], marker="x", s=20, label="nonsense labels")
        ax.scatter(x, g.acc_shuffled, color=OI["purple"], marker="v", s=18, label="shuffled names")
        K = {"agnews": 4, "bbc2025": 4, "sst2": 2, "dbpedia": 14, "massive": 18}
        for xi, ds in zip(x, g.dataset): ax.hlines(1 / K[ds], xi - 0.4, xi + 0.4, color="#666", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([f"{DSN[d]}\n{short(m)}" for d, m in zip(g.dataset, g.model)], rotation=90, fontsize=6); ax.set_title(KN[kind])
    axes[0].set_ylabel("accuracy (whole scored pool)"); axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Controls: labels carry the semantics (nonsense ≈ chance, shuffled ≪ chance); dotted = chance 1/K", y=1.02); fig.tight_layout(); save(fig, "fig10_controls_accuracy")

# ---------------------------------------------------------------- fig11 contamination
def fig11():
    fs = sorted(glob.glob(f"{ROOT}/results/contamination/minkprob__*.json"))
    r4 = S.get("verdicts", {}).get("RQ4", {}).get("pairs", [])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, f in enumerate(fs):
        z = json.load(open(f)); m = z["model"].split("/")[-1]
        for j, ds in enumerate(["agnews", "bbc2025", "dbpedia", "sst2", "massive"]):
            if ds not in z["datasets"]: continue
            x = [r["mink20"] for r in z["datasets"][ds]["rows"]]
            pos = i * 6 + j; bp = axes[0].boxplot([x], positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor([OI["blue"], OI["verm"], OI["green"], OI["orange"], OI["purple"]][j]); bp["boxes"][0].set_alpha(0.7)
        axes[0].text(i * 6 + 2, axes[0].get_ylim()[0] if i else -12, short(m), ha="center", fontsize=7)
    axes[0].set_xticks([]); axes[0].set_ylabel("Min-K% Prob (k = 20%), higher = more 'familiar'")
    axes[0].legend(handles=[Patch(color=c, label=DSN[d]) for d, c in zip(["agnews", "bbc2025", "dbpedia", "sst2", "massive"], [OI["blue"], OI["verm"], OI["green"], OI["orange"], OI["purple"]])], fontsize=7, loc="lower right")
    axes[0].set_title("Contamination proxy per LLM (proxy only: genre differs)")
    km = {"nli": OI["blue"], "bienc": OI["orange"], "llm": OI["purple"]}
    for x in r4:
        axes[1].plot([0, 1], [x["acc_valid_mean_ag"], x["acc_valid_mean_bbc"]], color=km[x["kind"]], marker="o", alpha=0.8)
        axes[2].plot([0, 1], [x["sd_w_ag"], x["sd_w_bbc"]], color=km[x["kind"]], marker="o", alpha=0.8)
    for ax, t in [(axes[1], "accuracy (mean over 30 valid sets)"), (axes[2], "σ_wording of set size (n = 250)")]:
        ax.set_xticks([0, 1]); ax.set_xticklabels(["AG News\n(2004-05)", "BBC\n(2024-25)"]); ax.set_title(t); ax.set_xlim(-0.3, 1.3)
    axes[2].legend(handles=[Line2D([], [], color=c, marker="o", label=KN[k]) for k, c in km.items()], fontsize=7)
    fig.suptitle("Same 4 classes, same wordings, old vs new news (one line per model; compare slopes within a colour)", y=1.03); fig.tight_layout(); save(fig, "fig11_contamination_ag_vs_bbc")

# ---------------------------------------------------------------- fig12 v1 re-analysis
def fig12():
    p = f"{ROOT}/results/v1_reanalysis.json"
    if not os.path.exists(p): return
    R = json.load(open(p)); fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = ["clef", "qwen3.5"]; pub = {"clef": 0.32, "qwen3.5": 0.55}
    for i, n in enumerate(names):
        r = R[n]; ci = r["published_style_share_bootstrap_95ci"]; v = r["v2_method_set_size"]
        axes[0].errorbar(i - 0.12, r["published_style_share_wording_vs_cal_only"], yerr=[[r["published_style_share_wording_vs_cal_only"] - ci[0]], [ci[1] - r["published_style_share_wording_vs_cal_only"]]], fmt="o", color=OI["verm"], capsize=3)
        axes[0].errorbar(i + 0.12, v["share_w"], yerr=[[v["share_w"] - v["ci_share_w"][0]], [v["ci_share_w"][1] - v["share_w"]]], fmt="s", color=OI["blue"], capsize=3)
        axes[0].scatter(i, pub[n], marker="*", color="k", s=60, zorder=5)
        s1 = r["sd_wording_ddof1"]; lo, hi = r["chi2_95ci_sd_wording_ddof1"]
        axes[1].errorbar(i, s1, yerr=[[s1 - lo], [hi - s1]], fmt="o", color=OI["verm"], capsize=3); axes[1].scatter(i, r["u_wording_published_popsd"], marker="*", color="k", s=60, zorder=5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Clef-flash (v1 budget B)", "qwen3.5 9B (v1 budget C)"]); axes[0].set_ylim(0, 1); axes[0].set_ylabel("wording share of set-size variance")
    axes[0].legend(handles=[Line2D([], [], marker="*", ls="", color="k", label="published (RSS share)"), Line2D([], [], marker="o", ls="", color=OI["verm"], label="same, bootstrap over wordings + items"),
                            Line2D([], [], marker="s", ls="", color=OI["blue"], label="v2 crossed random-effects")], fontsize=7, loc="upper left")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Clef-flash", "qwen3.5"]); axes[1].set_ylabel("σ_wording of set size"); axes[1].set_title("W = 4: χ² 95% CI for an SD with 3 df")
    byn = R["clef_share_by_n_fixed_test"]; ns = sorted(int(n) for n in byn)
    axes[2].plot(ns, [byn[str(n)]["share_w"] for n in ns], color=OI["verm"], marker="o"); axes[2].fill_between(ns, [byn[str(n)]["ci_share_w"][0] for n in ns], [byn[str(n)]["ci_share_w"][1] for n in ns], color=OI["verm"], alpha=0.15)
    axes[2].set_xlabel("calibration size n (v1 500-item pool, fixed test)"); axes[2].set_ylabel("wording share"); axes[2].set_title("Clef: the share depends on n")
    fig.suptitle("Re-analysis of the published v1 numbers from the committed v1 outputs", y=1.03); fig.tight_layout(); save(fig, "fig12_v1_reanalysis", foot=False)

# ---------------------------------------------------------------- fig13 configuration shift
def fig13():
    fig, axes = plt.subplots(1, len(ZK), figsize=(4.7 * len(ZK), 4), sharey=True, squeeze=False); axes = axes[0]
    for ax, kind in zip(axes, ZK):
        data, labs = [], []
        for k, v in zs.items():
            if v["kind"] != kind or not v.get("config_shift_official_to_valid"): continue
            data.append([x["coverage_mean"] for x in v["config_shift_official_to_valid"]]); labs.append(f"{DSN[v['dataset']]}\n{short(v['model'])}")
        if not data: continue
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 2}); ax.axhline(0.9, color=OI["verm"], lw=1); ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, rotation=90, fontsize=6)
        ax.set_title(KN[kind])
    axes[0].set_ylabel("coverage when calibrated with the official names\nand deployed with another valid wording (n = 250)")
    fig.suptitle("Configuration shift breaks the coverage guarantee: target 0.90 (red)", y=1.02); fig.tight_layout(); save(fig, "fig13_config_shift_coverage")

# ---------------------------------------------------------------- fig14 supervised
def fig14():
    if not sup: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, v in sup.items():
        ns = [n for n in v["n_grid"] if f"A|lac|trainboot|{n}" in v["floor"]]
        axes[0].plot(ns, [v["floor"][f"A|lac|trainboot|{n}"]["set_size"]["sd_s"] for n in ns], color=COMP["split"], alpha=0.6)
        axes[0].plot(ns, [v["floor"][f"A|lac|trainboot|{n}"]["set_size"]["sd_w"] for n in ns], color=COMP["train"], alpha=0.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("calibration size n"); axes[0].set_ylabel("SD of mean set size")
    axes[0].legend(handles=[Line2D([], [], color=COMP["split"], label="split sampling"), Line2D([], [], color=COMP["train"], label="training-sample bootstrap (20 refits)")], fontsize=7)
    axes[0].set_title("T4: no wording component exists; the analogue is training data")
    rows = [(v["dataset"], v["model"], v["point"]["reference"]["accuracy"]) for v in sup.values()]
    for i, m in enumerate(sorted({r[1] for r in rows})):
        a = {r[0]: r[2] for r in rows if r[1] == m}
        if "agnews" in a and "bbc2025" in a: axes[1].plot([0, 1], [a["agnews"], a["bbc2025"]], marker="o", color=[OI["blue"], OI["orange"], OI["green"]][i % 3], label=short(m))
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["AG News test", "BBC 2024-25"]); axes[1].set_ylabel("accuracy (trained on AG News)"); axes[1].legend(fontsize=7)
    axes[1].set_title("Real temporal/source shift for an AG-trained probe")
    fig.tight_layout(); save(fig, "fig14_supervised_reference")

# ---------------------------------------------------------------- fig15 surface vs valid flips
def fig15():
    r3 = S.get("verdicts", {}).get("RQ3", {}).get("cells", [])
    if not r3: return
    fig, ax = plt.subplots(figsize=(6, 5)); km = {"nli": OI["blue"], "bienc": OI["orange"], "llm": OI["purple"]}
    for x in r3:
        if not np.isfinite(x["surface_flip_mean"]): continue
        ax.scatter(max(x["valid_flip_mean"], 1e-4), max(x["surface_flip_mean"], 1e-4), color=km[x["kind"]], marker=MARK[DATASETS.index(x["dataset"])], s=30)
    lim = [1e-4, 1]; ax.plot(lim, lim, color="#888", lw=0.7); ax.plot(lim, [l / 2 for l in lim], color="#888", lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("argmax flip rate vs official: valid paraphrase sets (mean)"); ax.set_ylabel("flip rate: case-only changes (mean); 1e-4 = none")
    h = [Line2D([], [], marker="o", ls="", color=c, label=KN[k]) for k, c in km.items()] + [Line2D([], [], marker=MARK[i], ls="", color="k", label=DSN[d]) for i, d in enumerate(DATASETS)]
    h += [Line2D([], [], color="#888", ls="--", label="refutation line: surface = ½ valid")]
    ax.legend(handles=h, fontsize=7, loc="upper left"); ax.set_title("How much of the wording effect needs no change of meaning?")
    fig.tight_layout(); save(fig, "fig15_surface_vs_valid")

# ---------------------------------------------------------------- fig16 tau / raw-LL sensitivity
def fig16():
    rows = []
    for k, v in cells.items():
        for tag in ["_tau0.02", "_tau0.1", "_rawll"]:
            if k.endswith(tag):
                base = k[: -len(tag)]
                if base in zs:
                    b = zs[base]; n = n250(b); rows.append((b["kind"], tag, b["floor"][f"A|lac|valid|{n}"]["set_size"]["share_w"], v["floor"][f"A|lac|valid|{n}"]["set_size"]["share_w"], b["dataset"]))
    if not rows: return
    fig, ax = plt.subplots(figsize=(5.5, 5)); tc = {"_tau0.02": OI["blue"], "_tau0.1": OI["orange"], "_rawll": OI["purple"]}
    for r in rows: ax.scatter(r[2], r[3], color=tc[r[1]], marker=MARK[DATASETS.index(r[4])], s=25)
    ax.plot([0, 1], [0, 1], color="#888", lw=0.7); ax.set_xlabel("wording share, main specification"); ax.set_ylabel("wording share, sensitivity specification")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label={"_tau0.02": "bi-encoder τ = 0.02", "_tau0.1": "bi-encoder τ = 0.1", "_rawll": "LLM raw log-lik. (no PMI)"}[t]) for t, c in tc.items()], fontsize=7)
    ax.set_title("Sensitivity of the wording share to the probability scale (n = 250)"); fig.tight_layout(); save(fig, "fig16_sensitivity_tau_rawll")

for fn in [fig01, lambda: fig02(), lambda: fig02("coverage", name="fig02b_nscaling_coverage"), lambda: fig02("set_size", "B", name="fig02c_nscaling_setsize_designB_v1style"),
           lambda: fig02("set_size", "A", "aps", name="fig02d_nscaling_setsize_APS"), fig03, fig04, fig05, fig06, fig07, lambda: fig08("accuracy"), lambda: fig08("set_size"),
           fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16]:
    try: fn()
    except Exception as e: import traceback; print("FIGURE FAILED", getattr(fn, "__name__", "lambda"), repr(e)); traceback.print_exc()
