"""Figures for the exploratory T5 API-LLM cells (gpt-6-luna via Decisions, gpt-4.1-nano via chat logprobs).
Generative LLMs are shown ONLY next to other LLMs (T3 small local LLMs), never next to bi-encoder or NLI classifiers."""
import json, os, sys, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(__file__)); from core import ROOT, DATASETS, RAW, Scorer, load_sets
C = f"{ROOT}/results/cells"; F = f"{ROOT}/figures"
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
COMP = {"wording": OI["verm"], "split": OI["blue"], "resid": "#BBBBBB", "order": OI["green"]}
DSN = {"agnews": "AG News", "bbc2025": "BBC 2024-25", "sst2": "SST-2", "dbpedia": "DBpedia-14", "massive": "MASSIVE-18"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 220,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
FOOT = "LLMs only (T3 local small LLMs, T5 closed API LLMs; T5 exploratory, not preregistered; API scored 2026-10-17). Not compared with bi-encoders/NLI."
def save(fig, name):
    fig.text(0.01, 0.003, FOOT, fontsize=7, color="#555555", ha="left", va="bottom"); fig.savefig(f"{F}/{name}.png", bbox_inches="tight"); plt.close(fig); print("saved", name)
cells = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{C}/*.json")) if os.path.basename(f)[:-5].count("__") == 2}
llm = {k: v for k, v in cells.items() if v["kind"] in ("llm", "api") and not any(t in k for t in ("_tau", "_rawll", "_sub"))}
def lab(v): return {"gpt-6-luna": "gpt-6-luna (T5, Decisions)", "gpt-4.1-nano": "gpt-4.1-nano (T5, logprobs)"}.get(v["model"], v["model"].replace("-Instruct", "") + " (T3, PMI)")
MODEL_COL = lambda m: {"gpt-6-luna": OI["verm"], "gpt-4.1-nano": OI["blue"]}.get(m, OI["purple"])
def n250(v): return 250 if 250 in v["n_grid"] else v["n_grid"][0]
order_cells = sorted(llm, key=lambda k: (DATASETS.index(llm[k]["dataset"]), llm[k]["kind"], llm[k]["model"]))

def fig17():   # variance shares at n=250, LLM cells
    rows = [(f"{DSN[llm[k]['dataset']]} · {lab(llm[k])}", llm[k]["floor"][f"A|lac|valid|{n250(llm[k])}"]["set_size"]) for k in order_cells]
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows) + 1.4), gridspec_kw={"width_ratios": [1.2, 1]}); y = np.arange(len(rows))[::-1]
    w = np.array([r[1]["share_w"] for r in rows]); s = np.array([r[1]["share_s"] for r in rows])
    axes[0].barh(y, w, color=COMP["wording"]); axes[0].barh(y, s, left=w, color=COMP["split"]); axes[0].barh(y, 1 - w - s, left=w + s, color=COMP["resid"])
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], fontsize=7); axes[0].set_xlim(0, 1); axes[0].set_xlabel("share of variance of mean set size (LAC, α=0.10, n=250)")
    axes[0].legend(handles=[Patch(color=COMP["wording"], label="wording (30 VALID sets)"), Patch(color=COMP["split"], label="calibration/test split"), Patch(color=COMP["resid"], label="interaction + residual")], loc="lower right", fontsize=7)
    for yi, r in zip(y, rows):
        axes[1].errorbar(r[1]["sd_w"], yi, xerr=[[r[1]["sd_w"] - r[1]["ci_sd_w"][0]], [r[1]["ci_sd_w"][1] - r[1]["sd_w"]]], fmt="o", color=COMP["wording"], ms=4)
        axes[1].plot(r[1]["sd_s"], yi, "s", color=COMP["split"], ms=4)
    axes[1].set_xscale("log"); axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].set_xlabel("SD of mean set size (o: wording, 95% CI; ■: split)")
    fig.suptitle("LLM cells: wording vs sampling components of conformal set size", y=1.0); fig.tight_layout(); save(fig, "fig17_llm_api_variance_shares_n250")

def fig18():   # n-scaling of sd_w and sd_s for LLM cells
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for k in order_cells:
            v = llm[k]
            if v["dataset"] != ds: continue
            ns = [n for n in v["n_grid"] if f"A|lac|valid|{n}" in v["floor"]]; b = [v["floor"][f"A|lac|valid|{n}"]["set_size"] for n in ns]; c = MODEL_COL(v["model"])
            ax.plot(ns, [x["sd_w"] for x in b], "-o", color=c, ms=3, label=f"{lab(v)}: wording"); ax.plot(ns, [x["sd_s"] for x in b], "--s", color=c, ms=3, alpha=0.7, label=f"{lab(v)}: split")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(DSN[ds]); ax.set_xlabel("calibration size n")
    axes[0].set_ylabel("SD of mean set size (LAC, α=0.10)"); h, l = axes[0].get_legend_handles_labels()
    hs = {}; [hs.setdefault(b, a) for ax in axes for a, b in zip(*ax.get_legend_handles_labels())]
    fig.legend(list(hs.values()), list(hs.keys()), loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.12)); fig.tight_layout(); save(fig, "fig18_llm_api_nscaling")

def fig19():   # option order vs wording
    p = f"{ROOT}/results/api/option_order.json"
    if not os.path.exists(p): return
    R = json.load(open(p)); keys = list(R)
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.6 * len(keys) + 2.2), gridspec_kw={"width_ratios": [1.2, 1, 1]}); y = np.arange(len(keys))[::-1]
    for ax, met, title in [(axes[0], "lac_set_size", "LAC set size"), (axes[1], "aps_set_size", "APS set size")]:
        for yi, k in zip(y, keys):
            b = R[k][met]
            for off, comp, cik, col in [(0.22, "w", "ci_sd_w", COMP["wording"]), (0.0, "o", "ci_sd_o", COMP["order"]), (-0.22, "s", "ci_sd_s", COMP["split"])]:
                v = b["sd"][comp]; lo, hi = b[cik]; ax.errorbar(max(v, 1e-4), yi + off, xerr=[[max(v, 1e-4) - max(lo, 1e-4)], [max(hi, 1e-4) - max(v, 1e-4)]], fmt="o", color=col, ms=4)
        ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([f"{DSN[k.split('__')[0]]} · {k.split('__')[1]}" for k in keys] if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel(f"SD of mean {title} (α=0.10, n=250; 600 items)"); ax.set_title(title)
    fig.legend(handles=[Patch(color=COMP["wording"], label="wording (30 VALID sets)"), Patch(color=COMP["order"], label="option order (3 orders, nested in wording)"), Patch(color=COMP["split"], label="calibration/test split")], fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.45, -0.08))
    from matplotlib.ticker import NullFormatter
    for ax in axes[:2]: ax.xaxis.set_minor_formatter(NullFormatter())
    for yi, k in zip(y, keys):
        it = R[k]["item_level"]
        axes[2].barh(yi + 0.18, it["argmax_flip_between_orders_mean"], height=0.34, color=COMP["order"]); axes[2].barh(yi - 0.18, it["argmax_flip_vs_official_wording_mean"], height=0.34, color=COMP["wording"])
    axes[2].set_yticks(y); axes[2].set_yticklabels([]); axes[2].set_xlabel("mean argmax flip rate"); axes[2].set_title("item level: order change (same wording)\nvs wording change (vs official, same order)")
    fig.suptitle("Exploratory: option-order factor vs label wording, API LLMs (AG News, BBC 2024-25)", y=1.02); fig.tight_layout(); save(fig, "fig19_option_order_vs_wording")

def fig20():   # API test-retest
    p = f"{ROOT}/results/repeatability.json"
    if not os.path.exists(p): return
    R = json.load(open(p)); ks = [k for k in ("api-luna", "api-nano") if k in R]
    if not ks: return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, k in enumerate(ks):
        per = R[k]["per_set"]; c = OI["verm"] if "luna" in k else OI["blue"]; x = np.arange(len(per)) + (0.2 if i else -0.2)
        axes[0].bar(x, [max(r["max_abs_dp"], 1e-7) for r in per], width=0.4, color=c, label=f"{k} ({R[k]['runs']} runs, {R[k]['items']} items)")
        axes[1].bar(x, [1 - r["share_items_same_label_all_runs"] for r in per], width=0.4, color=c)
        axes[2].bar(x, [r["share_test_sets_changed"] for r in per], width=0.4, color=c)
    axes[0].set_yscale("log"); axes[0].set_ylabel("max |Δp| across runs"); axes[1].set_ylabel("share of items whose argmax changes"); axes[2].set_ylabel("share of test items whose LAC set changes")
    for ax in axes: ax.set_xlabel("wording set (0 = official, 1-30 = VALID)")
    axes[0].legend(fontsize=7); fig.suptitle("API test-retest on AG News (main run + 2 reruns, same day)"); fig.tight_layout(); save(fig, "fig20_api_repeatability")

def fig21():   # Luna quantisation mechanism
    p = f"{RAW}/agnews__api__gpt-6-luna.npz"
    if not os.path.exists(p) or "agnews__api__gpt-6-luna" not in llm: return
    z = np.load(p, allow_pickle=True); P = z["P"]; v = P[np.isfinite(P)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].hist(v, bins=np.arange(-0.005, 1.01, 0.01), color=OI["verm"]); axes[0].set_yscale("log"); axes[0].set_xlabel("returned probability (all sets, AG News)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"{len(np.unique(np.round(v, 6)))} distinct values; {100*np.mean(v==0):.0f}% exactly 0")
    for ds, mk in zip(DATASETS, "osD^v"):
        for m, c in [("gpt-6-luna", OI["verm"]), ("gpt-4.1-nano", OI["blue"])]:
            k = f"{ds}__api__{m}"
            if k not in llm: continue
            ps = llm[k]["per_set"]["lac"]; acc = llm[k]["point"]; names = [n for n in ps if n.startswith("valid_")]
            axes[1].scatter([1 - acc[n]["accuracy"] for n in names], [ps[n]["set_size"] / llm[k]["K"] for n in names], color=c, marker=mk, s=14, alpha=0.7, label=f"{DSN[ds]} · {m}")
            ps2 = llm[k]["per_set"]["aps"]; axes[2].scatter([1 - acc[n]["accuracy"] for n in names], [ps2[n]["set_size"] / llm[k]["K"] for n in names], color=c, marker=mk, s=14, alpha=0.7)
    for ax, t in [(axes[1], "LAC"), (axes[2], "APS")]:
        ax.axvline(0.10, color="#888", lw=0.7, ls=":"); ax.set_xlabel("wording's top-1 error rate (pool)"); ax.set_ylabel(f"mean {t} set size / K (n=250, α=0.10)"); ax.set_title(f"{t}: one point per VALID wording set")
    axes[1].legend(fontsize=6, ncol=2); fig.tight_layout(); save(fig, "fig21_luna_quantisation_setsize")

def fig22():   # accuracy across wordings, LLMs
    fig, ax = plt.subplots(figsize=(12, 3.8)); xt = []; x = 0
    for ds in DATASETS:
        for k in order_cells:
            v = llm[k]
            if v["dataset"] != ds: continue
            a = [v["point"][n]["accuracy"] for n in v["point"] if n.startswith("valid_")]
            ax.boxplot([a], positions=[x], widths=0.6, patch_artist=True, boxprops=dict(facecolor=MODEL_COL(v["model"]), alpha=0.5), medianprops=dict(color="k"), flierprops=dict(ms=2))
            ax.plot(x, v["point"]["official"]["accuracy"], "k*", ms=7); xt.append((x, f"{DSN[ds]}\n{v['model'].replace('-Instruct','')}")); x += 1
        x += 0.7
    ax.set_xticks([t[0] for t in xt]); ax.set_xticklabels([t[1] for t in xt], fontsize=6, rotation=90); ax.set_ylabel("top-1 accuracy over 30 VALID wordings (★ official)")
    ax.set_title("LLM accuracy across meaning-preserving label wordings"); fig.tight_layout(); save(fig, "fig22_llm_accuracy_by_wording")

for f in [fig17, fig18, fig19, fig20, fig21, fig22]:
    try: f()
    except Exception as e: print("FAIL", f.__name__, repr(e))
