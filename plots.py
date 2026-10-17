"""Plots from results.json."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(f"{HERE}/results.json"))
B = R["budgets"]; keys = [k for k in ["A_supervised_minilm", "B_zeroshot_clef", "C_zeroshot_qwen3.5_9b"] if k in B]
titles = {"A_supervised_minilm": "A. Supervised: MiniLM embedding + MLP head",
          "B_zeroshot_clef": "B. Zero-shot: Clef-flash (8-bit MLX)", "C_zeroshot_qwen3.5_9b": "C. Zero-shot LLM: qwen3.5:9b (Ollama, temperature 0)"}
fig, axes = plt.subplots(len(keys), 1, figsize=(9, 3.1 * len(keys) + 0.6))
axes = np.atleast_1d(axes)
for ax, k in zip(axes, keys):
    comps = sorted([c for c in B[k]["components"] if c.get("in_rss", True)], key=lambda c: c["u_set_size"])
    names = [c["component"] for c in comps]; u = [c["u_set_size"] for c in comps]
    col = ["#c0392b" if c["type"].startswith("B") else "#2c7fb8" for c in comps]
    ax.barh(names, u, color=col)
    for i, c in enumerate(comps):
        ax.text(c["u_set_size"], i, f"  {c['u_set_size']:.3f} ({100*c['share_var_set_size']:.0f}% of var)", va="center", fontsize=8)
    ax.set_title(f"{titles[k]}   u_c = {B[k]['u_c_set_size']:.3f} (ref. set size {B[k]['reference_metrics']['set_size']:.2f})", fontsize=10)
    ax.set_xlim(0, max(u) * 1.6 if max(u) > 0 else 1); ax.tick_params(axis="y", labelsize=8)
axes[-1].set_xlabel("standard uncertainty of mean conformal set size (alpha = 0.10)")
fig.text(0.01, 0.005, "blue = Type A (statistical: bootstrap/repeats); red = Type B (alternatives / systematic effect). AG News, 250 cal / 250 test.", fontsize=7)
plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig(f"{HERE}/budget_bar.png", dpi=160); plt.close()

# coverage / set size per model with 95% bootstrap CI, clean vs shift
M = R["models"]
ORDER = ["supervised probe", "deterministic schema classifier", "generative chat LLM"]
names = sorted(M, key=lambda n: (next((i for i, c in enumerate(ORDER) if c in M[n].get("instrument_class", "")), 9), n))
fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 1.8), sharey=True)
y = np.arange(len(names))
for ax, met, lab in [(axes[0], "set_size", "mean set size"), (axes[1], "coverage", "empirical coverage")]:
    v = np.array([M[n][met] for n in names]); lo = np.array([M[n][f"ci95_{met}"][0] for n in names]); hi = np.array([M[n][f"ci95_{met}"][1] for n in names])
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color="#2c7fb8", label="clean test (95% bootstrap CI)")
    sv = [M[n].get("shift30", {}).get(met, np.nan) for n in names]
    ax.plot(sv, y, "x", color="#c0392b", ms=8, label="shifted test (typos/word drop 0.3)")
    if met == "coverage": ax.axvline(0.9, ls="--", color="gray", lw=1)
    ax.set_xlabel(lab)
    cls = [M[n].get("instrument_class", "") for n in names]
    for i in range(1, len(names)):
        if cls[i] != cls[i-1]: ax.axhline(i - 0.5, color="black", lw=0.8)
axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8); axes[1].legend(fontsize=7, loc="lower left")
fig.suptitle("Split conformal (alpha = 0.10), AG News, same 250 cal / 250 test. Grouped by instrument class; NOT a leaderboard:\ncompare only within a group (output types differ: MLP probs / schema-head softmax / first-token logprobs)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{HERE}/coverage_setsize.png", dpi=160); plt.close()

if "scaling_clef_wording" in R:
    S = R["scaling_clef_wording"]; n = [s["n_cal"] for s in S]
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(n, [s["sd_cal_sampling"] for s in S], "o-", color="#2c7fb8", label="calibration-set sampling (SD over draws)")
    w = [s["sd_wording_systematic"] for s in S]
    plt.errorbar(n, w, yerr=[[a - s["ci95_sd_wording_systematic"][0] for a, s in zip(w, S)], [s["ci95_sd_wording_systematic"][1] - a for a, s in zip(w, S)]],
                 fmt="s-", color="#c0392b", capsize=3, label="label wording, systematic part (SD of per-wording means)")
    ref = S[0]["sd_cal_sampling"] * np.sqrt(S[0]["n_cal"] / np.array(n))
    plt.plot(n, ref, ":", color="gray", label="1/sqrt(n) reference")
    plt.xlabel("calibration set size n"); plt.ylabel("SD of mean set size (alpha = 0.10)"); plt.xscale("log"); plt.yscale("log")
    plt.title("Clef-flash, AG News: which component shrinks with more calibration data?", fontsize=9); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{HERE}/scaling.png", dpi=160); plt.close()
print("plots written")
