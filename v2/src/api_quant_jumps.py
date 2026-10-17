"""Exploratory: how often does a Luna/nano VALID wording set give LAC sets that cover (almost) all K labels (n=250, alpha=0.10, 50 Design-A splits)?
Explains the large Luna sigma_wording: with probabilities on a 0.01 grid and many exact zeros, when a wording's error rate exceeds alpha the LAC
threshold lands on the floored-zero tie and sets jump to all K labels. Output results/api/quant_jumps.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
out = {}
for ds in ["agnews", "bbc2025", "sst2", "dbpedia", "massive"]:
    D_ = load_ds(ds); sets = [s for s in load_sets(ds) if s["pop"] == "valid"]
    for tag in ["gpt-6-luna", "gpt-4.1-nano"]:
        sc = Scorer(f"{RAW}/{ds}__api__{tag}.npz"); P0 = sc.probs(sets[0]["labels"]); N, K = P0.shape
        y = np.array(D_["pool_y"])[:N]; u = np.random.default_rng(7).uniform(size=N); cal, te = splits_A(N, 250); cal, te = cal[:50], te[:50]
        rows = []
        for s in sets:
            P = sc.probs(s["labels"]); err = float(np.mean(P.argmax(1) != y)); sz = conformal_eval(scores_all(P, "lac", u), y, cal, te, 0.1)["set_size"]
            frac0 = float(np.mean(P[np.arange(N), y] <= 1e-4 + 1e-9))
            rows.append({"set": s["name"], "err": err, "size_over_K": float(np.mean(sz)) / K, "share_true_prob_at_floor": frac0})
        big = [r for r in rows if r["size_over_K"] >= 0.9]
        out[f"{ds}__{tag}"] = {"K": K, "n_sets": len(rows), "n_sets_size_ge_0.9K": len(big), "err_of_those": [round(r["err"], 3) for r in big],
            "share_true_prob_at_floor_mean": float(np.mean([r["share_true_prob_at_floor"] for r in rows])),
            "max_size_over_K": max(r["size_over_K"] for r in rows), "rows": rows}
        print(ds, tag, {k: v for k, v in out[f"{ds}__{tag}"].items() if k != "rows"}, flush=True)
json.dump(out, open(f"{ROOT}/results/api/quant_jumps.json", "w"), indent=1); print("DONE")
