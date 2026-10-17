"""Exploratory repeatability check under a hardware/backend change: the same NLI cell scored on the box CPU (torch 2.14.1+cpu, 4 threads)
and on an Apple M5 GPU (torch 2.14.1 MPS), both fp32, bs 64, identical pair order and batch composition, same transformers version.
Compares raw entailment logits, class probabilities for every scorable wording set, argmax labels and LAC conformal sets (n_cal 250, alpha 0.10,
200 Design-A splits: both platforms use the same splits, each calibrates its own threshold).
usage: xplat_compare.py <box npz> <mac npz> [--out results/xplat_<cell>.json]"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
ap = argparse.ArgumentParser(); ap.add_argument("box"); ap.add_argument("mac"); ap.add_argument("--out"); a = ap.parse_args()
A, B = Scorer(a.box), Scorer(a.mac); assert A.strings == B.strings and A.N == B.N
za, zb = np.load(a.box, allow_pickle=True), np.load(a.mac, allow_pickle=True)
dl = np.abs(za["logits"].astype(np.float64) - zb["logits"].astype(np.float64))
ds = A.ds; y = np.array(load_ds(ds)["pool_y"])[:A.N]; sets = [s for s in load_sets(ds) if A.has(s["labels"])]
cal, te = splits_A(A.N, 250)
res = {"cell": os.path.basename(a.box)[:-4], "N": A.N, "n_strings": len(A.strings), "box_meta": A.meta, "mac_meta": B.meta,
       "max_abs_dlogit": float(dl.max()), "mean_abs_dlogit": float(dl.mean()), "share_logits_bitwise_equal": float((dl == 0).mean()), "per_set": {}}
mx_dp, flips, setchg, qd, cov_d, size_d = [], [], [], [], [], []
for s in sets:
    Pa, Pb = A.probs(s["labels"]), B.probs(s["labels"]); dp = np.abs(Pa - Pb)
    f = float((Pa.argmax(1) != Pb.argmax(1)).mean())
    SA, SB = 1 - Pa, 1 - Pb; n = cal.shape[1]
    qa = np.quantile(SA[cal, y[cal]], qlevel(n, 0.1), axis=1, method="higher"); qb = np.quantile(SB[cal, y[cal]], qlevel(n, 0.1), axis=1, method="higher")
    sa = SA[te] <= qa[:, None, None]; sb = SB[te] <= qb[:, None, None]
    ch = (sa != sb).any(2).mean(1)   # share of test items whose LAC set differs, per split
    ea, eb = conformal_eval(SA, y, cal, te, 0.1), conformal_eval(SB, y, cal, te, 0.1)
    res["per_set"][s["name"]] = {"max_dp": float(dp.max()), "argmax_flip": f, "lac_set_changed_mean": float(ch.mean()), "lac_set_changed_max": float(ch.max()),
                                 "max_dq": float(np.abs(qa - qb).max()), "max_d_mean_size": float(np.abs(ea["set_size"] - eb["set_size"]).max()), "max_d_cov": float(np.abs(ea["coverage"] - eb["coverage"]).max())}
    mx_dp.append(dp.max()); flips.append(f); setchg.append(ch.mean())
r = res["per_set"]
res["summary"] = {"n_sets": len(sets), "max_dp": float(max(mx_dp)), "median_set_max_dp": float(np.median(mx_dp)), "argmax_flip_max_share": float(max(flips)),
                  "argmax_flip_items_total": int(round(sum(flips) * A.N)), "lac_set_changed_mean_over_sets": float(np.mean(setchg)), "lac_set_changed_max_split": float(max(v["lac_set_changed_max"] for v in r.values())),
                  "sets_with_any_lac_change": int(sum(v["lac_set_changed_max"] > 0 for v in r.values())), "max_d_mean_set_size": float(max(v["max_d_mean_size"] for v in r.values())),
                  "max_d_coverage": float(max(v["max_d_cov"] for v in r.values())), "official": r.get("official")}
out = a.out or f"{ROOT}/results/xplat_{res['cell']}.json"; json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_set", "box_meta", "mac_meta")}, indent=1)); print("WROTE", out)
