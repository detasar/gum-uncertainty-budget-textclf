"""Luna pilot: questions-per-request (1 vs 8 vs 31) and run-to-run noise on 100 AG News items x 31 sets (official + 30 valid).
Output results/api/luna_pilot.json"""
import json, os, itertools, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); C = f"{ROOT}/raw/api_cache/luna"
def load(cond):
    P = {}; ref = 0
    for l in open(f"{C}/agnews__{cond}.jsonl"):
        r = json.loads(l)
        for nm in r["sets"]:
            v = r["p"][nm]
            if v is None: ref += 1; continue
            P[(r["i"], nm)] = np.array(v, float)
    return P, ref
conds = ["pilot_q1", "pilot_q1b", "pilot_q8", "pilot_q31", "pilot_q31b"]; L = {c: load(c) for c in conds}
y = np.array(json.load(open(f"{ROOT}/data/agnews.json"))["pool_y"])
out = {"refusals": {c: L[c][1] for c in conds}, "n_pairs": {c: len(L[c][0]) for c in conds}}
allv = np.concatenate([np.concatenate(list(L[c][0].values())) for c in conds])
out["quantisation"] = {"n_values": int(allv.size), "share_on_0.01_grid": float(np.mean(np.abs(allv * 100 - np.round(allv * 100)) < 1e-9)),
                       "share_exact_0": float(np.mean(allv == 0)), "share_exact_1": float(np.mean(allv == 1)),
                       "distinct_values": int(len(np.unique(np.round(allv, 6)))), "sum_to_1_abs_dev_max": None}
sums = np.array([v.sum() for c in conds for v in L[c][0].values()]); out["quantisation"]["sum_to_1_abs_dev_max"] = float(np.abs(sums - 1).max())
out["quantisation"]["share_rows_sum_ne_1"] = float(np.mean(np.abs(sums - 1) > 1e-9))
acc = {}
for c in conds:
    ks = [k for k in L[c][0] if k[1] == "official"]; acc[c] = float(np.mean([L[c][0][k].argmax() == y[k[0]] for k in ks]))
out["accuracy_official"] = acc
pairs = {}
for a, b in [("pilot_q1", "pilot_q1b"), ("pilot_q31", "pilot_q31b"), ("pilot_q1", "pilot_q31"), ("pilot_q1b", "pilot_q31b"), ("pilot_q1", "pilot_q8"), ("pilot_q8", "pilot_q31")]:
    ks = sorted(set(L[a][0]) & set(L[b][0])); A = np.stack([L[a][0][k] for k in ks]); B = np.stack([L[b][0][k] for k in ks])
    d = np.abs(A - B)
    pairs[f"{a}_vs_{b}"] = {"n_pairs": len(ks), "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()), "share_rows_identical": float((d.max(1) < 1e-12).mean()),
                           "share_rows_dp_gt_0.05": float((d.max(1) > 0.05).mean()), "argmax_flip_rate": float((A.argmax(1) != B.argmax(1)).mean()),
                           "mean_abs_dp_true_class": float(np.mean([abs(L[a][0][k][y[k[0]]] - L[b][0][k][y[k[0]]]) for k in ks]))}
out["pairs"] = pairs
json.dump(out, open(f"{ROOT}/results/api/luna_pilot.json", "w"), indent=1)
print(json.dumps(out, indent=1))
