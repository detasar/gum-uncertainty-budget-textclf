"""Exploratory (not preregistered): option-order factor for the API LLMs (T5) on AG News and BBC-2025.
Data: first 600 pool items, official + 30 VALID wording sets, 3 option orders: ord0 = canonical class order (main run), ord1/ord2 = random
permutations fixed per (dataset, set, order) and identical for all items. Order levels are therefore nested within wording.
Model for Y = mean LAC set size (alpha 0.10, n_cal 250, test 300, Design A, 200 splits):
   Y[w,o,s] = mu + W_w + O_o(w) + S_s + WS_ws + E   (E = S x O(w) interaction + error; no replication)
Method-of-moments with expected mean squares
   MS_w = e + S*o + O*ws + O*S*w ; MS_o(w) = e + S*o ; MS_s = e + O*ws + W*O*s ; MS_ws = e + O*ws ; MS_e = e
CIs: two-way bootstrap over wordings and splits (500 draws). Item level: argmax flip rates and total-variation distance between orders
(same wording) vs between wordings (same order, vs official), and display-position bias. Output results/api/option_order.json"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
OUTP = f"{ROOT}/results/api/option_order.json"; MODELS = {"gpt-6-luna": "luna", "gpt-4.1-nano": "nano"}; M = 600; NCAL = 250
def vc3(Y):
    W, O, S = Y.shape; g = Y.mean(); mw = Y.mean((1, 2)); mwo = Y.mean(2); ms = Y.mean((0, 1)); mws = Y.mean(1)
    MSw = O * S * ((mw - g) ** 2).sum() / (W - 1); MSo = S * ((mwo - mw[:, None]) ** 2).sum() / (W * (O - 1))
    MSs = W * O * ((ms - g) ** 2).sum() / (S - 1); MSws = O * ((mws - mw[:, None] - ms[None, :] + g) ** 2).sum() / ((W - 1) * (S - 1))
    E = Y - mwo[:, :, None] - mws[:, None, :] + mw[:, None, None]; MSe = (E ** 2).sum() / (W * (O - 1) * (S - 1))
    v = {"e": MSe, "o": (MSo - MSe) / S, "ws": (MSws - MSe) / O, "s": (MSs - MSws) / (W * O)}; v["w"] = (MSw - MSo - MSws + MSe) / (O * S)
    neg = {k: bool(x < 0) for k, x in v.items()}; v = {k: max(x, 0.0) for k, x in v.items()}; t = sum(v.values())
    return v, {k: (x / t if t > 0 else 0.0) for k, x in v.items()}, neg
def load_api(path, names):
    sc = Scorer(path); return {nm: sc.probs(lab) for nm, lab in names}, sc.meta
res = {}
for ds in ["agnews", "bbc2025"]:
    y = np.array(load_ds(ds)["pool_y"])[:M]; sets = [s for s in load_sets(ds) if s["pop"] in ("official", "valid")]; names = [(s["name"], s["labels"]) for s in sets]
    for tag, short in MODELS.items():
        paths = [f"{RAW}/{ds}__api__{tag}.npz"] + [f"{RAW}/apiorder/{ds}__api__{tag}__ord{k}.npz" for k in (1, 2)]
        if not all(os.path.exists(p) for p in paths): print("missing", ds, tag); continue
        Ps = []; metas = []
        for p in paths:
            P, m = load_api(p, names); Ps.append({k: v[:M] for k, v in P.items()}); metas.append(m)
        if min(next(iter(P.values())).shape[0] for P in Ps) < M: print("too few items", ds, tag); continue
        valid = [nm for nm, _ in names if nm.startswith("valid_")]; W, O = len(valid), 3
        u = np.random.default_rng(99).uniform(size=M); cal, te = splits_A(M, NCAL)
        Y = np.zeros((W, O, cal.shape[0])); Yaps = np.zeros_like(Y); C = np.zeros_like(Y)
        for i, nm in enumerate(valid):
            for o in range(O):
                r = conformal_eval(scores_all(Ps[o][nm], "lac", u), y, cal, te, 0.1); Y[i, o] = r["set_size"]; C[i, o] = r["coverage"]
                Yaps[i, o] = conformal_eval(scores_all(Ps[o][nm], "aps", u), y, cal, te, 0.1)["set_size"]
        out = {"W": W, "O": O, "S": int(cal.shape[0]), "n_items": M, "n_cal": NCAL, "orders": ["canonical", "perm1", "perm2"]}
        for lab_, YY in [("lac_set_size", Y), ("aps_set_size", Yaps), ("lac_coverage", C)]:
            v, sh, neg = vc3(YY); rng = np.random.default_rng(1); B = []
            for _ in range(500):
                iw = rng.integers(0, W, W); js = rng.integers(0, YY.shape[2], YY.shape[2]); B.append(vc3(YY[iw][:, :, js]))
            ci = lambda f: [float(np.percentile([f(b) for b in B], 2.5)), float(np.percentile([f(b) for b in B], 97.5))]
            out[lab_] = {"mean": float(YY.mean()), "sd": {k: math.sqrt(x) for k, x in v.items()}, "share": sh, "negative_estimate": neg,
                         "ci_sd_w": ci(lambda b: math.sqrt(b[0]["w"])), "ci_sd_o": ci(lambda b: math.sqrt(b[0]["o"])), "ci_sd_s": ci(lambda b: math.sqrt(b[0]["s"])),
                         "ci_share_w": ci(lambda b: b[1]["w"]), "ci_share_o": ci(lambda b: b[1]["o"]),
                         "ratio_sd_o_over_sd_w": (math.sqrt(v["o"]) / math.sqrt(v["w"])) if v["w"] > 0 else None,
                         "mean_by_order": [float(YY[:, o].mean()) for o in range(O)]}
        # item level
        fo = []; tvo = []; fw = []; tvw = []; acc = np.zeros((W, O))
        for i, nm in enumerate(valid):
            for o in range(O): acc[i, o] = (Ps[o][nm].argmax(1) == y).mean()
            for a_, b_ in [(0, 1), (0, 2), (1, 2)]:
                A, Bm = Ps[a_][nm], Ps[b_][nm]; fo.append((A.argmax(1) != Bm.argmax(1)).mean()); tvo.append(0.5 * np.abs(A - Bm).sum(1).mean())
            for o in range(O):
                A, Bm = Ps[o][nm], Ps[o]["official"]; fw.append((A.argmax(1) != Bm.argmax(1)).mean()); tvw.append(0.5 * np.abs(A - Bm).sum(1).mean())
        offo = [(Ps[a_]["official"].argmax(1) != Ps[b_]["official"].argmax(1)).mean() for a_, b_ in [(0, 1), (0, 2), (1, 2)]]
        # display-position bias: probability mass on the option shown in display position j (only permuted orders; canonical = class order)
        pos = np.zeros(4 if ds in ("agnews", "bbc2025") else 0); cntp = 0
        sp = {s["name"]: s for s in sets}
        sys.path.insert(0, os.path.dirname(__file__)); from score_api import perm_for
        for o in (1, 2):
            for nm in valid:
                pm = perm_for(ds, nm, o, 4); P = Ps[o][nm]
                pos += np.array([P[:, pm[j]].mean() for j in range(4)]); cntp += 1
        out["item_level"] = {"argmax_flip_between_orders_mean": float(np.mean(fo)), "argmax_flip_between_orders_max": float(np.max(fo)),
                             "tv_between_orders_mean": float(np.mean(tvo)), "argmax_flip_vs_official_wording_mean": float(np.mean(fw)),
                             "tv_vs_official_wording_mean": float(np.mean(tvw)), "official_set_flip_between_orders": [float(x) for x in offo],
                             "accuracy_mean_by_order": [float(x) for x in acc.mean(0)], "accuracy_sd_over_wordings": float(acc.mean(1).std(ddof=1)),
                             "accuracy_sd_over_orders_within_wording_mean": float(acc.std(1, ddof=1).mean()),
                             "mean_prob_by_display_position_perm_orders": [float(x) for x in pos / cntp]}
        out["model_ids_returned"] = sorted({x for m in metas for x in m.get("model_ids_returned", [])}); out["request_window_utc"] = [min(m["first_request_utc"] for m in metas), max(m["last_request_utc"] for m in metas)]
        res[f"{ds}__{tag}"] = out
        print(ds, tag, {k: (round(out[k]["sd"]["w"], 4), round(out[k]["sd"]["o"], 4), round(out[k]["sd"]["s"], 4), round(out[k]["share"]["w"], 3), round(out[k]["share"]["o"], 3)) for k in ("lac_set_size", "aps_set_size")}, out["item_level"])
os.makedirs(os.path.dirname(OUTP), exist_ok=True); json.dump(res, open(OUTP, "w"), indent=1); print("DONE", OUTP)
