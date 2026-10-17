"""Exploratory template factor (gaps B12): bi-encoders, 3 hypothesis templates x 30 valid wording sets x 200 splits, n=250, LAC, alpha=0.10.
Three-way crossed random effects (template x wording x split, no replication), method-of-moments; plus sigma_wording within each template.
usage: template_factor.py  (expects raw/templates/<ds>__bienc__<model>__tpl{1,2}.npz; tpl0 = main raw file)"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from core import *
TPL = {"tpl0": "This text is about {}.", "tpl1": "{}", "tpl2": "Category: {}."}
def mom3(Y):   # Y: T x W x S
    T, W, S = Y.shape; m = Y.mean()
    mt, mw, ms = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean((0, 1))
    mtw, mts, mws = Y.mean(2), Y.mean(1), Y.mean(0)
    SS = lambda x, k: k * ((x) ** 2).sum()
    ss_t = SS(mt - m, W * S); ss_w = SS(mw - m, T * S); ss_s = SS(ms - m, T * W)
    ss_tw = SS(mtw - mt[:, None] - mw[None, :] + m, S); ss_ts = SS(mts - mt[:, None] - ms[None, :] + m, W); ss_ws = SS(mws - mw[:, None] - ms[None, :] + m, T)
    ss_e = ((Y - m) ** 2).sum() - ss_t - ss_w - ss_s - ss_tw - ss_ts - ss_ws
    MS = dict(t=ss_t / (T - 1), w=ss_w / (W - 1), s=ss_s / (S - 1), tw=ss_tw / ((T - 1) * (W - 1)), ts=ss_ts / ((T - 1) * (S - 1)), ws=ss_ws / ((W - 1) * (S - 1)),
              e=ss_e / ((T - 1) * (W - 1) * (S - 1)))
    v = dict(t=(MS["t"] - MS["tw"] - MS["ts"] + MS["e"]) / (W * S), w=(MS["w"] - MS["tw"] - MS["ws"] + MS["e"]) / (T * S), s=(MS["s"] - MS["ts"] - MS["ws"] + MS["e"]) / (T * W),
             tw=(MS["tw"] - MS["e"]) / S, ts=(MS["ts"] - MS["e"]) / W, ws=(MS["ws"] - MS["e"]) / T, e=MS["e"])
    return {k: float(np.sqrt(max(x, 0))) for k, x in v.items()}
out = {}
for f in sorted(glob.glob(f"{RAW}/templates/*__tpl1.npz")):
    b = os.path.basename(f).split("__"); ds, model = b[0], b[2]
    files = {"tpl0": f"{RAW}/{ds}__bienc__{model}.npz", "tpl1": f, "tpl2": f.replace("__tpl1", "__tpl2")}
    if not all(os.path.exists(x) for x in files.values()): continue
    sc = {k: Scorer(v) for k, v in files.items()}; y = np.array(load_ds(ds)["pool_y"])[: sc["tpl0"].N]; M = len(y)
    sets = [s for s in load_sets(ds) if s["pop"] == "valid" and all(sc[t].has(s["labels"]) for t in TPL)]
    off = [s for s in load_sets(ds) if s["name"] == "official"][0]
    cal, te = splits_A(M, 250); u = np.random.default_rng(99).uniform(size=M)
    Y = np.zeros((3, len(sets), len(cal))); acc = {}
    for ti, t in enumerate(TPL):
        for wi, s in enumerate(sets):
            Y[ti, wi] = conformal_eval(scores_all(sc[t].probs(s["labels"]), "lac", u), y, cal, te, 0.1)["set_size"]
        acc[t] = float((sc[t].probs(off["labels"]).argmax(1) == y).mean())
    within = {t: vc_summary(Y[ti], n_boot=200, seed=1) for ti, t in enumerate(TPL)}
    r = {"W": len(sets), "S": len(cal), "templates": TPL, "acc_official": acc, "mean_set_size": {t: float(Y[ti].mean()) for ti, t in enumerate(TPL)},
         "sd3": mom3(Y), "sd_w_within_template": {t: {k: within[t][k] for k in ("sd_w", "ci_sd_w", "sd_s")} for t in TPL}}
    out[f"{ds}__{model}"] = r; print(ds, model, json.dumps(r["sd3"]), acc, flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "template_factor.json"), "w"), indent=1)
print("DONE template_factor", len(out))
