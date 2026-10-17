"""GUM-style uncertainty budget for split-conformal text classification (AG News, 4 classes).

Measurands (all at alpha = 0.10, LAC / "1 - p_true" score, split conformal):
  Y1 = mean prediction-set size on the test set   (efficiency)
  Y2 = empirical coverage on the test set
  Y3 = top-1 accuracy (context only)

Inputs:
  cache/supervised_set.json, cache/zs_set.json        (prep_data.py)
  cache/emb_{minilm,bgesmall,embeddinggemma}.npz     (embed_box.py / mac embed_gemma.py)
  cache/mac/clef_logits.jsonl, cache/mac/llm_*.jsonl (mac run_clef.py / run_ollama.py)
Outputs: results.json, budget_bar.png, coverage_setsize.png, scaling.png
"""
import json, os, glob, math, time, warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
ALPHA = 0.10; RNG = np.random.default_rng(12345)
N_BOOT_CAL = 1000      # calibration-set bootstrap draws
N_BOOT_TEST = 1000     # test-set bootstrap draws
N_TRAIN_BOOT = 15      # training-set bootstrap refits (per supervised model)
N_SEEDS = 15           # seed refits (per supervised model)

# ----------------------------------------------------------------------------- conformal core
def qhat(cal_p, cal_y, alpha=ALPHA):
    s = 1.0 - cal_p[np.arange(len(cal_y)), cal_y]; n = len(s)
    lvl = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(s, lvl, method="higher")

def evaluate(cal_p, cal_y, te_p, te_y, alpha=ALPHA):
    q = qhat(cal_p, cal_y, alpha); sets = (1.0 - te_p) <= q
    return {"set_size": float(sets.sum(1).mean()), "coverage": float(sets[np.arange(len(te_y)), te_y].mean()),
            "accuracy": float((te_p.argmax(1) == te_y).mean())}

def boot_ci(vals, lo=2.5, hi=97.5):
    vals = np.asarray(vals); return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]

def softmax(z):
    z = np.asarray(z, float); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def entropy(p):
    p = np.clip(p, 1e-12, 1); return -(p * np.log(p)).sum(-1)

def decompose(P):
    """P: (members, n, K). Returns TU, AU (expected entropy), EU (mutual information), per example."""
    TU = entropy(P.mean(0)); AU = entropy(P).mean(0); return TU, AU, TU - AU

# ----------------------------------------------------------------------------- data
S = json.load(open(f"{C}/supervised_set.json")); Z = json.load(open(f"{C}/zs_set.json"))
ytr = np.array(S["train_y"]); ycal_full = np.array(S["cal_y"]); yte_full = np.array(S["test_y"])
NZ = len(Z["cal_y"]); ycal = ycal_full[:NZ]; yte = yte_full[:NZ]       # shared 250/250 evaluation subset
assert (ycal == np.array(Z["cal_y"])).all() and (yte == np.array(Z["test_y"])).all()
SH = "0.3"  # shift level used for the shared comparison

def fit_predict(Xtr, y, Xs, seed):
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, random_state=seed).fit(sc.transform(Xtr), y)
    return [m.predict_proba(sc.transform(X)) for X in Xs]

# ----------------------------------------------------------------------------- supervised models
SUP = {}
for tag in ["minilm", "bgesmall", "embeddinggemma"]:
    f = f"{C}/emb_{tag}.npz"
    if not os.path.exists(f): print("missing", f); continue
    E = np.load(f); t0 = time.time()
    Xs = [E["cal"][:NZ], E["test_0.0"][:NZ], E[f"test_{SH}"][:NZ]] + [E[f"test_{l}"][:NZ] for l in ["0.15", "0.45"]]
    ref = fit_predict(E["train"], ytr, Xs, seed=0)
    seeds = [fit_predict(E["train"], ytr, Xs, seed=s) for s in range(1, N_SEEDS + 1)]
    boots = []
    for b in range(N_TRAIN_BOOT):
        idx = RNG.integers(0, len(ytr), len(ytr)); boots.append(fit_predict(E["train"][idx], ytr[idx], Xs, seed=0))
    SUP[tag] = {"ref": ref, "seeds": seeds, "boots": boots}
    print(tag, "fitted in", round(time.time() - t0, 1), "s", flush=True)

# ----------------------------------------------------------------------------- zero-shot: Clef + LLMs
WK = list(Z["wordings"].keys()); QK = ["q0_topic", "q1_section", "q2_classify"]
def load_jsonl(path, key):
    out = {}
    if not os.path.exists(path): return out
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        out[(r["cfg"], r["split"], r["i"], r.get("rep", 0))] = np.array(r[key], float)
    return out

def assemble(D, cfg, split, n, transform):
    rows = [D.get((cfg, split, i, 0)) for i in range(n)]
    if any(r is None for r in rows): return None
    return transform(np.stack(rows))

def llm_norm(P):
    P = np.where(P.sum(1, keepdims=True) > 0, P, 0.25); return P / P.sum(1, keepdims=True)

CLEF = load_jsonl(f"{C}/mac/clef_logits.jsonl", "logits")
ZS = {}  # name -> cfg -> {"cal","test","shift"}
def add_zs(name, D, cfgs, transform):
    for cfg in cfgs:
        cal = assemble(D, cfg, "cal", NZ, transform); te = assemble(D, cfg, "test", NZ, transform)
        if cal is None or te is None: continue
        ZS.setdefault(name, {})[cfg] = {"cal": cal, "test": te, "shift": assemble(D, cfg, "shift30", NZ, transform)}
add_zs("clef-flash-8bit", CLEF, [f"{w}|{q}" for w in WK for q in QK], softmax)
for path in sorted(glob.glob(f"{C}/mac/llm_*.jsonl")):
    name = os.path.basename(path)[4:-6]; add_zs(name, load_jsonl(path, "p"), WK, llm_norm)
print("zero-shot configs:", {k: list(v) for k, v in ZS.items()})

# determinism checks (identical repeated calls)
def det_check(D):
    reps = {}
    for (cfg, split, i, rep), v in D.items():
        if cfg == "det": reps.setdefault(i, {})[rep] = v
    if not reps: return None
    diffs = [np.abs(v[r] - v[0]).max() for v in reps.values() for r in v if r > 0]
    return {"n_texts": len(reps), "n_repeat_pairs": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None}
DET = {"clef-flash-8bit (logits)": det_check(CLEF)}
for path in sorted(glob.glob(f"{C}/mac/llm_*.jsonl")):
    DET[os.path.basename(path)[4:-6] + " (A-D probs)"] = det_check(load_jsonl(path, "p"))

# ----------------------------------------------------------------------------- budget helpers
def u_of(vals, ddof):
    vals = np.asarray(vals, float); return float(vals.std(ddof=ddof)) if len(vals) > 1 else 0.0

def resample_component(cal_p, te_p, which, n_draw):
    out = {"set_size": [], "coverage": []}
    for _ in range(n_draw):
        if which == "cal":
            i = RNG.integers(0, NZ, NZ); r = evaluate(cal_p[i], ycal[i], te_p, yte)
        else:
            j = RNG.integers(0, NZ, NZ); r = evaluate(cal_p, ycal, te_p[j], yte[j])
        for k in out: out[k].append(r[k])
    return out

def component(name, typ, vals_ss, vals_cov, ddof, note, levels=None):
    return {"component": name, "type": typ, "u_set_size": u_of(vals_ss, ddof), "u_coverage": u_of(vals_cov, ddof),
            "n_levels": len(vals_ss), "note": note,
            **({"levels": levels} if levels is not None else {})}

def combine(comps):
    inc = [c for c in comps if c.get("in_rss", True)]
    for k in ["u_set_size", "u_coverage"]:
        tot = math.sqrt(sum(c[k] ** 2 for c in inc))
        for c in comps: c[k.replace("u_", "share_var_")] = (c[k] ** 2 / tot ** 2) if (tot > 0 and c.get("in_rss", True)) else 0.0
    return {"u_c_set_size": math.sqrt(sum(c["u_set_size"] ** 2 for c in inc)),
            "u_c_coverage": math.sqrt(sum(c["u_coverage"] ** 2 for c in inc))}

REP = json.load(open(f"{HERE}/repeatability.json")) if os.path.exists(f"{HERE}/repeatability.json") else {}
def rep_component(key, label, in_rss=True):
    r = REP.get(key)
    if r is None: return None
    return {"component": label, "type": "A (repeat, >=4 identical runs)", "u_set_size": r["sd_set_size_across_runs"], "u_coverage": r["sd_coverage_across_runs"],
            "n_levels": r["runs"], "in_rss": in_rss,
            "note": f"{r['runs']} runs x {r['items']} items (half cal / half test); max |dp| = {r['max_abs_dp_any_pair']:.3g}; "
                    f"items with changed conformal set = {100*r['pct_test_items_whose_conformal_set_changes']:.1f}%"}

def shift_component(ref_clean, ref_shift, label):
    # Type B: deployment shift severity unknown between 0 and the tested level -> rectangular, half-width |delta|
    dss = ref_shift["set_size"] - ref_clean["set_size"]; dcv = ref_shift["coverage"] - ref_clean["coverage"]
    return {"component": label, "type": "B (rectangular, half-width = abs(shift effect))",
            "u_set_size": abs(dss) / math.sqrt(3), "u_coverage": abs(dcv) / math.sqrt(3), "n_levels": 2,
            "delta_set_size": dss, "delta_coverage": dcv,
            "note": "Uncorrected systematic effect; assumes the actual shift lies anywhere between none and the tested level."}

RES = {"alpha": ALPHA, "n_cal_pool": NZ, "n_test": NZ, "n_train": int(len(ytr)), "determinism": DET, "budgets": {}, "models": {}}

# ----------------------------------------------------------------------------- per-model reference metrics + CIs
def ref_metrics(cal_p, te_p, sh_p=None):
    r = evaluate(cal_p, ycal, te_p, yte)
    cb = resample_component(cal_p, te_p, "cal", 300); tb = resample_component(cal_p, te_p, "test", 300)
    # joint bootstrap for CI: resample cal and test together
    js, jc, ja = [], [], []
    for _ in range(N_BOOT_TEST):
        i = RNG.integers(0, NZ, NZ); j = RNG.integers(0, NZ, NZ); e = evaluate(cal_p[i], ycal[i], te_p[j], yte[j])
        js.append(e["set_size"]); jc.append(e["coverage"]); ja.append(e["accuracy"])
    r.update({"ci95_set_size": boot_ci(js), "ci95_coverage": boot_ci(jc), "ci95_accuracy": boot_ci(ja)})
    if sh_p is not None:
        s = evaluate(cal_p, ycal, sh_p, yte); r["shift30"] = s
    return r

CLASS = {"sup": ("supervised probe", "single-pass class probabilities from an MLP trained on 2,000 labelled items"),
         "clef": ("deterministic schema classifier (zero-shot)", "single-pass option probabilities (softmax of head logits), no sampling"),
         "llm": ("generative chat LLM (zero-shot)", "first-token logprobs over letters A-D (renormalised); the sampled answer is a separate output")}
for tag, d in SUP.items():
    RES["models"][f"{tag}+MLP (supervised)"] = ref_metrics(d["ref"][0], d["ref"][1], d["ref"][2])
    RES["models"][f"{tag}+MLP (supervised)"].update(instrument_class=CLASS["sup"][0], output_type=CLASS["sup"][1])
for name, cfgs in ZS.items():
    refcfg = "w0_official|q0_topic" if name.startswith("clef") else "w0_official"
    if refcfg in cfgs:
        c = cfgs[refcfg]; RES["models"][f"{name} (zero-shot)"] = ref_metrics(c["cal"], c["test"], c["shift"])
        k = CLASS["clef"] if name.startswith("clef") else CLASS["llm"]
        RES["models"][f"{name} (zero-shot)"].update(instrument_class=k[0], output_type=k[1])

# ----------------------------------------------------------------------------- Budget A: supervised (reference: minilm+MLP)
def supervised_budget(tag):
    d = SUP[tag]; cal_p, te_p, sh_p = d["ref"][0], d["ref"][1], d["ref"][2]
    ref = evaluate(cal_p, ycal, te_p, yte)
    comps = []
    cb = resample_component(cal_p, te_p, "cal", N_BOOT_CAL)
    comps.append(component("calibration-set sampling (n=250)", "A (bootstrap)", cb["set_size"], cb["coverage"], 1, "Bootstrap of the calibration set."))
    tb = resample_component(cal_p, te_p, "test", N_BOOT_TEST)
    comps.append(component("test-set sampling (n=250)", "A (bootstrap)", tb["set_size"], tb["coverage"], 1, "Finite evaluation sample."))
    ev = [evaluate(s[0], ycal, s[1], yte) for s in d["seeds"]]
    comps.append(component("random seed (MLP init/order)", "A (repeat)", [e["set_size"] for e in ev], [e["coverage"] for e in ev], 1, f"{N_SEEDS} refits, same data."))
    ev = [evaluate(s[0], ycal, s[1], yte) for s in d["boots"]]
    comps.append(component("training-data sampling (bootstrap, n=2000)", "A (bootstrap)", [e["set_size"] for e in ev], [e["coverage"] for e in ev], 1, f"{N_TRAIN_BOOT} refits on bootstrap resamples."))
    ev = {t: evaluate(SUP[t]["ref"][0], ycal, SUP[t]["ref"][1], yte) for t in SUP}
    comps.append(component("model choice within class (3 embedders)", "B (discrete uniform over alternatives)", [e["set_size"] for e in ev.values()],
                           [e["coverage"] for e in ev.values()], 0, "minilm / bge-small / embeddinggemma, same head.", levels=ev))
    comps.append(shift_component(ref, evaluate(cal_p, ycal, sh_p, yte), "distribution shift (word drop + typos, level 0.3)"))
    return {"reference": f"{tag}+MLP", "reference_metrics": ref, "components": comps, **combine(comps)}

if "minilm" in SUP: RES["budgets"]["A_supervised_minilm"] = supervised_budget("minilm")

# ----------------------------------------------------------------------------- Budget B: zero-shot Clef (reference w0|q0)
def clef_budget():
    cf = ZS["clef-flash-8bit"]; r0 = cf["w0_official|q0_topic"]; cal_p, te_p = r0["cal"], r0["test"]
    ref = evaluate(cal_p, ycal, te_p, yte); comps = []
    rc = rep_component("clef-flash-8bit | w0_official | deterministic", "run-to-run repeatability (identical runs)")
    if rc: comps.append(rc)
    cb = resample_component(cal_p, te_p, "cal", N_BOOT_CAL)
    comps.append(component("calibration-set sampling (n=250)", "A (bootstrap)", cb["set_size"], cb["coverage"], 1, "Bootstrap of the calibration set."))
    tb = resample_component(cal_p, te_p, "test", N_BOOT_TEST)
    comps.append(component("test-set sampling (n=250)", "A (bootstrap)", tb["set_size"], tb["coverage"], 1, "Finite evaluation sample."))
    ev = {w: evaluate(cf[f"{w}|q0_topic"]["cal"], ycal, cf[f"{w}|q0_topic"]["test"], yte) for w in WK if f"{w}|q0_topic" in cf}
    comps.append(component("label/option wording (4 paraphrase sets)", "B (discrete uniform over alternatives)", [e["set_size"] for e in ev.values()],
                           [e["coverage"] for e in ev.values()], 0, "Same question, 4 verbalizer sets; each recalibrated.", levels=ev))
    ev = {q: evaluate(cf[f"w0_official|{q}"]["cal"], ycal, cf[f"w0_official|{q}"]["test"], yte) for q in QK if f"w0_official|{q}" in cf}
    comps.append(component("question phrasing (3 instructions)", "B (discrete uniform over alternatives)", [e["set_size"] for e in ev.values()],
                           [e["coverage"] for e in ev.values()], 0, "Same options, 3 instructions; each recalibrated.", levels=ev))
    comps.append({"component": "model choice within class (deterministic schema classifiers)", "type": "B", "u_set_size": 0.0, "u_coverage": 0.0,
                  "n_levels": 1, "in_rss": False, "note": "NOT EVALUATED: only one model of this instrument type was run (GLiNER2 install failed). Not zero, unknown."})
    if r0["shift"] is not None:
        comps.append(shift_component(ref, evaluate(cal_p, ycal, r0["shift"], yte), "distribution shift (word drop + typos, level 0.3)"))
    return {"reference": "clef-flash-8bit, w0_official|q0_topic", "reference_metrics": ref, "components": comps, **combine(comps)}

if "clef-flash-8bit" in ZS and "w0_official|q0_topic" in ZS["clef-flash-8bit"]:
    RES["budgets"]["B_zeroshot_clef"] = clef_budget()

# ----------------------------------------------------------------------------- Budget C: zero-shot LLM qwen3.5:9b (t0)
def llm_budget(name):
    cf = ZS[name]; r0 = cf["w0_official"]; cal_p, te_p = r0["cal"], r0["test"]
    ref = evaluate(cal_p, ycal, te_p, yte); comps = []
    rc = rep_component(f"{name} | w0_official | t0", "run-to-run repeatability, temperature 0 (operating mode)")
    if rc: comps.append(rc)
    rc = rep_component(f"{name} | w0_official | default", "run-to-run repeatability IF run at default temperature (not in RSS)", in_rss=False)
    if rc: comps.append(rc)
    cb = resample_component(cal_p, te_p, "cal", N_BOOT_CAL)
    comps.append(component("calibration-set sampling (n=250)", "A (bootstrap)", cb["set_size"], cb["coverage"], 1, "Bootstrap of the calibration set."))
    tb = resample_component(cal_p, te_p, "test", N_BOOT_TEST)
    comps.append(component("test-set sampling (n=250)", "A (bootstrap)", tb["set_size"], tb["coverage"], 1, "Finite evaluation sample."))
    ev = {w: evaluate(cf[w]["cal"], ycal, cf[w]["test"], yte) for w in WK if w in cf}
    others = [n for n in ZS if n != name and not n.startswith("clef") and "w0_official" in ZS[n]]
    if others:
        evm = {n: evaluate(ZS[n]["w0_official"]["cal"], ycal, ZS[n]["w0_official"]["test"], yte) for n in [name] + others}
        comps.append(component(f"model choice within class (generative LLMs: {', '.join([name] + others)})", "B (discrete uniform over alternatives)",
                               [e["set_size"] for e in evm.values()], [e["coverage"] for e in evm.values()], 0, "Same prompt and wording w0; each recalibrated.", levels=evm))
    if len(ev) > 1:
        comps.append(component(f"label/option wording ({len(ev)} paraphrase sets)", "B (discrete uniform over alternatives)", [e["set_size"] for e in ev.values()],
                               [e["coverage"] for e in ev.values()], 0, "Same prompt template, different verbalizers; each recalibrated.", levels=ev))
    if r0["shift"] is not None:
        comps.append(shift_component(ref, evaluate(cal_p, ycal, r0["shift"], yte), "distribution shift (word drop + typos, level 0.3)"))
    return {"reference": f"{name}, w0_official, temperature 0", "reference_metrics": ref, "components": comps, **combine(comps)}
for nm in ZS:
    if nm.startswith("qwen3.5") and "w0_official" in ZS[nm]: RES["budgets"]["C_zeroshot_qwen3.5_9b"] = llm_budget(nm)
RES["comparability_note"] = ("The three instrument classes differ in output type (supervised probe vs single-pass deterministic schema scores vs "
    "sampled generative LLM with logprob-derived scores) and in information used (2,000 labelled items vs none). Accuracy/coverage/set size are "
    "computed identically, but they are NOT a leaderboard: LLM probabilities are heavily saturated, which forces singleton sets, and Clef allows empty sets. "
    "Compare budgets (which components dominate), not headline numbers.")

# ----------------------------------------------------------------------------- Monte Carlo (GUM-S1 style) check of the RSS for Clef
def clef_mc(n=2000):
    cf = ZS["clef-flash-8bit"]; cfgs = [c for c in cf]; ss, cv = [], []
    for _ in range(n):
        c = cf[cfgs[RNG.integers(len(cfgs))]]; i = RNG.integers(0, NZ, NZ); j = RNG.integers(0, NZ, NZ)
        e = evaluate(c["cal"][i], ycal[i], c["test"][j], yte[j]); ss.append(e["set_size"]); cv.append(e["coverage"])
    bud = RES["budgets"].get("B_zeroshot_clef", {}).get("components", [])
    sub = [c for c in bud if c["component"].split(" (")[0] in ("calibration-set sampling", "test-set sampling", "label/option wording", "question phrasing")]
    rss = math.sqrt(sum(c["u_set_size"] ** 2 for c in sub)); rssc = math.sqrt(sum(c["u_coverage"] ** 2 for c in sub))
    return {"rss_same_components_set_size": rss, "rss_same_components_coverage": rssc, "configs_sampled": cfgs, "n_draws": n, "sd_set_size": float(np.std(ss, ddof=1)), "sd_coverage": float(np.std(cv, ddof=1)),
            "note": "Joint sampling of prompt config (uniform over available wording/question configs) x cal bootstrap x test bootstrap."}
if "clef-flash-8bit" in ZS: RES["mc_check_clef"] = clef_mc()

# ----------------------------------------------------------------------------- data vs structure: scaling with calibration size
def scaling(cfg_pools, n_grid, draws=300):
    """cfg_pools: wording -> (cal_p_pool, cal_y_pool, te_p). For each n: draw the SAME calibration indices for all
    wordings (paired), evaluate set size per wording.
      sd_cal      = SD over draws (per wording, averaged)        -> calibration-sampling component
      sd_word_raw = SD across wordings within a draw (averaged)  -> includes calibration noise
      sd_word_sys = SD across wordings of the per-wording mean   -> 'systematic' wording component"""
    out = []
    npool = min(len(v[1]) for v in cfg_pools.values())
    for n in n_grid:
        if n > npool: continue
        per = {k: [] for k in cfg_pools}
        for _ in range(draws):
            i = RNG.integers(0, npool, n)   # bootstrap from the pool for every n (avoids finite-population shrinkage at n close to pool size)
            for k, (cp, cy, tp) in cfg_pools.items():
                per[k].append(evaluate(cp[i], cy[i], tp, yte)["set_size"])
        A = np.array([per[k] for k in cfg_pools])                       # (wordings, draws)
        # bootstrap CI for sd_word_sys over draws
        bs = [np.std(A[:, RNG.integers(0, draws, draws)].mean(1), ddof=0) for _ in range(500)]
        out.append({"n_cal": n, "sd_cal_sampling": float(A.std(1, ddof=1).mean()),
                    "sd_wording_within_draw": float(A.std(0, ddof=0).mean()),
                    "sd_wording_systematic": float(A.mean(1).std(ddof=0)), "ci95_sd_wording_systematic": boot_ci(bs),
                    "mean_set_size_per_wording": {k: float(v) for k, v in zip(cfg_pools, A.mean(1))},
                    "sampling": "bootstrap with replacement from the pool"})
    return out

if "clef-flash-8bit" in ZS:
    CALX = load_jsonl(f"{C}/mac/clef_logits_calx.jsonl", "logits")
    pools = {}
    for w in WK:
        cfg = f"{w}|q0_topic"
        if cfg not in ZS["clef-flash-8bit"]: continue
        base = ZS["clef-flash-8bit"][cfg]["cal"]; extra = [CALX.get((cfg, "cal", i, 0)) for i in range(250, 500)]
        if all(e is not None for e in extra): base = np.vstack([base, softmax(np.stack(extra))])
        pools[w] = (base, ycal_full[:len(base)], ZS["clef-flash-8bit"][cfg]["test"])
    if len(pools) > 1:
        RES["scaling_clef_wording"] = scaling(pools, [25, 50, 100, 200, 400, 500])
        RES["scaling_pool_size"] = min(len(v[1]) for v in pools.values())

# ----------------------------------------------------------------------------- "aleatoric vs epistemic" depends on what you call the model
def ae_table():
    rows = {}
    for tag, d in SUP.items():
        P = np.stack([b[1] for b in d["boots"]]); TU, AU, EU = decompose(P)
        Ps = np.stack([b[2] for b in d["boots"]]); TUs, AUs, EUs = decompose(Ps)
        rows[f"{tag}+MLP, ensemble = training bootstraps"] = {"mean_TU": float(TU.mean()), "mean_AU": float(AU.mean()), "mean_EU": float(EU.mean()),
            "EU_share": float(EU.sum() / TU.sum()), "shift30_EU_share": float(EUs.sum() / TUs.sum()),
            "shift30_mean_AU": float(AUs.mean()), "shift30_mean_EU": float(EUs.mean()), "_EU": EU, "_AU": AU}
    if "clef-flash-8bit" in ZS:
        cf = ZS["clef-flash-8bit"]
        single = cf["w0_official|q0_topic"]["test"][None]; TU, AU, EU = decompose(single)
        rows["clef, single prompt (deterministic => no ensemble)"] = {"mean_TU": float(TU.mean()), "mean_AU": float(AU.mean()), "mean_EU": float(EU.mean()), "EU_share": 0.0}
        keys = [k for k in cf]
        P = np.stack([cf[k]["test"] for k in keys]); TU, AU, EU = decompose(P)
        rows[f"clef, ensemble = {len(keys)} prompt configs"] = {"mean_TU": float(TU.mean()), "mean_AU": float(AU.mean()), "mean_EU": float(EU.mean()),
                                                                 "EU_share": float(EU.sum() / TU.sum()), "_EU": EU, "_AU": AU}
    # cross-model agreement on which examples are "epistemic-dominated"
    tags = [k for k in rows if "_EU" in rows[k]]; agree = {}
    for a in range(len(tags)):
        for b in range(a + 1, len(tags)):
            ea, eb = rows[tags[a]]["_EU"], rows[tags[b]]["_EU"]
            from scipy.stats import spearmanr
            top_a = ea >= np.quantile(ea, 0.9); top_b = eb >= np.quantile(eb, 0.9)
            agree[f"{tags[a]}  vs  {tags[b]}"] = {"spearman_EU": float(spearmanr(ea, eb)[0]),
                                                  "overlap_top10pct_EU": float((top_a & top_b).sum() / top_a.sum())}
    for k in rows: rows[k] = {kk: vv for kk, vv in rows[k].items() if not kk.startswith("_")}
    return {"decomposition": rows, "cross_model_agreement_on_EU": agree}
RES["aleatoric_epistemic"] = ae_table()


# ----------------------------------------------------------------------------- configuration shift across wordings + wording ensemble
def cross_wording(name, keyfmt):
    cf = ZS.get(name, {}); ks = [w for w in WK if keyfmt.format(w) in cf]
    if len(ks) < 2: return None
    M = {}
    for a in ks:
        for b in ks:
            e = evaluate(cf[keyfmt.format(a)]["cal"], ycal, cf[keyfmt.format(b)]["test"], yte)
            M[f"cal={a} | test={b}"] = e
    off = [v["coverage"] for k, v in M.items() if k.split(" | ")[0][4:] != k.split(" | ")[1][5:]]
    diag = [v["coverage"] for k, v in M.items() if k.split(" | ")[0][4:] == k.split(" | ")[1][5:]]
    ens_cal = np.mean([cf[keyfmt.format(w)]["cal"] for w in ks], 0); ens_te = np.mean([cf[keyfmt.format(w)]["test"] for w in ks], 0)
    ens = evaluate(ens_cal, ycal, ens_te, yte)
    return {"matrix": M, "coverage_same_wording_mean": float(np.mean(diag)), "coverage_same_wording_range": [float(min(diag)), float(max(diag))],
            "coverage_cross_wording_mean": float(np.mean(off)), "coverage_cross_wording_range": [float(min(off)), float(max(off))],
            "wording_ensemble (mean prob over wordings)": ens}
RES["cross_wording"] = {k: v for k, v in {"clef-flash-8bit": cross_wording("clef-flash-8bit", "{}|q0_topic"),
                                           "qwen3.5": cross_wording("qwen3.5", "{}")}.items() if v}

# ----------------------------------------------------------------------------- option-name vs definition (after Sun et al. 2609.26758)
def nameswap():
    f = f"{C}/mac/clef_nameswap.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        rows[(r["cond"], r["i"])] = r
    names = ["World", "Sports", "Business", "Technology"]; out = {}; sel = {}
    for cond in ["matched", "swapped"]:
        if not all((cond, i) in rows for i in range(NZ)): continue
        sd, nm = [], []
        for i in range(NZ):
            r = rows[(cond, i)]; j = int(np.argmax(r["logits"])); sd.append(r["def_index_of_option"][j]); nm.append(names.index(r["option_ids"][j]))
        sd, nm = np.array(sd), np.array(nm); sel[cond] = sd
        out[cond] = {"follows_definition (accuracy vs true label)": float((sd == yte).mean()), "follows_name": float((nm == yte).mean()),
                     "ci95_follows_definition": boot_ci([(sd[j] == yte[j]).mean() for j in [RNG.integers(0, NZ, NZ) for _ in range(1000)]])}
    if "w2_topic|q0_topic" in ZS.get("clef-flash-8bit", {}):
        neu = ZS["clef-flash-8bit"]["w2_topic|q0_topic"]["test"].argmax(1); out["neutral_ids_A-D"] = {"follows_definition (accuracy vs true label)": float((neu == yte).mean())}
        sel["neutral"] = neu
    if "matched" in sel and "swapped" in sel:
        out["decision_flip_rate_matched_to_swapped"] = float((sel["matched"] != sel["swapped"]).mean())
    if "neutral" in sel and "matched" in sel:
        out["decision_flip_rate_neutral_to_matched"] = float((sel["neutral"] != sel["matched"]).mean())
    return out
ns = nameswap()
if ns: RES["name_vs_definition_clef"] = ns

json.dump(RES, open(f"{HERE}/results.json", "w"), indent=2, default=float)
print(json.dumps({k: RES[k] for k in ["models", "determinism"]}, indent=1, default=float)[:3000])
for b, v in RES["budgets"].items():
    print("\n==", b, v["reference_metrics"], "u_c(set)=%.4f u_c(cov)=%.4f" % (v["u_c_set_size"], v["u_c_coverage"]))
    for c in v["components"]: print("  %-55s u_set=%.4f (%.0f%%)  u_cov=%.4f (%.0f%%)" % (c["component"], c["u_set_size"], 100*c.get("share_var_set_size",0), c["u_coverage"], 100*c.get("share_var_coverage",0)))
print("MC:", {k: v for k, v in RES.get("mc_check_clef", {}).items() if k != "configs_sampled"})
for r in RES.get("scaling_clef_wording", []): print("scaling", {k: v for k, v in r.items() if k != "mean_set_size_per_wording"})
print(json.dumps(RES["aleatoric_epistemic"], indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "matrix"} for k, v in RES.get("cross_wording", {}).items()}, indent=1))
print(json.dumps(RES.get("name_vs_definition_clef"), indent=1))

# raw per-item probabilities of the supervised reference fits (shared 250 cal / 250 test / 250 shifted test)
np.savez_compressed(f"{HERE}/raw_supervised_probs.npz", **{f"{t}_{s}": SUP[t]["ref"][j] for t in SUP for j, s in enumerate(["cal", "test", "shift30"])},
                    cal_y=ycal, test_y=yte)
