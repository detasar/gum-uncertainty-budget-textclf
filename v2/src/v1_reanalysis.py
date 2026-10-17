"""Re-analysis of the PUBLISHED v1 numbers from the committed v1 outputs (no new model runs).
Clef-flash (budget B) and qwen3.5 (budget C): wording treated as a random factor with W = 4."""
import json, os, sys, math
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(__file__)); from core import vc_summary, vc_mom, softmax, ROOT
V1 = os.path.join(ROOT, "..")      # /workspace/x-master/uq-budget
Z = json.load(open(f"{V1}/cache/zs_set.json")); S = json.load(open(f"{V1}/cache/supervised_set.json"))
ycal = np.array(Z["cal_y"]); yte = np.array(Z["test_y"]); ycal_full = np.array(S["cal_y"]); NZ = 250; WK = list(Z["wordings"].keys())
def load(path, key):
    out = {}
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        out[(r["cfg"], r["split"], r["i"], r.get("rep", 0))] = np.array(r[key], float)
    return out
def qhat(cp, cy, a=0.1):
    s = 1 - cp[np.arange(len(cy)), cy]; n = len(s); return np.quantile(s, min(1, math.ceil((n + 1) * (1 - a)) / n), method="higher")
def ev(cp, cy, tp, ty):
    q = qhat(cp, cy); st = (1 - tp) <= q; return st.sum(1).mean(), st[np.arange(len(ty)), ty].mean()
def norm(P): P = np.where(P.sum(1, keepdims=True) > 0, P, 0.25); return P / P.sum(1, keepdims=True)
CL = load(f"{V1}/cache/mac/clef_logits.jsonl", "logits"); CX = load(f"{V1}/cache/mac/clef_logits_calx.jsonl", "logits"); QW = load(f"{V1}/cache/mac/llm_qwen3.5.jsonl", "p")
inst = {}
inst["clef"] = {w: (softmax(np.stack([CL[(f"{w}|q0_topic", "cal", i, 0)] for i in range(NZ)])), softmax(np.stack([CL[(f"{w}|q0_topic", "test", i, 0)] for i in range(NZ)]))) for w in WK}
inst["qwen3.5"] = {w: (norm(np.stack([QW[(w, "cal", i, 0)] for i in range(NZ)])), norm(np.stack([QW[(w, "test", i, 0)] for i in range(NZ)]))) for w in WK}
rng = np.random.default_rng(1); out = {}
for name, D in inst.items():
    r = {}
    ss = {w: ev(D[w][0], ycal, D[w][1], yte)[0] for w in WK}; vals = np.array(list(ss.values()))
    s0 = vals.std(ddof=0); s1 = vals.std(ddof=1)
    lo, hi = s1 * math.sqrt(3 / chi2.ppf(0.975, 3)), s1 * math.sqrt(3 / chi2.ppf(0.025, 3))
    r["set_size_per_wording"] = ss; r["u_wording_published_popsd"] = s0; r["sd_wording_ddof1"] = s1; r["chi2_95ci_sd_wording_ddof1"] = [lo, hi]
    ref = "w0_official"; cp, tp = D[ref]
    ucal = np.std([ev(cp[i], ycal[i], tp, yte)[0] for i in [rng.integers(0, NZ, NZ) for _ in range(1000)]], ddof=1)
    r["u_cal_bootstrap"] = ucal; r["published_style_share_wording_vs_cal_only"] = s0 ** 2 / (s0 ** 2 + ucal ** 2)
    shares = []
    for b in range(300):
        ws = [WK[k] for k in rng.integers(0, 4, 4)]; i = rng.integers(0, NZ, NZ); j = rng.integers(0, NZ, NZ)
        v = np.array([ev(D[w][0][i], ycal[i], D[w][1][j], yte[j])[0] for w in ws]); uw = v.std(ddof=0)
        uc = np.std([ev(D[ref][0][i][k], ycal[i][k], D[ref][1][j], yte[j])[0] for k in [rng.integers(0, NZ, NZ) for _ in range(60)]], ddof=1)
        shares.append(uw ** 2 / (uw ** 2 + uc ** 2) if uw + uc > 0 else 0)
    r["published_style_share_bootstrap_95ci"] = [float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))]
    Y = np.zeros((4, 500)); Yc = np.zeros((4, 500))
    for s in range(500):
        i = rng.integers(0, NZ, NZ); j = rng.integers(0, NZ, NZ)
        for k, w in enumerate(WK): Y[k, s], Yc[k, s] = ev(D[w][0][i], ycal[i], D[w][1][j], yte[j])
    v = vc_summary(Y, n_boot=1000); r["v2_method_set_size"] = {k: vv for k, vv in v.items() if not k.startswith("boot_")}
    v = vc_summary(Yc, n_boot=1000); r["v2_method_coverage"] = {k: vv for k, vv in v.items() if not k.startswith("boot_")}
    out[name] = r
    print(name, "popSD %.4f sd1 %.4f chi2CI [%.4f, %.4f] ucal %.4f pubshare %.2f CI %s | v2: share_w %.2f %s" % (s0, s1, lo, hi, ucal, r["published_style_share_wording_vs_cal_only"],
          np.round(r["published_style_share_bootstrap_95ci"], 2), r["v2_method_set_size"]["share_w"], np.round(r["v2_method_set_size"]["ci_share_w"], 2)), flush=True)
pool = {}
for w in WK:
    base = inst["clef"][w][0]; ex = np.stack([CX[(f"{w}|q0_topic", "cal", i, 0)] for i in range(250, 500)])
    pool[w] = np.vstack([base, softmax(ex)])
ypool = ycal_full[:500]; byn = {}
for n in [50, 100, 200, 250, 400, 500]:
    Y = np.zeros((4, 300))
    for s in range(300):
        i = rng.integers(0, 500, n)
        for k, w in enumerate(WK): Y[k, s] = ev(pool[w][i], ypool[i], inst["clef"][w][1], yte)[0]
    v = vc_summary(Y, n_boot=500); byn[n] = {k: vv for k, vv in v.items() if not k.startswith("boot_")}
    print("clef n=%d sd_w %.4f %s sd_cal %.4f share_w %.2f %s" % (n, v["sd_w"], np.round(v["ci_sd_w"], 4), v["sd_s"], v["share_w"], np.round(v["ci_share_w"], 2)), flush=True)
out["clef_share_by_n_fixed_test"] = byn
os.makedirs(f"{ROOT}/results", exist_ok=True); json.dump(out, open(f"{ROOT}/results/v1_reanalysis.json", "w"), indent=1, default=float)
