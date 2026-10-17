"""Core analysis library for v2: probabilities from raw scores, conformal methods (LAC/APS/RAPS), split designs,
variance components (two-way crossed random effects, method of moments) and bootstrap CIs."""
import json, os, math, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); D = os.path.join(ROOT, "data"); RAW = os.path.join(ROOT, "raw")
TEST_SIZE = 300; N_SPLITS = 200; TAU = 0.05; RAPS_LAMBDA = 0.01; RAPS_KREG = 1
TYPES = {"nli": "T1 zero-shot NLI cross-encoder", "bienc": "T2 zero-shot bi-encoder", "llm": "T3 small LLM (label log-likelihood, PMI)", "sup": "T4 supervised probe (LogReg)",
         "api": "T5 closed API LLM (exploratory, not preregistered)"}
API_EPS = 1e-4   # probability floor for API cells whose probabilities are quantised (gpt-6-luna: 0.01 grid, many exact zeros)
DATASETS = ["agnews", "bbc2025", "sst2", "dbpedia", "massive"]

def softmax(z, axis=-1):
    z = np.asarray(z, np.float64); z = z - z.max(axis, keepdims=True); e = np.exp(z); return e / e.sum(axis, keepdims=True)

def load_ds(ds):
    d = json.load(open(f"{D}/{ds}.json")); return d
def load_sets(ds): return json.load(open(f"{D}/wordsets_{ds}.json"))["sets"]
def load_labels(ds): return json.load(open(f"{D}/labels_{ds}.json"))

class Scorer:
    """Wraps one raw score file; probs(labels) returns P (N x K) for a wording set (list of K label strings) or None if unavailable."""
    def __init__(self, path, tau=TAU, llm_mode="pmi"):
        z = np.load(path, allow_pickle=True); self.path = path
        b = os.path.basename(path)[:-4].split("__"); self.ds, self.kind, self.model = b[0], b[1], b[2]
        self.strings = list(z["strings"]); self.idx = {s: i for i, s in enumerate(self.strings)}; self.meta = json.loads(str(z["meta"]))
        if self.kind == "nli": self.S = z["logits"][:, :, int(z["entail_index"])].astype(np.float64)
        elif self.kind == "bienc": self.S = z["cos"].astype(np.float64) / tau; self.emb_pool = z["emb_pool"]; self.emb_train = z["emb_train"]
        elif self.kind == "llm": self.S = (z["raw"] - z["null"][None, :]).astype(np.float64) if llm_mode == "pmi" else z["raw"].astype(np.float64)
        elif self.kind == "api":   # per-wording-set probabilities (N x n_sets x K, canonical class order); NaN rows = refusal/missing -> uniform
            self.P = z["P"].astype(np.float64); self.eps = float(self.meta.get("eps", 0.0))
            self.setidx = {}
            for j, lab in enumerate(json.loads(str(z["set_labels"]))): self.setidx.setdefault(tuple(lab), j)
            self.S = self.P[:, 0, :]
        self.N = self.S.shape[0]
    def has(self, labels):
        if self.kind == "api": return tuple(labels) in self.setidx
        return all(l in self.idx for l in labels)
    def probs(self, labels):
        if not self.has(labels): return None
        if self.kind == "api":
            P = self.P[:, self.setidx[tuple(labels)], :len(labels)].copy(); bad = ~np.isfinite(P).all(1); P[bad] = 1.0 / len(labels)
            P = np.maximum(P, self.eps); return P / P.sum(1, keepdims=True)
        return softmax(self.S[:, [self.idx[l] for l in labels]])

# ------------------------------------------------------------------ conformal
def qlevel(n, alpha): return min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)

def scores_all(P, method, u):
    """Nonconformity score of every label for every item. P: N x K; u: N uniforms (APS/RAPS randomisation)."""
    if method == "lac": return 1.0 - P
    order = np.argsort(-P, 1, kind="stable"); Ps = np.take_along_axis(P, order, 1); cum = np.cumsum(Ps, 1)
    above = cum - Ps                                   # mass strictly ranked above each label (ties broken by order)
    s_sorted = above + u[:, None] * Ps
    if method == "raps":
        rank = np.arange(1, P.shape[1] + 1)[None, :]; s_sorted = s_sorted + RAPS_LAMBDA * np.maximum(0, rank - RAPS_KREG)
    S = np.empty_like(P); np.put_along_axis(S, order, s_sorted, 1); return S

def conformal_eval(SC, y, cal_idx, test_idx, alpha):
    """SC: N x K nonconformity scores. cal_idx: S x n, test_idx: S x T. Returns dict of S-vectors."""
    n = cal_idx.shape[1]; cs = SC[cal_idx, y[cal_idx]]                 # S x n
    q = np.quantile(cs, qlevel(n, alpha), axis=1, method="higher")   # S
    st = SC[test_idx] <= q[:, None, None]                            # S x T x K
    yt = y[test_idx]; size = st.sum(2); cov_item = np.take_along_axis(st, yt[..., None], 2)[..., 0]
    K = SC.shape[1]; cls_cov = []
    for c in range(K):
        m = yt == c; cnt = m.sum(1)
        cls_cov.append(np.where(cnt > 0, (cov_item & m).sum(1) / np.maximum(cnt, 1), np.nan))
    return {"set_size": size.mean(1), "coverage": cov_item.mean(1), "singleton": (size == 1).mean(1), "empty": (size == 0).mean(1),
            "worst_class_cov": np.nanmin(np.stack(cls_cov, 1), 1), "q": q}

def point_metrics(P, y, idx=None):
    if idx is not None: P, y = P[idx], y[idx]
    pred = P.argmax(1); conf = P.max(1); acc = (pred == y).mean()
    bins = np.linspace(0, 1, 16); ece = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return {"accuracy": float(acc), "ece": float(ece), "mean_maxp": float(conf.mean()), "share_maxp_gt_099": float((conf > 0.99).mean())}

# ------------------------------------------------------------------ split designs
def splits_A(M, n, S=N_SPLITS, T=TEST_SIZE):
    """Design A: random partitions. Split s uses permutation rng(s): test = first T, calibration = next n (nested in n)."""
    cal, te = [], []
    for s in range(S):
        p = np.random.default_rng(s).permutation(M); te.append(p[:T]); cal.append(p[T:T + n])
    return np.array(cal), np.array(te)

def splits_B(M, n, S=N_SPLITS, T=TEST_SIZE):
    """Design B (v1 replication): fixed test set = items 0..T-1; calibration bootstrapped (with replacement) from items T..M-1."""
    te = np.tile(np.arange(T), (S, 1)); cal = np.array([T + np.random.default_rng(10_000 + s).integers(0, M - T, n) for s in range(S)])
    return cal, te

def n_grid(M, T=TEST_SIZE):
    g = [n for n in [50, 100, 250, 500, 1000, 2000] if n <= M - T]
    if M - T not in g and M - T > g[-1]: g.append(M - T)
    return g

# ------------------------------------------------------------------ variance components
def vc_mom(Y):
    """Two-way crossed random effects without replication, Y: W x S. Returns sigma^2 for wording, split, residual(interaction+error)."""
    W, S = Y.shape; g = Y.mean(); rw = Y.mean(1); cs = Y.mean(0)
    msw = S * ((rw - g) ** 2).sum() / (W - 1); mss = W * ((cs - g) ** 2).sum() / (S - 1)
    res = Y - rw[:, None] - cs[None, :] + g; mse = (res ** 2).sum() / ((W - 1) * (S - 1))
    vw = (msw - mse) / S; vs = (mss - mse) / W
    return {"var_w": max(vw, 0.0), "var_s": max(vs, 0.0), "var_e": mse, "neg_w": vw < 0, "neg_s": vs < 0}

def vc_summary(Y, n_boot=1000, seed=0):
    est = vc_mom(Y); W, S = Y.shape; rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        iw = rng.integers(0, W, W); js = rng.integers(0, S, S); B.append(vc_mom(Y[np.ix_(iw, js)]))
    def stats(key_fn):
        v = np.array([key_fn(b) for b in B]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    tot = lambda e: e["var_w"] + e["var_s"] + e["var_e"]
    out = {"sd_w": math.sqrt(est["var_w"]), "sd_s": math.sqrt(est["var_s"]), "sd_e": math.sqrt(est["var_e"]),
           "share_w": est["var_w"] / tot(est) if tot(est) > 0 else 0.0, "share_s": est["var_s"] / tot(est) if tot(est) > 0 else 0.0,
           "share_e": est["var_e"] / tot(est) if tot(est) > 0 else 0.0,
           "ratio_w_s": (math.sqrt(est["var_w"]) / math.sqrt(est["var_s"])) if est["var_s"] > 0 else float("inf"),
           "neg_w": bool(est["neg_w"]), "neg_s": bool(est["neg_s"]), "W": W, "S": S, "mean": float(Y.mean())}
    out["ci_sd_w"] = stats(lambda b: math.sqrt(b["var_w"])); out["ci_sd_s"] = stats(lambda b: math.sqrt(b["var_s"]))
    out["ci_share_w"] = stats(lambda b: b["var_w"] / tot(b) if tot(b) > 0 else 0.0); out["ci_share_s"] = stats(lambda b: b["var_s"] / tot(b) if tot(b) > 0 else 0.0)
    out["ci_ratio_w_s"] = stats(lambda b: math.sqrt(b["var_w"]) / math.sqrt(b["var_s"]) if b["var_s"] > 0 else np.inf)
    out["boot_sd_w"] = [math.sqrt(b["var_w"]) for b in B[:200]]   # kept for slope bootstrap
    out["boot_sd_s"] = [math.sqrt(b["var_s"]) for b in B[:200]]
    return out

def raw_files(kind=None, ds=None):
    fs = sorted(glob.glob(f"{RAW}/*.npz"))
    return [f for f in fs if (kind is None or f"__{kind}__" in f) and (ds is None or os.path.basename(f).startswith(ds + "__"))]
