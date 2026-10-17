"""Exploratory (added in the report session, 2026-10-17 ~16:30 Istanbul): compare repeat run 0 with the main-grid run of the
same model on the same items and shared label strings (different process, same dtype/threads/batch setting; the LLM chunks
contain different label strings). Output results/repeat_rep0_vs_main.json. Not the preregistered RQ5 analysis."""
import numpy as np, json, os
R = os.path.join(os.path.dirname(__file__), "..")
def sm(X): X = X - X.max(1, keepdims=True); E = np.exp(X); return E / E.sum(1, keepdims=True)
V1 = [["World", "Sports", "Business", "Sci/Tech"], ["World", "Sports", "Business", "Science/Technology"],
      ["world news", "sports news", "business news", "science and technology news"],
      ["international politics and conflicts", "athletes, teams and games", "companies, markets and the economy", "computers, the internet and scientific research"],
      ["politics", "sport", "money", "tech"]]
out = {}
for inst, a, b in [("bienc bge-small-en-v1.5", "raw/repeat/bienc_rep0.npz", "raw/agnews__bienc__bge-small-en-v1.5.npz"),
                   ("llm Qwen2.5-0.5B-Instruct", "raw/repeat/llm_rep0.npz", "raw/agnews__llm__Qwen2.5-0.5B-Instruct.npz")]:
    A = np.load(f"{R}/{a}", allow_pickle=True); B = np.load(f"{R}/{b}", allow_pickle=True)
    SA, SB = list(A["strings"]), list(B["strings"]); sh = [s for s in SA if s in SB]
    ia, ib = [SA.index(s) for s in sh], [SB.index(s) for s in sh]
    if "cos" in A.files: XA, XB = A["cos"][:, ia], B["cos"][: A["cos"].shape[0], ib]
    else: XA = A["raw"][:, ia] - A["null"][None, ia]; XB = B["raw"][: A["raw"].shape[0], ib] - B["null"][None, ib]
    N = XA.shape[0]; d = {"items": int(N), "shared_strings": len(sh), "max_abs_diff_raw_score": float(np.abs(XA.astype(np.float64) - XB).max())}
    mp, fl = [], []
    for lab in V1:
        if not all(l in sh for l in lab): continue
        j = [sh.index(l) for l in lab]; t = 0.05 if "cos" in A.files else 1.0
        PA, PB = sm(XA[:, j].astype(np.float64) / t), sm(XB[:, j].astype(np.float64) / t)
        mp.append(float(np.abs(PA - PB).max())); fl.append(float((PA.argmax(1) != PB.argmax(1)).mean()))
    d.update({"wording_sets_compared": len(mp), "max_abs_dp_over_sets": max(mp) if mp else None, "max_argmax_flip_rate": max(fl) if fl else None})
    out[inst] = d; print(inst, d)
json.dump(out, open(f"{R}/results/repeat_rep0_vs_main.json", "w"), indent=1)
