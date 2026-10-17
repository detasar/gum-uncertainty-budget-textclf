"""Reduced-design comparator for the decision-model group (exploratory): copy of a per-set score file restricted to the first N items
and the official + VALID sets, written to raw/sub200/<same name>.npz. Used to compare Clef-flash (200 items, item cap) with gpt-6-luna on the
same items, sets and design (UQ_TEST_SIZE=100, n_cal 50/100).
usage: make_sub200.py <npz> [--n 200]"""
import sys, os, json, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True); names = list(z["set_names"]); labs = json.loads(str(z["set_labels"]))
keep = [j for j, nm in enumerate(names) if nm == "official" or nm.startswith("valid_")]
d = {k: z[k] for k in z.files}; meta = json.loads(str(z["meta"])); meta.update(sub_items=a.n, sub_sets="official+valid", source=os.path.basename(a.npz))
d["P"] = z["P"][: a.n][:, keep]; d["set_names"] = np.array([names[j] for j in keep], dtype=object); d["set_labels"] = json.dumps([labs[j] for j in keep])
for k in ("lp", "miss", "lg"):
    if k in d: d[k] = z[k][: a.n][:, keep]
d["meta"] = json.dumps(meta)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); os.makedirs(f"{root}/raw/sub200", exist_ok=True)
out = f"{root}/raw/sub200/{os.path.basename(a.npz)}"; np.savez_compressed(out, **d); print("SUB", out, d["P"].shape)
