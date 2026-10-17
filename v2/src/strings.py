"""Unique label strings to score for a dataset.
mode 'all'  : every candidate variant + both nonsense controls
mode 'v1'   : official + the four v1 hand-picked wordings (repeatability runs)
mode 'valid': VALID variants (incl. official) + nonsense N1 only (used for T1/T3 on sst2/dbpedia/massive, for CPU cost)"""
import json, os
D = os.path.join(os.path.dirname(__file__), "..", "data")
def label_strings(ds, mode="all"):
    L = json.load(open(f"{D}/labels_{ds}.json"))
    s = []
    for v in L["variants"]:
        for r in v:
            keep = mode == "all" or (mode == "valid" and r["valid"]) or (mode == "v1" and any(g == "G0_official" or g.startswith("v1_") for g in r["generators"]))
            if keep and r["text"] not in s: s.append(r["text"])
    for k, lst in L["controls"].items():
        if mode == "v1" or (mode != "all" and k != "N1_nonsense"): continue
        for x in lst:
            if x not in s: s.append(x)
    return s, L
def load_ds(ds): return json.load(open(f"{D}/{ds}.json"))
