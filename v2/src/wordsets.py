"""Fix the wording sets used in the analysis (seed 2026), before any classifier output. Output data/wordsets_<ds>.json.
Each set = list of K label strings (index = class id) + a population tag."""
import json, os, sys, itertools
import numpy as np
D = os.path.join(os.path.dirname(__file__), "..", "data")
W = 30
def build(ds):
    L = json.load(open(f"{D}/labels_{ds}.json")); K = len(L["classes_official"]); V = L["variants"]
    off = [next(r["text"] for r in v if "G0_official" in r["generators"]) for v in V]
    rng = np.random.default_rng(2026); sets = [{"name": "official", "pop": "official", "labels": off}]
    def draw(pop, key, n):
        pools = [[r["text"] for r in v if key(r)] for v in V]
        total = int(np.prod([len(p) for p in pools], dtype=float)) if all(pools) else 0
        if total <= n:   # enumerate the whole population
            combos = list(itertools.product(*pools))
        else:
            seen, combos = set(), []
            while len(combos) < n:
                c = tuple(p[rng.integers(len(p))] for p in pools)
                if c not in seen: seen.add(c); combos.append(c)
        for i, c in enumerate(combos[:n]): sets.append({"name": f"{pop}_{i:02d}", "pop": pop, "labels": list(c)})
        return total
    sizes = {"valid": draw("valid", lambda r: r["valid"], W), "strict": draw("strict", lambda r: r["strict"], W), "all": draw("all", lambda r: True, W)}
    # homogeneous generator sets (dose-response / interpretability); classes lacking that generator keep the official name
    gens = [g for g in L["generators"]]
    for g in gens:
        if g == "G0_official": continue
        lab = [L["generators"][g][c] if L["generators"][g][c] else off[c] for c in range(K)]
        sets.append({"name": f"homog_{g}", "pop": "homog", "labels": lab})
    # surface-only control: change the case of a random non-empty subset of classes (G1)
    g1 = L["generators"]["G1_surface"]; seen = set()
    subsets = [s for r in range(1, K + 1) for s in itertools.combinations(range(K), r)] if K <= 4 else None
    for i in range(10):
        if subsets is not None:
            if i >= len(subsets): break
            sub = subsets[i] if len(subsets) <= 10 else subsets[rng.integers(len(subsets))]
        else:
            m = rng.integers(1, K + 1); sub = tuple(sorted(rng.choice(K, m, replace=False)))
        if sub in seen: continue
        seen.add(sub); sets.append({"name": f"surface_{i:02d}", "pop": "surface", "labels": [g1[c] if c in sub else off[c] for c in range(K)]})
    sets.append({"name": "identity", "pop": "identity", "labels": list(off)})
    for k, lst in L["controls"].items(): sets.append({"name": k, "pop": "nonsense", "labels": lst})
    for j, p in enumerate(L["shuffle_perms"]): sets.append({"name": f"shuffled_{j}", "pop": "shuffled", "labels": [off[p[c]] for c in range(K)]})
    json.dump({"dataset": ds, "W": W, "population_sizes": sizes, "sets": sets}, open(f"{D}/wordsets_{ds}.json", "w"), indent=1)
    print(ds, "population sizes", sizes, "n sets", len(sets))
for ds in sys.argv[1:] or ["agnews", "bbc2025", "sst2", "dbpedia", "massive"]: build(ds)
