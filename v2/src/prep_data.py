"""v2 data preparation. Fixes every evaluation pool and training set once (seed 0) and writes data/<ds>.json.
Pools are drawn from the official evaluation split(s); supervised training sets from the official train split.
No model outputs are used here."""
import json, re, os, sys, hashlib
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
OUT = os.path.join(os.path.dirname(__file__), "..", "data"); os.makedirs(OUT, exist_ok=True)
SEED = 0

def clean(t): return re.sub(r"\s+", " ", str(t).replace("\\", " ")).strip()

def pq(repo, path): return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))

def balanced(y, per, rng, classes):
    idx = []
    for c in classes:
        w = np.where(y == c)[0]
        if len(w) < per: raise ValueError(f"class {c} has {len(w)} < {per}")
        idx.append(rng.choice(w, per, replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx); return idx

def save(name, meta, pool_text, pool_y, train_text, train_y, extra=None):
    d = {"name": name, **meta, "pool_text": pool_text, "pool_y": [int(v) for v in pool_y],
         "train_text": train_text, "train_y": [int(v) for v in train_y]}
    if extra: d.update(extra)
    s = json.dumps(d); open(f"{OUT}/{name}.json", "w").write(s)
    print(name, "pool", len(pool_y), np.bincount(pool_y).tolist(), "train", len(train_y), "sha256", hashlib.sha256(s.encode()).hexdigest()[:16], flush=True)

def agnews():
    rng = np.random.default_rng(SEED)
    tr = pq("fancyzhx/ag_news", "data/train-00000-of-00001.parquet"); te = pq("fancyzhx/ag_news", "data/test-00000-of-00001.parquet")
    ip = balanced(te.label.values, 650, rng, range(4)); it = balanced(tr.label.values, 500, rng, range(4))
    save("agnews", {"source": "fancyzhx/ag_news (test split -> pool, train split -> train)", "classes_official": ["World", "Sports", "Business", "Sci/Tech"],
                    "genre": "news article", "facet": "topic"},
         [clean(t) for t in te.text.values[ip]], te.label.values[ip], [clean(t) for t in tr.text.values[it]], tr.label.values[it])

def sst2():
    rng = np.random.default_rng(SEED)
    def load(split):
        p = hf_hub_download("SetFit/sst2", f"{split}.jsonl", repo_type="dataset"); return pd.read_json(p, lines=True)
    ev = pd.concat([load("test"), load("dev")], ignore_index=True); tr = load("train")
    ip = balanced(ev.label.values, 1300, rng, range(2)); it = balanced(tr.label.values, 1000, rng, range(2))
    save("sst2", {"source": "SetFit/sst2 (test+dev -> pool, train -> train)", "classes_official": ["negative", "positive"], "genre": "movie review sentence", "facet": "sentiment"},
         [clean(t) for t in ev.text.values[ip]], ev.label.values[ip], [clean(t) for t in tr.text.values[it]], tr.label.values[it])

def dbpedia():
    rng = np.random.default_rng(SEED)
    files = list_repo_files("fancyzhx/dbpedia_14", repo_type="dataset")
    ftr = [f for f in files if "train" in f and f.endswith(".parquet")][0]; fte = [f for f in files if "test" in f and f.endswith(".parquet")][0]
    tr = pq("fancyzhx/dbpedia_14", ftr); te = pq("fancyzhx/dbpedia_14", fte)
    names = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
             "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
    ip = balanced(te.label.values, 100, rng, range(14)); it = balanced(tr.label.values, 143, rng, range(14))
    txt = lambda df, i: [clean(f"{a}. {b}") for a, b in zip(df.title.values[i], df.content.values[i])]
    save("dbpedia", {"source": f"fancyzhx/dbpedia_14 ({fte} -> pool, {ftr} -> train)", "classes_official": names, "genre": "encyclopedia article", "facet": "category"},
         txt(te, ip), te.label.values[ip], txt(tr, it), tr.label.values[it])

def massive():
    rng = np.random.default_rng(SEED)
    get = lambda s: pd.read_json(hf_hub_download("mteb/amazon_massive_scenario", f"{s}/en.json.gz", repo_type="dataset"), lines=True)
    ev = pd.concat([get("test"), get("validation")], ignore_index=True); tr = get("train")
    print("massive cols", ev.columns.tolist(), ev.head(2).to_dict("records"))
    names = sorted(ev.label.unique().tolist()); m = {n: i for i, n in enumerate(names)}
    yev = ev.label.map(m).values; ytr = tr.label.map(m).values
    ip = rng.choice(len(ev), 2600, replace=False)   # natural class proportions (several scenarios are small)
    it = rng.choice(len(tr), 2000, replace=False)
    save("massive", {"source": "mteb/amazon_massive_scenario en (test+validation -> pool, natural proportions; train -> train)", "classes_official": names,
                     "genre": "request to a voice assistant", "facet": "scenario"},
         [clean(t) for t in ev.text.values[ip]], yev[ip], [clean(t) for t in tr.text.values[it]], ytr[it])

def bbc():
    rng = np.random.default_rng(SEED)
    months = ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    df = pd.concat([pq("RealTimeData/bbc_news_alltime", f"{m}/train-00000-of-00001.parquet").assign(month=m) for m in months], ignore_index=True)
    world = {"World", "Europe", "US & Canada", "Middle East", "Asia", "Africa", "Latin America", "Australia", "China", "India"}
    def lab(r):
        if "/sport/" in str(r.link): return 1
        s = str(r.section)
        if s in world: return 0
        if s == "Business": return 2
        if s in {"Technology", "Science & Environment", "Climate"}: return 3
        return -1
    df["y"] = df.apply(lab, axis=1); df = df[df.y >= 0]
    df["text"] = [clean(f"{re.sub(r'\s*[-|]\s*BBC (News|Sport).*$', '', str(a))}. {b}") for a, b in zip(df.title.values, df.description.fillna("").values)]
    df = df[df.description.fillna("").str.len() > 20].drop_duplicates("title").drop_duplicates("text").reset_index(drop=True)
    print("bbc available per class", np.bincount(df.y.values).tolist())
    caps = [650, 650, 550, 550]; ip = []
    for c in range(4):
        w = np.where(df.y.values == c)[0]; ip.append(rng.choice(w, min(caps[c], len(w)), replace=False))
    ip = np.concatenate(ip); rng.shuffle(ip)
    it = np.array([], dtype=int)  # no in-domain training set: the supervised probe for bbc2025 is trained on AG News (a real distribution shift)
    save("bbc2025", {"source": "RealTimeData/bbc_news_alltime, months 2024-10..2025-06; label from section/link (Climate counted as Sci/Tech); class sizes capped, not balanced, mapped to AG News classes", "classes_official": ["World", "Sports", "Business", "Sci/Tech"],
                     "genre": "news article", "facet": "topic", "months": months},
         df.text.values[ip].tolist(), df.y.values[ip], df.text.values[it].tolist(), df.y.values[it],
         extra={"pool_link": df.link.values[ip].tolist(), "pool_date": df.published_date.astype(str).values[ip].tolist(), "pool_section": df.section.astype(str).values[ip].tolist()})

for f in sys.argv[1:] or ["agnews", "sst2", "dbpedia", "massive", "bbc"]:
    globals()[f]()
