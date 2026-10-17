"""Shared data setup for the uncertainty-budget experiment (AG News)."""
import json, os, random, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# Label-wording paraphrases (verbalizers) for zero-shot models. Index = AG News label id.
WORDINGS = {
    "w0_official": ["World", "Sports", "Business", "Science/Technology"],
    "w1_news":     ["world news", "sports news", "business news", "science and technology news"],
    "w2_topic":    ["international politics and conflicts", "athletes, teams and games",
                    "companies, markets and the economy", "computers, the internet and scientific research"],
    "w3_short":    ["politics", "sport", "money", "tech"],
}

def load_agnews():
    from huggingface_hub import hf_hub_download
    out = {}
    for s in ["train", "test"]:
        p = hf_hub_download("fancyzhx/ag_news", f"data/{s}-00000-of-00001.parquet", repo_type="dataset")
        out[s] = pd.read_parquet(p)
    return out

def balanced_sample(df, n, rng):
    per = n // 4
    idx = np.concatenate([rng.choice(np.where(df.label.values == c)[0], per, replace=False) for c in range(4)])
    rng.shuffle(idx)
    return idx

def perturb(text, level, seed):
    """Synthetic covariate shift: drop words w.p. level/2, then typo (swap/delete a char) w.p. level."""
    r = random.Random(seed)
    out = []
    for w in text.split():
        if r.random() < level / 2:
            continue
        if len(w) > 3 and r.random() < level:
            i = r.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:] if r.random() < 0.5 else w[:i] + w[i + 1:]
        out.append(w)
    return " ".join(out) if out else text

def clean(t):
    return re.sub(r"\s+", " ", t.replace("\\", " ")).strip()
