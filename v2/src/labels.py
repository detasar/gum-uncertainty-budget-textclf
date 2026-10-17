"""v2 label-paraphrase generation + validity check (documented procedure; runs BEFORE any classifier scoring).

Per class, candidate verbalizers come from fixed generators:
  G0 official  : dataset label string (camelCase split, '_' -> ' ')
  G1 surface   : case change only (lower-case, or Title Case if already lower) -- near-identity control
  G2 wn_syn    : first WordNet noun lemma of the head word that differs from it (automatic)
  G3 wn_hyper  : first lemma of the first hypernym of the head word's first noun synset (automatic)
  G4 keywords  : top-3 chi-square unigrams of the class in the training split (automatic, data-derived)
  G5 definition: short definition written from the dataset documentation (fixed in this file, before any scoring)
  G6,G7 llm1/2 : first two alternative names from Qwen2.5-3B-Instruct, greedy decoding, prompt below (automatic)
  (AG News/BBC only) v1_* : the four hand-picked wordings of the published v1 experiment
Controls: N1,N2 nonsense pseudo-words (seeded), shuffled sets (derangements of the official names).

Validity judges (NOT used as classifiers): sentence-transformers/all-mpnet-base-v2 (cosine) and FacebookAI/roberta-large-mnli
(bidirectional entailment between 'This text is about {a}.' and 'This text is about {b}.').
Preregistered rule (fixed before any classifier scoring):
  VALID  iff (i) discriminable: judge-cosine to its own official name > judge-cosine to every other class's official name, AND
             (ii) not flagged in the researcher audit below (written after seeing the candidate strings, before any classifier output).
  STRICT iff VALID and judge-cosine to the official name >= 0.50.
  ALL    = every generated candidate (controls excluded).
  G0 (official) is valid and strict by definition. Cosine and bidirectional-entailment values are kept as continuous covariates.
Wording distance of a candidate = 1 - judge cosine to the official name.
Outputs: data/labels_<ds>.json
"""
import json, os, re, sys, itertools, random
import numpy as np, torch
from nltk.corpus import wordnet as wn, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
torch.set_num_threads(8)
D = os.path.join(os.path.dirname(__file__), "..", "data")

DEFS = {
 "agnews": ["news about international affairs, politics and conflicts", "news about athletes, teams, matches and competitions",
            "news about companies, markets and the economy", "news about science, technology, computing and the internet"],
 "sst2": ["the reviewer expresses a negative opinion of the film", "the reviewer expresses a positive opinion of the film"],
 "dbpedia": ["a business organization or firm", "a school, college or university", "a musician, painter, actor or other creative artist",
             "a person who competes in sports", "a politician or person holding public office", "a vehicle, ship, aircraft or train",
             "a building or architectural structure", "a natural feature such as a mountain, river or lake", "a village or small rural settlement",
             "an animal species", "a plant species", "a music album", "a movie", "a book, novel, journal or other written work"],
 "massive": ["setting or managing alarms", "controlling volume or sound settings", "calendar events, meetings and reminders", "recipes and cooking",
             "the current date or time", "reading or sending email", "chit-chat, jokes and general commands to the assistant",
             "controlling smart home devices such as lights", "creating or managing lists", "music preferences and information about songs",
             "news headlines and current events", "playing music, podcasts, radio, audiobooks or games", "general knowledge questions and facts",
             "recommendations for places, events or movies", "social media posts and messages", "ordering takeaway food",
             "trains, taxis, tickets and traffic", "weather forecasts"],
}
DEFS["bbc2025"] = DEFS["agnews"]
V1 = {"v1_w0": ["World", "Sports", "Business", "Science/Technology"], "v1_w1": ["world news", "sports news", "business news", "science and technology news"],
      "v1_w2": ["international politics and conflicts", "athletes, teams and games", "companies, markets and the economy", "computers, the internet and scientific research"],
      "v1_w3": ["politics", "sport", "money", "tech"]}
LLM_PROMPT = ("A dataset of {genre}s is labelled by {facet} with these categories: {allc}. "
              "Give 5 alternative names for the category \"{name}\" that a person could use instead. Each alternative must be 1 to 4 words long "
              "and mean the same category. Output one per line, with no numbering and no explanations.")

# Researcher audit (D. E. Tasar's pipeline, applied by the v2 agent): candidates judged NOT to denote the same category for typical items of the
# dataset (wrong sense, much broader/narrower, or a different concept). Criterion: "would a competent annotator assign the same items to this name
# as to the official name?" Written before any classifier output existed.
AUDIT_REJECT = {
 "agnews": ["universe", "natural object", "concern", "technical school", "school", "Recreation", "diversion", "politics", "international politics and conflicts"],
 "sst2": ["denial", "Sad", "positive degree", "adverb"],
 "dbpedia": ["companionship", "institution", "establishment", "organization", "transportation system", "facility", "community", "organism", "works",
             "building complex", "medium", "product", "activity", "point", "topographic point", "bearer", "holding device", "Person", "edifice", "Picture"],
 "massive": ["dismay", "fear", "auditory communication", "arrangement", "full general", "general officer", "Common", "Everyday", "sociable", "party",
             "Voice", "change of state", "electronic communication", "database", "Notes", "euphony", "Chill", "Sound", "intelligence", "drama",
             "dramatic composition", "Listen", "Musician", "testimonial", "instrumentality", "Conditions"],
}
AUDIT_REJECT["bbc2025"] = AUDIT_REJECT["agnews"]

def official(s): return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("_", " ")
def head(s): return re.split(r"[ /]", official(s))[-1].lower()
def surface(s):   # case change only: cannot change what the label means (near-identity control)
    w = official(s); return w.lower() if w[:1].isupper() else w.title()
def wn_syn(s):
    h = head(s)
    for ss in wn.synsets(h, pos=wn.NOUN):
        for l in ss.lemma_names():
            l = l.replace("_", " ")
            if l.lower() != h and l.lower() not in official(s).lower(): return l
    return None
def wn_hyper(s):
    ss = wn.synsets(head(s), pos=wn.NOUN)
    if not ss or not ss[0].hypernyms(): return None
    return ss[0].hypernyms()[0].lemma_names()[0].replace("_", " ")
def keywords(texts, y, K):
    sw = stopwords.words("english")
    cv = CountVectorizer(stop_words=sw, min_df=3, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b", binary=True); X = cv.fit_transform(texts); voc = np.array(cv.get_feature_names_out())
    out = []
    for c in range(K):
        sc, _ = chi2(X, (np.asarray(y) == c).astype(int))
        pos = np.asarray(X[np.asarray(y) == c].mean(0)).ravel() > np.asarray(X[np.asarray(y) != c].mean(0)).ravel()
        sc = np.where(pos, np.nan_to_num(sc), -1); out.append(", ".join(voc[np.argsort(-sc)[:3]]))
    return out

def llm_alts(ds, names, genre, facet):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    m = "Qwen/Qwen2.5-3B-Instruct"; tok = AutoTokenizer.from_pretrained(m); mod = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).eval()
    allc = ", ".join(official(n) for n in names); res, raw = [], []
    for n in names:
        msg = [{"role": "user", "content": LLM_PROMPT.format(genre=genre, facet=facet, allc=allc, name=official(n))}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.inference_mode(): out = mod.generate(**ids, max_new_tokens=60, do_sample=False)
        txt = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True); raw.append(txt)
        cands = []
        for line in txt.splitlines():
            l = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"').strip()
            if not l or len(l.split()) > 5 or l.lower() == official(n).lower() or l.lower() in [c.lower() for c in cands]: continue
            cands.append(l)
        res.append((cands + [None, None])[:2])
    return res, raw

def nonsense(K, seed):
    r = random.Random(seed); C, V = "bdfgklmnprstvz", "aeiou"
    return ["".join(r.choice(C) + r.choice(V) for _ in range(3)) for _ in range(K)]

def judges(official_names, cands):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    tok = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli"); nli = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli").eval()
    ent = [k for k, v in nli.config.id2label.items() if v.upper().startswith("ENTAIL")][0]
    O = st.encode([official(n) for n in official_names], normalize_embeddings=True)
    def P(a, b):
        enc = tok([f"This text is about {x}." for x in a], [f"This text is about {x}." for x in b], return_tensors="pt", padding=True)
        with torch.inference_mode(): return nli(**enc).logits.softmax(-1)[:, ent].numpy()
    out = []
    for c, lst in enumerate(cands):
        if not lst: out.append([]); continue
        E = st.encode([x for x in lst], normalize_embeddings=True); cos_all = E @ O.T
        on = [official(official_names[c])] * len(lst)
        pab = P(on, lst); pba = P(lst, on)
        rows = []
        for j, x in enumerate(lst):
            cos = float(cos_all[j, c]); others = np.delete(cos_all[j], c)
            disc = bool(cos > others.max()) if len(others) else True
            ment = float(min(pab[j], pba[j]))
            rows.append({"text": x, "cos_official": cos, "max_cos_other_class": float(others.max()) if len(others) else None,
                         "discriminable": disc, "entail_min": ment, "distance": 1 - cos})
        out.append(rows)
    return out

def build(ds):
    d = json.load(open(f"{D}/{ds}.json")); names = d["classes_official"]; K = len(names)
    tr_text, tr_y = (d["train_text"], d["train_y"]) if d["train_y"] else (json.load(open(f"{D}/agnews.json"))["train_text"], json.load(open(f"{D}/agnews.json"))["train_y"])
    kw = keywords(tr_text, tr_y, K)
    prev = f"{D}/labels_{ds}.json"
    if os.path.exists(prev):   # greedy decoding is deterministic: reuse the stored LLM alternatives instead of regenerating
        P = json.load(open(prev)); llm, raw = list(zip(P["generators"]["G6_llm1"], P["generators"]["G7_llm2"])), P["llm_raw"]
    else:
        llm, raw = llm_alts(ds, names, d["genre"], d["facet"])
    gens = {"G0_official": [official(n) for n in names], "G1_surface": [surface(n) for n in names], "G2_wn_syn": [wn_syn(n) for n in names],
            "G3_wn_hyper": [wn_hyper(n) for n in names], "G4_keywords": kw, "G5_definition": DEFS[ds],
            "G6_llm1": [a[0] for a in llm], "G7_llm2": [a[1] for a in llm]}
    if ds in ("agnews", "bbc2025"): gens.update(V1)
    # per class candidate list (unique strings, keep generator provenance)
    per_class = []
    for c in range(K):
        seen = {}
        for g, lst in gens.items():
            v = lst[c]
            if v is None: continue
            seen.setdefault(v, []).append(g)
        per_class.append(seen)
    J = judges(names, [list(pc.keys()) for pc in per_class])
    variants = []
    for c in range(K):
        rows = []
        for r in J[c]:
            r["generators"] = per_class[c][r["text"]]
            r["audit_reject"] = r["text"] in AUDIT_REJECT.get(ds, [])
            r["valid"] = bool(r["discriminable"] and not r["audit_reject"]); r["strict"] = bool(r["valid"] and r["cos_official"] >= 0.5)
            if "G0_official" in r["generators"]: r["valid"] = r["strict"] = True
            rows.append(r)
        variants.append(rows)
    controls = {"N1_nonsense": nonsense(K, 11), "N2_nonsense": nonsense(K, 22)}
    rng = np.random.default_rng(7)
    ders = [p for p in itertools.permutations(range(K)) if all(p[i] != i for i in range(K))] if K <= 8 else None
    shuffles = []
    for s in range(3 if K > 2 else 1):
        if ders: p = list(ders[rng.integers(len(ders))])
        else:
            while True:
                p = rng.permutation(K).tolist()
                if all(p[i] != i for i in range(K)): break
        shuffles.append(p)
    out = {"dataset": ds, "classes_official": names, "generators": gens, "variants": variants, "controls": controls, "shuffle_perms": shuffles,
           "llm_raw": raw, "llm_prompt": LLM_PROMPT, "rule": "valid iff discriminable AND not audit-rejected; strict iff valid AND cos>=0.5; G0 valid+strict by definition", "audit_reject": AUDIT_REJECT.get(ds, [])}
    json.dump(out, open(f"{D}/labels_{ds}.json", "w"), indent=1)
    nv = [sum(r["valid"] for r in v) for v in variants]; ns = [sum(r["strict"] for r in v) for v in variants]; nt = [len(v) for v in variants]
    print(ds, "valid/strict/total per class", list(zip(nv, ns, nt)), flush=True)
    for c in range(K): print("  ", names[c], [(r["text"], round(r["cos_official"], 2), round(r["entail_min"], 2), r["valid"]) for r in variants[c]], flush=True)

for ds in sys.argv[1:] or ["agnews", "bbc2025", "sst2", "dbpedia", "massive"]:
    build(ds)
