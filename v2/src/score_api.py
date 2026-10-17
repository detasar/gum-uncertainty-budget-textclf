"""API LLM scorer (exploratory T5 instruments; deviation from preregistration, see report.md).
Two instruments, both scored per WORDING SET (the class probabilities depend on the whole option list, so scores are stored per set):
  luna : gpt-6-luna via POST /v1/decisions; one 'choice' question per wording set, choices = the K label strings of that set;
         returns per-choice probabilities (quantised to 0.01 by the API; see report).
  nano : gpt-4.1-nano via /v1/chat/completions, logprobs=true, top_logprobs=20, max_tokens=1, temperature=0; options listed as letters
         A, B, C, ...; class score = log p(first token == letter) (logsumexp over 'X' and ' X' variants); a letter absent from the top-20 gets
         the smallest returned top-20 logprob (an upper bound on its true logprob) and is counted as missing.
Conditions (cond): main (all requested sets, canonical order = class order), ordK (option order permutation K>=1, valid+official sets),
rtK (test-retest rerun of items 0..199, official+valid sets, canonical order).
Every request is appended to a JSONL cache raw/api_cache/<model>/<ds>__<cond>.jsonl (resumable; finished (item,set) pairs are skipped).
usage: score_api.py <luna|nano> <ds> <cond> [--items N] [--sets main|valid|core] [--threads 16] [--qpr 1] [--start 0]"""
import argparse, json, os, sys, time, threading, random, math, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); D = os.path.join(ROOT, "data")
MODELS = {"luna": "gpt-6-luna", "nano": "gpt-4.1-nano"}
PRICE = {"luna": (0.10, 0.0), "nano": (0.10, 0.40)}          # USD per 1M input / output tokens (cached-token discounts ignored: conservative)
CAP = float(os.environ.get("API_CAP_USD", "23.0"))
SPEND_DIR = os.path.join(ROOT, "logs", "api", "spend"); CACHE = os.path.join(ROOT, "raw", "api_cache")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PREFIX = "Read the following {genre} and state its {facet}.\n\nText: {text}"

def load(ds):
    return json.load(open(f"{D}/{ds}.json")), json.load(open(f"{D}/wordsets_{ds}.json"))["sets"]

def choose_sets(sets, mode):
    if mode == "main":  return sets                                               # every set (agnews/bbc2025)
    if mode == "core":  return [s for s in sets if s["pop"] in ("official", "valid", "surface", "identity", "nonsense", "shuffled")]
    if mode == "valid": return [s for s in sets if s["pop"] in ("official", "valid")]
    raise ValueError(mode)

def perm_for(ds, set_name, k, K):
    if k == 0: return list(range(K))
    rng = np.random.default_rng([k, sum(map(ord, ds)), sum((i + 1) * ord(c) for i, c in enumerate(set_name))])
    return [int(x) for x in rng.permutation(K)]

def cost(model, usage):
    pi, po = PRICE[model]
    if model == "luna": return usage.get("input_tokens", 0) * pi / 1e6 + usage.get("output_tokens", 0) * po / 1e6
    return usage.get("prompt_tokens", 0) * pi / 1e6 + usage.get("completion_tokens", 0) * po / 1e6

def cache_spend(path, model):
    s = 0.0
    if os.path.exists(path):
        for line in open(path):
            try: s += cost(model, json.loads(line)["usage"])
            except Exception: pass
    return s

def global_spend():
    t = 0.0
    for f in glob.glob(f"{SPEND_DIR}/*.json"):
        try: t += json.load(open(f))["usd"]
        except Exception: pass
    return t

_tls = threading.local()
def session():
    if not hasattr(_tls, "s"): _tls.s = requests.Session()
    return _tls.s

def post(url, body, max_tries=8):
    H = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"}
    for t in range(max_tries):
        try:
            r = session().post(url, headers=H, json=body, timeout=90)
            if r.status_code == 200:
                try: return r.json()
                except ValueError: time.sleep(1 + random.random()); continue      # truncated/malformed body: retry
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** t + random.random())); continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 2 ** t + random.random()))
    raise RuntimeError("max retries")

def luna_request(d, text, qsets):
    """qsets: list of (set_name, labels_canonical, perm). Returns {set: probs in canonical class order}, usage, raw_probs (display order)."""
    qs = []
    for j, (nm, lab, pm) in enumerate(qsets):
        qs.append({"type": "choice", "name": f"q{j}", "instructions": f"What is the {d['facet']} of this {d['genre']}? Choose the best option.",
                   "choices": [{"value": lab[c]} for c in pm]})
    j = post("https://api.openai.com/v1/decisions", {"model": MODELS["luna"], "input": PREFIX.format(genre=d["genre"], facet=d["facet"], text=text), "questions": qs})
    ans = {a["name"]: a for a in j["answers"]}; out = {}
    for jj, (nm, lab, pm) in enumerate(qsets):
        a_ = ans.get(f"q{jj}", {})
        if "probabilities" not in a_: out[nm] = None; continue          # refusal (type 'refusal') or missing answer: recorded as null
        pv = {p["value"]: p["probability"] for p in a_["probabilities"]}
        out[nm] = [pv.get(lab[c], float("nan")) for c in range(len(lab))]
    return out, j["usage"], j.get("model")

def nano_request(d, text, nm, lab, pm):
    K = len(lab); opts = "\n".join(f"{LETTERS[i]}. {lab[c]}" for i, c in enumerate(pm))
    content = (PREFIX.format(genre=d["genre"], facet=d["facet"], text=text) + f"\n\nOptions:\n{opts}\n\nAnswer with the letter of the option only.")
    j = post("https://api.openai.com/v1/chat/completions", {"model": MODELS["nano"], "messages": [{"role": "user", "content": content}],
             "logprobs": True, "top_logprobs": 20, "max_tokens": 1, "temperature": 0, "seed": 0})
    top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]; floor = min(x["logprob"] for x in top)
    lp_disp = []; miss = []
    for i in range(K):
        v = [x["logprob"] for x in top if x["token"].strip() == LETTERS[i]]
        if v: lp_disp.append(float(np.logaddexp.reduce(v))); miss.append(0)
        else: lp_disp.append(floor); miss.append(1)
    lp = [None] * K; ms = [None] * K
    for i, c in enumerate(pm): lp[c] = lp_disp[i]; ms[c] = miss[i]
    return {nm: lp}, {nm: ms}, j["usage"], j.get("model"), j.get("system_fingerprint"), j["choices"][0]["logprobs"]["content"][0]["token"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model", choices=["luna", "nano"]); ap.add_argument("ds"); ap.add_argument("cond")
    ap.add_argument("--items", type=int, default=1200); ap.add_argument("--start", type=int, default=0); ap.add_argument("--sets", default="core")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--qpr", type=int, default=1, help="questions per Decisions request (luna)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); d, sets = load(a.ds); os.makedirs(SPEND_DIR, exist_ok=True); os.makedirs(f"{CACHE}/{a.model}", exist_ok=True)
    chosen = choose_sets(sets, a.sets); k = int(a.cond[3:]) if a.cond.startswith("ord") else 0
    cpath = f"{CACHE}/{a.model}/{a.ds}__{a.cond}{a.tag}.jsonl"; key = f"{a.model}__{a.ds}__{a.cond}{a.tag}"
    import fcntl
    lockf = open(cpath + ".lock", "w")
    try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)      # another process is already filling this cache -> skip (it will finish it)
    except BlockingIOError: print(f"LOCKED {key}: another process is scoring it, skipping", flush=True); return
    done = set()
    if os.path.exists(cpath):
        for line in open(cpath):
            try: r = json.loads(line)
            except Exception: continue
            for nm in r["sets"]: done.add((r["i"], nm))
    spent = [cache_spend(cpath, a.model)]; lock = threading.Lock(); f = open(cpath, "a"); stop = [False]
    def flush_spend():
        json.dump({"usd": spent[0], "t": time.time()}, open(f"{SPEND_DIR}/{key}.json", "w"))
    flush_spend()
    items = range(a.start, a.start + a.items); texts = d["pool_text"]
    jobs = []
    for i in items:
        todo = [s for s in chosen if (i, s["name"]) not in done]
        if not todo: continue
        if a.model == "luna":
            for c0 in range(0, len(todo), a.qpr): jobs.append((i, todo[c0:c0 + a.qpr]))
        else:
            for s in todo: jobs.append((i, [s]))
    print(f"{key}: {len(jobs)} requests to do ({len(done)} (item,set) pairs cached), cached spend ${spent[0]:.4f}, global ${global_spend():.4f}", flush=True)
    t0 = time.time(); cnt = [0]; last = [time.time()]
    def work(job):
        if stop[0]: return
        i, ss = job; text = texts[i]
        qsets = [(s["name"], s["labels"], perm_for(a.ds, s["name"], k, len(s["labels"]))) for s in ss]
        if a.model == "luna":
            probs, usage, mid = luna_request(d, text, qsets); rec = {"i": i, "sets": [q[0] for q in qsets], "perm": [q[2] for q in qsets], "p": probs, "usage": usage, "model": mid}
        else:
            nm, lab, pm = qsets[0]; lp, ms, usage, mid, fp, tok = nano_request(d, text, nm, lab, pm)
            rec = {"i": i, "sets": [nm], "perm": [pm], "lp": lp, "miss": ms, "usage": usage, "model": mid, "fp": fp, "tok": tok}
        rec["ts"] = round(time.time(), 2)
        with lock:
            f.write(json.dumps(rec) + "\n"); spent[0] += cost(a.model, usage); cnt[0] += 1
            if time.time() - last[0] > 30:
                f.flush(); flush_spend(); last[0] = time.time(); g = global_spend()
                print(f"{key} {cnt[0]}/{len(jobs)} {time.time()-t0:.0f}s this=${spent[0]:.3f} global=${g:.3f}", flush=True)
                if g > CAP: stop[0] = True; print("BUDGET CAP REACHED, stopping", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for fut in [ex.submit(work, j) for j in jobs]:
            try: fut.result()
            except Exception as e: print("ERR", repr(e)[:300], flush=True)
    f.close(); flush_spend()
    print(("STOPPED " if stop[0] else "DONE ") + key, f"{cnt[0]} requests {time.time()-t0:.0f}s spend ${spent[0]:.4f} global ${global_spend():.4f}", flush=True)

if __name__ == "__main__": main()
