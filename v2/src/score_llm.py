"""Small instruction LLM scored by label log-likelihood (no generation, deterministic).
For every pool text t and label string s:  raw[t,s] = sum_k log p(token_k of s+<eos> | chat prompt(t))  ;  null[s] = same with content-free text 'N/A'.
Class score used downstream = PMI = raw - null (content-free calibration, after Holtzman et al. 2021 / Zhao et al. 2021); raw is also saved.
The prompt prefix is encoded once per text and its KV cache is reused for all label continuations.
usage: score_llm.py <model_id> <ds> [--items N] [--threads 8] [--dtype bf16] [--check]"""
import argparse, json, os, time, sys, copy, platform
import numpy as np, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); from strings import label_strings, load_ds
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("ds"); ap.add_argument("--items", type=int, default=0)
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--out", default=None)
ap.add_argument("--strings", default="all"); ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); torch.set_num_threads(a.threads)
d = load_ds(a.ds); S, L = label_strings(a.ds, a.strings if a.strings in ("valid", "v1") else "all")
if a.strings == "official": S = [r["text"] for v in L["variants"] for r in v if "G0_official" in r["generators"]]
texts = d["pool_text"][: a.items or None]
tok = AutoTokenizer.from_pretrained(a.model); dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
mod = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).eval()
PROMPT = "Read the following {genre} and state its {facet}.\n\nText: {text}\n\nAnswer with the {facet} only."
eos = tok.eos_token
cont = [tok(s + eos, add_special_tokens=False)["input_ids"] for s in S]
pad = tok.pad_token_id if tok.pad_token_id is not None else 0

def prefix_ids(text):
    msg = [{"role": "user", "content": PROMPT.format(genre=d["genre"], facet=d["facet"], text=text)}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]

@torch.inference_mode()
def score_text(text):
    ids = prefix_ids(text); P = ids.shape[1]
    out = mod(input_ids=ids, use_cache=True); cache = out.past_key_values
    last = torch.log_softmax(out.logits[0, -1].float(), -1)        # distribution of the first label token
    res = np.zeros(len(S), np.float64)
    for c0 in range(0, len(S), a.chunk):
        idx = list(range(c0, min(len(S), c0 + a.chunk))); n = len(idx); Lmax = max(len(cont[j]) for j in idx)
        X = torch.full((n, Lmax), pad, dtype=torch.long); M = torch.zeros((n, Lmax), dtype=torch.long)
        for r, j in enumerate(idx): X[r, :len(cont[j])] = torch.tensor(cont[j]); M[r, :len(cont[j])] = 1
        cch = copy.deepcopy(cache); cch.batch_repeat_interleave(n)
        am = torch.cat([torch.ones((n, P), dtype=torch.long), M], 1)
        pos = torch.arange(P, P + Lmax).unsqueeze(0).expand(n, -1)
        lg = torch.log_softmax(mod(input_ids=X, attention_mask=am, past_key_values=cch, position_ids=pos, use_cache=True).logits.float(), -1)
        for r, j in enumerate(idx):
            t = cont[j]; s = last[t[0]].item()
            if len(t) > 1: s += lg[r, torch.arange(len(t) - 1), torch.tensor(t[1:])].sum().item()
            res[j] = s
    return res

@torch.inference_mode()
def score_full(text, j):   # slow reference implementation (no cache) for --check
    ids = prefix_ids(text); t = torch.tensor(cont[j]).unsqueeze(0); X = torch.cat([ids, t], 1)
    lg = torch.log_softmax(mod(input_ids=X).logits[0].float(), -1); P = ids.shape[1]
    return sum(lg[P - 1 + k, cont[j][k]].item() for k in range(len(cont[j])))

if a.check:
    r = score_text(texts[0]); print("cached vs full:", [(round(r[j], 4), round(score_full(texts[0], j), 4)) for j in range(min(6, len(S)))]); sys.exit()
t0 = time.time(); null = score_text("N/A"); raw = np.zeros((len(texts), len(S)), np.float32)
for i, t in enumerate(texts):
    raw[i] = score_text(t)
    if i % 100 == 0: print(f"{i}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
tag = a.model.split("/")[-1]
path = a.out or os.path.join(os.path.dirname(__file__), "..", "raw", f"{a.ds}__llm__{tag}.npz")
np.savez_compressed(path, raw=raw, null=null.astype(np.float32), strings=np.array(S, dtype=object),
                    meta=json.dumps({"model": a.model, "prompt": PROMPT, "dtype": a.dtype, "threads": a.threads, "chunk": a.chunk, "torch": torch.__version__,
                                     "transformers": transformers.__version__, "cpu": platform.processor(), "n_items": len(texts), "seconds": time.time() - t0,
                                     "n_label_tokens": [len(c) for c in cont]}))
print("DONE", path, f"{time.time()-t0:.0f}s", flush=True)
