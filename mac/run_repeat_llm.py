"""Test-retest (repeatability) runs for Ollama LLMs: identical requests repeated R times.
mode t0: temperature 0, seed 42 every run.  mode default: model's default sampling params, seed = run index."""
import json, urllib.request, sys, os, math
zs = json.load(open("zs_set.json")); W = zs["wordings"]
model, wk, n_half, mode, R = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
# runtime caps (items per split): 27B -> 15; 9B -> 75 (t0 and default temperature, reference wording), 25 (other wording)  [values as actually run]
if "qwen38" in model: sys.exit("27B repeatability skipped for runtime (~2.3 s/call on this laptop)")
elif mode == "default": n_half = min(n_half, 75)
elif wk != "w0_official": n_half = min(n_half, 25)
else: n_half = min(n_half, 75)
tag = model.split(":")[0].replace("/", "_"); OUT = f"out/rep_llm_{tag}.jsonl"; done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: r = json.loads(line); done.add((r["w"], r["mode"], r["run"], r["split"], r["i"]))
        except Exception: pass
f = open(OUT, "a")
def call(text, labels, run):
    opts = "\n".join(f"{L}) {w}" for L, w in zip("ABCD", labels))
    msg = (f"Classify the news article into exactly one topic.\n\nArticle: {text}\n\nTopics:\n{opts}\n\n"
           "Answer with a single letter (A, B, C or D) and nothing else.")
    o = {"num_predict": 1}
    if mode == "t0": o.update({"temperature": 0, "seed": 42})
    else: o.update({"seed": 1000 + run})
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "stream": False, "think": False,
            "logprobs": True, "top_logprobs": 20, "keep_alive": "5m", "options": o}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900)); lp = {}
    for c in r["logprobs"][0]["top_logprobs"]:
        k = c["token"].strip()
        if len(k) == 1 and k in "ABCD": lp[k] = lp.get(k, 0.0) + math.exp(c["logprob"])
    return r["message"]["content"], [lp.get(k, 0.0) for k in "ABCD"], o
for run in range(R):
    for split, texts in [("cal", zs["cal_text"][:n_half]), ("test", zs["test_text"][:n_half])]:
        for i, t in enumerate(texts):
            if (wk, mode, run, split, i) in done: continue
            ans, p, o = call(t, W[wk], run)
            f.write(json.dumps({"w": wk, "mode": mode, "run": run, "split": split, "i": i, "answer": ans, "p": p, "options": o}) + "\n"); f.flush()
    print(model, wk, mode, "run", run, "done", flush=True)
