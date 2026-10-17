"""Zero-shot LLM classification via local Ollama (temperature 0, fixed seed, 1 token, letter options).
Class probabilities = first-token top-20 logprobs restricted to A-D and renormalised."""
import json, urllib.request, sys, os, time, math
zs = json.load(open("zs_set.json")); W = zs["wordings"]
model, wkeys, do_shift = sys.argv[1], sys.argv[2].split(","), sys.argv[3] == "1"
if "qwen38" in model: do_shift = False  # runtime cap for the 27B model
tag = model.split(":")[0].replace("/", "_")
OUT = f"out/llm_{tag}.jsonl"
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add((r["cfg"], r["split"], r["i"], r.get("rep", 0)))
f = open(OUT, "a")
def call(text, labels):
    opts = "\n".join(f"{L}) {w}" for L, w in zip("ABCD", labels))
    msg = (f"Classify the news article into exactly one topic.\n\nArticle: {text}\n\nTopics:\n{opts}\n\n"
           "Answer with a single letter (A, B, C or D) and nothing else.")
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "stream": False, "think": False,
            "logprobs": True, "top_logprobs": 20, "keep_alive": "5m",
            "options": {"temperature": 0, "seed": 42, "num_predict": 1}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    lp = {}
    for c in r["logprobs"][0]["top_logprobs"]:
        k = c["token"].strip()
        if k in "ABCD" and len(k) == 1:
            lp[k] = lp.get(k, 0.0) + math.exp(c["logprob"])
    return r["message"]["content"], [lp.get(k, 0.0) for k in "ABCD"]
def run(cfg, wk, split, texts, rep=0):
    for i, t in enumerate(texts):
        if (cfg, split, i, rep) in done: continue
        ans, p = call(t, W[wk])
        f.write(json.dumps({"cfg": cfg, "w": wk, "split": split, "i": i, "rep": rep, "answer": ans, "p": p}) + "\n"); f.flush()
t0 = time.time()
for rep in ([] if "qwen38" in model else range(3)):  # 27B: repeatability measured separately (runtime)
    run("det", wkeys[0], "test", zs["test_text"][:30], rep)
for wk in wkeys:
    run(wk, wk, "cal", zs["cal_text"]); run(wk, wk, "test", zs["test_text"]); print(model, wk, round(time.time()-t0), flush=True)
if do_shift:
    run(wkeys[0], wkeys[0], "shift30", zs["test_text_shift30"])
print("LLM DONE", model, round(time.time()-t0), flush=True)
