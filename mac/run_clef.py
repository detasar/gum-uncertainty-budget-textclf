"""Clef-flash (mlx-community/clef-flash-8bit) zero-shot AG News: raw option logits for every config.
Read-only use of the locally installed model; no files written outside this folder."""
import sys, json, time, os
P = os.environ["CLEF_DIR"]  # local snapshot dir of mlx-community/clef-flash-8bit (ships clef_mlx.py)
sys.dont_write_bytecode = True
sys.path.insert(0, P)
import clef_mlx, mlx.core as mx
QUESTIONS = {"q0_topic": "What is the topic of this news article?",
             "q1_section": "Which section of a newspaper would this article appear in?",
             "q2_classify": "Classify the text into one category."}
zs = json.load(open("zs_set.json")); W = zs["wordings"]
m = clef_mlx.load(P)
OUT = "out/clef_logits.jsonl"
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add((r["cfg"], r["split"], r["i"], r.get("rep", 0)))
f = open(OUT, "a")
def run(cfg, wk, qk, split, texts, rep=0):
    for i, t in enumerate(texts):
        if (cfg, split, i, rep) in done: continue
        rec = {"state": t, "questions": {"topic": {"type": "choice", "instructions": QUESTIONS[qk],
                                                    "criteria": {k: w for k, w in zip("ABCD", W[wk])}}}}
        enc, lg = m.logits(rec)
        assert enc.questions[0].option_ids == ("A", "B", "C", "D")
        f.write(json.dumps({"cfg": cfg, "w": wk, "q": qk, "split": split, "i": i, "rep": rep,
                            "logits": lg[0].astype(mx.float32).tolist()}) + "\n"); f.flush()
t0 = time.time()
# determinism: 50 texts x 3 identical repeats (reference config)
for rep in range(3):
    run("det", "w0_official", "q0_topic", "test", zs["test_text"][:50], rep)
CROSS = len(sys.argv) > 1 and sys.argv[1] == "cross"
plan = ([] if CROSS else  # cross configs dropped for runtime
        [(wk, "q0_topic") for wk in W] + [("w0_official", "q1_section"), ("w0_official", "q2_classify")])
for wk, qk in plan:
    cfg = f"{wk}|{qk}"
    run(cfg, wk, qk, "cal", zs["cal_text"]); run(cfg, wk, qk, "test", zs["test_text"])
    print(cfg, round(time.time() - t0), flush=True)
for wk in ([] if CROSS else W):
    run(f"{wk}|q0_topic", wk, "q0_topic", "shift30", zs["test_text_shift30"])
    print("shift", wk, round(time.time() - t0), flush=True)
print("CLEF DONE", round(time.time() - t0), flush=True)
