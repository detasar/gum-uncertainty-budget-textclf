"""Test-retest runs for Clef-flash: identical requests repeated R times (fresh calls, same process)."""
import sys, json, os
P = os.environ["CLEF_DIR"]  # local snapshot dir of mlx-community/clef-flash-8bit (ships clef_mlx.py)
sys.dont_write_bytecode = True; sys.path.insert(0, P)
import clef_mlx, mlx.core as mx
zs = json.load(open("zs_set.json")); W = zs["wordings"]
m = clef_mlx.load(P); OUT = "out/rep_clef.jsonl"; done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: r = json.loads(line); done.add((r["w"], r["run"], r["split"], r["i"]))
        except Exception: pass
f = open(OUT, "a")
plan = [("w0_official", 100, 4)] + [(w, 10, 4) for w in W if w != "w0_official"]
for wk, n_half, R in plan:
    for run in range(R):
        for split, texts in [("cal", zs["cal_text"][:n_half]), ("test", zs["test_text"][:n_half])]:
            for i, t in enumerate(texts):
                if (wk, run, split, i) in done: continue
                rec = {"state": t, "questions": {"topic": {"type": "choice", "instructions": "What is the topic of this news article?",
                       "criteria": {k: w for k, w in zip("ABCD", W[wk])}}}}
                enc, lg = m.logits(rec)
                f.write(json.dumps({"w": wk, "run": run, "split": split, "i": i, "logits": lg[0].astype(mx.float32).tolist()}) + "\n"); f.flush()
        print("clef rep", wk, run, flush=True)
print("CLEF REP DONE", flush=True)
