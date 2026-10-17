"""Extra calibration texts (cal pool items 250..499) for the calibration-size scaling test, 4 wordings, q0."""
import sys, json, os
P = os.environ["CLEF_DIR"]  # local snapshot dir of mlx-community/clef-flash-8bit (ships clef_mlx.py)
sys.dont_write_bytecode = True; sys.path.insert(0, P)
import clef_mlx, mlx.core as mx
S = json.load(open("supervised_set.json")); W = json.load(open("zs_set.json"))["wordings"]
m = clef_mlx.load(P); OUT = "out/clef_logits_calx.jsonl"; done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: r = json.loads(line); done.add((r["cfg"], r["i"]))
        except Exception: pass
f = open(OUT, "a")
for wk in W:
    cfg = f"{wk}|q0_topic"
    for i in range(250, 500):
        if (cfg, i) in done: continue
        rec = {"state": S["cal_text"][i], "questions": {"topic": {"type": "choice", "instructions": "What is the topic of this news article?",
               "criteria": {k: w for k, w in zip("ABCD", W[wk])}}}}
        enc, lg = m.logits(rec)
        f.write(json.dumps({"cfg": cfg, "split": "cal", "i": i, "rep": 0, "logits": lg[0].astype(mx.float32).tolist()}) + "\n"); f.flush()
    print("calx", wk, flush=True)
print("CALX DONE", flush=True)
