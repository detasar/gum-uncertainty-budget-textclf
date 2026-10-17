"""Option-name vs definition test on Clef-flash (after Sun et al., arXiv:2609.26758).
Definitions fixed (w2_topic texts). Conditions:
  matched : option ids = class names, each bound to its own definition
  swapped : option ids = class names cyclically shifted (id 'Sports' carries the World definition, etc.)
  (neutral ids A-D with the same definitions = existing run w2_topic|q0_topic)
We record which DEFINITION the model selects."""
import sys, json, os
P = os.environ["CLEF_DIR"]  # local snapshot dir of mlx-community/clef-flash-8bit (ships clef_mlx.py)
sys.dont_write_bytecode = True; sys.path.insert(0, P)
import clef_mlx, mlx.core as mx
zs = json.load(open("zs_set.json")); DEF = zs["wordings"]["w2_topic"]; NAMES = ["World", "Sports", "Business", "Technology"]
m = clef_mlx.load(P); OUT = "out/clef_nameswap.jsonl"; done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: r = json.loads(line); done.add((r["cond"], r["i"]))
        except Exception: pass
f = open(OUT, "a")
conds = {"matched": {NAMES[c]: c for c in range(4)}, "swapped": {NAMES[(c + 1) % 4]: c for c in range(4)}}  # name -> definition index
for cond, name2def in conds.items():
    for i, t in enumerate(zs["test_text"]):
        if (cond, i) in done: continue
        rec = {"state": t, "questions": {"topic": {"type": "choice", "instructions": "What is the topic of this news article?",
               "criteria": {name: DEF[d] for name, d in name2def.items()}}}}
        enc, lg = m.logits(rec); ids = enc.questions[0].option_ids
        f.write(json.dumps({"cond": cond, "i": i, "option_ids": list(ids), "def_index_of_option": [name2def[o] for o in ids],
                            "logits": lg[0].astype(mx.float32).tolist()}) + "\n"); f.flush()
    print("nameswap", cond, flush=True)
