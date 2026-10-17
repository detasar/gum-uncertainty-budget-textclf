"""Fix all subsets once (seed 0) and write them to cache/ so every model sees identical data."""
import json, numpy as np
from common import *
SHIFT_LEVELS = [0.0, 0.15, 0.3, 0.45]
d = load_agnews(); rng = np.random.default_rng(0)
tr = d["train"]; te = d["test"]
i_train = balanced_sample(tr, 2000, rng)
rest = np.setdiff1d(np.arange(len(te)), [])
i_te = balanced_sample(te, 2000, rng)            # 1000 cal pool + 1000 test
i_cal, i_test = i_te[:1000], i_te[1000:]
sup = {"train_text": [clean(t) for t in tr.text.values[i_train]], "train_y": tr.label.values[i_train].tolist(),
       "cal_text": [clean(t) for t in te.text.values[i_cal]], "cal_y": te.label.values[i_cal].tolist(),
       "test_y": te.label.values[i_test].tolist(), "shift_levels": SHIFT_LEVELS,
       "test_text": {str(l): [perturb(clean(t), l, 1000 + k) if l > 0 else clean(t)
                              for k, t in enumerate(te.text.values[i_test])] for l in SHIFT_LEVELS}}
json.dump(sup, open(f"{CACHE}/supervised_set.json", "w"))
# Zero-shot / LLM set: subset of the same cal pool and test set (250 + 250), plus shifted (0.3) test texts
zs = {"cal_text": sup["cal_text"][:250], "cal_y": sup["cal_y"][:250],
      "test_text": sup["test_text"]["0.0"][:250], "test_y": sup["test_y"][:250],
      "test_text_shift30": sup["test_text"]["0.3"][:250], "wordings": WORDINGS, "classes": CLASSES}
json.dump(zs, open(f"{CACHE}/zs_set.json", "w"))
print(len(sup["train_text"]), len(sup["cal_text"]), len(sup["test_y"]), np.bincount(zs["cal_y"]), np.bincount(zs["test_y"]))
print(sup["test_text"]["0.0"][0][:200]); print(sup["test_text"]["0.3"][0][:200])
