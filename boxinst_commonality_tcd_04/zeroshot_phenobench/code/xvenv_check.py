"""Cross-venv parity: run.process in the feature venv (t457, which pipeline.py uses) must reproduce the sample60 masks
written by the repo .venv (where gate_parity.py passed) byte-for-byte, from the same stored features.
  cd code && PB_RUN=xvenv_check PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 <t457 python> -u xvenv_check.py
"""
import json, os, sys
import numpy as np
from common import ROOT, RES, stems
import run as R

tree = R.setup(); out = {}
for t in stems():
    F4 = np.load(os.path.join(ROOT, "features", "sample60", t + ".npy")).astype(np.float32)
    R.process(t, F4, tree, "xvenv")
    for a in R.ARMS:
        x = json.load(open(os.path.join(RES, a, t + ".json"))); y = json.load(open(os.path.join(ROOT, "results", "sample60", a, t + ".json")))
        same = sum(p["counts"] == q["counts"] for p, q in zip(x["masks_rle"], y["masks_rle"]))
        out[f"{t}/{a}"] = [same, len(y["masks_rle"])]
ok = all(s == n for s, n in out.values())
import numpy, PIL, pycocotools, scipy
out["_versions"] = {"python": sys.version.split()[0], "numpy": numpy.__version__, "PIL": PIL.__version__, "scipy": scipy.__version__}
out["_pass"] = ok
json.dump(out, open(os.path.join(ROOT, "results", "gate_xvenv_t457.json"), "w"), indent=1)
print(json.dumps(out, indent=0)); print("XVENV PASS" if ok else "XVENV FAIL")
