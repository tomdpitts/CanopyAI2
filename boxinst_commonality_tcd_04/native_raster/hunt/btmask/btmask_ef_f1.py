"""Best-F1 operating point for BTMASK-EF arms A and D (test, sparse), with the exact routine behind the paper's TP/FP/FN/F1
columns (ablation/results/figs/table1_best_f1.py -> make_qualitative.best_f1_threshold). Descriptive only. -> f1_ef.json
  cd ~/dphil/CanopyAI2 && .venv/bin/python -u boxinst_commonality_tcd_04/native_raster/hunt/btmask/btmask_ef_f1.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PKG, "ablation", "results", "figs"))
import make_qualitative as MQ  # noqa: E402
import table1_best_f1 as T1  # noqa: E402

fp_ = os.path.join(HERE, "f1_ef.json"); out = json.load(open(fp_)) if os.path.exists(fp_) else {}
for sp in ("test", "sparse"):
    gt = json.load(open(T1.GT[sp]))
    for arm in ("A", "D"):
        k = f"{sp}|{arm}"
        if k in out:
            continue
        p = json.load(open(os.path.join(HERE, f"preds_{arm}_ef_{sp}_r2048.json")))["preds"]
        th, f1, c = MQ.best_f1_threshold(gt, p)
        out[k] = dict(threshold=th, f1=round(f1, 4), **c, precision=round(c["tp"] / max(c["tp"] + c["fp"], 1), 4),
                      recall=round(c["tp"] / max(c["tp"] + c["fn"], 1), 4))
        print(k, out[k], flush=True); json.dump(out, open(fp_, "w"), indent=1)
