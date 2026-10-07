"""Gate for det_retrain_v2 before any GPU spend (CPU, local val features, tiny subset).

  (a) LEGACY: v2 with the published settings (w_size 0.1, w_giou 1, es_metric mAP50, min 12 /
      patience 2 / delta 0.005 scaled down to the toy run) must give a det_*.pt whose weights,
      best epoch and score_thr are BITWISE equal to the published train_detector_tiles.train
      (driven exactly as phase4_lib_tcd.train_4p drives it).
  (b) RESUME: a v2 run interrupted after epoch 2 and restarted must end with weights bitwise
      equal to an uninterrupted v2 run (CPU is deterministic).
  (c) PILOT KNOBS LIVE: w_size 0.5 / w_giou 2 / mAP50_95 must change the weights vs (a).

Toy data: 3 of the 108 val tiles relabelled "train", 2 relabelled "val" (features in
feat_cache/feat_4p_val). This tests code equivalence only; it is not a training result.

    VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m \
        boxinst_commonality_tcd_04.native_raster.gate_det_retrain_v2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import tempfile

import torch

import boxinst_commonality_tcd_04.train_detector_tiles as T
from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_lib_tcd import Detector4Phase
from boxinst_commonality_tcd_04.native_raster.det_retrain_v2 import train_v2

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FEAT = "/Users/tompitts/dphil/feat_cache/feat_4p_val"
OUT_JSON = os.path.join(HERE, "gate_det_retrain_v2_forced.json")
# Run 1 (gate_det_retrain_v2.json) passed but was weak: toy val AP is 0, so the best checkpoint
# was always epoch 1 and (a) compared one epoch of weights. This run wraps full_report (shared by
# both loops through module T) so the val metrics rise by +1e-2 per call: every eval is a new best,
# the checkpoint is re-saved each epoch, and (a)/(b) compare weights after the LAST epoch.
_REAL_FR = T.full_report
_CALLS = [0]


def forced_full_report(preds, gts, thr):
    rep = dict(_REAL_FR(preds, gts, thr)); _CALLS[0] += 1
    rep["mAP50"] += 1e-2 * _CALLS[0]; rep["mAP50_95"] += 1e-2 * _CALLS[0]
    return rep


def base_args(tag, gt_path, **kw):
    a = dict(tag=tag, epochs=4, seed=0, lr=1e-3, wd=1e-4, bs=3, width=256, tower=3,
             eval_every=1, arm="web", device="cpu", canvas=2048, gt_path=gt_path,
             early_stop=True, min_epochs=2, es_patience=2, es_min_delta=0.005,
             w_size=0.1, w_giou=1.0, es_metric="mAP50")
    a.update(kw)
    return argparse.Namespace(**a)


def same(p, q):
    a = torch.load(p, weights_only=False); b = torch.load(q, weights_only=False)
    eq = all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
    return eq, a["cfg"], b["cfg"]


class Stop(Exception):
    pass


def main():
    assert not os.path.exists(OUT_JSON), f"{OUT_JSON} exists -- refusing to overwrite"
    torch.use_deterministic_algorithms(True)
    gt = json.load(open(os.path.join(PKG, "train_tiles_gt.json")))
    val = sorted(t for t, v in gt.items() if v["partition"] == "val"
                 and os.path.exists(os.path.join(FEAT, t + ".npy")))[:5]
    toy = {t: {**gt[t], "partition": "train" if i < 3 else "val"} for i, t in enumerate(val)}
    tmp = tempfile.mkdtemp(prefix="gate_v2_")
    gtp = os.path.join(tmp, "toy_gt.json"); json.dump(toy, open(gtp, "w"))
    T.Detector8 = Detector4Phase                      # exactly as train_4p does
    T.cache_dir = lambda arm: FEAT
    T.full_report = forced_full_report
    res = {"tiles": val, "tmp": tmp, "forced_metric": "+1e-2 per eval call, reset per run"}

    # (a) published train vs v2 legacy
    T.ART = os.path.join(tmp, "pub")
    _CALLS[0] = 0
    with contextlib.redirect_stdout(io.StringIO()) as pub_log:
        T.train(base_args("pub", gtp))
    _CALLS[0] = 0
    train_v2(base_args("v2", gtp), os.path.join(tmp, "v2"), log=lambda *_: None)
    eq, c1, c2 = same(os.path.join(tmp, "pub", "det_pub.pt"),
                      os.path.join(tmp, "v2", "det_v2.pt"))
    res["a_legacy"] = {"weights_bitwise_equal": eq,
                       "best_epoch": [c1["best_epoch"], c2["best_epoch"]],
                       "score_thr": [c1["score_thr"], c2["score_thr"]],
                       "val_boxAP50": [c1["val_boxAP50"], c2["val_boxAP50"]],
                       "pass": eq and c1["best_epoch"] == c2["best_epoch"]
                       and c1["score_thr"] == c2["score_thr"]}
    res["a_legacy"]["published_log"] = pub_log.getvalue().splitlines()
    print("(a)", {k: v for k, v in res["a_legacy"].items() if k != "published_log"}, flush=True)

    # (b) resume: pilot knobs, interrupted after epoch 2, vs uninterrupted
    kw = dict(w_size=0.5, w_giou=2.0, es_metric="mAP50_95", es_min_delta=0.002,
              min_epochs=4, es_patience=3)
    full_dir, cut_dir = os.path.join(tmp, "full"), os.path.join(tmp, "cut")
    _CALLS[0] = 0
    train_v2(base_args("p", gtp, **kw), full_dir, log=lambda *_: None)
    _CALLS[0] = 0

    def stop_after_2(path):
        if path.endswith("state_p.pt") and torch.load(path, weights_only=False)["epoch"] == 2:
            raise Stop
    try:
        train_v2(base_args("p", gtp, **kw), cut_dir, persist=stop_after_2, log=lambda *_: None)
    except Stop:
        pass
    logs = []
    _CALLS[0] = 2                                   # epochs 1-2 already consumed 2 calls
    train_v2(base_args("p", gtp, **kw), cut_dir, log=lambda s: logs.append(s))
    eq_ck, _, _ = same(os.path.join(full_dir, "det_p.pt"), os.path.join(cut_dir, "det_p.pt"))
    sa = torch.load(os.path.join(full_dir, "state_p.pt"), weights_only=False)
    sb = torch.load(os.path.join(cut_dir, "state_p.pt"), weights_only=False)
    eq_last = all(torch.equal(sa["model"][k], sb["model"][k]) for k in sa["model"])
    res["b_resume"] = {"resumed": any("RESUME from epoch 2" in s for s in logs),
                       "best_ckpt_equal": eq_ck, "final_weights_equal": eq_last,
                       "pass": eq_ck and eq_last and any("RESUME" in s for s in logs)}
    print("(b)", res["b_resume"], flush=True)

    # (c) pilot knobs change the result vs legacy
    eq_c, _, cp = same(os.path.join(tmp, "v2", "det_v2.pt"), os.path.join(full_dir, "det_p.pt"))
    res["c_knobs_live"] = {"weights_differ_from_legacy": not eq_c,
                           "cfg": {k: cp[k] for k in ("w_size", "w_giou", "es_metric",
                                                      "es_min_delta", "val_boxAP50_95")},
                           "pass": not eq_c}
    print("(c)", res["c_knobs_live"], flush=True)

    res["pass"] = all(res[k]["pass"] for k in ("a_legacy", "b_resume", "c_knobs_live"))
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(f"GATE {'PASS' if res['pass'] else 'FAIL'} -> {OUT_JSON}", flush=True)
    if res["pass"]:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
