"""Isolated single-seed runner for the 5-seed MPS variance estimate.

Everything this touches for OUTPUT lives under mps_multiseed/ — nothing is written
to ../artifacts or ../vault. It does that WITHOUT editing the shared scripts: it
imports train_detector_tiles / evaluate and monkeypatches their artifact dirs
(`T.ART`, `E.OUT`) to point inside `_out/` before calling them. `_out/test_gt.json`
is a symlink back to the real (read-only) test GT, so the evaluator still scores the
true 439-tile benchmark. Deleting mps_multiseed/ undoes the entire experiment.

Fixed inputs (read-only, shared):
  ../train_tiles_gt.json                     900-tile cohort (train/val partitions)
  ../cache/web/feat_traintile                4096-dim DINOv3-web train features
  ../cache/web/{feat_test,feat_test_down}    test + downscale features
  ../vault/em_model.npz                      the fixed box->mask EM masker

Recipe = the det_t8 run behind the vaulted 0.504 (NOT the regularized full-4k
recipe): epochs 40, bs 3, eval_every 5, width 256, tower 3, Adam lr 1e-3 wd 1e-4,
cosine schedule, best-on-val checkpoint -- PLUS aggressive early stopping
(--early_stop, min_epochs 12, es_patience 2 -> stop after 10 flat epochs). ES only
trims dead tail epochs (checkpoint stays best-on-val); it deviates slightly from the
exact full-40 0.504 recipe and can nudge a seed down if a late peak is clipped. Only
--seed varies across the 5 runs.

Usage (from repo root):
    .venv/bin/python -u -m boxinst_commonality_tcd_04.mps_multiseed.run_seed \
        --seed 0 --stage both
"""
import argparse
import json
import os
import shutil

HERE = os.path.abspath(os.path.dirname(__file__))
OUTDIR = os.path.join(HERE, "_out")               # shadow OUT for evaluate.py
ARTDIR = os.path.join(OUTDIR, "artifacts")        # shadow artifacts (ckpts + eval json)
VAULT_EM = os.path.abspath(os.path.join(HERE, "..", "vault", "em_model.npz"))


def train_seed(seed):
    from boxinst_commonality_tcd_04 import train_detector_tiles as T
    tag = f"t8_s{seed}"
    ckpt = os.path.join(ARTDIR, f"det_{tag}.pt")
    if os.path.exists(ckpt):
        print(f"[seed {seed}] TRAIN: checkpoint exists, skipping -> {ckpt}", flush=True)
    else:
        T.ART = ARTDIR                            # redirect ALL checkpoint writes
        # det_t8 recipe (cosine, best-on-val) + AGGRESSIVE early stopping: stop
        # after es_patience=2 evals (eval_every=5 -> 10 epochs) with no >0.005 val
        # boxAP50 gain, min 12 epochs. Checkpoint is still best-on-val, so ES only
        # trims dead tail epochs; it can slightly lower the number if a late peak
        # is clipped. Deliberately deviates from the exact full-40 0.504 recipe.
        args = argparse.Namespace(
            tag=tag, epochs=40, seed=seed, lr=1e-3, wd=1e-4, bs=3,
            width=256, tower=3, eval_every=5, arm="web", device="mps",
            early_stop=True, min_epochs=12, es_patience=2, es_min_delta=0.005)
        print(f"[seed {seed}] TRAIN det_t8 recipe (cosine, best-on-val, "
              f"early-stop p2/min12) -> {ckpt}", flush=True)
        T.train(args)
    # top-level convenience symlink with the name the task asked for
    link = os.path.join(HERE, f"det_{tag}.pt")
    if not os.path.lexists(link):
        os.symlink(os.path.join("_out", "artifacts", f"det_{tag}.pt"), link)


def eval_seed(seed):
    from boxinst_commonality_tcd_04 import evaluate as E
    tag = f"t8_s{seed}"
    out_json = os.path.join(HERE, f"eval_s{seed}.json")
    if os.path.exists(out_json):
        print(f"[seed {seed}] EVAL: {out_json} exists, skipping", flush=True)
        return
    E.OUT = OUTDIR                                # redirect det-load + eval-json write
    args = argparse.Namespace(
        det=f"det_{tag}", model=VAULT_EM, arm="web", limit=None, topk=600,
        eval_score_thr=0.05, op_thr=None, multiscale=True, ms_nms_iou=0.5,
        device="mps")
    print(f"[seed {seed}] EVAL multiscale on full 439 (EM={VAULT_EM})", flush=True)
    E.run(args)
    auto = os.path.join(ARTDIR, f"eval_{tag}.json")   # evaluate.py auto-name
    shutil.copyfile(auto, out_json)
    res = json.load(open(out_json))
    print(f"[seed {seed}] EVAL done -> {out_json} | mask_mAP50={res['mask_mAP50']} "
          f"box_mAP50={res['box_mAP50']} semF1={res['semantic_F1']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", choices=["train", "eval", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(ARTDIR, exist_ok=True)
    # guard: refuse to run if the shadow test GT symlink isn't in place
    assert os.path.exists(os.path.join(OUTDIR, "test_gt.json")), \
        "missing _out/test_gt.json symlink"
    if a.stage in ("train", "both"):
        train_seed(a.seed)
    if a.stage in ("eval", "both"):
        eval_seed(a.seed)
