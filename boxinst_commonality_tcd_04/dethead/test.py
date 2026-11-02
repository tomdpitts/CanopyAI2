"""439 test for a dethead detector: decode -> LACE masker lace4px_simple (+ own4 s') -> frozen score_coco.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.test gate
  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.dethead.test run --tag fcos_s0
Per-tile checkpoints: feat_cache/dethead_<tag>_{det,mask}/<tid>.json (resume). Outputs (refuse to overwrite):
dethead/results/{preds,scored}_<tag>_{sprime,detscore}_r2048.json and results/summary_<tag>.json.
"""
import argparse, json, multiprocessing as mp, os, time
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.join(PKG, "dethead"); REPO = os.path.dirname(PKG)
FC_ = "/Users/tompitts/dphil/feat_cache"
TEST_FEAT = f"{FC_}/feat_4p_test_dl/feat_4p_test"
PUB_S0 = os.path.join(PKG, "native_raster", "preds_lace_s0_r2048_cgf_r12_tau035_detscore.json")
OUT = os.path.join(HERE, "results")


def load_model(ckpt, head):
    import torch
    from boxinst_commonality_tcd_04.dethead.train import build
    m = build(head); ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state"]); m.eval()
    return m


def detect(ckpt, head, tids, out_dir, device, nms=None):
    import torch
    from boxinst_commonality_tcd_04.dethead.train import predict
    os.makedirs(out_dir, exist_ok=True)
    m = load_model(ckpt, head).to(device)
    t0 = time.time()
    for k, t in enumerate(tids, 1):
        fp = os.path.join(out_dir, t + ".json")
        if os.path.exists(fp):
            continue
        f = torch.from_numpy(np.load(os.path.join(TEST_FEAT, t + ".npy")))
        b, s = predict(m, head, f, device, nms)
        json.dump({"boxes_2048": b.numpy().astype(np.float64).tolist(), "scores": s.numpy().astype(np.float64).tolist()},
                  open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        if k % 50 == 0:
            print(f"  det {k}/{len(tids)} {time.time()-t0:.0f}s", flush=True)


def gate():
    """Local decode of the published (Modal-trained) LACE s0 checkpoint on the local copy of the Modal test features must
    reproduce the published s0 detector boxes/scores (first 3 tiles)."""
    import torch
    from boxinst_commonality_tcd_04.dethead.train import predict
    P = json.load(open(PUB_S0))["preds"]; tids = sorted(P)[:3]
    m = load_model(f"{FC_}/det_phase4_L24_s0.pt", "centernet")
    dmax_b = dmax_s = 0.0; nmis = 0; diffs = []
    for t in tids:
        b, s = predict(m, "centernet", torch.from_numpy(np.load(os.path.join(TEST_FEAT, t + ".npy"))), torch.device("cpu"))
        pb, ps = np.asarray(P[t]["boxes_2048"]), np.asarray(P[t]["scores"])
        if len(pb) != len(b):
            nmis += 1; continue
        if not len(pb):
            continue
        # CPU vs CUDA float differences re-order near-tied scores: match each published box to its nearest local box
        from scipy.optimize import linear_sum_assignment
        lb, ls = b.numpy(), s.numpy()
        cost = np.abs(pb[:, None, :] - lb[None, :, :]).max(2)
        r, c = linear_sum_assignment(cost)
        dmax_b = max(dmax_b, float(cost[r, c].max())); dmax_s = max(dmax_s, float(np.abs(ps[r] - ls[c]).max()))
        diffs += cost[r, c].tolist()
    frac = float(np.mean(np.asarray(diffs) < 0.05)) if diffs else 1.0
    # CPU vs CUDA: rare NMS tie flips move a box; require equal counts, >= 99% of boxes within 0.05 px, scores within 1e-3
    ok = nmis == 0 and frac >= 0.99 and dmax_s < 1e-3
    rec = {"gate": "local decode of det_phase4_L24_s0.pt == published s0 detector boxes", "tiles": tids, "count_mismatch_tiles": nmis,
           "max_abs_box_px": dmax_b, "frac_boxes_within_0.05px": frac, "n_boxes": len(diffs), "max_abs_score": dmax_s, "pass": ok, "date": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(OUT, exist_ok=True); json.dump(rec, open(os.path.join(OUT, "gate_decode.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)
    return ok


def _mask_job(args):
    t, boxes, scores, d = args
    from boxinst_commonality_tcd_04.native_raster.hunt.btmask import btmask as BM
    fp = os.path.join(d, t + ".json")
    if not os.path.exists(fp):
        if "m4" not in BM._W:
            BM._init("test")
        r = BM.render_boxes("test", t, np.asarray(boxes, np.float32).reshape(-1, 4), scores, [BM.FROZEN])[BM.cfg_name(BM.FROZEN)]
        json.dump(r, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    return t


def run(run_tag, head, nms=None):
    import torch
    tag = run_tag if nms is None else f"{run_tag}_nms{int(round(nms * 10)):02d}"      # outputs keyed by decode too
    sfp = os.path.join(OUT, f"summary_{tag}.json"); assert not os.path.exists(sfp), sfp
    tids = sorted(json.load(open(PUB_S0))["preds"]); assert len(tids) == 439
    ckpt = os.path.join(HERE, "runs", run_tag, "best.pt")
    ddir, mdir = f"{FC_}/dethead_{tag}_det", f"{FC_}/dethead_{tag}_mask"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    detect(ckpt, head, tids, ddir, device, nms)
    D = {t: json.load(open(os.path.join(ddir, t + ".json"))) for t in tids}
    print(f"[{tag}] detections: {sum(len(D[t]['scores']) for t in tids)} on 439 tiles", flush=True)
    os.makedirs(mdir, exist_ok=True)
    jobs = [(t, D[t]["boxes_2048"], D[t]["scores"], mdir) for t in tids if not os.path.exists(os.path.join(mdir, t + ".json"))]
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(12) as pool:
            for k, _ in enumerate(pool.imap_unordered(_mask_job, jobs), 1):
                if k % 20 == 0 or k == len(jobs):
                    print(f"  mask {k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    M = {t: json.load(open(os.path.join(mdir, t + ".json"))) for t in tids}
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for rank, floor in (("sprime", 0.0), ("detscore", 0.05)):
        fp = os.path.join(OUT, f"preds_{tag}_{rank}_r2048.json"); files[rank] = (fp, floor)
        if not os.path.exists(fp):
            preds = {t: {"boxes_2048": D[t]["boxes_2048"], "scores": M[t]["s_prime"] if rank == "sprime" else D[t]["scores"],
                         "masks_rle": M[t]["masks_rle"]} for t in tids}
            json.dump({"meta": {"mask_res": 2048, "model": f"dethead {tag} ({head}) + LACE lace4px_simple, ranking {rank}"},
                       "preds": preds}, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
    from boxinst_commonality_tcd_04 import score_coco as SC
    os.chdir(REPO)
    out = {"tag": tag, "head": head, "nms": nms, "ckpt_cfg": torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"], "rows": {}}
    for rank, (fp, floor) in files.items():
        sp = fp.replace("preds_", "scored_")
        if not os.path.exists(sp):
            json.dump(SC.score_many([fp], os.path.join(PKG, "test_gt.json"), res=2048, max_dets=600, score_floor=floor)[0], open(sp, "w"))
        c = json.load(open(sp))["canopy_neutral_crowd"]
        out["rows"][rank] = {"mask": {k: c["mask"][k] for k in ("AP50", "AP75", "AP50_95", "n_det")},
                             "box": {k: c["box"][k] for k in ("AP50", "AP75", "AP50_95")}, "floor": floor}
    json.dump(out, open(sfp, "w"), indent=1); print(json.dumps(out, indent=1, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate", "run"]); ap.add_argument("--tag"); ap.add_argument("--head", default="fcos"); ap.add_argument("--nms", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "gate":
        assert gate(), "GATE FAILED"
    else:
        run(a.tag, a.head, a.nms)
