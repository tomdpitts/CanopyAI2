"""A6 (PLAN.md §13, pre-registered 2026-10-15): GT-box masker comparison on the sparse-236 slice.
Protocol = A4 (tab:gtbox code: gtbox_lib boxes, E.raster GT, 512-ref validity, per_crown_iou_rle).
  lace       deployed LACE masker (em_model_4p_fix.npz, alpha 0.3, kappa 1.6 from the npz, colour-guided
             r 12 / eps 1e-4 / tau 0.35, box clip) -- functions imported from guided_ablate_modal.py;
             Modal-cached features tcd-sparse-vol:feat_4p_sparse (downloaded), local CPU
  ft_*/base_* SAM 3 fine-tuned / zero-shot, whole tile and per subtile (gtbox_sam_persub.arm), local MPS
GATE: on LACE sparse s0's own boxes (ours_prod_s0.json), first 3 non-empty tiles, the local render must
reproduce the published guided masks (preds_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json).
Outputs: tierA/a6_gate.json, tierA/a6_tiles/<tid>.json (valid + per-crown IoU per arm),
tierA/a6_lace_sparse.jsonl (A5-compatible), tierA/a6_sparse_gtbox.json (summary + CIs).

  PYTHONPATH=. .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.tierA.a6_sparse_gtbox --feat_dir <dir>
"""
import argparse, importlib.util, json, os, time
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as G

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
RGB_DIR = os.path.join(REPO, "data", "tcd_sparse", "test")
GT = os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")
MAN = os.path.join(REPO, "data", "tcd_sparse", "slice_manifest.json")
NPZ = "/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"
spec = importlib.util.spec_from_file_location(
    "gsp", os.path.join(NR, "audit_2026-10-14", "selvabox_evidence", "gtbox_sam_persub.py"))
GSP = importlib.util.module_from_spec(spec); spec.loader.exec_module(GSP)


def lace_setup():
    from boxinst_commonality_tcd_04 import evaluate as E
    from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
    m = E.TCDMasker(NPZ)
    assert abs(m.prior_weight - GA.ALPHA) < 1e-9 and abs(m.kappa_scale - GA.KAPPA) < 1e-9
    return GA, m, json.load(open(MAN))["tiles"]


def lace_masks(GA, m, man, tid, boxes, feat_dir, knobs):
    rgb = GA._rgb(RGB_DIR, tid, man)
    feat = np.load(os.path.join(feat_dir, f"{tid}.npy")).astype(np.float32)
    g = feat.shape[-1]; zn = m.project(feat); del feat
    out = []
    for b in np.asarray(boxes, np.float32).reshape(-1, 4):
        clip, prob, _ = GA._prob(m, zn, g, b, *knobs)
        out.append(GA._guided(rgb, clip, prob))
    return out


def gate(feat_dir, GA, m, man):
    fp = os.path.join(HERE, "a6_gate.json")
    if os.path.exists(fp):
        return json.load(open(fp))["pass"]
    src = json.load(open(os.path.join(PKG, "modal_sparse_tcd_multiseed", "preds", "ours_prod_s0.json")))["preds"]
    pub = json.load(open(os.path.join(NR, "preds_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json")))["preds"]
    tiles = [t for t in sorted(src) if src[t]["scores"]][:3]
    n = eq = 0; ious = []
    for t in tiles:
        assert src[t]["boxes_2048"] == pub[t]["boxes_2048"], t
        ms = lace_masks(GA, m, man, t, src[t]["boxes_2048"], feat_dir, (GA.ALPHA, GA.KAPPA))
        for a, b in zip(ms, pub[t]["masks_rle"]):
            B = mu.decode({"size": b["size"], "counts": b["counts"].encode()}).astype(bool)
            n += 1; eq += bool((a == B).all()); u = (a | B).sum()
            ious.append(1.0 if u == 0 else float((a & B).sum() / u))
    res = {"tiles": tiles, "n_boxes": n, "masks_identical": eq, "mean_iou": float(np.mean(ious)),
           "pass": float(np.mean(ious)) >= 0.999}
    json.dump(res, open(fp, "w"), indent=1); print(json.dumps(res), flush=True)
    return res["pass"]


def load_rgb(tid):
    im = Image.open(os.path.join(RGB_DIR, f"{tid}.tif")).convert("RGB"); assert im.size == (2048, 2048), tid
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat_dir", required=True); a = ap.parse_args()
    GA, m, man = lace_setup()
    assert gate(a.feat_dir, GA, m, man), "A6 GATE FAILED -- not running"
    gt = json.load(open(GT))
    tids = [t for t in sorted(gt) if gt[t]["trees"]]
    tdir = os.path.join(HERE, "a6_tiles"); os.makedirs(tdir, exist_ok=True)
    ft, base = GSP.L.load_sam(GSP.DEV), GSP.load_base()
    t0 = time.time()
    for n, t in enumerate(tids):
        fp = os.path.join(tdir, t + ".json")
        if os.path.exists(fp):
            continue
        polys = gt[t]["trees"]
        B = G.gt_boxes_2048(polys).astype(np.float64)
        g, valid = G.gt_rles_and_valid(polys, res=2048)
        rec = {"valid": valid.tolist()}
        rec["lace"] = G.per_crown_iou_rle(G.to_rle(lace_masks(GA, m, man, t, B, a.feat_dir, (None, None))), g).round(5).tolist()
        rgb = load_rgb(t)
        for name, (sp, persub, pre) in {"ft_whole": (ft, False, 1777), "ft_sub": (ft, True, 1777),
                                        "base_whole": (base, False, None), "base_sub": (base, True, None)}.items():
            rec[name] = G.per_crown_iou_rle(G.to_rle(GSP.arm(sp[0], sp[1], rgb, B, persub, pre)), g).round(5).tolist()
        json.dump(rec, open(fp + ".tmp", "w")); os.replace(fp + ".tmp", fp)
        print(f"{n+1}/{len(tids)} {t} {len(polys)} crowns  {time.time()-t0:.0f}s", flush=True)
    # A5-compatible LACE jsonl + summary
    lp = os.path.join(HERE, "a6_lace_sparse.jsonl")
    if not os.path.exists(lp):
        with open(lp, "w") as fh:
            for t in tids:
                r = json.load(open(os.path.join(tdir, t + ".json")))
                fh.write(json.dumps({"tile": t, "iou": r["lace"], "valid": r["valid"]}) + "\n")
    out = os.path.join(HERE, "a6_sparse_gtbox.json"); assert not os.path.exists(out), out
    arms = ["lace", "ft_whole", "ft_sub", "base_whole", "base_sub"]
    vals = {x: {} for x in arms}
    for t in tids:
        r = json.load(open(os.path.join(tdir, t + ".json"))); v = np.array(r["valid"], bool)
        for x in arms:
            vals[x][t] = np.asarray(r[x], float)[v]
    summ = {"n_tiles": len(tids), "n_crowns": int(sum(len(x) for x in vals["lace"].values()))}
    for x in arms:
        c = np.concatenate(list(vals[x].values()))
        summ[x] = {"mean_iou": round(float(c.mean()), 4), "ge50": round(float((c >= .5).mean()), 4),
                   "ge75": round(float((c >= .75).mean()), 4), "ge90": round(float((c >= .9).mean()), 4)}
    for x in ("ft_sub", "base_sub", "base_whole"):
        d = {t: vals["lace"][t] - vals[x][t] for t in tids}
        summ[f"lace_minus_{x}"] = {"mean": round(float(np.concatenate(list(d.values())).mean()), 4), "ci95_tile_boot": GSP.boot(d)}
    json.dump(summ, open(out, "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
