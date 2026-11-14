"""Masker cost: LACE masker vs SAM 3 zero-shot, per 2048 px OAM-TCD test tile, on IDENTICAL frozen D1 boxes.

One method per process, each in the venv that produced the paper's numbers (attempt 1 ran LACE in .venv: its DINOv3
features drift from the deployed Modal ones, cos 0.41 raw; logs/bench_attempt1_venv_drift.log):
  lace  venv_feat_t457 (transformers 4.57.1 = Modal): 16-phase DINOv3 ViT-L/16 L24 features (phase4_features_tcd.
        _extract_2048 on the tile shifted by {0,4,8,12}^2 = native_raster/hunt/feat16p.py, 64 ViT-L passes / tile)
        -> TCDMasker.project -> fp16 round-trip (deployed storage) -> lace4px_simple render of every box
        (zeroshot_phenobench/code/lace_core, parity-gated byte-identical to the deployed masks) -> RLE.
  sam   repo .venv (transformers 5.12.1, the paper's SAM 3 run): dino_det/sam3_on_d1.sam_tile (= Tier A1: facebook/sam3
        HF Sam3Tracker, box on most central 1024 subtile else whole tile, multimask off, pasted at 2048) -> RLE.
The D1 detector (DINO-Swin-L) is common to both and not timed; it yields no DINOv3 features, so nothing is shared.

Timed: compute only (image decode, model load excluded). torch.mps.synchronize at every timer edge. Warm-up tile run
first and discarded. Run order SAM, LACE, SAM (pass 2 = drift check on the same tiles).
Gates (per tile, stored, untimed): LACE projected features vs feat_cache/hunt_z4p_test cosine; LACE masks vs deployed D1
test masks (RLE identity, IoU); SAM masks vs feat_cache/sam3_d1_test (the paper's SAM 3 run) IoU. RGB identity vs manifest (sha1 or image_id, as guided_ablate_modal._rgb).

  cd boxinst_commonality_tcd_04/masker_cost
  PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u bench.py sam 1
  PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 /Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python -u bench.py lace 1
  PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 ../../.venv/bin/python -u bench.py sam 2
Per-tile checkpoints results/<method>_p<pass>/<tid>.json (resume by listing).
"""
import hashlib, json, os, platform, sys, time

import numpy as np
from PIL import Image
from pycocotools import mask as mu

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); REPO = os.path.dirname(PKG)
FC = "/Users/tompitts/dphil/feat_cache"; OUT = os.path.join(HERE, "results")
N, SEED, SHIFTS = 20, 0, (0, 4, 8, 12)
sys.path.insert(0, os.path.join(PKG, "zeroshot_phenobench", "code"))


def sync():
    import torch
    torch.mps.synchronize()


def nparams(m):
    try:
        return int(sum(p.numel() for p in m.parameters()))
    except Exception:
        return None


def main(method, pas):
    import torch, transformers
    P = json.load(open(os.path.join(PKG, "dino_det", "results", "preds_dino_swinl_s0_test_R3_ef_r2048.json")))["preds"]
    tids = sorted(P); assert len(tids) == 439
    pick = np.random.default_rng(SEED).permutation(len(tids)); sample = [tids[i] for i in pick[:N]]; warm = tids[pick[N]]
    selp = os.path.join(OUT, "selection.json"); sel = {"seed": SEED, "warmup": warm, "tiles": sample}
    if os.path.exists(selp):
        assert json.load(open(selp)) == sel
    json.dump(sel, open(selp, "w"), indent=1)
    man = json.load(open(os.path.join(PKG, "modal_tcd_multiseed", "phase4", "manifest.json")))["feat_test"]
    tdir = os.path.join(OUT, f"{method}_p{pas}"); os.makedirs(tdir, exist_ok=True)

    lace = method.startswith("lace")
    if lace:
        import lace_core as LC, _verbatim as V
        from boxinst_commonality_tcd_04 import evaluate as E
        from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4.phase4_features_tcd import _shift, _extract_2048
        from dapt.backbone import FrozenDinoV3Features
        from boxinst_commonality_tcd_04.dino_det.interlace_test import _interp_matrix
        assert V.verify() and LC.check_frozen()
        # lace = deployed 16 offsets; lace4 / lace1 = IL-test arms interp8 / interp16 (dino_det/interlace_test.arm_feats):
        # 4 offsets {0,8}^2 (16 passes) or the native grid (4 passes), bilinearly interpolated to the 4 px lattice.
        shifts = {"lace": SHIFTS, "lace4": (0, 8), "lace1": (0,)}[method]; ns = 128 * len(shifts)
        A = _interp_matrix(ns, 512 // ns).astype(np.float32) if method != "lace" else None      # constant, untimed
        il_arm = {"lace4": "interp8", "lace1": "interp16"}.get(method)
        nrm = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        em = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); m4 = LC.Masker4(np.load(LC.DEPLOYED_NPZ))
        net = FrozenDinoV3Features("web", layers=(24,), device="mps"); d = np.load(LC.DEPLOYED_NPZ)
        params = {"dinov3_vitl16": nparams(getattr(net, "model", net)),
                  "lace_masker_npz": int(sum(d[k].size for k in ("mu", "U", "scale", "C", "Gbg", "wbg", "pi", "size_edges")))}
    else:
        from boxinst_commonality_tcd_04.dino_det import sam3_on_d1 as S
        sam, proc = S.A1.load_base(); params = {"sam3_tracker": nparams(sam)}
        enc_t = []; _gie = sam.get_image_embeddings
        def gie(*a, **k):                         # encoder timing: wrap the instance method, sync both edges
            sync(); t0 = time.perf_counter(); r = _gie(*a, **k); sync(); enc_t.append(time.perf_counter() - t0); return r
        sam.get_image_embeddings = gie
    env = {"method": method, "pass": pas, "machine": platform.machine(), "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
           "mem_gb": int(os.popen("sysctl -n hw.memsize").read()) // 2**30, "platform": platform.platform(), "python": platform.python_version(),
           "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
           "device": "mps", "date": time.strftime("%Y-%m-%d %H:%M"), "params": params}
    json.dump(env, open(os.path.join(tdir, "_env.json"), "w"), indent=1); print("[env]", json.dumps(env), flush=True)

    T0 = time.time()
    for k, t in enumerate([warm] + sample):
        fp = os.path.join(tdir, t + ".json")
        if os.path.exists(fp):
            continue
        B = np.asarray(P[t]["boxes_2048"], np.float64).reshape(-1, 4)
        pil = Image.open(os.path.join(REPO, "data", "tcd", "test", t + ".tif")).convert("RGB")       # decode: untimed
        u8 = np.asarray(pil); assert u8.shape == (2048, 2048, 3)
        if "rgb_sha1" in man[t]:                                      # = guided_ablate_modal._rgb identity check
            assert hashlib.sha1(np.ascontiguousarray(u8).tobytes()).hexdigest() == man[t]["rgb_sha1"], t
        else:
            assert json.load(open(os.path.join(REPO, "data", "tcd", "test", t + "_meta.json")))["image_id"] == man[t]["image_id"], t
        rec = {"tile": t, "warmup": t == warm, "n_boxes": len(B)}
        if lace:
            arr = u8.astype(np.float32) / 255.0                       # = guided_ablate_modal._rgb (render guide) and feat16p input
            sync(); t0 = time.perf_counter()
            if method == "lace":
                Z = np.zeros((512, 512, 128), np.float32)
                for dy in SHIFTS:
                    for dx in SHIFTS:
                        f = _extract_2048(net, _shift(arr, dy, dx)).astype(np.float16).astype(np.float32)
                        Z[dy // 4::4, dx // 4::4] = em.project(f).reshape(128, 128, 128)
                zn = Z.astype(np.float16).astype(np.float32).reshape(-1, 128)               # deployed fp16 storage
            else:
                k = len(shifts); Vs = np.zeros((ns, ns, 128), np.float32)                    # unnormalised projection
                for dy in shifts:
                    for dx in shifts:
                        f = _extract_2048(net, _shift(arr, dy, dx)).astype(np.float16).astype(np.float32)
                        Vs[dy // 8::k, dx // 8::k] = ((f.reshape(1024, -1).T - em.mu) @ em.U / em.scale).reshape(128, 128, 128)
                zn = nrm(np.einsum("xj,yjc->yxc", A, np.einsum("yi,ijc->yjc", A, Vs))).astype(np.float16).astype(np.float32).reshape(-1, 128)
            t1 = time.perf_counter(); rles = []
            for b in B.astype(np.float32):
                clip, prob, r = LC.prob4(m4, zn, 512, 512, b, 2048, 2048)
                m = LC.render_post4(arr, clip, prob, 2048, 2048)
                rles.append({"size": [2048, 2048], "counts": mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")})
            t2 = time.perf_counter()
            rec["time"] = {"feat_s": t1 - t0, "render_s": t2 - t1, "total_s": t2 - t0}
            if method == "lace":
                ref = np.load(f"{FC}/hunt_z4p_test/{t}.npy").astype(np.float32).reshape(-1, 128)
                cos = (zn * ref).sum(1); dep = P[t]["masks_rle"]
            else:                                                     # gate vs the IL-test arm's stored masks (Modal feat_4p)
                cos = np.ones(1); dep = json.load(open(f"{FC}/interlace_test_{il_arm}/{t}.json"))["masks_rle"]
            iou = mu.iou(rles, dep, [0] * len(B)).diagonal() if len(B) else np.ones(1)
            rec["gate"] = {"feat_cos_mean": float(cos.mean()), "feat_cos_min": float(cos.min()),
                           "rle_identical": int(sum(a["counts"] == b["counts"] for a, b in zip(rles, dep))), "iou_mean": float(iou.mean()), "iou_min": float(iou.min())}
            msg = f"LACE {t2-t0:.1f}s (feat {t1-t0:.1f}, render {t2-t1:.1f}) | cos {cos.mean():.6f} ident {rec['gate']['rle_identical']}/{len(B)}"
        else:
            enc_t.clear(); sync(); t0 = time.perf_counter()
            rles, _, _ = S.sam_tile(sam, proc, pil, B)
            sync(); t1 = time.perf_counter()
            rec["time"] = {"encoder_s": sum(enc_t), "n_encodes": len(enc_t), "decode_post_s": (t1 - t0) - sum(enc_t), "total_s": t1 - t0}
            sref = json.load(open(f"{FC}/sam3_d1_test/{t}.json")); assert sref["boxes_2048"] == P[t]["boxes_2048"]
            iou = mu.iou(rles, sref["masks_rle"], [0] * len(B)).diagonal() if len(B) else np.ones(1)
            rec["gate"] = {"iou_mean": float(iou.mean()), "iou_min": float(iou.min()), "frac_iou_ge_0.99": float((iou >= 0.99).mean())}
            msg = f"SAM3 {t1-t0:.1f}s (enc {sum(enc_t):.1f} x{len(enc_t)}, dec+post {t1-t0-sum(enc_t):.1f}) | iou {iou.mean():.4f}"
        json.dump(rec, open(fp + ".tmp", "w"), indent=1); os.replace(fp + ".tmp", fp)
        print(f"[{method} p{pas} {k}/{N}] {t} n={len(B)} {'WARMUP ' if t == warm else ''}{msg}  [{time.time()-T0:.0f}s]", flush=True)
    print(f"[bench {method} p{pas}] done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
