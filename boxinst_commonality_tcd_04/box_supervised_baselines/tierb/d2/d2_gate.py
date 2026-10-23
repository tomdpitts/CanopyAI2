"""1-GPU gate for B1/B2. Every assertion here must pass before a training GPU-hour is spent.

G0 environment + compiled ops          G6 P1 canopy ignore: == released when no canopy; LACE's
G1 pinned sources, patches installed       cell rule reproduced exactly at stride 8; positives kept
G2 staged pixels (val) hash-verified    G7 P5 chunked postprocess == released postprocess
G3 COCO checkpoint: only cls re-init       (masks, boxes, scores, mask scores; bitwise)
G4 mapper: crop/canopy/boxes aligned    G8 P6 CPU assignment == released GPU loop (random +
   with the tile; box-only supervision      real teacher outputs)
G5 train fwd/bwd: dense, empty and      G9 P4 teacher input = crop at native scale
   all-empty batches; finite grads      G10 inference on 2 val tiles, both geometries; timing
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

from tierb import crops as C
from tierb import tierb_common as TC


def _eq(a, b):
    return bool(torch.equal(a, b)) if isinstance(a, torch.Tensor) else a == b


def _detached(inst):
    """Copy of an Instances with every tensor detached (deepcopy refuses non-leaf tensors)."""
    from detectron2.structures import Instances
    out = Instances(inst.image_size)
    for k, v in inst.get_fields().items():
        out.set(k, v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
    return out


def gate(args):
    from tierb.d2 import d2_run as R
    from tierb.d2 import d2_patches as P
    method = args.method
    R._tf32()                                     # same numerics as train/predict
    rep = {"method": method, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    ck = rep["checks"]
    t00 = time.time()

    # G0 ------------------------------------------------------------------------------
    import detectron2
    import PIL
    if method == "boxteacher":
        sys.path.insert(0, f"{R.BT_ROOT}/projects/BoxTeacher")
    import adet
    from adet import _C as adet_C          # noqa: F401  (compiled op must import)
    from detectron2 import _C as d2_C      # noqa: F401
    assert torch.cuda.is_available()
    ck["G0_env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "gpu": torch.cuda.get_device_name(0), "detectron2": detectron2.__version__,
                    "numpy": np.__version__, "pillow": PIL.__version__}
    assert np.__version__.startswith("1."), np.__version__
    assert PIL.__version__.startswith("9."), PIL.__version__

    # G1 ------------------------------------------------------------------------------
    ck["G1_patches"] = P.apply(method)

    # G2 ------------------------------------------------------------------------------
    ck["G2_val_pixels"] = TC.verify_split("val")

    # G3 ------------------------------------------------------------------------------
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from tierb.d2.d2_data import TierBMapper, register, train_dicts
    sched = C.schedule()
    cfg = R.build_cfg(method, "/tmp/gate_out", sched, num_gpus=8)
    model = build_model(cfg)
    ck["G3_ckpt"] = R.ckpt_key_check(model, cfg.MODEL.WEIGHTS, method)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    ck["G3_ckpt"]["iter_buffers_before_reset"] = R.reset_loss_warmups(model, method)
    R.check_released_test_attrs(model, method)
    ck["G3_sched"] = sched
    ck["G3_cfg"] = {"lr": cfg.SOLVER.BASE_LR, "opt": cfg.SOLVER.get("OPTIMIZER", "SGD(default)"),
                    "steps": list(cfg.SOLVER.STEPS), "max_iter": cfg.SOLVER.MAX_ITER,
                    "warmup": cfg.SOLVER.WARMUP_ITERS, "wd": cfg.SOLVER.WEIGHT_DECAY,
                    "pairwise_warmup": cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS,
                    "bottom_pixels_removed": cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED,
                    "num_classes": cfg.MODEL.FCOS.NUM_CLASSES}

    # G4 ------------------------------------------------------------------------------
    gt = json.load(open(R.TRAIN_GT))
    dicts = train_dicts(R.TRAIN_GT)
    assert len(dicts) == 3168
    register(R.TRAIN_GT)

    def nboxes(d):
        oy, ox = C.quadrant_offset(C.TILE, d["tierb_quadrant"])
        b = np.asarray(gt[d["tierb_tid"]]["boxes"], float).reshape(-1, 4)
        return int(((b[:, 2] > ox) & (b[:, 0] < ox + C.CROP) & (b[:, 3] > oy) & (b[:, 1] < oy + C.CROP)).sum())
    counts = [nboxes(d) for d in dicts]
    dense = dicts[int(np.argmax(counts))]
    empties = [d for d, n in zip(dicts, counts) if n == 0]
    can_rich = max((d for d in dicts if gt[d["tierb_tid"]]["canopy"]), key=lambda d: len(gt[d["tierb_tid"]]["canopy"]))
    ck["G4_quadrants"] = {"n": len(dicts), "empty": len(empties), "max_boxes": max(counts),
                          "dense": dense["image_id"], "canopy_rich": can_rich["image_id"]}
    det = TierBMapper(cfg.INPUT.FORMAT, scales=(1.0,), hflip=False)
    al = []
    for d in (dense, can_rich, empties[0]):
        o = det(d)
        t = d["tierb_tid"]
        oy, ox = C.quadrant_offset(C.TILE, d["tierb_quadrant"])
        rgb = TC.load_rgb(t)
        bgr = torch.as_tensor(np.ascontiguousarray(rgb[oy:oy + C.CROP, ox:ox + C.CROP, ::-1].transpose(2, 0, 1)))
        assert torch.equal(o["image"], bgr), "image crop misaligned"
        cp = TC.canopy_px(gt[t]["canopy"])[oy:oy + C.CROP, ox:ox + C.CROP]
        assert np.array_equal(o["canopy"].numpy().astype(bool), cp), "canopy crop misaligned"
        b = np.asarray(gt[t]["boxes"], float).reshape(-1, 4) - [ox, oy, ox, oy]
        b = np.clip(b, 0, C.CROP)
        b = b[(b[:, 2] - b[:, 0] > 1e-5) & (b[:, 3] - b[:, 1] > 1e-5)]
        got = o["instances"].gt_boxes.tensor.numpy()
        assert got.shape == b.shape and np.allclose(got, b, atol=1e-3), "boxes misaligned"
        assert set(o["instances"].get_fields()) == {"gt_boxes", "gt_classes"}, o["instances"].get_fields().keys()
        al.append({"id": d["image_id"], "boxes": len(b), "canopy_frac": float(cp.mean())})
    rnd = TierBMapper(cfg.INPUT.FORMAT,
                      aug_type=cfg.INPUT.AUG_TYPE if method == "boxteacher" else "none",
                      aug_extra=cfg.INPUT.AUG_EXTRA if method == "boxteacher" else True)
    random.seed(0)
    sample = random.sample(dicts, 40) + empties[:8]
    outs = [rnd(d) for d in sample]
    for o in outs:
        assert tuple(o["image"].shape) == (3, C.CROP, C.CROP) and tuple(o["canopy"].shape) == (C.CROP, C.CROP)
        assert set(np.unique(o["canopy"].numpy())) <= {0, 1}
        bx = o["instances"].gt_boxes.tensor
        assert (bx >= 0).all() and (bx <= C.CROP).all()
        assert ("aug_image" in o) == (method == "boxteacher")
    ck["G4_mapper"] = {"aligned_s1": al, "random_ok": len(outs),
                       "scales_seen": sorted({round(o["tierb_scale"], 4) for o in outs}),
                       "coverage_check": C.coverage_check()}

    # G5 + G6 + G8 + G9: training forward/backward with captures --------------------------
    from adet.modeling.fcos.fcos_outputs import FCOSOutputs
    cap = {}
    patched_losses = FCOSOutputs.fcos_losses

    def capturing(self, instances):
        cap["fcos"] = (self, _detached(instances), self._tierb_canopy)   # last batch = canopy-rich
        return patched_losses(self, instances)
    FCOSOutputs.fcos_losses = capturing
    tsizes = []
    m = model
    # collective count per training forward (P7): wrap the one function every all-reduce goes through
    import adet.utils.comm as AC
    n_coll = [0]
    orig_reduce_sum = AC.reduce_sum

    def counting_reduce_sum(t):
        n_coll[0] += 1
        return orig_reduce_sum(t)
    AC.reduce_sum = counting_reduce_sum
    if method == "boxteacher":
        orig_rbs = P._resize_by_scale

        def rbs_capture(images, scales):
            out, sizes = orig_rbs(images, scales)
            tsizes.extend([(tuple(o.shape[-2:]), float(s)) for o, s in zip(out, scales)])
            return out, sizes
        P._resize_by_scale = rbs_capture
        orig_pl = type(m).pseudo_labeling

        def pl_capture(self, bi, inst, iou_thr=0.5, score_thr=0.0):
            cap.setdefault("teacher", []).extend(
                [(_detached(x["instances"]).to("cpu"), _detached(b["instances"]).to("cpu"))
                 for x, b in zip(inst, bi)])
            return orig_pl(self, bi, inst, iou_thr, score_thr)
        type(m).pseudo_labeling = pl_capture
    m.train()
    batches = {"dense+empty": [det(dense), det(empties[0])],
               "empty+empty": [det(empties[1]), det(empties[2])],
               "canopy+random": [rnd(can_rich), rnd(dicts[7])]}
    fb, colls = {}, {}
    from detectron2.utils.events import EventStorage
    with EventStorage(0):                          # BoxTeacher's mask head logs to the storage
        for name, b in batches.items():
            if method == "boxteacher":                 # det mapper has no colour aug: as released
                for x in b:
                    x.setdefault("aug_image", x["image"])
            m.zero_grad()
            n_coll[0] = 0
            losses = m(b)
            colls[name] = n_coll[0]
            tot = sum(losses.values())
            assert torch.isfinite(tot), (name, {k: float(v) for k, v in losses.items()})
            tot.backward()
            g = [p.grad for p in m.parameters() if p.grad is not None]
            assert g and all(torch.isfinite(x).all() for x in g), name
            fb[name] = {k: round(float(v), 5) for k, v in losses.items()}
    FCOSOutputs.fcos_losses = patched_losses
    AC.reduce_sum = orig_reduce_sum
    assert len(set(colls.values())) == 1, f"collective count differs across batches (DDP hang risk): {colls}"
    ck["G5_fwd_bwd"] = fb
    ck["G5_collectives_per_forward"] = colls
    if method == "boxteacher":
        P._resize_by_scale = orig_rbs
        type(m).pseudo_labeling = P._bt_pseudo_labeling
        scales = [x["tierb_scale"] for bb in batches.values() for x in bb]
        assert [sc for _, sc in tsizes] == scales, (tsizes, scales)
        want = [C.teacher_size(C.scaled_size(sc)) for sc in scales]
        got = [hw for hw, _ in tsizes]
        assert got == [(w, w) for w in want], (got, want)
        ck["G9_teacher_scale"] = {"scales": scales, "teacher_hw": got, "expected": want, "exact": True}

    # G6 canopy ignore on the captured FCOS instances (dense+empty batch, then canopy batch)
    fo, inst, canopy = cap["fcos"]
    fo._tierb_canopy = None
    a = P.ORIG["fcos_losses"](fo, _detached(inst))[1]
    b_ = P._fcos_losses(fo, _detached(inst))[1]
    assert all(_eq(a[k], b_[k]) for k in a), "P1 without canopy != released"
    fo._tierb_canopy = [torch.zeros_like(c) for c in canopy]
    z = P._fcos_losses(fo, _detached(inst))[1]
    zdiff = max(float((a[k] - z[k]).abs()) for k in a)
    assert zdiff <= 1e-6, f"P1 zero canopy != released ({zdiff})"
    # LACE's rule, independently, at stride 8 on a canopy-rich crop
    cr = rnd(can_rich)
    cmap = cr["canopy"].numpy().astype(bool)
    lace = cmap.reshape(128, 8, 128, 8).mean((1, 3)) > 0.5
    from adet.utils.comm import compute_locations
    from detectron2.structures import Instances
    locs = compute_locations(128, 128, 8, "cpu")
    I = Instances((0, 0))
    I.locations, I.fpn_levels = locs, torch.zeros(len(locs), dtype=torch.long)
    I.im_inds = torch.zeros(len(locs), dtype=torch.long)
    ign = P.canopy_ignore(I, [cr["canopy"]], [8]).reshape(128, 128).numpy()
    assert np.array_equal(ign, lace), "P1 cell rule != LACE canopy_cell_mask"
    fo._tierb_canopy = canopy
    r = P._fcos_losses(fo, _detached(inst))
    n_ign = fo._tierb_n_ignored
    fo._tierb_canopy = None
    ck["G6_canopy_ignore"] = {"equal_to_released_without_canopy": True, "zero_canopy_max_abs_diff": zdiff,
                              "lace_cell_rule_exact": True, "lace_cells_canopy": int(lace.sum()),
                              "captured_batch_negatives_ignored": n_ign,
                              "loss_cls_released": float(a["loss_fcos_cls"]),
                              "loss_cls_with_ignore": float(r[1]["loss_fcos_cls"])}

    # G8 P6 assignment loop
    g8 = {"random_cases": 0, "real_cases": 0}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for trial in range(300):
        M, N = int(torch.randint(0, 121, (1,), generator=gen)), int(torch.randint(0, 151, (1,), generator=gen))
        iou = (torch.rand((M, N), generator=gen) * (torch.rand((M, N), generator=gen) > 0.6)).cuda()
        iou = torch.round(iou * 20) / 20                                     # many exact ties
        sc = (torch.round(torch.rand(M, generator=gen) * 10) / 10).cuda()
        ref = P.greedy_match_released(iou, sc, 0.5, 0.0)
        new = P.greedy_match_cpu(iou.cpu(), sc.cpu(), torch.argsort(sc, descending=True).cpu().tolist(), 0.5, 0.0)
        assert ref == new, (trial, M, N)
        g8["random_cases"] += 1
    if method == "boxteacher":
        from detectron2.structures import pairwise_iou
        for pred, gti in cap["teacher"]:
            if len(gti) == 0 or not pred.has("pred_masks"):
                continue
            iou = pairwise_iou(pred.pred_boxes.to("cuda"), gti.gt_boxes.to("cuda"))
            sc = pred.scores.cuda()
            ref = P.greedy_match_released(iou, sc, 0.5, 0.0)
            new = P.greedy_match_cpu(iou.cpu(), sc.cpu(), torch.argsort(sc, descending=True).cpu().tolist(), 0.5, 0.0)
            assert ref == new
            g8["real_cases"] += 1
    ck["G8_assignment"] = g8

    # G7 P5 postprocess equality (cap 64 so the released path cannot overflow) --------------
    m.eval()
    net = R.infer_net(m, method)
    cls = type(net)
    pp_patched = cls.postprocess
    seen = []

    def pp_capture(self, results, *a, **kw):
        seen.append((copy.deepcopy(results), a, kw))
        return pp_patched(self, results, *a, **kw)
    cls.postprocess = pp_capture
    tid = TC.manifest()["ids"]["val"][0]
    rgb = TC.load_rgb(tid)
    with R.caps(m, method, "whole"):
        for n in ([net.proposal_generator.fcos_outputs] if method == "boxinst"
                  else [m.teacher.proposal_generator.fcos_outputs]):
            n.post_nms_topk_test = 64
            n.pre_nms_thresh_test = 0.0
        R.run_window(m, method, rgb, masks=False)
    cls.postprocess = pp_patched
    res, pa, pkw = seen[-1]
    orig = P.ORIG["adet_postprocess" if method == "boxinst" else "bt_postprocess"]
    P.MASK_CHUNK = 16
    with torch.no_grad():
        o = orig(net, copy.deepcopy(res), *pa, **pkw)
        n_ = pp_patched(net, copy.deepcopy(res), *pa, **pkw)
    P.MASK_CHUNK = 64
    assert len(o) == len(n_) > 0
    assert torch.equal(o.pred_boxes.tensor, n_.pred_boxes.tensor)
    if method == "boxinst":
        assert torch.equal(o.scores, n_.scores)          # no reduction in the adet postprocess
        sdiff = 0.0
    else:   # BoxTeacher's mask_score is a CUDA sum whose block split depends on the chunk size
        sdiff = float(((o.scores - n_.scores).abs() / o.scores.abs().clamp(min=1e-12)).max())
        assert torch.allclose(o.scores, n_.scores, rtol=1e-6, atol=0), sdiff
    assert torch.equal(o.pred_masks.bool(), n_.pred_masks)
    if method == "boxteacher":
        assert torch.allclose(o.mask_score, n_.mask_score, rtol=1e-6, atol=0)
    ck["G7_postprocess"] = {"tile": tid, "n_instances": len(o), "chunk": 16,
                            "masks_boxes_bitwise_equal": True, "scores_max_rel_diff": sdiff}

    # G10 inference, both geometries, 2 val tiles --------------------------------------
    torch.cuda.reset_peak_memory_stats()
    inf = {}
    for tid in TC.manifest()["ids"]["val"][:2]:
        rgb = TC.load_rgb(tid)
        t0 = time.time()
        w = R.predict_tile(m, method, rgb, "whole", masks=True)
        t1 = time.time()
        s = R.predict_tile(m, method, rgb, "sub", masks=True)
        t2 = time.time()
        assert len(w["scores"]) <= 600 and len(w["masks_rle"]) == len(w["scores"])
        assert all(r["size"] == [2048, 2048] for r in w["masks_rle"])
        if method == "boxinst" and w["scores"]:
            assert min(w["scores"]) >= 0.05 - 1e-6, min(w["scores"])
        inf[tid] = {"whole_n": len(w["scores"]), "whole_s": round(t1 - t0, 2),
                    "sub_n": sum(len(v["scores"]) for v in s.values()), "sub_s": round(t2 - t1, 2)}
    R.check_released_test_attrs(m, method)
    ck["G10_inference"] = {"tiles": inf, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

    rep["pass"] = True
    rep["seconds"] = round(time.time() - t00, 1)
    TC.report(method, "gate", rep)
    TC.mark_done(method, "gate", {"pass": True})
    print(json.dumps(rep, indent=1, default=str), flush=True)
    print("GATE PASS", flush=True)
