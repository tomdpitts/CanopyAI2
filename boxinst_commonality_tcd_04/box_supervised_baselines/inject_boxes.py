"""Box-conditioned inference for CondInst/BoxInst, built from the method's OWN assigner.

WHY THIS EXISTS. BoxInst has no box prompt: its mask head is conditioned on a feature-map
LOCATION (a controller vector read at one FPN cell), never on box coordinates. So it cannot
enter the GT-box bake-off the way SAM or LACE can. But its training-time assigner is a pure
function of a box -- `get_targets` picks the FPN level from box size via `regress_ranges` and
the locations from the box extent / centre-sampling radius, using no model prediction at all.
Running that assigner at INFERENCE therefore yields the location the method would itself hold
responsible for a given box, which is as close to "prompting" as the architecture admits.

    box -> get_targets  -> (level, location) -> controller vector -> dynamic conv -> mask

THIS IS OUR ADAPTATION, NOT THE AUTHORS'. It must never be reported without the round-trip
check below, and never without the native system number beside it.

THE ROUND-TRIP GATE. Feed the model its OWN detected boxes and compare the injected masks with
the masks it natively produced for those same detections. This tests the code path, not the
weights, so it is meaningful on an untrained model -- an untrained CondInst emits garbage masks,
but faithful injection must emit the SAME garbage. A high median IoU means the injected mask is
genuinely "what the model says about this box"; a low one means the two paths disagree and any
injected-box comparison would be measuring our adaptation rather than the method.

Note the one way this can legitimately fail: native inference selects a location by peak
classification score, whereas the assigner selects by box geometry. Where those disagree the
masks differ for a real reason, not a bug -- which is precisely why the gate is empirical.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _assign(bbox_head, featmap_sizes, boxes, device, dtype):
    """Injected boxes -> one responsible (flat index, point, level) each, via THEIR assigner.

    Returns (sel_idx, points, level_inds, kept) where `kept` indexes the boxes that received
    any location at all. A box can receive none -- if its size falls outside every level's
    regress_range, or centre-sampling excludes every cell -- and such a box has no defined
    mask under this method. Those are reported, never silently dropped."""
    pts = bbox_head.prior_generator.grid_priors(featmap_sizes, dtype, device)
    n = boxes.size(0)
    labels_in = boxes.new_zeros(n, dtype=torch.long)          # single class
    labels, bbox_targets, gt_inds = bbox_head.get_targets(pts, [boxes], [labels_in])

    flat_labels = torch.cat(labels)
    flat_targets = torch.cat(bbox_targets)
    flat_gt = torch.cat(gt_inds)
    flat_points = torch.cat(pts)
    lvl = torch.cat([torch.full((p.size(0),), i, device=device, dtype=torch.long)
                     for i, p in enumerate(pts)])

    pos = (flat_labels >= 0) & (flat_labels < bbox_head.num_classes)
    # Centre-most assigned cell = the one the method weights most heavily for this box; the
    # same quantity that weights its regression loss.
    ctr = torch.zeros_like(flat_labels, dtype=torch.float32)
    if pos.any():
        ctr[pos] = bbox_head.centerness_target(flat_targets[pos])

    sel, kept = [], []
    for i in range(n):
        m = pos & (flat_gt == i)
        if not m.any():
            continue
        cand = m.nonzero().reshape(-1)
        sel.append(int(cand[torch.argmax(ctr[cand])]))
        kept.append(i)
    if not sel:
        return None
    sel = torch.as_tensor(sel, device=device, dtype=torch.long)
    return sel, flat_points[sel], lvl[sel], torch.as_tensor(kept, device=device,
                                                            dtype=torch.long)


@torch.no_grad()
def masks_for_boxes(model, img, img_metas, boxes, rescale=True):
    """-> (masks, kept_box_indices). `masks` are the model's mask-head output for `boxes`.

    Mirrors `CondInst.simple_test` exactly, substituting only WHERE the controller vectors are
    read from: the assigner's responsible location instead of the detector's peak."""
    feats = model.extract_feat(img)
    cls_scores, bbox_preds, centernesses, param_preds = model.bbox_head.forward(
        feats, model.mask_head.param_conv)
    featmap_sizes = [c.size()[-2:] for c in cls_scores]
    device, dtype = bbox_preds[0].device, bbox_preds[0].dtype

    got = _assign(model.bbox_head, featmap_sizes, boxes.to(device).to(dtype), device, dtype)
    if got is None:
        return [], torch.zeros(0, dtype=torch.long)
    sel, points, level_inds, kept = got

    # param_preds are per-level (B, C, H, W); flatten to (sum HW, C) in the SAME level-major
    # order the assigner's points use, so `sel` indexes both consistently. Batch size 1.
    flat_params = torch.cat([p[0].permute(1, 2, 0).reshape(-1, p.size(1))
                             for p in param_preds])
    det_params = flat_params[sel]
    det_labels = torch.zeros(det_params.size(0), device=device, dtype=torch.long)

    mask_feat = model.mask_branch(feats)
    segm = model.mask_head.simple_test(
        mask_feat, [det_labels], [det_params], [points], [level_inds],
        img_metas, model.bbox_head.num_classes, rescale=rescale)
    return segm[0][0], kept.cpu()          # single image, single class


@torch.no_grad()
def round_trip(model, img, img_metas, score_thr=0.05, max_inst=64):
    """Inject the model's OWN detections and compare against its native masks.

    -> {"n", "median_iou", "mean_iou", "frac_ge_0.95", "unassigned"} or None if the model
    detected nothing at this threshold."""
    from pycocotools import mask as mu

    native = model.simple_test(img, img_metas, rescale=True)[0]
    bboxes, segms = native[0][0], native[1][0]
    if len(bboxes) == 0:
        return None
    keep = [i for i, b in enumerate(bboxes) if float(b[4]) >= score_thr][:max_inst]
    if not keep:
        return None
    boxes = torch.as_tensor([bboxes[i][:4] for i in keep], dtype=torch.float32)

    inj, kept = masks_for_boxes(model, img, img_metas, boxes)
    if len(inj) == 0:
        return {"n": 0, "median_iou": None, "unassigned": len(keep)}

    def _rle(m):
        import numpy as np
        if isinstance(m, dict):
            c = m["counts"]
            return {"size": m["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}
        return mu.encode(np.asfortranarray(np.asarray(m).astype("uint8")))

    ious = []
    for j, bi in enumerate(kept.tolist()):
        a, b = _rle(inj[j]), _rle(segms[keep[bi]])
        inter = float(mu.area(mu.merge([a, b], intersect=1)))
        union = float(mu.area(a)) + float(mu.area(b)) - inter
        ious.append(inter / union if union > 0 else 0.0)
    import numpy as np
    v = np.asarray(ious)
    return {"n": int(len(v)), "median_iou": round(float(np.median(v)), 4),
            "mean_iou": round(float(v.mean()), 4),
            "frac_ge_0.95": round(float((v >= 0.95).mean()), 4),
            "unassigned": int(len(keep) - len(kept))}
