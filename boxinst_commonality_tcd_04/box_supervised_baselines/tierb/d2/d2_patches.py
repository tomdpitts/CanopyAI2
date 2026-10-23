"""Source patches for B1 (BoxInst, AdelaiDet) and B2 (BoxTeacher), both from the BoxTeacher repo
pinned at 6692d92 (it vendors AdelaiDet; B1 uses that copy, whose BoxInst path is identical to
upstream AdelaiDet -- only fcos.py/fcos_outputs.py gained a `forward_test`, which BoxInst never
calls). Applied by monkeypatching at import, never by editing the clone, and refused unless every
target file's sha256 equals the pinned file's.

Two kinds of patch, kept apart:

PROTOCOL (changes what is computed; each is a declared Tier B rule, TIERB.md section 3):
  P1  Canopy is IGNORED, not background, in the FCOS classification loss -- LACE's own rule
      (dapt/head.py focal_heatmap_loss): a location whose stride cell is >50% canopy contributes
      no NEGATIVE loss; positives are always supervised; the normaliser is unchanged.
  P4  BoxTeacher's teacher sees each training crop at the TEST scale, as released: there the
      student image is resized to short side 640-800 and the teacher re-resizes it to 800, i.e.
      x 1/s. Released code derives the target from MIN/MAX_SIZE_TEST, which on our crops would be
      the wrong number, so the per-crop factor 1/s is passed explicitly (`tierb_scale`).

NEUTRAL (must not change any number; each is proved equal by a gate in d2_run.py `gate`):
  P2  stashes the per-image canopy rasters where P1 can read them (training only).
  P3  crops with no crown (15.7% of quadrants; LACE trains on these pixels too) crash the released
      code (`torch.stack`/`torch.cat` of an empty list, upstream BoxTeacher issue #7). Each site
      gets the correctly-shaped empty tensor instead.
  P5  mask post-processing in chunks of instances: at 2048^2 x 600 detections the released code
      overflows int32 indexing in `aligned_bilinear` (BoxTeacher issue #11) and needs ~10 GB per
      tile. Per-instance arithmetic is unchanged; masks are returned as bool (the only consumer
      thresholds them anyway).
  P6  BoxTeacher's pseudo-label assignment loop reads GPU scalars one at a time (~3 syncs per
      (pred, gt) pair; ~15k pairs on a dense crop). The identical loop runs on CPU copies.
  P7  BoxTeacher's mask head all-reduces two counters ONLY when its GPU has proposals; a GPU whose
      two crops are both empty (~18% of 8-GPU iterations here) would skip them and pair its DDP
      gradient all-reduce with the other ranks' counters -> hang or corrupted gradients. In that
      case the same two 1-element all-reduces are issued (values discarded). Numerically neutral;
      the gate checks the collective count is identical for dense, mixed and all-empty batches.
"""
from __future__ import annotations

import hashlib
import inspect
import math

import torch
import torch.nn.functional as F

EXPECTED_SHA256 = {    # files at BoxTeacher@6692d92 (computed from the pinned clone)
    "adet.modeling.fcos.fcos_outputs":
        "81f608098f968b6c916eecc355f7c50454ea6347ddb1656c378c053df994601e",
    "adet.modeling.condinst.condinst":
        "7ae1a8920ad21c3a814c96d30453df4c40db1bb1b3bf7be67cb02f340b886ece",
    "boxteacher.condinst":
        "4ed8fb3c7935b78849a91e0c2d5c7e24627203c6aa7e4fedb0ad2fa19f028bb4",
    "boxteacher.boxteacher":
        "db9da433247a82008c28a0854f8e2e50f20a95a7565c3e7f457ea8a80950251e",
}
MASK_CHUNK = 64
APPLIED = {}
ORIG = {}          # released functions, kept for the equality gates (d2_gate.py)


def verify_sources():
    import importlib
    out = {}
    for mod, want in EXPECTED_SHA256.items():
        m = importlib.import_module(mod)
        got = hashlib.sha256(open(inspect.getsourcefile(m), "rb").read()).hexdigest()
        assert got == want, f"{mod}: source sha256 {got[:12]} != pinned {want[:12]}; refusing to patch"
        out[mod] = got
    return out


# ------------------------------------------------------------------ P1 canopy ignore
def canopy_ignore(instances, canopy, strides):
    """Bool per FCOS location: its stride cell is >50% canopy (LACE's canopy_cell_mask rule).
    Locations are cell centres x = j*s + s//2 (adet.utils.comm.compute_locations)."""
    locs, lv, im = instances.locations, instances.fpn_levels, instances.im_inds
    ign = torch.zeros(len(locs), dtype=torch.bool, device=locs.device)
    for l, s in enumerate(strides):
        sel = lv == l
        if not bool(sel.any()):
            continue
        xs = torch.div(locs[sel, 0] - s // 2, s, rounding_mode="floor").long()
        ys = torch.div(locs[sel, 1] - s // 2, s, rounding_mode="floor").long()
        H, W = int(ys.max()) + 1, int(xs.max()) + 1
        maps = torch.zeros((len(canopy), H, W), device=locs.device)
        for i, c in enumerate(canopy):
            p = F.avg_pool2d(c[None, None].float(), kernel_size=s, stride=s, ceil_mode=True)[0, 0]
            h, w = min(p.shape[0], H), min(p.shape[1], W)
            maps[i, :h, :w] = p[:h, :w]
        ign[sel] = maps[im[sel], ys, xs] > 0.5
    return ign


def _fcos_losses(self, instances):
    """adet FCOSOutputs.fcos_losses @6692d92, verbatim except the marked canopy lines."""
    from adet.utils.comm import compute_ious, reduce_mean
    from adet.modeling.fcos.fcos_outputs import compute_ctrness_targets
    from fvcore.nn import sigmoid_focal_loss_jit
    losses, extras = {}, {}

    # 1. compute the cls loss
    num_classes = instances.logits_pred.size(1)
    assert num_classes == self.num_classes

    labels = instances.labels.flatten()

    pos_inds = torch.nonzero(labels != num_classes).squeeze(1)

    num_pos_local = torch.ones_like(pos_inds).sum()
    num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

    # prepare one_hot
    class_target = torch.zeros_like(instances.logits_pred)
    class_target[pos_inds, labels[pos_inds]] = 1

    canopy = getattr(self, "_tierb_canopy", None)                          # TIERB P1
    if canopy is None:                                                      # TIERB P1
        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum"
        )
    else:                                                                   # TIERB P1
        neg_ign = canopy_ignore(instances, canopy, self.strides) & (labels == num_classes)
        self._tierb_n_ignored = int(neg_ign.sum())
        el = sigmoid_focal_loss_jit(
            instances.logits_pred, class_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
        class_loss = (el * (~neg_ign).to(el.dtype)[:, None]).sum()

    if self.loss_normalizer_cls == "moving_fg":
        self.moving_num_fg = self.moving_num_fg_momentum * self.moving_num_fg + (
                1 - self.moving_num_fg_momentum
        ) * num_pos_avg
        class_loss = class_loss / self.moving_num_fg
    elif self.loss_normalizer_cls == "fg":
        class_loss = class_loss / num_pos_avg
    else:
        num_samples_local = torch.ones_like(labels).sum()
        num_samples_avg = max(reduce_mean(num_samples_local).item(), 1.0)
        class_loss = class_loss / num_samples_avg

    losses["loss_fcos_cls"] = class_loss * self.loss_weight_cls

    # 2. compute the box regression and quality loss
    instances = instances[pos_inds]
    instances.pos_inds = pos_inds

    ious, gious = compute_ious(instances.reg_pred, instances.reg_targets)

    if self.box_quality == "ctrness":
        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        instances.gt_ctrs = ctrness_targets

        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_mean(ctrness_targets_sum).item(), 1e-6)
        extras["loss_denorm"] = loss_denorm

        reg_loss = self.loc_loss_func(ious, gious, ctrness_targets) / loss_denorm
        losses["loss_fcos_loc"] = reg_loss

        ctrness_loss = F.binary_cross_entropy_with_logits(
            instances.ctrness_pred, ctrness_targets,
            reduction="sum"
        ) / num_pos_avg
        losses["loss_fcos_ctr"] = ctrness_loss
    elif self.box_quality == "iou":
        reg_loss = self.loc_loss_func(ious, gious) / num_pos_avg
        losses["loss_fcos_loc"] = reg_loss

        quality_loss = F.binary_cross_entropy_with_logits(
            instances.ctrness_pred, ious.detach(),
            reduction="sum"
        ) / num_pos_avg
        losses["loss_fcos_iou"] = quality_loss
    else:
        raise NotImplementedError

    extras["instances"] = instances

    return extras, losses


# ------------------------------------------------------------------ P2 canopy hand-off
def _wrap_forward(cls):
    orig = cls.forward

    def forward(self, batched_inputs):
        fo = self.proposal_generator.fcos_outputs
        fo._tierb_canopy = ([x["canopy"].to(self.device) for x in batched_inputs]
                            if self.training and "canopy" in batched_inputs[0] else None)
        try:
            return orig(self, batched_inputs)
        finally:
            fo._tierb_canopy = None
    forward.__tierb_wrapped__ = orig
    cls.forward = forward


# ------------------------------------------------------------------ P3 empty crops
def _adet_add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
    """adet CondInst.add_bitmasks_from_boxes @6692d92 + an early branch for 0 boxes."""
    from skimage import color
    from adet.modeling.condinst.condinst import get_images_color_similarity
    stride = self.mask_out_stride
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0

    downsampled_images = F.avg_pool2d(
        images.float(), kernel_size=stride,
        stride=stride, padding=0
    )[:, [2, 1, 0]]
    image_masks = image_masks[:, start::stride, start::stride]

    for im_i, per_im_gt_inst in enumerate(instances):
        images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
        images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
        images_lab = images_lab.permute(2, 0, 1)[None]
        images_color_similarity = get_images_color_similarity(
            images_lab, image_masks[im_i],
            self.pairwise_size, self.pairwise_dilation
        )

        per_im_boxes = per_im_gt_inst.gt_boxes.tensor
        if len(per_im_boxes) == 0:                                          # TIERB P3
            z = torch.zeros((0, im_h, im_w), device=self.device).float()
            per_im_gt_inst.gt_bitmasks = z[:, start::stride, start::stride]
            per_im_gt_inst.gt_bitmasks_full = z
            per_im_gt_inst.image_color_similarity = images_color_similarity[:0]
            continue
        per_im_bitmasks = []
        per_im_bitmasks_full = []
        for per_box in per_im_boxes:
            bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
            bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

            bitmask = bitmask_full[start::stride, start::stride]

            assert bitmask.size(0) * stride == im_h
            assert bitmask.size(1) * stride == im_w

            per_im_bitmasks.append(bitmask)
            per_im_bitmasks_full.append(bitmask_full)

        per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
        per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
        per_im_gt_inst.image_color_similarity = torch.cat([
            images_color_similarity for _ in range(len(per_im_gt_inst))
        ], dim=0)


def _bt_add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
    """boxteacher CondInst.add_bitmasks_from_boxes @6692d92 + 0-box branch."""
    from boxteacher.condinst import get_images_color_similarity
    from boxteacher.utils import rgb2lab
    stride = self.mask_out_stride
    start = int(stride // 2)

    assert images.size(2) % stride == 0
    assert images.size(3) % stride == 0

    downsampled_images = F.avg_pool2d(
        images.float(), kernel_size=stride,
        stride=stride, padding=0
    )
    # RGB to BGR
    downsampled_images = downsampled_images[:, [2, 1, 0]]

    image_masks = image_masks[:, start::stride, start::stride]
    # float to byte to float
    downsampled_images_lab = rgb2lab(downsampled_images.byte().float())
    for im_i, per_im_gt_inst in enumerate(instances):
        images_lab = downsampled_images_lab[im_i][None]
        images_color_similarity = get_images_color_similarity(
            images_lab, image_masks[im_i],
            self.pairwise_size, self.pairwise_dilation,
        )
        per_im_boxes = per_im_gt_inst.gt_boxes.tensor

        bitmasks_full = torch.zeros((len(per_im_boxes), im_h, im_w)).to(self.device).float()
        for idx, per_box in enumerate(per_im_boxes):
            bitmasks_full[idx, int(per_box[1]):int(per_box[3] + 1),
                          int(per_box[0]):int(per_box[2] + 1)] = 1.0

        bit_masks = bitmasks_full[:, start::stride, start::stride]
        per_im_gt_inst.boxinst_gt_bitmasks = bit_masks
        per_im_gt_inst.boxinst_gt_bitmasks_full = bitmasks_full
        if len(per_im_gt_inst) == 0:                                        # TIERB P3
            per_im_gt_inst.boxinst_image_color_similarity = images_color_similarity[:0]
            continue
        per_im_gt_inst.boxinst_image_color_similarity = torch.cat([
            images_color_similarity for _ in range(len(per_im_gt_inst))
        ], dim=0)


def _bt_add_pseudo_bitmasks(self, instances, im_h, im_w, ori_sizes):
    """boxteacher CondInst.add_pseudo_bitmasks @6692d92 + 0-instance branch."""
    for per_im_gt_inst, ori_size in zip(instances, ori_sizes):
        start = int(self.mask_out_stride // 2)
        if not per_im_gt_inst.has("gt_bitmasks"):
            gt_masks_flags = per_im_gt_inst.gt_masks_flags
            if gt_masks_flags.size(0) > 0:
                bitmasks_full = torch.zeros(
                    (gt_masks_flags.size(0), im_h, im_w)).to(self.device)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                         start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
            else:                                                           # TIERB P3
                z = torch.zeros((0, im_h, im_w)).to(self.device)
                per_im_gt_inst.gt_bitmasks = z[:, start::self.mask_out_stride,
                                               start::self.mask_out_stride]
            continue

        bitmasks = per_im_gt_inst.get("gt_bitmasks")

        h, w = bitmasks.size()[1:]
        bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
        bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                 start::self.mask_out_stride]
        per_im_gt_inst.gt_bitmasks = bitmasks


def _bt_pseudo_mask_pairwise_affinity(self, instances, image_masks):
    """boxteacher CondInst.pseudo_mask_pairwise_affinity @6692d92 + 0-instance branch."""
    from boxteacher.condinst import get_instance_masks_affinity
    stride = self.mask_out_stride
    start = int(stride // 2)
    image_masks = image_masks[:, start::stride, start::stride]
    for im_i, per_im_gt_inst in enumerate(instances):
        valid_gt_bitmasks = per_im_gt_inst.gt_bitmasks
        if len(valid_gt_bitmasks) == 0:                                     # TIERB P3
            k = self.pairwise_size ** 2 - 1
            h, w = image_masks[im_i].shape
            per_im_gt_inst.gt_mask_affinity = valid_gt_bitmasks.new_zeros((0, k, h, w)).float()
            per_im_gt_inst.gt_bitmasks = per_im_gt_inst.gt_bitmasks > self.teacher_mask_threshold
            continue
        mask_affinity = get_instance_masks_affinity(
            valid_gt_bitmasks, image_masks[im_i],
            self.pairwise_size, self.pairwise_dilation)
        per_im_gt_inst.gt_mask_affinity = mask_affinity
        # binary masks
        per_im_gt_inst.gt_bitmasks = per_im_gt_inst.gt_bitmasks > self.teacher_mask_threshold


# ------------------------------------------------------------------ P4 teacher scale
def _bt_forward_teacher(self, batched_inputs):
    """boxteacher CondInst.forward_teacher @6692d92; the ONLY change is where the resize target
    comes from: x 1/s from `tierb_scale` (released semantics: student -> test scale)."""
    from detectron2.structures import ImageList
    from boxteacher.condinst import resize_images
    original_images = [x["image"].to(self.device) for x in batched_inputs]
    images_norm = [self.normalizer(x) for x in original_images]
    if all("tierb_scale" in x for x in batched_inputs):                     # TIERB P4
        images_norm, image_sizes = _resize_by_scale(images_norm,
                                                    [x["tierb_scale"] for x in batched_inputs])
    else:
        images_norm, image_sizes = resize_images(images_norm, self.resize_size)

    images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

    features = self.backbone(images_norm.tensor)

    gt_instances = None
    mask_feats, _ = self.mask_branch(features, gt_instances)

    proposals, _ = self.proposal_generator.forward_test(
        images_norm, features, gt_instances, self.controller)

    padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
    processed_results = []
    for im_id, (input_per_image, proposal, image_size) in enumerate(zip(batched_inputs, proposals, image_sizes)):

        height = image_size[0]
        width = image_size[1]
        instances_per_im = self._forward_mask_heads_test(
            [proposal], mask_feats[im_id].unsqueeze(0))

        teacher_threshold = self.teacher_mask_threshold
        if self.return_float_masks:
            teacher_threshold = -1.0

        instances_per_im = self.postprocess(
            instances_per_im, height, width,
            padded_im_h, padded_im_w, teacher_threshold
        )
        processed_results.append({
            "instances": instances_per_im
        })

    return processed_results


def _resize_by_scale(images, scales):
    """Same op as boxteacher `resize_images` (bilinear, align_corners=False, int(x+0.5)), with
    the target = current size / s."""
    out, sizes = [], []
    for img, s in zip(images, scales):
        h, w = img.shape[-2:]
        newh, neww = int(h / float(s) + 0.5), int(w / float(s) + 0.5)
        out.append(F.interpolate(img.unsqueeze(0), (newh, neww), mode="bilinear",
                                 align_corners=False).squeeze(0))
        sizes.append((h, w))
    return out, sizes


# ------------------------------------------------------------------ P5 chunked postprocess
def _adet_postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w,
                      mask_threshold=0.5):
    """adet CondInst.postprocess @6692d92, masks in chunks (bool)."""
    from detectron2.structures import Instances
    from adet.utils.comm import aligned_bilinear
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    resized_im_h, resized_im_w = results.image_size
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_global_masks"):
        mask_h, mask_w = results.pred_global_masks.size()[-2:]
        factor_h = padded_im_h // mask_h
        factor_w = padded_im_w // mask_w
        assert factor_h == factor_w
        factor = factor_h
        chunks = []
        for c in range(0, len(results), MASK_CHUNK):                       # TIERB P5
            g = aligned_bilinear(results.pred_global_masks[c:c + MASK_CHUNK], factor)
            g = g[:, :, :resized_im_h, :resized_im_w]
            g = F.interpolate(g, size=(output_height, output_width),
                              mode="bilinear", align_corners=False)[:, 0, :, :]
            chunks.append(g > mask_threshold)
        results.pred_masks = (torch.cat(chunks) if chunks else torch.zeros(
            (0, output_height, output_width), dtype=torch.bool, device=results.pred_global_masks.device))
        results.remove("pred_global_masks")
    return results


def _bt_postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w,
                    mask_threshold=0.50):
    """boxteacher CondInst.postprocess @6692d92, masks in chunks. Scores are computed exactly as
    released: scores = sqrt(sqrt-FCOS-score * mean in-mask probability)."""
    from detectron2.structures import Instances
    from adet.utils.comm import aligned_bilinear
    scale_x, scale_y = (
        output_width / results.image_size[1], output_height / results.image_size[0])
    resized_im_h, resized_im_w = results.image_size
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_global_masks"):
        mask_h, mask_w = results.pred_global_masks.size()[-2:]
        factor_h = padded_im_h // mask_h
        factor_w = padded_im_w // mask_w
        assert factor_h == factor_w
        factor = factor_h
        masks, mscores = [], []
        for c in range(0, len(results), MASK_CHUNK):                       # TIERB P5
            g = aligned_bilinear(results.pred_global_masks[c:c + MASK_CHUNK], factor)
            g = g[:, :, :resized_im_h, :resized_im_w]
            g = F.interpolate(g, size=(output_height, output_width),
                              mode="bilinear", align_corners=False)[:, 0, :, :]
            binary = (g > mask_threshold).float()
            mscores.append((g * binary).sum([1, 2]) / (binary.sum([1, 2]) + 1e-6))
            masks.append(g if mask_threshold == -1 else binary.bool())
        dev = results.pred_global_masks.device
        mask_score = torch.cat(mscores) if mscores else torch.zeros((0,), device=dev)
        pred_scores = results.scores
        pred_scores_sqrt = pred_scores.sqrt()
        results.tierb_fcos_scores = pred_scores.clone()
        results.mask_score = (pred_scores_sqrt * mask_score).sqrt()
        results.scores = (pred_scores * mask_score).sqrt()
        results.pred_masks = (torch.cat(masks) if masks else torch.zeros(
            (0, output_height, output_width), device=dev,
            dtype=torch.float32 if mask_threshold == -1 else torch.bool))
        results.remove("pred_global_masks")
    return results


# ------------------------------------------------------------------ P6 pseudo-label loop
def greedy_match_cpu(iou, scores, order, iou_thr, score_thr):
    """The released assignment loop (BoxTeacher.pseudo_labeling) on CPU python values. `order`
    must come from the SAME `torch.argsort(scores, descending=True)` call as released (on the
    scores' own device), so ties resolve identically. Returns gtm (list, len N): matched prediction
    index per GT or -1. Keeps the released `gtm[j] > 0` test verbatim (a GT matched to prediction 0
    stays matchable) -- semantics, not a fix."""
    M, N = iou.shape
    iou_l = iou.tolist()
    sc = scores.tolist()
    gtm = [-1] * N
    for i in order:
        if sc[i] < score_thr:
            continue
        max_iou = -1
        m = -1
        row = iou_l[i]
        for j in range(N):
            iou_ = row[j]
            if gtm[j] > 0 or iou_ < iou_thr or iou_ < max_iou:
                continue
            max_iou = iou_
            m = j
        if m == -1:
            continue
        gtm[m] = i
    return gtm


def greedy_match_released(iou, scores, iou_thr, score_thr):
    """BoxTeacher.pseudo_labeling's loop, verbatim on tensors (reference for the P6 gate)."""
    M, N = iou.shape
    device = iou.device
    sort_index = torch.argsort(scores, descending=True)
    dtm = torch.zeros((M,), dtype=torch.long, device=device)
    gtm = torch.zeros((N,), dtype=torch.long, device=device) - 1
    biou = torch.zeros((N,), dtype=torch.float, device=device)
    for i in sort_index:
        if scores[i] < score_thr:
            continue
        max_iou = -1
        m = -1
        for j in range(N):
            iou_ = iou[i, j]
            if gtm[j] > 0 or iou_ < iou_thr or iou_ < max_iou:
                continue
            max_iou = iou_
            m = j
        if m == -1:
            continue
        dtm[i] = m
        gtm[m] = i
        biou[m] = max_iou
    return gtm.tolist()


def _bt_pseudo_labeling(self, batched_inputs, instances, iou_thr=0.5, score_thr=0.0):
    """BoxTeacher.pseudo_labeling @6692d92 with the assignment loop on CPU (P6)."""
    from detectron2.structures import pairwise_iou
    for idx, instances_per_im in enumerate(instances):
        gt_instances = batched_inputs[idx]["instances"]
        pred_instances = instances_per_im["instances"]
        M = len(pred_instances)
        N = len(gt_instances)
        if N == 0 or not pred_instances.has("pred_masks"):
            batched_inputs[idx]["instances"].gt_masks_flags = torch.zeros(
                (N,), dtype=torch.float, device=self.device)
            continue
        pred_masks = pred_instances.pred_masks.to(self.device)

        gtboxes = gt_instances.gt_boxes.to(self.device)
        pred_boxes = pred_instances.pred_boxes.to(self.device)
        pred_scores = pred_instances.scores.to(self.device)
        iou = pairwise_iou(pred_boxes, gtboxes)

        sort_index = torch.argsort(pred_scores, descending=True)
        gtm = torch.tensor(greedy_match_cpu(iou.cpu(), pred_scores.cpu(),
                                            sort_index.cpu().tolist(), iou_thr, score_thr),
                           dtype=torch.long, device=self.device)            # TIERB P6

        new_instances = gt_instances
        new_masks_inds = gtm[gtm > -1]
        gt_masks = torch.zeros((N, pred_masks.shape[-2], pred_masks.shape[-1])).to(pred_masks)
        gt_masks[gtm > -1] = pred_masks[new_masks_inds]
        new_instances.gt_bitmasks = gt_masks
        new_instances.gt_masks_flags = (gtm > -1).float()
        new_instances.gt_masks_flags[gtm > -1] = pred_instances.mask_score[new_masks_inds]
        batched_inputs[idx]["instances"] = new_instances


# ------------------------------------------------------------------ P7 matched collectives
def _wrap_bt_mask_head_call(cls):
    orig = cls.__call__

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training and len(pred_instances) == 0:                      # TIERB P7
            from adet.utils.comm import reduce_mean
            reduce_mean(mask_feats.new_zeros([1]))      # = reduce_mean(num_instances_local)
            reduce_mean(mask_feats.new_zeros(()))       # = reduce_mean(num_valid_masks)
        return orig(self, mask_feats, mask_feat_stride, pred_instances, gt_instances)
    __call__.__tierb_wrapped__ = orig
    cls.__call__ = __call__


# ------------------------------------------------------------------ apply
def apply(method):
    """Install every patch for `method` in {'boxinst', 'boxteacher'}. Idempotent."""
    if APPLIED.get(method):
        return APPLIED[method]
    hashes = verify_sources()
    from adet.modeling.fcos.fcos_outputs import FCOSOutputs
    ORIG.setdefault("fcos_losses", FCOSOutputs.fcos_losses)
    FCOSOutputs.fcos_losses = _fcos_losses                                   # P1
    if method == "boxinst":
        from adet.modeling.condinst.condinst import CondInst
        ORIG.setdefault("adet_postprocess", CondInst.postprocess)
        ORIG.setdefault("adet_add_bitmasks_from_boxes", CondInst.add_bitmasks_from_boxes)
        _wrap_forward(CondInst)                                              # P2
        CondInst.add_bitmasks_from_boxes = _adet_add_bitmasks_from_boxes     # P3
        CondInst.postprocess = _adet_postprocess                             # P5
    elif method == "boxteacher":
        from boxteacher.condinst import CondInst as BTCondInst
        from boxteacher.boxteacher import BoxTeacher
        ORIG.setdefault("bt_postprocess", BTCondInst.postprocess)
        ORIG.setdefault("bt_forward_teacher", BTCondInst.forward_teacher)
        ORIG.setdefault("bt_pseudo_labeling", BoxTeacher.pseudo_labeling)
        _wrap_forward(BTCondInst)                                            # P2
        BTCondInst.add_bitmasks_from_boxes = _bt_add_bitmasks_from_boxes     # P3
        BTCondInst.add_pseudo_bitmasks = _bt_add_pseudo_bitmasks             # P3
        BTCondInst.pseudo_mask_pairwise_affinity = _bt_pseudo_mask_pairwise_affinity  # P3
        BTCondInst.forward_teacher = torch.no_grad()(_bt_forward_teacher)    # P4
        BTCondInst.postprocess = _bt_postprocess                             # P5
        BoxTeacher.pseudo_labeling = _bt_pseudo_labeling                     # P6
        from boxteacher.condinst import DynamicMaskHead as BTMaskHead
        _wrap_bt_mask_head_call(BTMaskHead)                                  # P7
    else:
        raise ValueError(method)
    APPLIED[method] = {"method": method, "source_sha256": hashes,
                       "patches": ["P1 canopy ignore (protocol)", "P2 canopy hand-off",
                                   "P3 empty crops", "P5 chunked postprocess"]
                       + (["P4 teacher at test scale (protocol)", "P6 CPU assignment loop",
                           "P7 matched collectives on empty GPUs"]
                          if method == "boxteacher" else [])}
    return APPLIED[method]
