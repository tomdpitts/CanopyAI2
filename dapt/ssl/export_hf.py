"""Native DAPT teacher .pth -> full HF AutoModel dir, registered as a dapt arm.

Steps: unmap native->HF keys (convert_dinov3_to_hf), load into a real web AutoModel
(catches any shape/key drift), save_pretrained + the web AutoImageProcessor
(ImageNet norm — the run-1 decision: dapt differs from web by WEIGHTS ONLY), then
register the dir in dapt/ssl/checkpoints.json so dapt/backbone.py picks it up as
an arm with zero pipeline change. Weights are cast to fp32 to match how the web
checkpoint flows through feature extraction.

Sanity gate: the exported model's features must DIFFER from web's (adaptation
happened) but not be garbage (cosine to web within (lo, hi) band) on a real tile.

Usage:
    .venv/bin/python -m dapt.ssl.export_hf --teacher dapt/ssl/export/teacher_s0_i4999.pth
    # -> dapt/ckpt/dapt_s0_i4999_hf/, registered as arm "dapt_s0_i4999"
"""
import argparse
import json
import os
import re

import torch

from dapt.data.cohort import REPO
from dapt.ssl.convert_dinov3_to_hf import unmap

WEB_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REGISTRY = os.path.join(REPO, "dapt/ssl/checkpoints.json")


def export(teacher_pth: str, out_dir: str | None = None, register: bool = True,
           cos_band=(0.15, 0.9999)):
    # cos lower bound = CORRUPTION tripwire only. Run-1 (hot LR 7e-5) showed real
    # adaptation drifts smoothly to cos≈0.385 by iter ~3000 — a smooth monotone decay
    # is training, not corruption; corruption looks like ~0/NaN/random. Whether the
    # drifted features are BETTER is the val probes' question, not this gate's.
    from transformers import AutoImageProcessor, AutoModel

    name = re.sub(r"^teacher_", "dapt_", os.path.splitext(
        os.path.basename(teacher_pth))[0])
    out_dir = out_dir or os.path.join(REPO, "dapt/ckpt", name + "_hf")

    sd = torch.load(teacher_pth, map_location="cpu")["teacher"]
    native = {k[len("backbone."):]: v.float() for k, v in sd.items()
              if k.startswith("backbone.")}
    if native["mask_token"].dim() == 1:            # model stores (C,); unmap wants 2D
        native["mask_token"] = native["mask_token"].reshape(1, -1)
    hf_sd = unmap(native)

    model = AutoModel.from_pretrained(WEB_ID)
    msd = model.state_dict()
    load = {}
    for k in msd:                                   # live model prefixes 'model.'
        base = k[len("model."):] if k.startswith("model.") else k
        if base in hf_sd:
            load[k] = hf_sd[base].reshape(msd[k].shape)
    missing, unexpected = model.load_state_dict(load, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    n_missing_params = [m for m in missing if not m.endswith(
        ("position_ids",)) and "rope" not in m]
    print(f"loaded {len(load)} tensors; missing(non-param/alias)={missing[:3]}")

    # sanity gate: adapted features should differ from web but stay on-manifold
    import glob
    import torch.nn.functional as F
    web = AutoModel.from_pretrained(WEB_ID).eval()
    model = model.eval()
    tile = sorted(glob.glob(os.path.join(REPO, "dapt/ssl/pool/tiles/*/*.png")))[0]
    from PIL import Image
    import numpy as np
    img = Image.open(tile).convert("RGB").resize((256, 256))
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / \
        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        a = model(pixel_values=x).last_hidden_state
        b = web(pixel_values=x).last_hidden_state
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1)).item()
    print(f"feature cosine vs web on {os.path.basename(tile)}: {cos:.4f}")
    assert cos_band[0] < cos, f"cosine {cos:.3f} too LOW — export corrupted?"
    if cos > cos_band[1]:
        print("WARNING: features ~identical to web — did adaptation train at all?")

    model.save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(WEB_ID).save_pretrained(out_dir)
    print(f"wrote HF dir {os.path.relpath(out_dir, REPO)}")

    if register:
        reg = json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}
        reg[name] = os.path.relpath(out_dir, REPO)
        json.dump(reg, open(REGISTRY, "w"), indent=2, sort_keys=True)
        print(f"registered arm {name!r} in dapt/ssl/checkpoints.json")
    return name, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    export(a.teacher, a.out, register=not a.no_register)
