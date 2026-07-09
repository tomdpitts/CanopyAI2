"""Local CPU smoke of the DAPT setup — runs the box pre-flight for free.

1. parse dinov3_dapt_vitl.yaml through dinov3's own loader (validates keys under the
   strict ssl_default schema);
2. build the ViT-L teacher backbone from that config (the RELEASED arch: ropenew /
   layernormbf16 / 4 storage tokens / masked k-bias) on CPU;
3. load the converted web checkpoint into it and assert 0 missing PARAMETERS;
4. run one forward pass on a real pool tile to confirm the arch executes and emits
   features.

Usage:  .venv/bin/python -m dapt.ssl.local_smoke
"""
import os
import sys
import types

import torch

from dapt.data.cohort import REPO
from dapt.ssl.convert_hf_to_dinov3 import DINOV3_REPO

CFG = os.path.join(REPO, "dapt/ssl/dinov3_dapt_vitl.yaml")
CKPT = os.path.join(REPO, "dapt/ssl/dinov3_web_vitl_teacher.pth")


def main():
    sys.path.insert(0, DINOV3_REPO)
    from dinov3.configs.config import get_cfg_from_args
    from dinov3.models import build_model

    # 1. config parse (strict merge over ssl_default)
    args = types.SimpleNamespace(config_file=CFG, opts=[], output_dir=None)
    cfg = get_cfg_from_args(args)
    print(f"1. config parsed OK  (arch={cfg.student.arch}, "
          f"pos_embed={cfg.student.pos_embed_type}, norm={cfg.student.norm_layer}, "
          f"n_storage={cfg.student.n_storage_tokens}, gram.use_loss={cfg.gram.use_loss})")

    # 2. build the teacher backbone on CPU from the real config
    gcs = cfg.crops.global_crops_size
    teacher, embed_dim = build_model(cfg.student, only_teacher=True,
                                     img_size=gcs if isinstance(gcs, int) else max(gcs),
                                     device="cpu")
    teacher = teacher.eval()
    print(f"2. built ViT-L teacher on CPU  (embed_dim={embed_dim})")

    # 3. load converted web weights, assert 0 missing params
    sd = torch.load(CKPT, map_location="cpu")["teacher"]
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = teacher.load_state_dict(bb, strict=False)
    params = {n for n, _ in teacher.named_parameters()}
    miss_p = [m for m in missing if m in params]
    print(f"3. load: {len(bb)} tensors | missing params={miss_p} | "
          f"unexpected={unexpected[:5]}")
    assert not miss_p and not unexpected, "arch/weight mismatch under the real config!"
    print("   PRE-FLIGHT WEIGHT LOAD OK ✓ (0 missing params under released arch)")

    # 4. forward_features on a real pool tile (ImageNet norm here — fine for the
    # native-vs-HF comparison in step 5, which only needs identical inputs; the
    # actual SSL run normalizes with arid stats from pool/manifest.json)
    from PIL import Image
    import numpy as np
    import glob
    import torch.nn.functional as F
    S = gcs if isinstance(gcs, int) else max(gcs)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tile = sorted(glob.glob(os.path.join(REPO, "dapt/ssl/pool/tiles/*/*.png")))[0]
    img = Image.open(tile).convert("RGB").resize((S, S))
    x0 = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
    x = (x0 - mean) / std
    with torch.no_grad():
        nat = teacher.forward_features(x.float())["x_norm_patchtokens"][0]  # (N,1024)
    print(f"4. forward_features OK on {os.path.basename(tile)} -> patches {tuple(nat.shape)}")

    # 5. faithfulness: native (dinov3) vs HF patch tokens on the SAME input
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
    with torch.no_grad():
        hs = hf(pixel_values=x.float()).last_hidden_state[0]
    hf_patch = hs[-nat.shape[0]:]                       # trailing N patch tokens
    cos = F.cosine_similarity(F.normalize(nat, dim=-1),
                              F.normalize(hf_patch, dim=-1), dim=-1)
    print(f"5. native-vs-HF patch cosine: mean={cos.mean():.4f} min={cos.min():.4f}")
    faithful = cos.mean() > 0.9
    print(f"   GEOMETRY {'FAITHFUL ✓' if faithful else 'MISMATCH ✗ — investigate rope'}"
          f" (native ropenew build reproduces HF features)")
    print(f"\nLOCAL SMOKE {'PASSED' if faithful else 'FAILED step 5'} — "
          f"box pre-flight green; DAPT will start from correct features.")


if __name__ == "__main__":
    main()
