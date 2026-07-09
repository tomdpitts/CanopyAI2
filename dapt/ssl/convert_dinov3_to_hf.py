"""Reverse converter: native dinov3 teacher backbone -> HF-style state_dict.

Purpose: after DAPT, load the adapted weights through the SAME HF/AutoModel path that
dapt/backbone.py uses for web/sat, so feature extraction is byte-identical across
arms. Inverts convert_hf_to_dinov3.remap (splits fused qkv, drops k-bias + bias_mask).

Round-trip self-test:  .venv/bin/python -m dapt.ssl.convert_dinov3_to_hf --selftest
(HF -> native -> HF, assert exact tensor equality, then feature-cosine via AutoModel.)
"""
import argparse
import os
import sys

import torch

from dapt.data.cohort import REPO
from dapt.ssl.convert_hf_to_dinov3 import DEPTH, load_hf, remap


def unmap(native):
    """native backbone keys (no 'backbone.' prefix) -> HF keys."""
    hf = {}
    hf["embeddings.patch_embeddings.weight"] = native["patch_embed.proj.weight"]
    hf["embeddings.patch_embeddings.bias"] = native["patch_embed.proj.bias"]
    hf["embeddings.cls_token"] = native["cls_token"]
    # HF mask_token is (1,1,C); native stores (1,C)
    hf["embeddings.mask_token"] = native["mask_token"].reshape(1, 1, -1)
    hf["embeddings.register_tokens"] = native["storage_tokens"]
    hf["norm.weight"] = native["norm.weight"]
    hf["norm.bias"] = native["norm.bias"]
    for i in range(DEPTH):
        s, d = f"blocks.{i}", f"layer.{i}"
        C = native[f"{s}.attn.qkv.weight"].shape[1]
        qw, kw, vw = native[f"{s}.attn.qkv.weight"].split(C, dim=0)
        qb, kb, vb = native[f"{s}.attn.qkv.bias"].split(C, dim=0)
        hf[f"{d}.attention.q_proj.weight"] = qw
        hf[f"{d}.attention.k_proj.weight"] = kw   # k bias dropped (HF has none)
        hf[f"{d}.attention.v_proj.weight"] = vw
        hf[f"{d}.attention.q_proj.bias"] = qb
        hf[f"{d}.attention.v_proj.bias"] = vb
        hf[f"{d}.attention.o_proj.weight"] = native[f"{s}.attn.proj.weight"]
        hf[f"{d}.attention.o_proj.bias"] = native[f"{s}.attn.proj.bias"]
        hf[f"{d}.mlp.up_proj.weight"] = native[f"{s}.mlp.fc1.weight"]
        hf[f"{d}.mlp.up_proj.bias"] = native[f"{s}.mlp.fc1.bias"]
        hf[f"{d}.mlp.down_proj.weight"] = native[f"{s}.mlp.fc2.weight"]
        hf[f"{d}.mlp.down_proj.bias"] = native[f"{s}.mlp.fc2.bias"]
        hf[f"{d}.layer_scale1.lambda1"] = native[f"{s}.ls1.gamma"]
        hf[f"{d}.layer_scale2.lambda1"] = native[f"{s}.ls2.gamma"]
        for k in (f"{d}.norm1.weight", f"{d}.norm1.bias",
                  f"{d}.norm2.weight", f"{d}.norm2.bias"):
            hf[k] = native[k.replace(d, s)]
    return hf


def convert(teacher_pth, out_pt):
    sd = torch.load(teacher_pth, map_location="cpu")["teacher"]
    native = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}
    torch.save(unmap(native), out_pt)
    print(f"wrote {out_pt}")


def selftest():
    hf0 = load_hf("web")
    native = remap(hf0)                      # HF -> native (adds bias_mask)
    # native stores mask_token flattened when loaded into the model; simulate
    native["mask_token"] = native["mask_token"].reshape(1, -1)
    hf1 = unmap(native)                      # native -> HF
    bad = [k for k in hf1 if not torch.equal(hf1[k].reshape(hf0[k].shape), hf0[k])]
    assert not bad, f"round-trip mismatch: {bad[:5]}"
    assert set(hf1) == set(hf0), (set(hf0) ^ set(hf1))
    print(f"round-trip exact-equal OK ({len(hf1)} tensors)")

    # feature check through the REAL AutoModel path used by dapt/backbone.py
    import torch.nn.functional as F
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = m(pixel_values=x).last_hidden_state
    # live AutoModel prefixes block keys with 'model.'; key against its own names
    ld = {}
    for k in m.state_dict():
        base = k[len("model."):] if k.startswith("model.") else k
        if base in hf1:
            ld[k] = hf1[base].reshape(m.state_dict()[k].shape)
    missing, unexpected = m.load_state_dict(ld, strict=False)
    assert not unexpected, unexpected[:5]
    print(f"AutoModel reload: matched {len(ld)}, missing={missing[:3]} (aliases ok)")
    with torch.no_grad():
        out = m(pixel_values=x).last_hidden_state
    cos = F.cosine_similarity(ref.flatten(1), out.flatten(1)).item()
    assert cos > 0.9999, cos
    print(f"feature cosine after round-trip: {cos:.6f}  SELFTEST PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--teacher", help="native teacher .pth (from DAPT run)")
    ap.add_argument("--out", help="output HF-style state_dict .pt")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        convert(a.teacher, a.out or a.teacher.replace(".pth", "_hf.pt"))
