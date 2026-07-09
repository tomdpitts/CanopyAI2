"""Convert a HuggingFace DINOv3 ViT-L checkpoint -> native dinov3 training checkpoint.

The dinov3 training code initializes a continued-SSL run from
`student.resume_from_teacher_chkpt=<file.pth>`, where the file must be
`{"teacher": {"backbone.<param>": tensor, ...}}` (see checkpointer.py
init_fsdp_model_from_checkpoint). HF ships the same weights under a different naming
scheme (split q/k/v, `layer.N`, `embeddings.*`), so we remap keys and fuse qkv.

Also serves as a LOCAL CPU smoke test of the mapping: it instantiates the real
dinov3 ViT-L and load_state_dict(strict=False), asserting no unexpected keys and only
the expected non-parameter buffers missing.

Usage:
    .venv/bin/python -m dapt.ssl.convert_hf_to_dinov3 --arm web
"""
import argparse
import glob
import os
import sys

import torch
from safetensors import safe_open

from dapt.data.cohort import REPO

# Local dinov3 clone, needed only for the load_state_dict self-test / local_smoke
# (Modal clones its own fresh copy). Set $DINOV3_REPO or clone to ~/.cache/dinov3:
#   git clone --depth 1 https://github.com/facebookresearch/dinov3 ~/.cache/dinov3
DINOV3_REPO = os.environ.get("DINOV3_REPO",
                             os.path.expanduser("~/.cache/dinov3"))
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
HF_DIR = {"web": "models--facebook--dinov3-vitl16-pretrain-lvd1689m",
          "sat": "models--facebook--dinov3-vitl16-pretrain-sat493m"}
DEPTH = 24


def load_hf(arm):
    p = glob.glob(os.path.join(HF_CACHE, HF_DIR[arm], "snapshots/*/model.safetensors"))[0]
    t = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            t[k] = f.get_tensor(k)
    return t


def remap(hf):
    """HF keys -> dinov3 backbone keys (no 'backbone.' prefix yet)."""
    out = {}
    out["patch_embed.proj.weight"] = hf["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf["embeddings.patch_embeddings.bias"]
    out["cls_token"] = hf["embeddings.cls_token"]
    out["mask_token"] = hf["embeddings.mask_token"]
    out["storage_tokens"] = hf["embeddings.register_tokens"]
    out["norm.weight"] = hf["norm.weight"]
    out["norm.bias"] = hf["norm.bias"]
    embed = hf["embeddings.cls_token"].shape[-1]
    for i in range(DEPTH):
        s, d = f"layer.{i}", f"blocks.{i}"
        out[f"{d}.norm1.weight"] = hf[f"{s}.norm1.weight"]
        out[f"{d}.norm1.bias"] = hf[f"{s}.norm1.bias"]
        out[f"{d}.norm2.weight"] = hf[f"{s}.norm2.weight"]
        out[f"{d}.norm2.bias"] = hf[f"{s}.norm2.bias"]
        # fuse split projections -> qkv (order q,k,v on dim 0)
        q, k, v = (hf[f"{s}.attention.{x}_proj.weight"] for x in "qkv")
        out[f"{d}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = hf.get(f"{s}.attention.q_proj.bias", torch.zeros(embed))
        vb = hf.get(f"{s}.attention.v_proj.bias", torch.zeros(embed))
        kb = hf.get(f"{s}.attention.k_proj.bias", torch.zeros(embed))  # absent -> 0
        out[f"{d}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        # LinearKMaskedBias needs its mask buffer (HF has no k-bias; dinov3 masks it).
        # HF ships no bias_mask, so synthesize: keep q,v bias, zero k bias.
        out[f"{d}.attn.qkv.bias_mask"] = torch.cat(
            [torch.ones(embed), torch.zeros(embed), torch.ones(embed)], dim=0)
        out[f"{d}.attn.proj.weight"] = hf[f"{s}.attention.o_proj.weight"]
        out[f"{d}.attn.proj.bias"] = hf[f"{s}.attention.o_proj.bias"]
        out[f"{d}.mlp.fc1.weight"] = hf[f"{s}.mlp.up_proj.weight"]
        out[f"{d}.mlp.fc1.bias"] = hf[f"{s}.mlp.up_proj.bias"]
        out[f"{d}.mlp.fc2.weight"] = hf[f"{s}.mlp.down_proj.weight"]
        out[f"{d}.mlp.fc2.bias"] = hf[f"{s}.mlp.down_proj.bias"]
        out[f"{d}.ls1.gamma"] = hf[f"{s}.layer_scale1.lambda1"]
        out[f"{d}.ls2.gamma"] = hf[f"{s}.layer_scale2.lambda1"]
    return out


def build_model():
    sys.path.insert(0, DINOV3_REPO)
    from dinov3.models.vision_transformer import vit_large
    return vit_large(patch_size=16, n_storage_tokens=4, mask_k_bias=True,
                     layerscale_init=1e-5, norm_layer="layernorm",
                     ffn_layer="mlp", qkv_bias=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="web", choices=["web", "sat"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf = load_hf(args.arm)
    mapped = remap(hf)
    model = build_model()
    msd = model.state_dict()

    # shape-adapt where HF stores an extra singleton dim (e.g. mask_token)
    for k in list(mapped):
        if k in msd and mapped[k].shape != msd[k].shape and \
                mapped[k].numel() == msd[k].numel():
            mapped[k] = mapped[k].reshape(msd[k].shape)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    param_names = {n for n, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]   # buffers are ok to miss

    print(f"[{args.arm}] mapped {len(mapped)} tensors into ViT-L "
          f"({len(msd)} model entries)")
    print(f"  unexpected keys: {len(unexpected)} {unexpected[:6]}")
    print(f"  missing (buffers, expected): "
          f"{[m for m in missing if m not in param_names][:6]}")
    print(f"  missing PARAMETERS (should be empty): {missing_params}")
    ok = not unexpected and not missing_params
    print(f"  MAPPING {'OK ✓' if ok else 'FAILED ✗'}")
    if not ok:
        sys.exit(1)

    teacher = {f"backbone.{k}": v for k, v in mapped.items()}
    ckpt = {"teacher": teacher}
    out = args.out or os.path.join(REPO, "dapt/ssl",
                                   f"dinov3_{args.arm}_vitl_teacher.pth")
    torch.save(ckpt, out)
    print(f"  wrote {os.path.relpath(out, REPO)} "
          f"({sum(v.numel() for v in teacher.values())/1e6:.0f}M params)")


if __name__ == "__main__":
    main()
