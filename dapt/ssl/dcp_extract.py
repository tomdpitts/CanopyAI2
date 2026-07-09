"""Extract the TEACHER BACKBONE from a dinov3 DCP checkpoint -> native teacher .pth.

Pure-torch, metadata-driven PARTIAL load: reads the DCP metadata, allocates empty
tensors only for keys under `model.teacher.backbone.` (~0.6 GB bf16), and dcp.load()s
just those — no model instantiation, no optimizer materialization, no dinov3 import.
Output format = {"teacher": {"backbone.<k>": tensor}}, i.e. exactly what
convert_dinov3_to_hf.py / init_fsdp_model_from_checkpoint consume.

Runs anywhere (Modal CPU or locally on a downloaded ckpt dir):
    python dcp_extract.py --ckpt-dir /vol/out_s0/ckpt/4999 --out teacher_s0_i4999.pth
"""
import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

PREFIX = "model.teacher.backbone."


def extract(ckpt_dir: str, out_path: str, prefix: str = PREFIX):
    reader = FileSystemReader(ckpt_dir)
    meta = reader.read_metadata().state_dict_metadata
    wanted = {k: m for k, m in meta.items() if k.startswith(prefix)}
    if not wanted:
        have = sorted({k.split(".")[0] for k in meta})
        raise SystemExit(f"no keys under {prefix!r} in {ckpt_dir} (top-level: {have})")

    state = {}
    for k, m in wanted.items():
        if hasattr(m, "size"):                    # TensorStorageMetadata
            state[k] = torch.empty(m.size, dtype=m.properties.dtype)
        else:                                     # non-tensor (unlikely here)
            state[k] = None
    dcp.load(state, storage_reader=reader)        # single-process, CPU

    teacher = {"backbone." + k[len(prefix):]: v for k, v in state.items()}
    torch.save({"teacher": teacher}, out_path)
    n_par = sum(v.numel() for v in teacher.values())
    print(f"wrote {out_path}  ({len(teacher)} tensors, {n_par/1e6:.0f}M params, "
          f"dtypes={sorted({str(v.dtype) for v in teacher.values()})})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="DCP dir, e.g. .../ckpt/4999")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extract(args.ckpt_dir, args.out)
