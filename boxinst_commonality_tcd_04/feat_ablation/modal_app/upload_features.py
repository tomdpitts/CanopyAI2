"""Stream-slice block-1024 features into the Modal Volume (no local staging).

Each cached parent .npy is (4096,128,128) fp16; the last DINOv3 block is the
contiguous tail slice [3072:4096] -> (1024,128,128), 33.6 MB. We mmap, slice,
np.save into a BytesIO, and batch_upload the buffer — local disk is never
touched (only ~65 GB free; the full sliced set is ~45 GB).

Resumable: already-present volume files are skipped, and each batch commits
independently, so a killed upload loses at most one batch.

Usage:
    .venv/bin/python -u boxinst_commonality_tcd_04/feat_ablation/modal_app/upload_features.py
"""
import io
import os
import time

import modal
import numpy as np

VOL_NAME = "tcd04-block1024-vol"
BLOCK_LO = 3072
BATCH = 25                                       # ~840 MB per commit

HERE = os.path.abspath(os.path.dirname(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))     # boxinst_commonality_tcd_04
SPLITS = {                                       # remote dir -> local parent cache
    "feat_traintile": os.path.join(PKG, "cache", "web", "feat_traintile"),
    "feat_test": os.path.join(PKG, "cache", "web", "feat_test"),
}


def existing(vol, remote_dir):
    try:
        return {os.path.basename(e.path) for e in vol.listdir(remote_dir)}
    except Exception:
        return set()


def main():
    vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
    for remote_dir, local_dir in SPLITS.items():
        files = sorted(f for f in os.listdir(local_dir) if f.endswith(".npy"))
        done = existing(vol, remote_dir)
        todo = [f for f in files if f not in done]
        print(f"[{remote_dir}] {len(todo)}/{len(files)} to upload "
              f"({len(done)} already present)", flush=True)
        t0, sent = time.time(), 0
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            with vol.batch_upload() as batch:
                for f in chunk:
                    arr = np.load(os.path.join(local_dir, f), mmap_mode="r")
                    assert arr.shape[0] == 4096, f"{f}: unexpected {arr.shape}"
                    buf = io.BytesIO()
                    np.save(buf, np.ascontiguousarray(arr[BLOCK_LO:]))
                    sent += buf.tell()
                    buf.seek(0)
                    batch.put_file(buf, f"/{remote_dir}/{f}")
            n = min(i + BATCH, len(todo))
            mbps = sent / 1e6 / max(time.time() - t0, 1e-9)
            eta = (len(todo) - n) * (time.time() - t0) / n / 60
            print(f"  {n}/{len(todo)}  {sent/1e9:.1f} GB  {mbps:.0f} MB/s  "
                  f"ETA {eta:.0f} min", flush=True)
    print("upload complete", flush=True)


if __name__ == "__main__":
    main()
