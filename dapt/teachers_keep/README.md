# Retained DAPT teachers (2026-10-20)

DAPT is abandoned. Both Modal volumes (dinov3-dapt-arid-vol, dinov3-dapt-v3-vol, 479 GB) are being deleted to save cost. These are the only surviving weights. They are kept so the v3 null result in `dapt/REPORT.md` ("DAPT − web −0.000 CI[−0.003,+0.003]") can be re-run.

Contents: the val-selected MLP-readout winners (`dapt/v3/artifacts/winners.json`) for **3 of the 5** v3 SSL seeds, Tom's choice: s101_i499, s201_i999, s301_i999. A 3-seed re-run is a replication check. It will not reproduce the 5-seed pooled number exactly (s401_i999 and s501_i499 are gone).

Format: the output of `extract` in `dapt/v3/ssl/app.py`, i.e. the raw DINOv3 teacher backbone (368 tensors, 303.2M params). Convert it with `python -m dapt.ssl.export_hf --teacher <pth>` → `dapt/v3/ckpt/dapt_v3_s<seed>_i<it>_hf/` (that config.json is tracked in git; the weights were not).

Integrity: `checksums.json`. Every file loads with torch, and each sha256 was confirmed by two independent downloads. One download of s201 came out corrupt at the right byte size, so always load-check after `modal volume get`.

`*.pth` is gitignored, so these files live ONLY on this disk. Back them up off-machine (the 2026-08 migration lost the previous `_hf` weights).
