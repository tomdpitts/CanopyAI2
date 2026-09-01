"""Modal app: run the seed-0 detector + EM diagnostics over the 108 HELD-OUT VAL tiles.

WHY
    The confidence reranker must be calibrated on data that is not the 439 test set,
    otherwise the resulting AP is not a publishable number. The phase4 split is
    792 train / 108 val (train_tiles_gt.json `partition`), and the val tiles are held out
    from the detector's training -- so they are the correct calibration set. Their cached
    features already live on the volume at /vol/feat_4p_train.

    Emits exactly the same per-box feature set as em_diag_modal.py, plus the boxes and
    scores, so rerank_val.py can fit on val and apply to test with no leakage in either
    direction.

GPU is needed only for the detector head over 108 cached feature grids (the masker is
numpy). Everything downstream is CPU.

Run:
    modal run val_diag_modal.py --limit 4     # smoke
    modal run val_diag_modal.py               # all 108
"""
import json
import os

import modal

APP = "tcd04-val-diag"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
P4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")

P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"

app = modal.App(APP)
vol = modal.Volume.from_name(VOL_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1", "numpy==2.2.6", "scipy",
                 "pycocotools", "pillow")   # torchvision: dapt.decode uses ops.nms
    .env({"PYTHONPATH": P})
)
for rel in ("dapt/__init__.py", "dapt/backbone.py", "dapt/decode.py", "dapt/head.py",
            "dapt/eval.py", "dapt/targets.py"):
    image = image.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}")
for rel in ("__init__.py", "detector.py", "evaluate.py", "em.py"):
    image = image.add_local_file(os.path.join(PKG, rel), f"{PKG_R}/{rel}")
image = image.add_local_file(os.path.join(P4, "val_gt.json"), "/root/val_gt.json")

VOL = "/vol"
FEAT_TRAIN = f"{VOL}/feat_4p_train"
EM_NPZ = f"{VOL}/out/em_model_4p_fix.npz"
CKPT = f"{VOL}/out/det_phase4_L24_s0.pt"


@app.function(gpu="A100", image=image, volumes={"/vol": vol}, timeout=2 * 3600,
              cpu=8, memory=32768)
def val_diag(limit: int = 0):
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from dapt.decode import decode
    from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8

    class Detector4Phase(Detector8):
        """VERBATIM copy of phase4_lib_tcd.Detector4Phase (phase4_lib_tcd.py:33) -- same
        modules and params as Detector8, no internal interpolate because the 4-phase grid
        is already at stride 8. Copied rather than imported so this job does not drag in
        phase4_lib_tcd's whole eval stack (and its dense-mask accumulation)."""

        def forward(self, feat):
            x = self.tower(self.stem(feat))
            x = self.up(x)                          # no interpolate (already 8px)
            return torch.cat([self.hm(x), self.reg(x)], dim=1)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Detector4Phase(cfg["in_dim"], width=cfg["width"],
                           tower=cfg["tower"]).cuda().eval()
    model.load_state_dict(ck["state"])
    print(f"[val] detector loaded: in_dim={cfg['in_dim']} width={cfg['width']} "
          f"tower={cfg['tower']} score_thr(op)={cfg['score_thr']}", flush=True)

    d = np.load(EM_NPZ, allow_pickle=False)
    mu, U, scale = d["mu"], d["U"], d["scale"]
    C, Gbg, wbg = d["C"], d["Gbg"], d["wbg"]
    pi, kappa = d["pi"], float(d["kappa"])
    size_edges = d["size_edges"]
    s = int(d["s_px"])
    origin = float(d["cell_origin"]) if "cell_origin" in d else s / 2.0
    prior_weight = float(d["prior_weight"]) if "prior_weight" in d else 1.0
    kappa_scale = float(d["kappa_scale"]) if "kappa_scale" in d else 1.0
    NB = pi.shape[2]
    k = kappa * kappa_scale

    gt = json.load(open("/root/val_gt.json"))
    tids = sorted(t for t in gt
                  if os.path.exists(os.path.join(FEAT_TRAIN, t + ".npy")))
    if limit:
        tids = tids[:limit]
    print(f"[val] {len(tids)} val tiles", flush=True)

    out = {}
    for n, tid in enumerate(tids):
        feat = np.load(os.path.join(FEAT_TRAIN, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        with torch.no_grad():
            det = model(torch.from_numpy(feat)[None].cuda())
        bx, sc = decode(det.cpu(), score_thr=0.05, stride=STRIDE8, topk=600)
        bx, sc = bx.numpy(), sc.numpy()
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - mu) @ U / scale
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        del feat, z, det
        cy, cx = np.mgrid[0:g, 0:g] * s + origin
        cyr, cxr = cy.ravel(), cx.ravel()
        rec = {q: [] for q in ("a_mean", "a_std", "a_p90", "a_max", "bimod",
                               "pfg_mean", "n_cells")}
        for b in bx:
            x0, y0, x1, y1 = b
            pad = s / 2.0
            m = ((cxr >= x0 - pad) & (cxr < x1 + pad) &
                 (cyr >= y0 - pad) & (cyr < y1 + pad))
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                d2 = (cxr - (x0 + x1) / 2) ** 2 + (cyr - (y0 + y1) / 2) ** 2
                idx = np.array([int(d2.argmin())])
            u = np.clip((cxr[idx] - x0) / max(x1 - x0, 1), 0, 1)
            v = np.clip((cyr[idx] - y0) / max(y1 - y0, 1), 0, 1)
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            sb = int(np.searchsorted(size_edges,
                                     np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1))))
            zi = zn[idx]
            bgll = logsumexp(np.log(wbg)[None] + k * (zi @ Gbg.T), 1)
            pis = pi[sb][:, bv, bu]
            zc = k * (zi @ C.T)
            psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
            lw = zc + np.log((pis / psum).T + 1e-9)
            A = logsumexp(lw, 1) - bgll
            Ac = A - A.mean() if len(A) > 1 else A
            pfg = 1.0 / (1.0 + np.exp(-(Ac + prior_weight * np.log(psum / (1 - psum)))))
            rec["a_mean"].append(float(A.mean()))
            rec["a_std"].append(float(A.std()))
            rec["a_p90"].append(float(np.percentile(A, 90)))
            rec["a_max"].append(float(A.max()))
            rec["bimod"].append(float(np.abs(2 * pfg - 1).mean()))
            rec["pfg_mean"].append(float(pfg.mean()))
            rec["n_cells"].append(int(len(idx)))
        out[tid] = {"boxes_2048": np.round(bx, 2).tolist(),
                    "scores": np.round(sc, 4).tolist(),
                    **{q: [round(x, 5) if isinstance(x, float) else x for x in v]
                       for q, v in rec.items()}}
        del zn
        if (n + 1) % 10 == 0 or n + 1 == len(tids):
            print(f"  {n+1}/{len(tids)} tiles", flush=True)
    return out


@app.local_entrypoint()
def main(limit: int = 0):
    res = val_diag.remote(limit)
    n = sum(len(v["scores"]) for v in res.values())
    fp = os.path.join(HERE, "em_diag_val_s0.json")
    with open(fp, "w") as f:
        json.dump(res, f)
    print(f"[val] {len(res)} tiles, {n} boxes -> {fp}")
