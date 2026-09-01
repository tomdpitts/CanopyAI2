"""Modal CPU app: extract the EM masker's DISCARDED box-level evidence for the saved seed-0
boxes, so it can be used to re-rank predictions.

THE IDEA
    boxinst_commonality/em.py:193 `estep` computes, per cell inside a box,

        A = logsumexp(zc + log spatial_prior, 1) - bgll        # fg-vs-bg log-likelihood ratio

    and then immediately does `A = A - A.mean()`. The docstring is explicit about why: the
    ABSOLUTE ratio is "confounded by ViT context (all in-box cells score fg)", so it is
    recentred to recover clean mask SHAPE.

    That reasoning is correct for shape and irrelevant for SCORING. `A.mean()` is precisely
    "how much does this box look like a tree at all" -- box-level object evidence, computed
    and then thrown away one line later. LACE currently ranks masks by the CenterNet
    heatmap peak alone (dapt/decode.py:26, sigmoid(hm_logit)), which measures "is there a
    crown centre here", not "is this box real" and not "is this mask good".

    Crucially the masker is INDEPENDENT of the detector: it is training-free and sees only
    feature-space commonality, never the heatmap. Mask R-CNN cannot do this -- its mask head
    is conditioned on the same ROI features that produced the box score. So EM's opinion is
    a genuine second estimator, and it is free.

NO GPU NEEDED. The detector is not re-run; boxes come from the saved seed-0 predictions and
the masker is pure numpy. Sharded across containers only because the cached features are
~268 MB/tile (1024,256,256) and the read dominates.

Emits per box: a_mean/a_std/percentiles of the RAW (pre-recentring) A, plus posterior
bimodality and cell count. Nothing else about the pipeline changes.

Run:
    modal run em_diag_modal.py                    # all 439 tiles
    modal run em_diag_modal.py --n-shards 2 --limit 8
"""
import json
import os

import modal

APP = "tcd04-em-diag"
VOL_NAME = "tcd04-phase4-vol"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
P4 = os.path.join(PKG, "modal_tcd_multiseed", "phase4")

app = modal.App(APP)
vol = modal.Volume.from_name(VOL_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy")
    .add_local_file(os.path.join(P4, "preds", "knobbed_s0.json"), "/root/preds.json")
)

VOL = "/vol"
FEAT_TEST = f"{VOL}/feat_4p_test"
EM_NPZ = f"{VOL}/out/em_model_4p_fix.npz"       # the masker behind the published row
OUT = f"{VOL}/out/em_diag"


@app.function(image=image, volumes={"/vol": vol}, timeout=3 * 3600, cpu=4, memory=32768)
def diag_shard(shard: int, n_shards: int, limit: int = 0):
    """Recompute estep for each saved box, keeping the quantity estep discards."""
    import numpy as np
    from scipy.special import logsumexp

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

    preds = json.load(open("/root/preds.json"))["preds"]
    tids = sorted(t for t in preds
                  if os.path.exists(os.path.join(FEAT_TEST, t + ".npy")))
    if limit:
        tids = tids[:limit]
    tids = tids[shard::n_shards]
    print(f"[diag] shard {shard}/{n_shards}: {len(tids)} tiles  s={s} origin={origin} "
          f"kappa={k:.3f} alpha={prior_weight}", flush=True)

    out = {}
    for n, tid in enumerate(tids):
        feat = np.load(os.path.join(FEAT_TEST, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - mu) @ U / scale
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        del feat, z
        cy, cx = np.mgrid[0:g, 0:g] * s + origin
        cyr, cxr = cy.ravel(), cx.ravel()
        boxes = np.asarray(preds[tid]["boxes_2048"], np.float64).reshape(-1, 4)
        rec = {q: [] for q in ("a_mean", "a_std", "a_p90", "a_max", "bimod",
                               "pfg_mean", "n_cells")}
        for b in boxes:
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
            A = logsumexp(lw, 1) - bgll                  # <-- the discarded evidence
            Ac = A - A.mean() if len(A) > 1 else A       # what estep actually keeps
            pfg = 1.0 / (1.0 + np.exp(-(Ac + prior_weight * np.log(psum / (1 - psum)))))
            rec["a_mean"].append(float(A.mean()))
            rec["a_std"].append(float(A.std()))
            rec["a_p90"].append(float(np.percentile(A, 90)))
            rec["a_max"].append(float(A.max()))
            rec["bimod"].append(float(np.abs(2 * pfg - 1).mean()))
            rec["pfg_mean"].append(float(pfg.mean()))
            rec["n_cells"].append(int(len(idx)))
        out[tid] = {q: [round(x, 5) if isinstance(x, float) else x for x in v]
                    for q, v in rec.items()}
        del zn
        if (n + 1) % 20 == 0 or n + 1 == len(tids):
            print(f"  shard {shard}: {n+1}/{len(tids)}", flush=True)
    return out


@app.local_entrypoint()
def main(n_shards: int = 8, limit: int = 0):
    merged = {}
    for part in diag_shard.starmap([(i, n_shards, limit) for i in range(n_shards)]):
        merged.update(part)
    n = sum(len(v["a_mean"]) for v in merged.values())
    print(f"[diag] merged {len(merged)} tiles, {n} boxes")
    import json as J
    with open(os.path.join(HERE, "em_diag_s0.json"), "w") as f:
        J.dump(merged, f)
    print(f"-> {os.path.join(HERE, 'em_diag_s0.json')}")
