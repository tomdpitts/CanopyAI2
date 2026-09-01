"""Modal CPU app: EM posterior statistics for MANY seeds / splits in one pass.

Generalises em_diag_modal.py. The posterior product needs only two numbers per box —

    bimod    = mean|2*pfg - 1|      how far from undecided the masker's posterior is
    pfg_mean = pfg.mean()           how much of the box looks foreground

— both two lines over the `pfg` that `estep` already returns (see confidence/README.md; the
pre-recentring `a_*` statistics turned out redundant, so nothing inside the masker changes).

WHY ONE JOB FOR ALL SEEDS
    Loading a tile's cached features (1024,256,256 = 268 MB) dominates the runtime; the
    masker pass over boxes is trivial by comparison. Every seed of a split shares the same
    features, so one pass serves all of them. Boxes are shipped stripped of their mask
    payloads (12 MB not 18 MB per seed).

SPLITS
    test    tcd04-phase4-vol:/feat_4p_test    439 tiles, boxes from phase4 preds/knobbed_s*.json
    sparse  tcd-sparse-vol:/feat_4p_sparse    236 tiles, boxes from modal_sparse .../ours_s*.json
            (the sparse slice shares NO tile with the 900 used for training)

Run:
    modal run em_diag_multi.py --split test   --tags test_s1,test_s2
    modal run em_diag_multi.py --split sparse --tags sparse_s0,sparse_s1,sparse_s2
"""
import json
import os

import modal

APP = "tcd04-em-diag-multi"
HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App(APP)
p4vol = modal.Volume.from_name("tcd04-phase4-vol")
spvol = modal.Volume.from_name("tcd-sparse-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy")
)
for f in sorted(os.listdir(os.path.join(HERE, "boxes"))):
    image = image.add_local_file(os.path.join(HERE, "boxes", f), f"/root/boxes/{f}")

FEAT = {"test": "/p4/feat_4p_test", "sparse": "/sp/feat_4p_sparse"}
# Default = the masker behind the published row (β=0.5, geometry-fixed). Override with
# --em-npz to ablate the masker itself, e.g. /p4/out/em_model_4p_b0_fix.npz for genuine β=0
# (contrastive term REMOVED -- `contrastive_update` returns the plain generative M-step).
# β acts at FIT time, so the boxes are unchanged: only the posterior differs.
EM_NPZ = "/p4/out/em_model_4p_fix.npz"


@app.function(image=image, volumes={"/p4": p4vol, "/sp": spvol},
              timeout=3 * 3600, cpu=4, memory=32768)
def diag_shard(shard: int, n_shards: int, split: str, tags: str,
               em_npz: str = EM_NPZ, suffix: str = ""):
    import numpy as np
    from scipy.special import logsumexp

    d = np.load(em_npz, allow_pickle=False)
    mu, U, scale = d["mu"], d["U"], d["scale"]
    C, Gbg, wbg = d["C"], d["Gbg"], d["wbg"]
    pi, kappa = d["pi"], float(d["kappa"])
    size_edges = d["size_edges"]
    s = int(d["s_px"])
    origin = float(d["cell_origin"]) if "cell_origin" in d else s / 2.0
    prior_weight = float(d["prior_weight"]) if "prior_weight" in d else 1.0
    k = kappa * (float(d["kappa_scale"]) if "kappa_scale" in d else 1.0)
    NB = pi.shape[2]

    tag_list = [t for t in tags.split(",") if t]
    boxes = {t: json.load(open(f"/root/boxes/{t}.json")) for t in tag_list}
    feat_dir = FEAT[split]
    tids = sorted({t for b in boxes.values() for t in b
                   if os.path.exists(os.path.join(feat_dir, t + ".npy"))})
    tids = tids[shard::n_shards]
    print(f"[diag] shard {shard}/{n_shards} split={split} tags={tag_list} "
          f"em={os.path.basename(em_npz)} {len(tids)} tiles", flush=True)

    out = {t: {} for t in tag_list}
    for n, tid in enumerate(tids):
        feat = np.load(os.path.join(feat_dir, tid + ".npy")).astype(np.float32)
        g = feat.shape[-1]
        z = (feat.reshape(feat.shape[0], -1).T.astype(np.float32) - mu) @ U / scale
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        del feat, z
        cy, cx = np.mgrid[0:g, 0:g] * s + origin
        cyr, cxr = cy.ravel(), cx.ravel()
        for tag in tag_list:
            if tid not in boxes[tag]:
                continue
            bm, pm = [], []
            for b in np.asarray(boxes[tag][tid]["boxes_2048"], np.float64).reshape(-1, 4):
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
                psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
                lw = k * (zi @ C.T) + np.log((pis / psum).T + 1e-9)
                A = logsumexp(lw, 1) - bgll
                Ac = A - A.mean() if len(A) > 1 else A
                pfg = 1.0 / (1.0 + np.exp(-(Ac + prior_weight * np.log(psum / (1 - psum)))))
                bm.append(round(float(np.abs(2 * pfg - 1).mean()), 5))
                pm.append(round(float(pfg.mean()), 5))
            out[tag][tid] = {"bimod": bm, "pfg_mean": pm}
        del zn
        if (n + 1) % 20 == 0 or n + 1 == len(tids):
            print(f"  shard {shard}: {n+1}/{len(tids)}", flush=True)
    return out


@app.local_entrypoint()
def main(split: str = "test", tags: str = "", n_shards: int = 8,
         em_npz: str = EM_NPZ, suffix: str = ""):
    assert tags, "--tags required, e.g. test_s1,test_s2"
    merged = {t: {} for t in tags.split(",") if t}
    for part in diag_shard.starmap([(i, n_shards, split, tags, em_npz, suffix)
                                    for i in range(n_shards)]):
        for tag, v in part.items():
            merged[tag].update(v)
    for tag, v in merged.items():
        fp = os.path.join(HERE, f"em_post_{tag}{suffix}.json")
        json.dump(v, open(fp, "w"))
        n = sum(len(x["bimod"]) for x in v.values())
        print(f"[diag] {tag}: {len(v)} tiles, {n} boxes -> {os.path.relpath(fp)}")
