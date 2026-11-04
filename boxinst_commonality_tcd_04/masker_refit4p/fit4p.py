"""Refit the LACE box->mask masker on the REAL 4 px lattice (pre-registration: masker_refit4p/README.md).

The deployed masker (em_model_4p_fix.npz, sha1 2f0eaa3f...) was fitted on the 8 px (4-offset) lattice and is applied
unchanged on the 4 px (16-offset) lattice. Here the same EM recipe is refitted on 4 px cells.

Input: z4p cells (tcd04-phase4-vol:z4p_train/<tid>.npy, (512,512,128) fp16) = the 4 px lattice already projected
with the DEPLOYED masker's PCA whitening and L2-normalised (dethead/z4p_modal.py = hunt/feat16p.py). The whitening
is therefore held FIXED (mu/U/scale copied from the deployed npz). Inference reads the same projected cells
(btmask.render_boxes) and never re-applies mu/U/scale, so fit and inference share one space.

fit_nopca() is em.fit VERBATIM except:
  (a) the PCA block is removed (cells are already whitened; NOT re-normalised, exactly as inference loads them);
  (b) geometry is the 4 px lattice: s = 4, cell centre 4X + 8 (origin 8; = feat16p / btmask m4.origin);
  (c) the ring / clear-background margin stays 8 px (2 cells), the 8 px fit's physical width;
  (d) checkpoint after init and after every EM iteration (resume-safe; RNG is used only before the loop);
  (e) mu/U/scale come from the deployed npz; s_px = 4, cell_origin = 8;
  (f) bg/neg subsampling gathers only the chosen rows (same RNG calls; bit-identical, verified on n120).
Labels/ring helpers: copied from modal_tcd_multiseed/phase4/phase4_fit_tcd.py with origin and margin made explicit.
"""
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

S_PX, ORIGIN, MARGIN, G = 4, 8.0, 8.0, 512


def _canopy_cell_mask(canopy_px, g):
    if canopy_px is None or not np.any(canopy_px):
        return np.zeros((g, g), bool)
    a = np.asarray(canopy_px, bool)
    blk = a.shape[0] // g
    a = a[:g * blk, :g * blk].reshape(g, blk, g, blk)
    return a.mean((1, 3)) > 0.5


def _raster_canopy(polys, res):
    if not polys:
        return np.zeros((res, res), bool)
    m = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.asarray(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


def _geom(boxes, canopy_px, g=G, s=S_PX, origin=ORIGIN, margin=MARGIN):
    cy, cx = np.mgrid[0:g, 0:g] * s + origin
    inbox = np.zeros((g, g), bool); near = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= ((cx >= x0 - margin) & (cx < x1 + margin) & (cy >= y0 - margin) & (cy < y1 + margin))
    canopy = _canopy_cell_mask(canopy_px, g)
    lab = -np.ones((g, g), np.int8); lab[inbox] = 1
    lab[(~near) & (~canopy)] = 0
    ring = near & ~inbox & ~canopy
    return lab.ravel(), ring.ravel()


def train_tiles(gt_path, n_tiles):
    gt = json.load(open(gt_path))
    tids = sorted(t for t, v in gt.items() if v["partition"] == "train")
    assert len(tids) == 792, len(tids)
    return gt, tids[:n_tiles]          # same rule as the deployed fit: first n of the sorted train ids


def load_train(feat_dir, gt_path, n_tiles):
    gt, tids = train_tiles(gt_path, n_tiles)
    feats, labs, rings, bxs = [], [], [], []
    for i, t in enumerate(tids):
        z = np.load(os.path.join(feat_dir, t + ".npy"))                   # (512,512,128) fp16
        assert z.shape == (G, G, 128), (t, z.shape)
        bx = np.array(gt[t]["boxes"], np.float32).reshape(-1, 4)
        can = _raster_canopy(gt[t].get("canopy", []), res=G * S_PX)
        lab, ring = _geom(bx, can)
        feats.append(z.reshape(-1, 128).astype(np.float32))
        labs.append(lab); rings.append(ring); bxs.append(bx)
        if (i + 1) % 20 == 0 or i + 1 == len(tids):
            print(f"  loaded {i+1}/{len(tids)}", flush=True)
    return tids, feats, labs, rings, bxs


def fit_nopca(feat_dir, gt_path, n_tiles, deployed_npz, out_dir, commit=lambda: None, seed=0, k=16, k_bg=12,
              bins=8, kappa=10.0, iters=30, prune_after=5, prune_frac=0.25, contrastive_beta=0.5):
    from boxinst_commonality.em import (BIN_CAP, FG_PRIOR_MASS, contrastive_update, estep, logsumexp,
                                        spherical_kmeans)
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"em_model_z4p_n{n_tiles}.npz")
    if os.path.exists(out_npz):
        print(f"[fit4p] {out_npz} exists -> skip", flush=True); return out_npz
    st0, ck = os.path.join(out_dir, "state0.npz"), os.path.join(out_dir, "ckpt.npz")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tiles, Zn, labs, rings, boxes = load_train(feat_dir, gt_path, n_tiles)
    g, s, origin = G, S_PX, ORIGIN
    print(f"[fit4p] tiles={len(tiles)} grid={g} stride={s}px origin={origin}px margin={MARGIN}px "
          f"load {time.time()-t0:.0f}s", flush=True)

    if os.path.exists(st0):
        d = np.load(st0); Gbg, wbg, negZ, size_edges, C0 = d["Gbg"], d["wbg"], d["negZ"], d["size_edges"], d["C0"]
        print("[fit4p] resumed init state", flush=True)
    else:
        # (f) memory: em.fit materialises concat(clear), concat(ring) and their concatenations before subsampling
        # (~27 GB of clear cells at 4 px / 250 tiles). Here the same rng.choice calls, in the same order, index a
        # VIRTUAL concatenation and only the chosen rows are gathered -> bit-identical bgZ / negZ to em.fit.
        clear_idx = [np.flatnonzero(l == 0) for l in labs]; ring_idx = [np.flatnonzero(r) for r in rings]
        n_clear = sum(len(i) for i in clear_idx); n_ring = sum(len(i) for i in ring_idx)

        def rows(parts):        # parts: list of (tile list of row-index arrays) concatenated in order
            seg = [(ti, ix) for part in parts for ti, ix in enumerate(part)]
            off = np.cumsum([0] + [len(ix) for _, ix in seg])
            def gather(sel):
                out = np.empty((len(sel), Zn[0].shape[1]), np.float32)
                k = np.searchsorted(off, sel, side="right") - 1
                for j in np.unique(k):
                    m = k == j; ti, ix = seg[j]
                    out[m] = Zn[ti][ix[sel[m] - off[j]]]
                return out
            return int(off[-1]), gather

        nb, gb = rows([clear_idx, ring_idx])
        bgZ = gb(rng.choice(nb, 80000, replace=False)) if nb > 80000 else gb(np.arange(nb))
        nn, gn = rows([ring_idx, ring_idx, clear_idx])
        negZ = gn(rng.choice(nn, 60000, replace=False)) if nn > 60000 else gn(np.arange(nn))
        Gbg, a = spherical_kmeans(bgZ, k_bg, rng)
        wbg = np.bincount(a, minlength=k_bg) / len(a)
        wbg = np.clip(wbg, 1e-3, None); wbg /= wbg.sum()
        print(f"bg mixture K_bg={k_bg} on {n_clear} clear + {n_ring} ring", flush=True)
        del bgZ, clear_idx, ring_idx
        sizes = np.array([np.sqrt((b[2] - b[0]) * (b[3] - b[1])) for bx in boxes for b in bx])
        size_edges = np.percentile(sizes, [100 / 3, 200 / 3])
    # in-box instance cells (strict centre-in-box)
    inst = []
    cy, cx = np.mgrid[0:g, 0:g] * s + origin
    for ti, bx in enumerate(boxes):
        for b in bx:
            x0, y0, x1, y1 = b
            m = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
            idx = np.flatnonzero(m.ravel())
            if not len(idx):
                continue
            u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
            v = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
            sb = int(np.searchsorted(size_edges, np.sqrt((x1 - x0) * (y1 - y0))))
            inst.append((ti, idx, u, v, sb))
    print(f"EM over {len(inst)} boxes / {sum(len(i[1]) for i in inst)} cells; "
          f"size terciles {np.round(size_edges, 1)}px", flush=True)

    bg_all = [logsumexp(np.log(wbg)[None] + kappa * (z @ Gbg.T), 1) for z in Zn]

    def bg_ll(ti, idx):
        return bg_all[ti][idx]

    if not os.path.exists(st0):
        fgZ = np.concatenate([Zn[ti][idx] for ti, idx, *_ in inst])
        fg_bgll = np.concatenate([bg_ll(ti, idx) for ti, idx, *_ in inst])
        fgZ = fgZ[fg_bgll < np.median(fg_bgll)]
        if len(fgZ) > 60000:
            fgZ = fgZ[rng.choice(len(fgZ), 60000, replace=False)]
        C0, _ = spherical_kmeans(fgZ, k, rng)
        del fgZ, fg_bgll
        np.savez(st0 + ".tmp.npz", Gbg=Gbg, wbg=wbg, negZ=negZ, size_edges=size_edges, C0=C0)
        os.replace(st0 + ".tmp.npz", st0); commit()

    NB, SB = bins, 3
    if os.path.exists(ck):
        d = np.load(ck); C, pi, it0, trace = d["C"], d["pi"], int(d["it"]), list(d["trace"])
        print(f"[fit4p] resumed at iteration {it0}", flush=True)
    else:
        C = C0
        pi = np.full((SB, C.shape[0], NB, NB), FG_PRIOR_MASS / C.shape[0]); it0, trace = 0, []
    for it in range(it0, iters):
        ti0 = time.time()
        K = C.shape[0]
        acc = np.zeros((SB, K, NB, NB)); cnt = np.zeros((SB, NB, NB))
        newC = np.zeros_like(C); fg_mass, n_tot = 0.0, 0
        for ti, idx, u, v, sb in inst:
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            pfg, r = estep(Zn[ti][idx], pi[sb][:, bv, bu], kappa, C, bg_ll(ti, idx), contrast=True)
            np.add.at(acc[sb], (slice(None), bv, bu), r.T)
            np.add.at(cnt[sb], (bv, bu), 1)
            newC += r.T @ Zn[ti][idx]; fg_mass += pfg.sum(); n_tot += len(idx)
        C = contrastive_update(newC, negZ, kappa, contrastive_beta)
        pi = np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-4, None)
        for sb in range(SB):
            pi[sb] *= FG_PRIOR_MASS / max(pi[sb].sum(0).mean(), 1e-6)
            bs = pi[sb].sum(0); over = bs > BIN_CAP
            if over.any():
                pi[sb][:, over] *= BIN_CAP / bs[over]
        if it >= prune_after and C.shape[0] > 2:
            share = acc.sum((0, 2, 3)); share = share / max(share.sum(), 1e-9)
            keep = share >= prune_frac / C.shape[0]
            if (~keep).any():
                C, pi = C[keep], pi[:, keep]
                print(f"  it{it+1:2d}: prune -> K={C.shape[0]}", flush=True)
        trace.append(round(fg_mass / n_tot, 4))
        np.savez(ck + ".tmp.npz", C=C, pi=pi, it=it + 1, trace=np.array(trace)); os.replace(ck + ".tmp.npz", ck)
        commit()
        print(f"  it{it+1:2d}: mean P(fg)={trace[-1]:.3f} K={C.shape[0]} {time.time()-ti0:.0f}s", flush=True)

    dep = np.load(deployed_npz)
    np.savez(out_npz, mu=dep["mu"], U=dep["U"], scale=dep["scale"], C=C, Gbg=Gbg, wbg=wbg, pi=pi, kappa=kappa,
             size_edges=size_edges, s_px=s, cell_origin=origin, prior_weight=float(dep["prior_weight"]),
             kappa_scale=float(dep["kappa_scale"]))
    json.dump({"seed": seed, "n_tiles": len(tiles), "first_tile": tiles[0], "last_tile": tiles[-1],
               "stride_px": s, "cell_origin_px": origin, "ring_margin_px": MARGIN, "pca": "fixed (deployed)",
               "k_init": k, "k_effective": int(C.shape[0]), "k_bg": k_bg, "bins": bins, "kappa": kappa,
               "contrastive_beta": contrastive_beta, "size_edges_px": [round(float(e), 1) for e in size_edges],
               "n_boxes": len(inst), "fg_mass_trace": trace, "minutes": round((time.time() - t0) / 60, 1),
               "note": "boxes-only, canopy-ignore, no polygon ever read"},
              open(os.path.join(out_dir, "fit_report.json"), "w"), indent=2)
    commit()
    print(f"[fit4p] saved {out_npz} K={C.shape[0]} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_npz
