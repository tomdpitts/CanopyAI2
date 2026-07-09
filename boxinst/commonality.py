"""Latent-space commonality across annotated boxes (DDT/BANA-style, no training).

Step 1 `direction`: pool in-box vs out-of-box patch features over TRAIN tiles,
fit (a) mean-difference and (b) shrinkage-LDA directions; score held-out tiles
per-pixel; report in-box/out-box separation AUC on TEST and render overlays.
The direction is fit on train only; nothing sees a mask label anywhere.

Step 2 `em`: dataset-level figure/ground EM inside boxes ("box topic model"):
K crown prototypes with spatial priors pi_k(u,v) over box-normalized coords +
one background prototype (initialized from out-of-box stats). Renders GT-box-
prompted posterior masks on test tiles and the learned spatial priors.

Both operate in a centred, PCA-reduced feature space — the raw last-4 features
share a dominant mean component (cosines saturate at ~0.96), and centring is
what removes it.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dapt.cache_features import cache_key
from dapt.data.cohort import REPO, load_boxes
from boxinst.cache_feats import FEAT_DIR

OUT = os.path.join(REPO, "claude_outputs/boxinst/commonality")
G, S = 32, 16                       # patch grid / stride


def load_split():
    split = json.load(open(os.path.join(REPO, "dapt/data/split.json")))
    boxes, _ = load_boxes(split["csv"])
    return split, boxes


def cell_labels(bx, H0, W0):
    """(G,G) labels: 1 in some box, 0 clear background (valid, >=1 cell from any
    box), -1 boundary/pad."""
    cy, cx = np.mgrid[0:G, 0:G] * S + S / 2.0
    lab = -np.ones((G, G), np.int8)
    valid = (cy < H0) & (cx < W0)
    inbox = np.zeros((G, G), bool)
    near = np.zeros((G, G), bool)
    for x0, y0, x1, y1 in bx:
        inbox |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        near |= (cx >= x0 - S) & (cx < x1 + S) & (cy >= y0 - S) & (cy < y1 + S)
    lab[valid & inbox] = 1
    lab[valid & ~near] = 0
    return lab


def collect(split, boxes, part):
    """Stack features + labels for a partition. Returns X (N,C), y (N,), plus
    per-tile arrays for scoring."""
    tiles = sorted(p for p, t in split["tiles"].items() if t["partition"] == part)
    feats, labs = [], []
    for p in tiles:
        f = np.load(os.path.join(FEAT_DIR, cache_key(p) + ".npy")).astype(np.float32)
        W0, H0 = Image.open(p).size
        lab = cell_labels(boxes.get(p, np.zeros((0, 4))), H0, W0)
        feats.append(f.reshape(f.shape[0], -1).T)          # (G*G, C)
        labs.append(lab.ravel())
    return tiles, feats, labs


def fit_directions(feats, labs, shrink=0.1):
    X = np.concatenate(feats)
    y = np.concatenate(labs)
    fg, bg = X[y == 1], X[y == 0]
    mu_all = X[y >= 0].mean(0)
    d = mu_all.shape[0]
    w_md = fg.mean(0) - bg.mean(0)
    Xc = np.concatenate([fg - fg.mean(0), bg - bg.mean(0)])
    cov = (Xc.T @ Xc) / len(Xc)
    cov = (1 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d, dtype=np.float32)
    w_lda = np.linalg.solve(cov, w_md)
    print(f"fit: fg={len(fg)} bg={len(bg)} cells, dim={d}, shrink={shrink}")
    return {"md": w_md, "lda": w_lda}, mu_all


def auc(scores, y):
    from numpy import argsort
    s, t = scores[y >= 0], y[y >= 0]
    order = argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_direction(args):
    split, boxes = load_split()
    tr = collect(split, boxes, "train")
    te = collect(split, boxes, "test")
    ws, mu = fit_directions(tr[1], tr[2])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "direction.npz"), **ws, mu=mu)

    report = {}
    for name, w in ws.items():
        sc = [(f - mu) @ w for f in te[1]]
        report[name] = {
            "test_auc_inbox_vs_bg": round(float(auc(np.concatenate(sc),
                                                    np.concatenate(te[2]))), 4)}
    # per-domain AUC (lda)
    for dom in ("WON", "BRU", "NEON"):
        idx = [i for i, p in enumerate(te[0]) if split["tiles"][p]["domain"] == dom]
        if idx:
            sc = np.concatenate([(te[1][i] - mu) @ ws["lda"] for i in idx])
            yy = np.concatenate([te[2][i] for i in idx])
            report[f"lda_auc_{dom}"] = round(float(auc(sc, yy)), 4)
    print(json.dumps(report, indent=1))
    json.dump(report, open(os.path.join(OUT, "direction_report.json"), "w"), indent=1)

    # render: RGB+GT | LDA heatmap | overlay, for the densest test tiles per domain
    chosen = []
    for dom in ("WON", "BRU", "NEON"):
        cand = [p for p in te[0] if split["tiles"][p]["domain"] == dom]
        cand.sort(key=lambda p: -split["tiles"][p]["n_boxes"])
        chosen += cand[:2]
    fg_frac = np.mean(np.concatenate(tr[2]) == 1)
    for p in chosen:
        i = te[0].index(p)
        sc = ((te[1][i] - mu) @ ws["lda"]).reshape(G, G)
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]
        up = np.array(Image.fromarray(sc.astype(np.float32)).resize(
            (G * S, G * S), Image.BILINEAR))[:H0, :W0]
        thr = np.quantile(sc.ravel(), 1 - fg_frac)         # untuned: train fg rate
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), dpi=120)
        axs[0].imshow(img)
        for x0, y0, x1, y1 in boxes.get(p, []):
            axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, ec="white", lw=1.0))
        axs[0].set_title("RGB + GT boxes", fontsize=9)
        im = axs[1].imshow(up, cmap="magma")
        plt.colorbar(im, ax=axs[1], fraction=0.045)
        axs[1].set_title("LDA commonality score (w·(f−μ))", fontsize=9)
        axs[2].imshow(img)
        axs[2].imshow(np.where(up >= thr, 1.0, np.nan), cmap="spring",
                      alpha=0.5, vmin=0, vmax=1)
        axs[2].set_title(f"score ≥ train-fg-rate quantile ({fg_frac:.2f})", fontsize=9)
        for a in axs:
            a.axis("off")
        name = cache_key(p)
        fig.suptitle(f"{split['tiles'][p]['domain']}/{split['tiles'][p]['site']} "
                     f"{name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"dir_{name}.png"))
        plt.close(fig)
        print(f"rendered dir_{name}.png")


def cmd_em(args):
    split, boxes = load_split()
    tiles, feats, labs = collect(split, boxes, "train")
    # PCA-reduce centred features (removes the dominant shared component)
    X = np.concatenate(feats)
    y = np.concatenate(labs)
    mu = X[y >= 0].mean(0)
    sub = X[y >= 0][::7] - mu
    U = np.linalg.svd(sub, full_matrices=False)[2][:args.pca].T     # (C,D)
    Z = [(f - mu) @ U for f in feats]                                # per tile (G*G,D)
    Zn = [z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8) for z in Z]

    # per-box cells: (tile_idx, cell_idx, u, v)
    inst = []
    for ti, p in enumerate(tiles):
        for b in boxes.get(p, []):
            x0, y0, x1, y1 = b
            cy, cx = np.mgrid[0:G, 0:G] * S + S / 2.0
            m = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
            idx = np.flatnonzero(m.ravel())
            if len(idx) == 0:
                continue
            u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
            v = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
            inst.append((ti, idx, u, v))
    print(f"EM: {len(inst)} boxes, pca={args.pca}, K={args.k}, kappa={args.kappa}")

    rng = np.random.default_rng(0)
    bgZ = np.concatenate([z[l == 0] for z, l in
                          zip(Zn, [lab for lab in labs])])
    g = bgZ.mean(0); g /= np.linalg.norm(g)
    # init crown prototypes: kmeans-ish on box-mean embeddings
    bm = np.stack([Zn[ti][idx].mean(0) for ti, idx, _, _ in inst])
    bm /= np.linalg.norm(bm, axis=1, keepdims=True)
    C = bm[rng.choice(len(bm), args.k, replace=False)]
    for _ in range(10):
        a = np.argmax(bm @ C.T, 1)
        C = np.stack([bm[a == k].mean(0) if (a == k).any() else C[k]
                      for k in range(args.k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)

    NB = args.bins
    pi = np.full((args.k, NB, NB), 0.5 / args.k)                   # P(fg_k | u,v)
    kappa = args.kappa
    for it in range(args.iters):
        R, acc, cnt = [], np.zeros((args.k, NB, NB)), np.zeros((NB, NB))
        newC = np.zeros_like(C); newG = np.zeros_like(g)
        for ti, idx, u, v in inst:
            z = Zn[ti][idx]
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            lf = kappa * (z @ C.T) + np.log(pi[:, bv, bu].T + 1e-9)  # (n,K)
            lb = kappa * (z @ g) + np.log(1 - pi[:, bv, bu].sum(0) + 1e-9)
            m = np.maximum(lf.max(1), lb)
            pf = np.exp(lf - m[:, None]); pb = np.exp(lb - m)
            tot = pf.sum(1) + pb
            r = pf / tot[:, None]                                   # (n,K)
            R.append(r)
            np.add.at(acc, (slice(None), bv, bu), r.T)
            np.add.at(cnt, (bv, bu), 1)
            newC += r.T @ z
            newG += (pb / tot) @ z
        C = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)
        g = newG / (np.linalg.norm(newG) + 1e-8)
        pi = np.clip(acc / np.maximum(cnt, 1), 1e-4, 1 - 1e-4)
        pi *= (1 - 1e-3) / np.maximum(pi.sum(0), 1)                 # keep sum<1
        mean_r = float(np.mean([r.sum(1).mean() for r in R]))
        print(f"  it{it+1}: mean fg-resp={mean_r:.3f}")

    np.savez(os.path.join(OUT, "em_model.npz"), C=C, g=g, pi=pi, U=U, mu=mu,
             kappa=kappa)
    # spatial priors figure
    fig, axs = plt.subplots(1, args.k, figsize=(3 * args.k, 3), dpi=120)
    for k, a in enumerate(np.atleast_1d(axs)):
        im = a.imshow(pi[k], cmap="viridis", vmin=0, vmax=pi.max())
        a.set_title(f"pi_{k}(u,v)", fontsize=9); a.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.02)
    fig.savefig(os.path.join(OUT, "em_spatial_priors.png"))
    plt.close(fig)

    # GT-box-prompted posterior masks on densest test tiles
    tiles_t, feats_t, _ = collect(split, boxes, "test")
    chosen = []
    for dom in ("WON", "BRU", "NEON"):
        cand = [p for p in tiles_t if split["tiles"][p]["domain"] == dom]
        cand.sort(key=lambda p: -split["tiles"][p]["n_boxes"])
        chosen += cand[:2]
    for p in chosen:
        i = tiles_t.index(p)
        z = (feats_t[i] - mu) @ U
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        img = np.asarray(Image.open(p).convert("RGB"))
        H0, W0 = img.shape[:2]
        fig, ax = plt.subplots(figsize=(7, 7 * H0 / W0), dpi=130)
        ax.imshow(img)
        cy, cx = np.mgrid[0:G, 0:G] * S + S / 2.0
        for j, b in enumerate(boxes.get(p, [])):
            x0, y0, x1, y1 = b
            m = ((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)).ravel()
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            u = (cx.ravel()[idx] - x0) / max(x1 - x0, 1)
            v = (cy.ravel()[idx] - y0) / max(y1 - y0, 1)
            bu = np.minimum((u * NB).astype(int), NB - 1)
            bv = np.minimum((v * NB).astype(int), NB - 1)
            zz = z[idx]
            lf = kappa * (zz @ C.T) + np.log(pi[:, bv, bu].T + 1e-9)
            lb = kappa * (zz @ g) + np.log(1 - pi[:, bv, bu].sum(0) + 1e-9)
            mm = np.maximum(lf.max(1), lb)
            pf = np.exp(lf - mm[:, None]).sum(1)
            r = pf / (pf + np.exp(lb - mm))
            grid = np.full(G * G, np.nan); grid[idx] = r
            rs = np.array(Image.fromarray(
                np.nan_to_num(grid.reshape(G, G)).astype(np.float32)).resize(
                (G * S, G * S), Image.BILINEAR))[:H0, :W0]
            col = np.array(CMAP_COLORS[j % len(CMAP_COLORS)])
            sel = rs >= 0.5
            ov = np.zeros((H0, W0, 4), np.float32)
            ov[sel] = [*col, 0.5]
            ax.imshow(ov)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec="white", lw=0.8))
        ax.axis("off")
        name = cache_key(p)
        ax.set_title(f"EM posterior (GT-box prompted, r>=0.5) {name}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"em_{name}.png"))
        plt.close(fig)
        print(f"rendered em_{name}.png")


CMAP_COLORS = [plt.get_cmap("tab20")(i)[:3] for i in range(20)]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("direction")
    em = sub.add_parser("em")
    em.add_argument("--pca", type=int, default=128)
    em.add_argument("--k", type=int, default=4)
    em.add_argument("--bins", type=int, default=8)
    em.add_argument("--kappa", type=float, default=10.0)
    em.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    {"direction": cmd_direction, "em": cmd_em}[args.cmd](args)


if __name__ == "__main__":
    main()
