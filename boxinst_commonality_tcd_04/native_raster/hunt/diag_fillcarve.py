"""Which per-detection signal separates 'fill the box' (rhalf GAIN: tight box, deployed under-segments) from
'carve' (rhalf LOSS: loose box, filling over-segments)? Small crowns, val, from diag/pair_deployed_vs_rhalf.json.
Signals (all computable at inference): box short side; deployed mask fill of box; posterior mean inside the box,
in the inner half, in the box border band, and in a 16-px ring outside the box; colour distance between the box's
inner half and its border band, and between inner half and outside ring (Lab-free, RGB, normalised by in-box std).
Prints per-signal medians and a rank AUC (gain vs loss). -> diag/fillcarve_signals.json"""
import json, os
import numpy as np
from pycocotools import mask as mu
from boxinst_commonality_tcd_04 import evaluate as E
from boxinst_commonality_tcd_04.native_raster import guided_ablate_modal as GA
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); FC = "/Users/tompitts/dphil/feat_cache"
OUT = os.path.join(HERE, "diag", "fillcarve_signals.json"); assert not os.path.exists(OUT)


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b); x = np.concatenate([a, b]); r = x.argsort().argsort() + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def signals(prob, rgb, b, dep):
    x0, y0, x1, y1 = b; by0, by1, bx0, bx1 = max(0, int(y0)), int(np.ceil(y1)), max(0, int(x0)), int(np.ceil(x1))
    H, W = by1 - by0, bx1 - bx0
    inb = np.zeros((2048, 2048), bool); inb[by0:by1, bx0:bx1] = True
    iy0, iy1, ix0, ix1 = by0 + H // 4, by1 - H // 4, bx0 + W // 4, bx1 - W // 4
    inner = np.zeros_like(inb); inner[iy0:max(iy1, iy0 + 1), ix0:max(ix1, ix0 + 1)] = True
    border = inb & ~inner
    ring = np.zeros_like(inb); ring[max(0, by0 - 16):by1 + 16, max(0, bx0 - 16):bx1 + 16] = True; ring &= ~inb
    sd = rgb[inb].std(0).mean() + 1e-3
    cd = lambda A, B: float(np.linalg.norm(rgb[A].mean(0) - rgb[B].mean(0)) / sd) if A.any() and B.any() else 0.0
    return {"short": float(min(x1 - x0, y1 - y0)), "dep_fill": float(dep[inb].mean()), "p_box": float(prob[inb].mean()),
            "p_inner": float(prob[inner].mean()), "p_border": float(prob[border].mean()) if border.any() else 0.0,
            "p_ring": float(prob[ring].mean()), "p_border/p_inner": float(prob[border].mean() / max(prob[inner].mean(), 1e-3)) if border.any() else 1.0,
            "col_inner_border": cd(inner, border), "col_inner_ring": cd(inner, ring), "col_border_ring": cd(border, ring)}


def main():
    rows = [r for r in json.load(open(os.path.join(HERE, "diag", "pair_deployed_vs_rhalf.json")))["rows"] if r["areaG"] < 622]
    m = E.TCDMasker(f"{FC}/em_model_4p_fix.npz"); cache = {}; S = {"gain": [], "loss": []}
    for r in sorted(rows, key=lambda r: r["tile"]):
        t = r["tile"]
        if t not in cache:
            cache.clear(); feat = np.load(f"{FC}/feat_4p_val/{t}.npy").astype(np.float32); g = feat.shape[-1]
            cache[t] = (m.project(feat), g, np.asarray(Image.open(f"/Users/tompitts/dphil/CanopyAI2/data/tcd/train/{t}.tif").convert("RGB")).astype(np.float32) / 255,
                        json.load(open(f"{FC}/boxfix_ap_val2_tiles/{t}.json")))
        zn, g, rgb, L = cache[t]
        b = np.asarray(L["boxes_2048"][r["det"]], np.float32)
        _, prob, _ = GA._prob(m, zn, g, b, GA.ALPHA, GA.KAPPA)
        dep = mu.decode({"size": [2048, 2048], "counts": L["masks_rle"]["deployed"][r["det"]]["counts"].encode()}).astype(bool)
        S["gain" if r["gain"] else "loss"].append(signals(prob, rgb, b, dep))
    keys = list(S["gain"][0])
    out = {k: {"gain_median": round(float(np.median([s[k] for s in S["gain"]])), 3), "loss_median": round(float(np.median([s[k] for s in S["loss"]])), 3),
               "auc_gain_gt_loss": round(auc([s[k] for s in S["gain"]], [s[k] for s in S["loss"]]), 3)} for k in keys}
    out["n"] = {"gain": len(S["gain"]), "loss": len(S["loss"])}
    json.dump({"summary": out, "signals": S}, open(OUT, "w"), indent=1)
    for k, v in out.items(): print(k, v)


if __name__ == "__main__":
    main()
