"""Local Phase-0 sanity check: build COCO for a few cohort tiles from data/tcd and render
crown (green) vs canopy-ignore (red) overlays on subtiles, so we can eyeball the supervision
before scaling to Modal. Writes overlays to claude_outputs/detectree2_baseline/.
"""
import json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_coco as B

ROOT = "/Users/tompitts/dphil/CanopyAI2"
OUT = os.path.join(ROOT, "claude_outputs/detectree2_baseline")
os.makedirs(OUT, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None


def load_tile(tid, split):
    img = np.asarray(Image.open(f"{ROOT}/data/tcd/{split}/{tid}.tif").convert("RGB"))
    meta = json.load(open(f"{ROOT}/data/tcd/{split}/{tid}_meta.json"))
    ca = meta["coco_annotations"]
    anns = json.loads(ca) if isinstance(ca, str) else ca
    return img, anns


def render(tid, img, images, annotations, crops, k=0):
    """Overlay one subtile's crown polygons (green) + canopy-ignore (red) on its pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycocotools import mask as maskUtils
    fn, pix = crops[k]
    iid = images[k]["id"]
    a_sub = [a for a in annotations if a["image_id"] == iid]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(pix)
    ncrown = ncan = 0
    for a in a_sub:
        ignore = a["iscrowd"] == 1
        seg = a["segmentation"]                       # compressed COCO RLE
        m = maskUtils.decode({"size": seg["size"],
                              "counts": seg["counts"].encode("ascii")}).astype(float)
        # outline crowns (green), fill canopy-ignore (red translucent)
        ax.contour(m, levels=[0.5], colors="red" if ignore else "#20ff20", linewidths=1.2)
        if ignore:
            ax.imshow(np.dstack([m, np.zeros_like(m), np.zeros_like(m), m * 0.25]))
        ncrown += not ignore
        ncan += ignore
    ax.set_title(f"{fn}  crowns(green)={ncrown}  canopy-ignore(red)={ncan}", fontsize=10)
    ax.axis("off")
    p = os.path.join(OUT, f"overlay_{fn}")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, ncrown, ncan


def main():
    gt = json.load(open(f"{ROOT}/boxinst_commonality_tcd_04/train_tiles_gt.json"))
    train = [t for t, v in gt.items() if v["partition"] == "train"]
    # pick a few crown-rich tiles
    tids = train[:4]
    iid = aid = 0
    tot_sub = tot_crown = tot_can = 0
    sizes = []
    rendered = []
    for ti, tid in enumerate(tids):
        img, anns = load_tile(tid, "train")
        imgs, ann, crops = B.build_tile(tid, img, anns, iid, aid)
        iid += len(imgs); aid += len(ann)
        tot_sub += len(imgs)
        tot_crown += sum(a["iscrowd"] == 0 for a in ann)
        tot_can += sum(a["iscrowd"] == 1 for a in ann)
        sizes += [np.sqrt(a["bbox"][2] * a["bbox"][3]) for a in ann if a["iscrowd"] == 0]
        # render the subtile with the most crowns
        if crops:
            by_iid = {im["id"]: sum(a["image_id"] == im["id"] and a["iscrowd"] == 0
                                    for a in ann) for im in imgs}
            best = max(range(len(imgs)), key=lambda k: by_iid[imgs[k]["id"]])
            rendered.append(render(tid, img, imgs, ann, crops, best))
    sizes = np.array(sizes)
    print(f"tiles={len(tids)}  subtiles(kept)={tot_sub}  crowns={tot_crown}  "
          f"canopy-ignore={tot_can}")
    print(f"crown sqrt-area px @1024: median={np.median(sizes):.1f} "
          f"p10={np.percentile(sizes,10):.1f} p90={np.percentile(sizes,90):.1f} "
          f"max={sizes.max():.1f}")
    print(f"mean crowns/subtile={tot_crown/max(tot_sub,1):.1f}  "
          f"(dense -> raise DETECTIONS_PER_IMAGE)")
    for p, nc, ncan in rendered:
        print(f"  rendered {p}  crowns={nc} canopy={ncan}")


if __name__ == "__main__":
    main()
