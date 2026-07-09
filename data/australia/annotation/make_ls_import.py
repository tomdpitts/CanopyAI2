"""Convert SAM v3 proposals -> Label Studio import (PNG tiles + editable polygon
pre-annotations) + the protocol-aware labeling config.

Browsers can't render the .tif tiles, so we export PNGs to tiles/<domain>/ and
reference them via LS local-files serving (doc root = repo root).

Usage:
  python make_ls_import.py --domains WON --limit 15 --tag won_try
"""
import argparse, json, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # repo root = LS doc root
PROP = os.path.join(ROOT, "tmp/samtest/sam_polygons_v3.json")

CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small" opacity="0.3">
    <Label value="tree" background="#33dd33"/>
  </PolygonLabels>
  <Choices name="flag" toName="image" perRegion="true" choice="multiple" showInline="true">
    <Choice value="truncated"/>
    <Choice value="small"/>
    <Choice value="uncertain"/>
  </Choices>
</View>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="WON")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--tag", default="won_try")
    a = ap.parse_args()
    doms = set(a.domains.split(","))
    d = json.load(open(PROP))
    imgs = [im for im in d["images"] if im["domain"] in doms and im["annotations"]]
    imgs = imgs[: a.limit]

    tasks = []
    for im in imgs:
        W, H = im["width"], im["height"]
        stem = os.path.splitext(os.path.basename(im["image_path"]))[0]
        pngdir = os.path.join(HERE, "tiles", im["domain"]); os.makedirs(pngdir, exist_ok=True)
        pngrel = f"data/australia/annotation/tiles/{im['domain']}/{stem}.png"
        pngabs = os.path.join(ROOT, pngrel)
        if not os.path.exists(pngabs):
            Image.open(im["image_path"]).convert("RGB").save(pngabs)
        result, scores = [], []
        for ai, ann in enumerate(im["annotations"]):
            scores.append(ann.get("sam_score", 0))
            for pi, poly in enumerate(ann.get("polygons", [])):
                if len(poly) < 3:
                    continue
                pts = [[round(x / W * 100, 3), round(y / H * 100, 3)] for x, y in poly]
                result.append({
                    "id": f"p{ai}_{pi}", "type": "polygonlabels",
                    "from_name": "label", "to_name": "image",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"points": pts, "closed": True, "polygonlabels": ["tree"]},
                })
        tasks.append({
            "data": {"image": f"/data/local-files/?d={pngrel}",
                     "domain": im["domain"], "split": im.get("split", ""),
                     "source_tif": im["image_path"], "n_proposals": len(result)},
            # import as ANNOTATIONS (editable), not predictions (read-only) — this is
            # a correction workflow, so the annotator drags/deletes these directly.
            "annotations": [{"result": result}],
        })

    exp = os.path.join(HERE, "exports"); os.makedirs(exp, exist_ok=True)
    tj = os.path.join(exp, f"ls_tasks_{a.tag}.json")
    json.dump(tasks, open(tj, "w"))
    cj = os.path.join(exp, "labeling_config.xml")
    open(cj, "w").write(CONFIG)
    print(f"tiles: {len(tasks)} PNGs -> tiles/{'/'.join(doms)}/")
    print(f"tasks: {tj}  ({sum(len(t['predictions'][0]['result']) for t in tasks)} polygons)")
    print(f"config: {cj}")


if __name__ == "__main__":
    main()
