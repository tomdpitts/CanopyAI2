"""Build the pinned NEON scored plot list + ground-truth boxes (Step 1).

Reads the PINNED NeonTreeEvaluation checkout (tag 1.8.0), takes the INTERSECTION of
annotation XMLs and evaluation/RGB tiles (scored set = XMLs that have imagery), parses
PASCAL-VOC boxes (xmin,ymin,xmax,ymax, image pixel coords), and writes neon_gt.json.

RGB-ONLY: reads only annotations/*.xml and evaluation/RGB/*.tif. No LiDAR/CHM/HSI.

Usage: .venv/bin/python -m boxinst_commonality_tcd_04.mps_neon_multiseed.prepare_neon
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(HERE, "NeonTreeEvaluation")
RGB_DIR = os.path.join(REPO, "evaluation", "RGB")
ANN_DIR = os.path.join(REPO, "annotations")
OUT_JSON = os.path.join(HERE, "neon_gt.json")
PINNED_TAG = "1.8.0"


def parse_voc(xml_path):
    """PASCAL-VOC -> (list of [xmin,ymin,xmax,ymax] float), (width,height)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text)) if size is not None else None
    h = int(float(size.find("height").text)) if size is not None else None
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append([float(b.find("xmin").text), float(b.find("ymin").text),
                      float(b.find("xmax").text), float(b.find("ymax").text)])
    return boxes, (w, h)


def site_of(plot):
    """NEON site code, e.g. 2018_SJER_3_..._image_628 -> SJER."""
    parts = plot.split("_")
    return parts[1] if len(parts) > 1 and parts[0].isdigit() else parts[0]


def main():
    xmls = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(ANN_DIR, "*.xml"))}
    rgbs = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(RGB_DIR, "*.tif"))}
    plots = sorted(set(xmls) & rgbs)             # scored set: XML AND RGB present

    gt, sizes, per_site, total = {}, {}, {}, 0
    nonstd_size = []
    for plot in plots:
        boxes, (w, h) = parse_voc(xmls[plot])
        gt[plot] = boxes
        total += len(boxes)
        per_site[site_of(plot)] = per_site.get(site_of(plot), 0) + len(boxes)
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        if (w, h) != (400, 400):
            nonstd_size.append((plot, w, h))

    json.dump(gt, open(OUT_JSON, "w"))
    print(f"pinned tag       : {PINNED_TAG}")
    print(f"scored plots     : {len(plots)}  (intersection of {len(xmls)} XML "
          f"and {len(rgbs)} RGB tiles)")
    print(f"total GT boxes   : {total}")
    print(f"NEON sites       : {len(per_site)}")
    print(f"tile sizes       : {dict(sizes)}")
    if nonstd_size:
        print(f"NON-400x400 tiles ({len(nonstd_size)}): {nonstd_size[:10]}")
    print(f"per-site boxes   : "
          + ", ".join(f"{s}={n}" for s, n in sorted(per_site.items())))
    print(f"wrote            : {OUT_JSON}")


if __name__ == "__main__":
    main()
