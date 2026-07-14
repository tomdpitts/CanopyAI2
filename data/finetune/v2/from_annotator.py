#!/usr/bin/env python3
"""Convert an annotator CSV export back to v2_combined format.

The annotator works on annotate_png/ (browser-viewable PNG copies, basename paths).
This restores absolute paths into images/, mapping basenames back to the original
.tif where one exists. Empty-tile rows and per-tile domains pass through unchanged.

Usage: python3 from_annotator.py <exported.csv> [output.csv]
       (defaults: ~/Downloads/crown_annotations.csv -> v2_combined.csv)
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/crown_annotations.csv")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "v2_combined.csv")

rows, missing = [], []
with open(src) as f:
    reader = csv.reader(f)
    header = next(reader)
    assert header[:5] == ["image_path", "xmin", "ymin", "xmax", "ymax"], f"unexpected header: {header}"
    for row in reader:
        if not row or not row[0].strip():
            continue
        base = os.path.basename(row[0])
        stem = os.path.splitext(base)[0]
        for cand in (stem + ".tif", stem + ".png"):
            if os.path.isfile(os.path.join(IMG_DIR, cand)):
                rows.append([os.path.join(IMG_DIR, cand)] + row[1:])
                break
        else:
            missing.append(base)

if missing:
    sys.exit(f"ERROR: {len(missing)} image(s) not found in {IMG_DIR}: {missing[:5]}...")

with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "label", "domain"])
    w.writerows(rows)

n_empty = sum(1 for r in rows if r[1] == "")
n_tiles = len({r[0] for r in rows})
print(f"wrote {dst}: {len(rows)} rows ({n_empty} empty-tile) across {n_tiles} tiles")
