"""Read per-tile bounding boxes from the cohort CSV.

Boxes are in tile pixel coords (xmin,ymin,xmax,ymax). Because tiles are zero-padded
bottom-right to 512 (top-left aligned), these coords are unchanged in the padded
canvas — no shift needed downstream.
"""
import csv
import os
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_boxes(csv_path):
    """path -> (N,4) float array [xmin,ymin,xmax,ymax], and path -> domain.

    Rows with empty box fields are negative (background) tiles: the tile is still
    registered, with a (0,4) array.
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(REPO, csv_path)
    boxes = defaultdict(list)
    domain = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            path = row["image_path"].strip()
            domain[path] = row["domain"].strip()
            boxes[path]  # touch so empty tiles get a key
            if row["xmin"].strip() == "":
                continue
            boxes[path].append([float(row["xmin"]), float(row["ymin"]),
                                float(row["xmax"]), float(row["ymax"])])
    return ({p: np.asarray(v, dtype=np.float32).reshape(-1, 4)
             for p, v in boxes.items()}, domain)
