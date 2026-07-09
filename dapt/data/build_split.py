"""Build the seeded, domain-stratified 60/20/20 tile split for the detector cohort.

Reads the bbox cohort CSV (phase22X_combined.csv), groups boxes per tile, derives a
site key per tile, and assigns each tile to train/val/test. Stratified within each
domain (WON/BRU/NEON) so every partition holds a proportional mix of all three.

Tiles are non-overlapping (BRU 400px@stride400, WON 500px), so a random subset split
leaks no crown between partitions -- see dapt/PLAN.md D1. Each tile is one image file
(one fixed rotation), so there is no rotation-duplication leakage either.

Usage:
    .venv/bin/python -m dapt.data.build_split \
        --csv data/finetune/phase22X_combined.csv \
        --out dapt/data/split.json --seed 20260702
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .sites import tile_site

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_tiles(csv_path):
    """path -> {domain, site, n_boxes}. One entry per distinct image_path.

    Rows with empty box fields are negative (background) tiles: registered with
    n_boxes unchanged (they contribute a tile but zero boxes).
    """
    tiles = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            path = row["image_path"].strip()
            domain = row["domain"].strip()
            has_box = row["xmin"].strip() != ""
            t = tiles.get(path)
            if t is None:
                tiles[path] = {"domain": domain, "site": tile_site(path, domain),
                               "n_boxes": 1 if has_box else 0}
            else:
                if t["domain"] != domain:
                    raise ValueError(f"tile {path} has mixed domains "
                                     f"{t['domain']} / {domain}")
                t["n_boxes"] += 1 if has_box else 0
    return tiles


def assign_partitions(tiles, ratios, seed):
    """Stratify by domain; seeded shuffle; cut ratios. Returns path -> partition."""
    rng = np.random.default_rng(seed)
    by_domain = defaultdict(list)
    for path, t in tiles.items():
        by_domain[t["domain"]].append(path)

    part = {}
    for domain in sorted(by_domain):
        paths = sorted(by_domain[domain])          # deterministic pre-shuffle order
        rng.shuffle(paths)
        n = len(paths)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        # test takes the remainder so the three always sum to n exactly.
        for i, path in enumerate(paths):
            if i < n_train:
                part[path] = "train"
            elif i < n_train + n_val:
                part[path] = "val"
            else:
                part[path] = "test"
    return part


def summarize(tiles, part):
    """Nested counts: partition -> site -> {tiles, boxes}, plus per-domain totals."""
    per_site = defaultdict(lambda: defaultdict(lambda: {"tiles": 0, "boxes": 0}))
    per_domain = defaultdict(lambda: defaultdict(lambda: {"tiles": 0, "boxes": 0}))
    for path, t in tiles.items():
        p = part[path]
        s = per_site[p][t["site"]]
        s["tiles"] += 1
        s["boxes"] += t["n_boxes"]
        d = per_domain[p][t["domain"]]
        d["tiles"] += 1
        d["boxes"] += t["n_boxes"]
    return per_site, per_domain


def print_table(tiles, part):
    per_site, per_domain = summarize(tiles, part)
    parts = ["train", "val", "test"]
    site_domain = {t["site"]: t["domain"] for t in tiles.values()}

    print(f"\n{'site':10s} {'dom':4s} " + " ".join(f"{p:>13s}" for p in parts)
          + f" {'total':>7s}")
    print("-" * 66)
    order = sorted(site_domain, key=lambda s: (site_domain[s], s))
    for site in order:
        cells, tot = [], 0
        for p in parts:
            c = per_site[p][site]
            cells.append(f"{c['tiles']:3d}t/{c['boxes']:4d}b")
            tot += c["tiles"]
        print(f"{site:10s} {site_domain[site]:4s} "
              + " ".join(f"{c:>13s}" for c in cells) + f" {tot:5d}t")
    print("-" * 66)
    for domain in sorted(per_domain["train"].keys() | per_domain["val"].keys()
                         | per_domain["test"].keys()):
        cells, tot = [], 0
        for p in parts:
            c = per_domain[p][domain]
            cells.append(f"{c['tiles']:3d}t/{c['boxes']:4d}b")
            tot += c["tiles"]
        print(f"{'ALL '+domain:10s} {'':4s} "
              + " ".join(f"{c:>13s}" for c in cells) + f" {tot:5d}t")
    gt = sum(1 for _ in tiles)
    gb = sum(t["n_boxes"] for t in tiles.values())
    print("-" * 66)
    print(f"{'TOTAL':10s} {'':4s} " + " " * 41 + f" {gt:5d}t/{gb}b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/finetune/phase22X_combined.csv")
    ap.add_argument("--out", default="dapt/data/split.json")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    args = ap.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(REPO, args.csv)
    tiles = load_tiles(csv_path)
    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    part = assign_partitions(tiles, ratios, args.seed)
    print_table(tiles, part)

    out = {
        "seed": args.seed,
        "ratios": list(ratios),
        "csv": os.path.relpath(csv_path, REPO),
        "n_tiles": len(tiles),
        "tiles": {path: {**t, "partition": part[path]}
                  for path, t in sorted(tiles.items())},
    }
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(out_path, REPO)}  "
          f"({len(tiles)} tiles, seed={args.seed})")


if __name__ == "__main__":
    main()
