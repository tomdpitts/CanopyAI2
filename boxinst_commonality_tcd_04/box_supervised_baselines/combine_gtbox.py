"""Merge the GT-box bake-off arms into one table.

Every arm was prompted with the same boxes and scored on the same GT-derived valid set, so the
merge is a straight per-crown join -- and this script ENFORCES that rather than assuming it:
`_align` fails if two arms disagree about which crowns exist or which are valid. A silent
mismatch there would produce a table whose rows were computed on different crowns.

Reports the same shape as the superseded masker_lab bake-off (mean IoU, frac >= 0.5/0.75/0.9,
double-assignment) plus a paired per-crown delta against LACE, which is the honest statistic:
the arms are matched on every crown, so a paired comparison is available and a difference of
means is not the right test.

Usage:
    .venv/bin/modal volume get tcd04-baselines-vol out/gtbox_lace.json      <dir>
    .venv/bin/modal volume get tcd04-baselines-vol out/gtbox_sam3_base.json <dir>
    .venv/bin/modal volume get tcd04-baselines-vol out/gtbox_sam3_ft.json   <dir>
    .venv/bin/python -m boxinst_commonality_tcd_04.box_supervised_baselines.combine_gtbox \
        --dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from boxinst_commonality_tcd_04.box_supervised_baselines import gtbox_lib as L

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"lace": "LACE commonality-EM (shipped)",
        "sam3_base": "SAM 3, zero-shot",
        "sam3_ft": "SAM 3, SelvaMask-FT"}
REF = "lace"


def _align(loaded):
    """-> (tids, {arm: iou array}) over the crowns every arm scored, valid ones only."""
    tids = sorted(set.intersection(*(set(d["per_tile"]) for d in loaded.values())))
    missing = {a: sorted(set(d["per_tile"]) - set(tids)) for a, d in loaded.items()}
    for a, m in missing.items():
        if m:
            print(f"[combine] {a}: {len(m)} tiles not shared with every arm, dropped", flush=True)
    cols = {a: [] for a in loaded}
    for t in tids:
        ref_valid = None
        for a, d in loaded.items():
            rec = d["per_tile"][t]
            v = np.asarray(rec["valid"], bool)
            if ref_valid is None:
                ref_valid = v
            elif not np.array_equal(v, ref_valid):
                raise ValueError(f"{t}: arms disagree on the valid set -- {a} differs. The "
                                 f"valid set is GT-derived and must be identical everywhere.")
            iou = np.asarray(rec["iou"], np.float64)
            if len(iou) != len(v):
                raise ValueError(f"{t}: {a} returned {len(iou)} IoUs for {len(v)} crowns")
            cols[a].append(iou[v])
    return tids, {a: np.concatenate(v) if v else np.zeros(0) for a, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PKG, "results_439"))
    ap.add_argument("--res", type=int, default=0,
                    help="0 = the original un-suffixed 512 files; otherwise gtbox_<arm>_r<res>")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suf = f"_r{a.res}" if a.res else ""
    if a.out is None:
        a.out = os.path.join(PKG, "results_439", f"gtbox_bakeoff{suf}.json")

    loaded = {}
    for arm in ARMS:
        fp = os.path.join(a.dir, f"gtbox_{arm}{suf}.json")
        if os.path.exists(fp):
            loaded[arm] = json.load(open(fp))
        else:
            print(f"[combine] SKIP {arm}: {fp} not found", flush=True)
    if not loaded:
        raise SystemExit("no arms found -- pull the gtbox_*.json files off the volume first")

    tids, cols = _align(loaded)
    n = len(next(iter(cols.values())))
    print(f"[combine] {len(tids)} tiles · {n} valid crowns · arms: {', '.join(loaded)}")

    rows = {}
    for arm, iou in cols.items():
        dbl = [v["double"] for v in loaded[arm]["per_tile"].values() if v["double"] is not None]
        r = L.summarise(iou, {"mean_double_assignment": round(float(np.mean(dbl)), 4)
                              if dbl else None})
        if arm != REF and REF in cols:
            d = iou - cols[REF]
            r["paired_delta_vs_lace"] = {
                "mean": round(float(d.mean()), 4),
                "win_rate": round(float((d > 0).mean()), 4),   # frac of crowns this arm wins
                "sd": round(float(d.std(ddof=1)), 4) if len(d) > 1 else None}
        rows[ARMS[arm]] = r

    json.dump({"n_tiles": len(tids), "n_crowns": int(n), "res": a.res or 512,
               "prompt": "ground-truth crown boxes, identical across arms",
               "rows": rows}, open(a.out, "w"), indent=2)

    w = max(len(x) for x in rows)
    print(f"\n{'arm'.ljust(w)} | mean IoU | >=0.5  | >=0.75 | >=0.9  | dbl-assign | vs LACE")
    print("-" * (w + 68))
    for lab, r in rows.items():
        d = r.get("paired_delta_vs_lace")
        print(f"{lab.ljust(w)} |  {r['mean_iou']:.4f}  | {r['rates']['>=0.5']:.4f} | "
              f"{r['rates']['>=0.75']:.4f} | {r['rates']['>=0.9']:.4f} | "
              f"{str(r['mean_double_assignment']).ljust(10)} | "
              + (f"{d['mean']:+.4f} (wins {d['win_rate']:.1%})" if d else "--"))
    floor = {512: 0.0048, 2048: 0.0015}.get(a.res or 512)
    print(f"\nGT crowns overlap each other by {floor:.4f} at this raster (mean over 120 tiles), "
          f"so read double-assignment against that floor, not zero.\n-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
