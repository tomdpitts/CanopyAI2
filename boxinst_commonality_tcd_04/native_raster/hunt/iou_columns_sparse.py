"""Matched-instance mean mask IoU / mean box IoU (tab:strata-maskiou instrument, box_iou_strata.py) on the
sparse-236 holdout, adopted 4 px masker (lace4px_simple) vs seam-fixed DetecTree2. Same matcher as the 439
column: greedy by score, box IoU >= 0.5, canopy-landing preds dropped; mask IoU is the dependent variable.

--gate: re-derives LACE s0 lace4px_simple on the 439 test set; must equal
ablation/results_r2048/box_iou_strata_lace4px_simple_r2048.json (n 19814, mask 0.7161, box 0.7209).

  .venv/bin/python -m boxinst_commonality_tcd_04.native_raster.hunt.iou_columns_sparse \
      --out boxinst_commonality_tcd_04/ablation/results_r2048/box_iou_strata_sparse236_lace4px_simple_r2048.json
"""
import argparse, json, os, sys

from boxinst_commonality_tcd_04.box_supervised_baselines import box_iou_strata as B

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.dirname(NR)
REPO = os.path.dirname(PKG)
SPARSE_GT = os.path.join(REPO, "data", "tcd_sparse", "sparse_gt.json")
ROWS = {f"LACE s{s} (product)": f"{NR}/hunt/test/preds_lace_sparse_s{s}_r2048_lace4px_simple.json" for s in (0, 1, 2)}
ROWS["DetecTree2 s0"] = f"{NR}/seamfix/preds_dt2_sparse_s0v2_r2048_seamfix.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    B.RES, B.SCALE = 2048, 1.0
    if a.gate:
        gt = json.load(open(B.GT))
        d = json.load(open(f"{NR}/hunt/test/preds_lace_s0_r2048_lace4px_simple.json"))
        pr = B.pairs_for_row(d.get("preds", d), gt)
        n = len(pr); bm = sum(v[0] for v in pr.values()) / n; mm = sum(v[1] for v in pr.values()) / n
        print(f"[gate] n={n} box={bm:.4f} mask={mm:.4f}  expect 19814 / 0.7209 / 0.7161", flush=True)
        assert (n, round(bm, 4), round(mm, 4)) == (19814, 0.7209, 0.7161), "GATE FAILED"
        print("[gate] PASS", flush=True)
        return
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    missing = [v for v in ROWS.values() if not os.path.exists(v)]
    assert not missing, missing
    B.GT, B.ROWS = SPARSE_GT, dict(ROWS)
    print(f"[iou_columns_sparse] {B.ROWS}", flush=True)
    sys.argv = [sys.argv[0], "--out", a.out]
    B.main()


if __name__ == "__main__":
    main()
