"""D1 copy of run_strata_guided.py (untouched): tab:strata (t03) / touching crowns (t04) on LACE (D1 + edge fix + ŝ), seed 0 only.
Pre-registration ABL-D1 items 1-2 (dino_det/README.md). Input: dino_det/results/preds_dino_swinl_s0_test_R3_ef_r2048.json (the
Table 1 row, scored 0.7141 by score_coco); t03 and t04 run unmodified via rle_io at 2048. Gate: pooled-all / re-scored AP50 = 0.7141.

  PYTHONPATH=. .venv/bin/python -u -m boxinst_commonality_tcd_04.native_raster.ablation_r2048.run_strata_d1 --script t03|t04 --out <json>
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)
PREDS = os.path.join(PKG, "dino_det", "results", "preds_dino_swinl_s0_test_R3_ef_r2048.json")
REF_AP50 = 0.7141


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", choices=("t03", "t04"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    from boxinst_commonality_tcd_04.native_raster.ablation_r2048 import rle_io
    from boxinst_commonality_tcd_04.ablation.lib import io
    side = os.path.join(os.path.dirname(a.out), f"side_{a.script}_r2048_d1")
    rle_io.install(2048, {0: PREDS}, side)
    if a.script == "t03":
        from boxinst_commonality_tcd_04.ablation.scripts import t03_strata_tile as T
        from boxinst_commonality_tcd_04.ablation.lib import strata as S
        real_join = os.path.join
        def join(*p):
            if len(p) == 3 and p[0] == io.HERE and p[1] == "confidence" and p[2].startswith("preds_knobbed_s"):
                return PREDS
            return real_join(*p)
        T.os.path.join = join
        real_pooled = S.pooled_ap; seen = {}
        def pooled(per, tids, *args, **kw):                  # gate: the first call is t03's pooled-all AP50
            r = real_pooled(per, tids, *args, **kw)
            if "all" not in seen:
                seen["all"] = float(r[0]); assert abs(seen["all"] - REF_AP50) < 5e-4, f"pooled-all {seen['all']:.4f} != {REF_AP50}"
                print(f"[d1] gate PASS: pooled-all mask AP50 {seen['all']:.4f}", flush=True)
            return r
        S.pooled_ap = pooled; T.S.pooled_ap = pooled
        sys.argv = [sys.argv[0], "--product", "--seeds", "0", "--out", a.out]
        T.main()
    else:
        from boxinst_commonality_tcd_04.ablation.scripts import t04_strata_instance as T
        sys.argv = [sys.argv[0], "--seeds", "0", "--out", a.out]
        T.main()
        got = json.load(open(a.out))["per_seed"]["0"]["ap"]
        v = got.get("mask_mAP50", got.get("AP50"))
        assert v is not None and abs(v - REF_AP50) < 5e-4, f"t04 AP re-score {got} != {REF_AP50}"
        print(f"[d1] gate PASS: t04 AP re-score {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
