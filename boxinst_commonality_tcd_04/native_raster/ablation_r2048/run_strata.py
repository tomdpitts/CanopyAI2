"""Run ablation t03 (tab:strata) / t04 (touching vs isolated prose) at 512 or 2048 via rle_io.

  python -m ...run_strata --script t03|t04 --res 512|2048 --out <json>
512 uses the published inputs (gate: output == ablation/results/<name>.json).
2048 uses native_raster/preds_lace_s{s}_r2048_tau040{,_detscore}.json.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.dirname(HERE)
PKG = os.path.dirname(NR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", choices=("t03", "t04"), required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"REFUSING to overwrite {a.out}"
    from boxinst_commonality_tcd_04.native_raster.ablation_r2048 import rle_io
    from boxinst_commonality_tcd_04.ablation.lib import io
    side = os.path.join(os.path.dirname(a.out), f"side_{a.script}_r{a.res}")
    if a.res == 512:
        kmap = {s: os.path.join(io.PREDS, f"knobbed_s{s}.json") for s in (0, 1, 2)}
        prod = {s: os.path.join(PKG, "confidence", f"preds_knobbed_s{s}_product.json") for s in (0, 1, 2)}
    else:
        kmap = {s: os.path.join(NR, f"preds_lace_s{s}_r{a.res}_tau040_detscore.json") for s in (0, 1, 2)}
        prod = {s: os.path.join(NR, f"preds_lace_s{s}_r{a.res}_tau040.json") for s in (0, 1, 2)}
    rle_io.install(a.res, kmap, side)
    if a.script == "t03":
        from boxinst_commonality_tcd_04.ablation.scripts import t03_strata_tile as T
        io.HERE_ORIG = io.HERE
        # t03 --product builds os.path.join(io.HERE, "confidence", f"preds_knobbed_s{seed}_product.json")
        if a.res != 512:
            real_join = os.path.join
            def join(*p):
                if len(p) == 3 and p[0] == io.HERE and p[1] == "confidence" and p[2].startswith("preds_knobbed_s"):
                    return prod[int(p[2][len("preds_knobbed_s")])]
                return real_join(*p)
            T.os.path.join = join
        sys.argv = [sys.argv[0], "--product", "--out", a.out]
    else:
        from boxinst_commonality_tcd_04.ablation.scripts import t04_strata_instance as T
        sys.argv = [sys.argv[0], "--out", a.out]
    T.main()


if __name__ == "__main__":
    main()
