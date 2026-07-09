"""Bounded exploration grid on the EXPLORE split (greenness, center target).

Pure-Python driver (no shell word-splitting). Picks the single best config by the
explore-set R2-vs-R3 effect; that config is then taken to confirm.py once.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shadow_prior.config import ShadowFeatureConfig  # noqa: E402
from run_efficiency import run  # noqa: E402

FEATS = ["raw", "web"]
OFFSETS = [(2, 20), (10, 50), (20, 80)]
NCH = [1, 2]
GRID = [6, 12, 24, 48]
DRAWS = 20
NTEST = 30


def main():
    rows = []
    for feat in FEATS:
        for (omin, omax) in OFFSETS:
            for nch in NCH:
                label = f"grid_{feat}_o{omin}-{omax}_n{nch}"
                cfg = ShadowFeatureConfig(offset_min=omin, offset_max=omax, offset_steps=8,
                                          aggregation="max", n_channels=nch,
                                          brightness_proxy="greenness")
                out = run(feat, cfg, False, label, GRID, DRAWS, NTEST,
                          os.path.join(HERE, "artifacts", f"eff_{label}.json"), target="ctr")
                by = {c["N"]: c for c in out["curve"]}
                n48 = by.get(48, {})
                rows.append((label, feat,
                             n48.get("ap_mean", {}).get("r1", 0),
                             n48.get("ap_mean", {}).get("r2", 0),
                             n48.get("d23_mean", 0),
                             n48.get("nb_23", {}).get("p_value", 1)))
                print(f"  done {label}", flush=True)

    rows.sort(key=lambda r: -r[4])
    print("\n=== EXPLORE GRID SUMMARY (sorted by d23 at N=48) ===")
    print(f"{'config':24s} {'r1AP':>6s} {'r2AP':>6s} {'d23':>8s} {'p':>7s}")
    for lab, feat, r1, r2, d23, p in rows:
        print(f"{lab:24s} {r1:6.3f} {r2:6.3f} {d23:+8.4f} {p:7.4f}")
    json.dump([{"config": r[0], "feat": r[1], "r1AP": r[2], "r2AP": r[3],
                "d23": r[4], "p": r[5]} for r in rows],
              open(os.path.join(HERE, "artifacts", "grid_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
