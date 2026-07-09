"""Is the directional shadow feature informative about crowns at patch resolution?

No probe: use the shadow crown-response itself as the score and measure AP vs the
occupancy / center targets, correct azimuth vs within-acq-shuffled vs base rate.
If correct ~ shuffled ~ base-rate, the patch-resolution probing simply cannot see
the directional cue (the matched filter is washed out by 16x pooling) -- that
bounds the whole Track-1 framing. Also reports it split by acquisition (dryland
WON/BRU vs temperate NEON).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from shadow_prior.config import ShadowFeatureConfig
from data import CACHE, confirm_explore_split, load_cohort
from probe import average_precision, shadow_patches
from run_efficiency import az_dicts

CFGS = {
    "default(2-20,lum,max)": ShadowFeatureConfig(),
    "offset10-50": ShadowFeatureConfig(offset_min=10, offset_max=50),
    "offset20-80": ShadowFeatureConfig(offset_min=20, offset_max=80),
    "greenness": ShadowFeatureConfig(brightness_proxy="greenness"),
    "logsumexp": ShadowFeatureConfig(aggregation="logsumexp"),
    "nch2-off10-50": ShadowFeatureConfig(offset_min=10, offset_max=50, n_channels=2),
}


def shadow_score_ap(recs, az, cfg, target):
    """AP of the (channel-0) shadow response as a direct score, pooled over scenes."""
    S, Y = [], []
    for r in recs:
        rgb = np.load(os.path.join(CACHE, "rgb512", r.scene + ".npy")).astype(np.float32) / 255.0
        sp = shadow_patches(rgb, az[r.scene], cfg)[:, 0]    # crown response per patch
        y = np.load(os.path.join(CACHE, target, r.scene + ".npy")).reshape(-1)
        S.append(sp); Y.append(y)
    S = np.concatenate(S); Y = np.concatenate(Y)
    return average_precision(S, Y), float(Y.mean())


def main():
    recs, _ = confirm_explore_split(load_cohort())
    az_c, az_s = az_dicts(recs)
    won_bru = [r for r in recs if r.acq in ("WON", "BRU")]
    neon = [r for r in recs if r.acq == "NEON"]
    print(f"explore={len(recs)} (WON+BRU dryland={len(won_bru)}, NEON temperate={len(neon)})\n")
    for tgt in ("occ", "ctr"):
        print(f"=== target={tgt} : AP of shadow response alone (correct / shuffled), base-rate ===")
        for name, cfg in CFGS.items():
            apc, base = shadow_score_ap(recs, az_c, cfg, tgt)
            aps, _ = shadow_score_ap(recs, az_s, cfg, tgt)
            apc_d, _ = shadow_score_ap(won_bru, az_c, cfg, tgt)
            aps_d, _ = shadow_score_ap(won_bru, az_s, cfg, tgt)
            print(f"  {name:18s} all: correct={apc:.3f} shuf={aps:.3f} (base={base:.3f}) | "
                  f"dryland: correct={apc_d:.3f} shuf={aps_d:.3f}")
        print()


if __name__ == "__main__":
    main()
