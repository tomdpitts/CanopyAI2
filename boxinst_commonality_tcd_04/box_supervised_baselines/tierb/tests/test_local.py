"""Local CPU checks (no GPU, no detectron2). Writes gates/local_tests.json; all must pass.

    PYTHONPATH=.:boxinst_commonality_tcd_04/box_supervised_baselines \
        .venv/bin/python -m tierb.tests.test_local
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from tierb import crops as C
from tierb import lace_gate as LG
from tierb.d2 import d2_patches as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_locations(h, w, stride):
    """adet.utils.comm.compute_locations @6692d92, verbatim."""
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return torch.stack((shift_x, shift_y), dim=1) + stride // 2


def t_crops():
    assert C.coverage_check()
    s = C.schedule()
    assert s == {"iters_per_epoch": 198, "max_iter": 3960, "max_epochs": 20, "steps": (),
                 "schedule": "plateau (plateau.py)", "warmup_iters": 26, "pairwise_warmup_iters": 264,
                 "boxteacher_mask_warmup_iters": 264, "fraction_of_released_1x": 0.0264}, s
    # every original pixel covered >=1 and <=2 times per epoch, for every combination of scales
    import itertools
    for sa, sb in itertools.product(C.SCALES, repeat=2):
        cov = np.zeros(C.TILE, int)
        for L, off in ((C.scaled_size(sa), 0), (C.scaled_size(sb), C.scaled_size(sb) - C.CROP)):
            a, b = C.original_interval(L, off)
            cov[int(np.floor(a)):int(np.ceil(b))] += 1
        assert cov.min() >= 1 and cov.max() <= 2, (sa, sb, cov.min(), cov.max())
    assert [C.teacher_size(C.scaled_size(x)) for x in C.SCALES] == [1280, 1219, 1164, 1113, 1067, 1024]
    return {"coverage_1_to_2x": True, "schedule": s}


def t_p6():
    g = torch.Generator().manual_seed(0)
    n = 0
    for trial in range(2000):
        M, N = int(torch.randint(0, 60, (1,), generator=g)), int(torch.randint(0, 80, (1,), generator=g))
        iou = torch.round(torch.rand((M, N), generator=g) * (torch.rand((M, N), generator=g) > 0.5) * 20) / 20
        sc = torch.round(torch.rand(M, generator=g) * 10) / 10
        ref = P.greedy_match_released(iou, sc, 0.5, 0.0)
        new = P.greedy_match_cpu(iou, sc, torch.argsort(sc, descending=True).tolist(), 0.5, 0.0)
        assert ref == new, trial
        n += 1
    return {"random_cases_equal": n}


def t_canopy_rule():
    from boxinst_commonality_tcd_04.detector import canopy_cell_mask
    rng = np.random.default_rng(1)
    for trial in range(20):
        c = np.zeros((1024, 1024), np.uint8)
        for _ in range(6):                                  # blobs with ragged edges
            y, x, r = rng.integers(0, 1024, 2).tolist() + [int(rng.integers(20, 200))]
            yy, xx = np.ogrid[:1024, :1024]
            c[(yy - y) ** 2 + (xx - x) ** 2 < r * r] = 1
        ct = torch.from_numpy(c)
        strides = [8, 16, 32, 64, 128]
        locs, lv, im = [], [], []
        for l, s in enumerate(strides):
            loc = compute_locations(1024 // s, 1024 // s, s)
            locs.append(loc); lv.append(torch.full((len(loc),), l, dtype=torch.long))
            im.append(torch.zeros(len(loc), dtype=torch.long))
        inst = SimpleNamespace(locations=torch.cat(locs), fpn_levels=torch.cat(lv), im_inds=torch.cat(im))
        ign = P.canopy_ignore(inst, [ct], strides)
        k = 0
        for l, s in enumerate(strides):
            g = 1024 // s
            want = canopy_cell_mask(c.astype(bool), g)            # LACE's own function
            got = ign[k:k + g * g].reshape(g, g).numpy()
            assert np.array_equal(got, want), (trial, s)
            k += g * g
    return {"lace_canopy_cell_mask_equal_all_levels": 20}


def t_gate():
    ref = LG.LACE[("test", "product")]
    cases = [({"AP50": ref["AP50"] + 0.0099, "AP50_95": ref["AP50_95"]}, False),
             ({"AP50": ref["AP50"] + 0.0100, "AP50_95": 0.0}, True),
             ({"AP50": ref["AP50"], "AP50_95": ref["AP50_95"] + 0.0100}, True),
             ({"AP50": ref["AP50"] - 0.0001, "AP50_95": ref["AP50_95"] + 0.05}, False)]
    for row, want in cases:
        r = LG.evaluate([dict(row, name="x", split="test", ranking="product")])
        assert r["triggered"] == want, (row, r)
    assert LG.evaluate([], gtbox={"mean_diff": 0.0101, "ci95": [0.001, 0.02]})["triggered"]
    assert not LG.evaluate([], gtbox={"mean_diff": 0.0101, "ci95": [-0.001, 0.02]})["triggered"]
    assert not LG.evaluate([], gtbox={"mean_diff": 0.0099, "ci95": [0.001, 0.02]})["triggered"]
    return {"boundary_cases": len(cases) + 3}


def t_mal_clip():
    from tierb.mal import mal_run as MR
    inside = [12.3, 40.0, 30.5, 22.25]
    assert MR.clip_xywh(inside) == inside                      # untouched floats
    assert MR.clip_xywh([-3.5, 10.0, 20.0, 5.0]) == [0.0, 10.0, 16.5, 5.0]
    assert MR.clip_xywh([2040.0, -2.0, 20.0, 10.0]) == [2040.0, 0.0, 8.0, 8.0]
    assert MR.clip_xywh([-30.0, 5.0, 10.0, 5.0]) is None       # entirely outside
    import json as _j
    bs = _j.load(open(os.path.join(HERE, "tierb_data", "boxsets", "lace_s1.json")))["tiles"]
    tb = {t: v["xywh"] for t, v in list(bs.items())[:60]}
    items, empty = MR.make_items(tb)
    seen = {}
    for tid, idx, xy in items:
        assert len(idx) == len(xy)
        for k, c in zip(idx, xy):
            assert c == MR.clip_xywh(tb[tid][k])
            seen.setdefault(tid, []).append(k)
    for tid, ks in empty.items():
        seen.setdefault(tid, []).extend(ks)
    for tid, b in tb.items():
        assert sorted(seen.get(tid, [])) == list(range(len(b))), tid
    n_clip = sum(1 for b in tb.values() for x in b if MR.clip_xywh(x) != [float(v) for v in x])
    return {"tiles": len(tb), "boxes": sum(map(len, tb.values())), "clipped": n_clip,
            "empty_after_clip": sum(map(len, empty.values()))}


def t_plateau():
    import tempfile
    from tierb import plateau as PL
    H = lambda ms: [{"epoch": i + 1, "metric": m} for i, m in enumerate(ms)]
    # gains below min_delta do not count; 2 flat epochs (from epoch>=4) -> decay; again -> decay; again -> stop
    st = PL.replay(H([.40, .45, .50, .503, .504, .52, .521, .522, .523, .524, .525]))
    ev = [(e["epoch"], e["event"]) for e in st["events"]]
    assert ev == [(5, "decay 1"), (8, "decay 2"), (10, "stop")], ev
    assert abs(st["lr_factor"] - 0.01) < 1e-12 and st["best_epoch"] == 10 and st["stop"]
    # nothing fires before min_epochs, even if flat from the start
    st = PL.replay(H([.5, .5, .5]))
    assert st["decays"] == 0 and not st["stop"]
    # cap: still improving at epoch 20 -> flagged lower bound
    st = PL.replay(H([.30 + .01 * i for i in range(20)]))
    assert st["stop"] and "lower bound" in st["reason"] and st["decays"] == 0
    # replay idempotence: re-recording an epoch replaces it
    d = tempfile.mkdtemp()
    p = PL.Plateau(os.path.join(d, "p.json"))
    for e, m in enumerate([.4, .45, .5, .5, .5], 1):
        p.record(e, m)
    s1 = dict(p.state)
    p.record(5, .5)
    p2 = PL.Plateau(os.path.join(d, "p.json"))
    assert p2.state == s1 == p.state and p2.lr_factor == 0.1
    # smoke params: decay at epoch 2 and 3 (d2 smoke) / epoch 2 (MAL smoke)
    st = PL.replay(H([.4, .4, .4]), **PL.SMOKE)
    assert st["decays"] == 2 and not st["stop"], st
    return {"cases": 5}


def main():
    out = {"date": time.strftime("%Y-%m-%d %H:%M")}
    for name, fn in (("crops", t_crops), ("p6_assignment", t_p6), ("canopy_rule", t_canopy_rule),
                     ("lace_gate", t_gate), ("mal_clip", t_mal_clip),
                     ("plateau", t_plateau)):
        out[name] = fn()
        print(f"PASS {name}: {out[name]}", flush=True)
    out["pass"] = True
    p = os.path.join(HERE, "gates", "local_tests.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    print("LOCAL TESTS PASS ->", p)


if __name__ == "__main__":
    main()
