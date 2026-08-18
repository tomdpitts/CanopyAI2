"""Structural guards for the SavannaTree v2 pipeline.

These are not unit tests of behaviour so much as tests of the DISCIPLINE the v2 plan
rests on. Each one corresponds to a specific way this project has previously produced a
number that could not be trusted:

  * test_single_metric_path        -- three times, a value computed by one metric path was
                                      compared against a value computed by another and the
                                      difference was read as an effect.
  * test_gamma_zero_equivalence    -- the IoU-quality head must be a pure ranking change,
                                      or its ablation against the plain detector is not a
                                      controlled comparison.
  * test_masker_origin_consistency -- phase4 fitted every savanna masker with its
                                      labels half a cell away from its instance cells.
  * test_no_val_selection          -- v2 has no val site; a `best epoch on val` code path
                                      must be impossible, not merely unused.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pytest
import torch

V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "boxinst_commonality_tcd_04", "modal_savanna_multiseed", "v2")


def _sources(exclude=("score.py", "__init__.py")):
    return [p for p in sorted(glob.glob(os.path.join(V2, "*.py")))
            if os.path.basename(p) not in exclude]


def test_single_metric_path():
    """Only score.py may open the test GT or define a metric.

    stage0.py is exempt from the GT rule alone: its diagnostics score GEOMETRY (what a
    filled box or a perfect cell-grid carver could achieve against the polygons), which
    needs the polygons but produces no model number.
    """
    bad = []
    for p in _sources():
        base = os.path.basename(p)
        src = open(p).read()
        if "gt_test.json" in src and base not in ("data.py", "stage0.py"):
            bad.append(f"{base}: opens gt_test.json")
        # viz.py reads the GT to DRAW polygons, and computes AP only by importing
        # score.py's functions -- it defines no metric of its own, which the checks
        # below still enforce.
        if "TEST_GT" in src and base not in ("data.py", "stage0.py", "infer.py",
                                             "calibrate.py", "viz.py"):
            bad.append(f"{base}: references TEST_GT")
        # Prose may name the metric and other modules may CALL score.py's copy; only a
        # second implementation, or a route to phase4's, is a defect.
        if "automl_metric" in src:
            bad.append(f"{base}: imports phase4's automl_metric")
        for m in re.finditer(r"^def (automl\w*|\w*_ap50|greedy_ap\w*)\(", src, re.M):
            bad.append(f"{base}: defines metric fn {m.group(1)}")
    assert not bad, "second metric path detected:\n  " + "\n  ".join(bad)


def test_no_val_selection():
    """No v2 module may implement validation-based checkpoint selection.

    v2 folds S10 into train, so there is no held-out site to select on, and selecting on
    the 449 test tiles would be selection on test. The phase4 README measured that
    val-selection was worse than a fixed epoch anyway (0.067 +- 0.051 vs 0.103 +- 0.025).
    """
    bad = []
    for p in _sources(exclude=("__init__.py",)):
        src = open(p).read()
        base = os.path.basename(p)
        if base == "score.py":
            continue          # score.py *describes* the matched protocol in prose
        for pat in ("select_on", "select-on", "best_epoch", "_best.pt", "best_ckpt"):
            if pat in src:
                bad.append(f"{base}: {pat}")
    assert not bad, "validation-selection code path present:\n  " + "\n  ".join(bad)


def test_matched_requires_fixed():
    """score.cmd_matched must refuse to emit a matched-protocol number on its own."""
    src = open(os.path.join(V2, "score.py")).read()
    assert "refusing" in src and 'kind"] == "fixed"' in src, (
        "cmd_matched no longer gates on a registered `fixed` result -- the matched "
        "number could then be quoted without its conservative counterpart, or used to "
        "choose between arms")


def test_gamma_zero_equivalence():
    """Detector8IoU at gamma=0 must decode identically to the plain Detector8 path.

    Guards the claim that the quality head changes only the ORDER of detections: peaks
    are picked on the raw heatmap, so which boxes survive cannot depend on q.
    """
    from boxinst_commonality_tcd_04.detector import STRIDE8, Detector8
    from boxinst_commonality_tcd_04.modal_savanna_multiseed.v2 import model as MO

    torch.manual_seed(0)
    base = Detector8(32, width=16, tower=1).eval()
    q = MO.Detector8IoU(32, width=16, tower=1).eval()
    q.load_state_dict(base.state_dict(), strict=False)

    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        ob, oq = base(x), q(x)
    assert torch.allclose(ob, oq[:, :5], atol=0), "shared channels diverged"

    b0, s0 = MO.decode_q(ob, score_thr=0.0, topk=64, stride=STRIDE8, gamma=0.0)
    b1, s1 = MO.decode_q(oq, score_thr=0.0, topk=64, stride=STRIDE8, gamma=0.0)
    assert torch.equal(b0, b1) and torch.equal(s0, s1)

    # gamma>0 may reorder and rescore, but must not change the SET of boxes.
    b2, _ = MO.decode_q(oq, score_thr=0.0, topk=64, stride=STRIDE8, gamma=0.5)
    assert len(b2) == len(b1), "quality head changed which peaks survived"
    assert torch.allclose(b2.sort(0).values, b1.sort(0).values, atol=1e-5)


def test_masker_origin_consistency():
    """The v2 fit must use ONE origin for labels, rings and instance cells.

    This is the bug being fixed: phase4's `fit_masker_scaled` took labels and rings from
    helpers hardcoded at origin=s (the 4-phase interleaved grid) while `em.fit` used s/2
    for the instance cells and stored s/2 in the npz -- against em.py's own instruction
    that the two must match.
    """
    import boxinst_commonality_tcd_04.em as EM
    from boxinst_commonality_tcd_04.modal_tcd_multiseed.phase4 import phase4_fit_tcd as P4

    g, s = 8, 16
    rng = np.random.RandomState(0)
    differ = 0
    for _ in range(40):
        x0, y0 = rng.uniform(0, 60, 2)
        boxes = np.array([[x0, y0, x0 + rng.uniform(20, 60),
                           y0 + rng.uniform(20, 60)]], np.float32)
        ours = EM.cell_labels_canopy(boxes, None, g, s, origin=s / 2.0)
        theirs = P4._cell_labels(boxes, None, g, s)
        differ += int(not np.array_equal(ours, theirs))
        # ours is the grid em.fit's instance cells actually live on
        cy, cx = np.mgrid[0:g, 0:g] * s + s / 2.0
        inbox = ((cx >= boxes[0, 0]) & (cx < boxes[0, 2]) &
                 (cy >= boxes[0, 1]) & (cy < boxes[0, 3])).ravel()
        assert np.array_equal(ours == 1, inbox)
    assert differ > 0, (
        "phase4's loader now agrees with the s/2 grid everywhere -- if it was fixed "
        "upstream, delete this test and the A/B arm in stage0.py")


def test_split_is_site_disjoint_and_complete():
    """Train = every non-S4 tile (S10 included); test = the 449 released S4 tiles."""
    from boxinst_commonality_tcd_04.modal_savanna_multiseed.v2 import data as D
    if not (os.path.exists(D.TRAIN_GT) and os.path.exists(D.TEST_GT)):
        pytest.skip("GT not built yet")
    tr, te = json.load(open(D.TRAIN_GT)), json.load(open(D.TEST_GT))
    assert len(te) == 449 and sum(len(v["trees"]) for v in te.values()) == 1911
    assert len(tr) == 2887
    assert not (set(tr) & set(te)), "train/test tile overlap"
    assert {v["site"] for v in tr.values()} == set(D.TRAIN_SITES)
    assert "S4" not in {v["site"] for v in tr.values()}
    assert all(v["partition"] == "train" for v in tr.values()), "a val partition exists"
