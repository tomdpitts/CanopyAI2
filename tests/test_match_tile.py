"""Guard for `evaluate.match_tile`, the single greedy matcher behind every AP / P-R /
crown-pairing number in the repo.

Written 2026-08-26, after an audit found the matcher had been re-typed at eight call
sites and every copy carried the same non-COCO rule: take j = argmax(iou[i]) and give up
when that GT is already claimed. COCOeval instead falls back to the best still-UNMATCHED
GT above threshold. On OAM-TCD the two agree exactly (NMS at IoU 0.5 removes the
competing duplicates), which is why it went unnoticed -- so these tests use synthetic
tiles built specifically to make the rules disagree, rather than repo data.
"""
import numpy as np
import pytest

from boxinst_commonality_tcd_04 import evaluate as E


def _legacy(iou, ps, ign, thr):
    """The pre-2026-08-26 rule, kept ONLY so the tests can assert we no longer do this."""
    o = np.argsort(-ps)
    iou, ps, ign = iou[o], ps[o], ign[o]
    matched = np.zeros(iou.shape[1], bool)
    tp = np.zeros(len(ps), bool)
    for i in range(len(ps)):
        j = int(np.argmax(iou[i])) if iou.shape[1] else -1
        if j >= 0 and iou[i, j] >= thr and not matched[j]:
            matched[j] = True
            tp[i] = True
    return tp


def test_falls_back_to_next_best_free_gt():
    """pred1's best crown is taken, but a FREE crown still clears threshold -> TP."""
    iou = np.array([[0.9, 0.2],       # takes crown0
                    [0.8, 0.6]])      # best is crown0 (taken); crown1 free at 0.6
    sc = np.array([0.9, 0.5], np.float32)
    _, _, tp, _, gidx = E.match_tile(iou, sc, np.zeros(2, bool), 0.5)
    assert tp.tolist() == [True, True]
    assert gidx.tolist() == [0, 1]
    assert _legacy(iou, sc, np.zeros(2, bool), 0.5).tolist() == [True, False]


def test_no_free_gt_above_threshold_is_a_false_positive():
    """The fallback must not invent matches: crown1 at 0.1 does not clear 0.5."""
    iou = np.array([[0.9, 0.2],
                    [0.8, 0.1]])
    sc = np.array([0.9, 0.5], np.float32)
    _, _, tp, _, gidx = E.match_tile(iou, sc, np.zeros(2, bool), 0.5)
    assert tp.tolist() == [True, False]
    assert gidx.tolist() == [0, -1]


def test_each_gt_claimed_at_most_once():
    iou = np.full((4, 2), 0.9)
    sc = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    _, _, tp, _, gidx = E.match_tile(iou, sc, np.zeros(4, bool), 0.5)
    assert tp.sum() == 2
    claimed = [j for j in gidx if j >= 0]
    assert sorted(claimed) == [0, 1]


def test_higher_score_gets_first_pick():
    iou = np.array([[0.6, 0.9],
                    [0.0, 0.7]])
    sc = np.array([0.2, 0.8], np.float32)     # row1 outranks row0
    order, _, tp, _, gidx = E.match_tile(iou, sc, np.zeros(2, bool), 0.5)
    assert order.tolist() == [1, 0]           # returns are in score order
    assert gidx.tolist() == [1, 0]            # row1 -> crown1, then row0 -> crown0
    assert tp.tolist() == [True, True]


def test_unmatched_ignored_pred_is_dropped_not_a_false_positive():
    iou = np.array([[0.9, 0.0],
                    [0.1, 0.1]])              # row1 matches nothing
    sc = np.array([0.9, 0.5], np.float32)
    for ign, keep_expected in (([False, False], [True, True]),
                               ([False, True], [True, False])):
        _, _, tp, keep, _ = E.match_tile(iou, sc, np.array(ign), 0.5)
        assert tp.tolist() == [True, False]
        assert keep.tolist() == keep_expected


def test_matched_pred_is_kept_even_when_ignored():
    """Ignore only ever drops UNMATCHED predictions; a TP in canopy stays a TP."""
    iou = np.array([[0.9, 0.0]])
    _, _, tp, keep, _ = E.match_tile(iou, np.array([0.9], np.float32),
                                     np.array([True]), 0.5)
    assert tp.tolist() == [True] and keep.tolist() == [True]


def test_threshold_is_inclusive():
    iou = np.array([[0.5]])
    _, _, tp, _, _ = E.match_tile(iou, np.array([0.9], np.float32),
                                  np.zeros(1, bool), 0.5)
    assert tp.tolist() == [True]


@pytest.mark.parametrize("n_p,n_g", [(0, 3), (3, 0), (0, 0)])
def test_degenerate_shapes(n_p, n_g):
    iou = np.zeros((n_p, n_g))
    _, ps, tp, keep, gidx = E.match_tile(iou, np.zeros(n_p, np.float32),
                                         np.zeros(n_p, bool), 0.5)
    assert len(ps) == len(tp) == len(keep) == len(gidx) == n_p
    assert not tp.any()


def test_score_ties_are_broken_stably():
    """Equal scores must resolve by input order (mergesort), as COCO's accumulate does."""
    iou = np.array([[0.9, 0.6],
                    [0.9, 0.6]])
    sc = np.array([0.7, 0.7], np.float32)
    order, _, _, _, gidx = E.match_tile(iou, sc, np.zeros(2, bool), 0.5)
    assert order.tolist() == [0, 1]
    assert gidx.tolist() == [0, 1]


def test_never_scores_below_the_legacy_rule():
    """Random tiles: the COCO rule can only ever match MORE, never fewer, crowns."""
    for seed in range(300):
        r = np.random.default_rng(seed)
        n_p, n_g = int(r.integers(0, 10)), int(r.integers(0, 8))
        iou = r.random((n_p, n_g))
        sc = r.random(n_p).astype(np.float32)
        ign = np.zeros(n_p, bool)
        for thr in E.IOU_50_95:
            _, _, tp, _, gidx = E.match_tile(iou, sc, ign, thr)
            assert tp.sum() >= _legacy(iou, sc, ign, thr).sum()
            assert tp.sum() <= min(n_p, n_g)
            claimed = [int(j) for j in gidx if j >= 0]
            assert len(claimed) == len(set(claimed))    # no GT double-claimed
