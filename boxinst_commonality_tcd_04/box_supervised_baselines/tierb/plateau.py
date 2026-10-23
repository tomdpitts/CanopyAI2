"""Convergence-driven schedule for all three Tier B methods (TIERB.md section 4; fixed 2026-10-15
before any number): LR decay on plateau + early stopping, both on the per-epoch val metric
(B1/B2: val box AP50 on the 108 whole val tiles; B3: val mean per-crown mask IoU on GT boxes).

  * improvement  = metric > best-so-far + MIN_DELTA (0.005, LACE's own es_min_delta)
  * PATIENCE (2) epochs without improvement -> LR x 0.1; at most N_DECAYS (2) decays, as in the
    released two-step schedules; after the last decay, PATIENCE more such epochs -> stop
  * nothing fires before MIN_EPOCHS (4); MAX_EPOCHS (20) is a budget cap (a run that reaches it
    while still improving is flagged as a possible lower bound)
  * selection is separate: the best val metric of any epoch (ties -> earlier)

The state is a pure function (replay) of the per-epoch history, stored as plateau.json on the
volume: a retried or resumed epoch replaces its own entry and the state is recomputed, so nothing
is ever double-counted.
"""
from __future__ import annotations

import json
import os

MIN_DELTA, PATIENCE, N_DECAYS, MIN_EPOCHS, MAX_EPOCHS, GAMMA = 0.005, 2, 2, 4, 20, 0.1
SMOKE = {"min_delta": 1.0, "patience": 1, "min_epochs": 1}   # forces both decays + stop in a smoke


def replay(history, min_delta=MIN_DELTA, patience=PATIENCE, n_decays=N_DECAYS,
           min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS, gamma=GAMMA):
    """history: [{"epoch": e (1-based), "metric": m}, ...] -> state after the last epoch."""
    st = {"best": None, "best_epoch": None, "bad": 0, "decays": 0, "lr_factor": 1.0,
          "stop": False, "reason": None, "events": []}
    for h in sorted(history, key=lambda h: h["epoch"]):
        e, m = h["epoch"], h["metric"]
        if st["stop"]:
            break
        if st["best"] is None or m > st["best"] + min_delta:
            st["bad"] = 0
        else:
            st["bad"] += 1
        if st["best"] is None or m > st["best"]:
            st["best"], st["best_epoch"] = m, e
        if e >= min_epochs and st["bad"] >= patience:
            if st["decays"] < n_decays:
                st["decays"] += 1
                st["lr_factor"] *= gamma
                st["bad"] = 0
                st["events"].append({"epoch": e, "event": f"decay {st['decays']}"})
            else:
                st["stop"], st["reason"] = True, f"plateau after {n_decays} decays (epoch {e})"
                st["events"].append({"epoch": e, "event": "stop"})
        if not st["stop"] and e >= max_epochs:
            st["stop"] = True
            st["reason"] = ("max epochs; STILL IMPROVING -> possible lower bound"
                            if st["bad"] == 0 else "max epochs")
            st["events"].append({"epoch": e, "event": "stop (max)"})
    return st


class Plateau:
    """File-backed state shared by every rank (all ranks read the file rank 0 wrote)."""

    def __init__(self, path, **params):
        self.path, self.params = path, params
        self.history, self.state = [], replay([], **params)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            d = json.load(open(self.path))
            self.history = d["history"]
        self.state = replay(self.history, **self.params)
        return self.state

    def record(self, epoch, metric):
        """Rank 0 only: replace this epoch's entry, recompute, write atomically."""
        self.history = [h for h in self.history if h["epoch"] != epoch] + [{"epoch": epoch, "metric": metric}]
        self.state = replay(self.history, **self.params)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"params": self.params or "default", "history": sorted(self.history, key=lambda h: h["epoch"]),
                       "state": self.state}, f, indent=1)
        os.replace(tmp, self.path)
        return self.state

    @property
    def lr_factor(self):
        return self.state["lr_factor"]
