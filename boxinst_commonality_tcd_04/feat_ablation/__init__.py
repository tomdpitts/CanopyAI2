"""Self-contained feature-compression ablation for the tcd_04 detector.

Everything (reduced caches, checkpoints, eval jsons, README) lives in this
folder; `rm -rf feat_ablation/` undoes the experiment. The parent pipeline is
never modified — trainers/evaluators are thin wrappers that monkeypatch paths.
"""
