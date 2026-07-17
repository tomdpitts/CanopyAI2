"""Modal stub for boxinst_commonality.em — imported by tcd_04/em.py, which the
box-only eval path never executes (TCDMasker is instantiated only in
evaluate.run, not eval_box). Names exist so imports resolve; calls raise."""


def _unavailable(*a, **k):
    raise RuntimeError("boxinst_commonality.em is stubbed on Modal "
                       "(EM mask stage not used in the box-only ablation)")


BIN_CAP = None
FG_PRIOR_MASS = None
contrastive_update = _unavailable
estep = _unavailable
logsumexp = _unavailable
spherical_kmeans = _unavailable
