"""Minimal REAL boxinst_commonality.em for Modal — mask EVAL and masker FIT.

The full module carries heavy fit-time deps (matplotlib, dapt.data.cohort,
boxinst.commonality) that neither inference nor the numpy EM needs. Everything the
TCD masker uses — logsumexp, estep (eval) + spherical_kmeans, contrastive_update,
softmax (fit) — is copied VERBATIM from boxinst_commonality/em.py so the masker is
byte-identical to local. Only the heavy I/O/plot helpers are dropped.
"""
import numpy as np

FG_PRIOR_MASS = np.pi / 4
BIN_CAP = 0.95


def softmax(a, axis):
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def estep(z, pis, kappa, C, bgll, contrast=True):
    zc = kappa * (z @ C.T)
    psum = np.clip(pis.sum(0), 1e-4, 1 - 1e-4)
    lw = zc + np.log((pis / psum).T + 1e-9)
    A = logsumexp(lw, 1) - bgll
    if contrast and len(A) > 1:
        A = A - A.mean()
    pfg = 1.0 / (1.0 + np.exp(-(A + np.log(psum / (1 - psum)))))
    e = lw - lw.max(1, keepdims=True)
    wk = np.exp(e)
    wk /= wk.sum(1, keepdims=True)
    return pfg, pfg[:, None] * wk


def spherical_kmeans(X, k, rng, iters=25):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        C = np.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                      for j in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    return C, np.argmax(X @ C.T, 1)


def contrastive_update(newC, negZ, kappa, beta):
    pos = newC / (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)
    if beta <= 0:
        return pos
    a = softmax(kappa * (negZ @ pos.T), axis=1)
    neg = (a.T @ negZ) / (a.sum(0)[:, None] + 1e-8)
    neg = neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-8)
    C = pos - beta * neg
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
