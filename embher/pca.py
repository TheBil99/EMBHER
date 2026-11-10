"""Top-k PCA whitening of a donors x features matrix."""
from __future__ import annotations

import numpy as np
import scipy.linalg


def svd_robust(A):
    """np.linalg.svd (LAPACK gesdd), falling back to the slower but more robust gesvd
    driver on the intermittent "SVD did not converge" LinAlgError gesdd is prone to on
    some inputs (seen in practice on a handful of (gene, layer) matrices)."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(A, full_matrices=False, lapack_driver="gesvd")


def whiten_embeddings(Z, n_pcs=None, pc_frac=None, verbose=True):
    """Z: donors x features (float64). Standardise the features, drop zero-variance ones,
    SVD, and return the top-k whitened PC scores (donors x k, unit variance per column),
    with k = n_pcs or round(pc_frac * numerical rank), capped at the rank.
    verbose=False silences the [info] lines (warnings still print)."""
    n = Z.shape[0]

    # ── 1. standardise the ORIGINAL feature columns; drop dead ones ──────────── //
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    live = sd > 0
    n_dead = int((~live).sum())
    Z = (Z[:, live] - mu[live]) / sd[live]
    if verbose:
        print(f"[info] {n} donors, {Z.shape[1]} live features "
              f"({n_dead} zero-variance dropped)", flush=True)

    # ── 2. PCA via SVD on the standardised matrix ───────────────────────────── //
    U, S, _ = svd_robust(Z)
    # Numerical rank: components below this carry no embedding variance at all.
    # Thresholded on EIGENvalues (S**2), not singular values, so it matches the
    # rank/effective-rank convention used everywhere else in this analysis.
    rank = int((S ** 2 > (S ** 2).max() * 1e-8).sum())
    if pc_frac is not None:
        requested = round(pc_frac * rank)
        k = min(requested, rank)
        if verbose:
            print(f"[info] pc_frac {pc_frac} of rank {rank} -> requesting {requested} "
                  f"PCs", flush=True)
    else:
        requested = n_pcs
        k = min(requested, rank)
    if k < requested:
        print(f"[warn] requested {requested} PCs but the embedding's numerical rank "
              f"is only {rank}; retaining {k}", flush=True)

    var_share = float((S[:k] ** 2).sum() / (S ** 2).sum())
    ev = (S ** 2) / n
    eff_rank_before = float(ev.sum() ** 2 / (ev ** 2).sum())
    if verbose:
        print(f"[info] rank={rank}, retaining k={k} PCs covering {100 * var_share:.2f}% "
              f"of total feature variance; effective rank of the UNWHITENED kernel was "
              f"{eff_rank_before:.1f} (whitened it is exactly k)", flush=True)

    # ── 3. whiten: each retained component rescaled to unit variance ────────── //
    # U's columns are orthonormal, so U[:, :k] * sqrt(n) has unit variance per column.
    # This is exactly scores / singular value, i.e. dividing out the eigenvalue that
    # would otherwise weight this direction in K.
    scores = U[:, :k] * np.sqrt(n)
    return scores
