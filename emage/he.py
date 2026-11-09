"""Haseman-Elston regression with closed-form delete-one jackknife.

Model: y = X b + sum_k g_k + e, g_k ~ N(0, s_k K_k), e ~ N(0, s_e I). The variance
components are fit by least squares on the products of the residual-projected kernels.
"""
from __future__ import annotations

import numpy as np

# ── HE regression with closed-form delete-one jackknife ─────────────────────── //

def _pair_terms(As, X, G, loo=True):
    """Full-sample and per-donor (delete-one) pieces of tr(M A_l M A_m) for every pair.
    Returns (S, S_loo): S (m, m); S_loo (n, m, m) with S_loo[i] = Q'Q of the fit
    without donor i. A matrix in As may be None, meaning the identity. loo=False skips the
    per-donor pieces (S_loo, Gu and h are None): the point estimate needs only S, and the
    per-donor work, O(n p^2) per kernel pair, is most of the cost once there are many
    covariates. he_fit sets loo from its jackknife argument (--jackknife in he_heritability.py)."""
    n, p = X.shape
    m = len(As)
    eye = [A is None for A in As]
    AX = [X if e else A @ X for A, e in zip(As, eye)]
    dA = [np.ones(n) if e else np.diag(A).copy() for A, e in zip(As, eye)]
    PA = [X.T @ ax for ax in AX]

    if loo:
        Gu = X @ G                               # rows: G u_i
        h = np.einsum("ij,ij->i", Gu, X)         # leverages
        Gi = G[None] + np.einsum("ia,ib->iab", Gu, Gu) / (1 - h)[:, None, None]
        # PA restricted to the fold, per donor: X_-i' A_-i X_-i
        PAi = [P[None] - np.einsum("ia,ib->iab", X, ax) - np.einsum("ia,ib->iab", ax, X)
               + d[:, None, None] * np.einsum("ia,ib->iab", X, X)
               for P, ax, d in zip(PA, AX, dA)]
        W = [np.einsum("iab,ibc->iac", Gi, P) for P in PAi]      # G_i PA_i
        V = [ax - d[:, None] * X for ax, d in zip(AX, dA)]       # X_-i' A_-i,i
        S_loo = np.zeros((n, m, m))
    else:
        S_loo = Gu = h = None

    S = np.zeros((m, m))
    for l in range(m):
        for k in range(l, m):
            Al, Ak = As[l], As[k]
            if eye[l] and eye[k]:
                prod = np.ones(n)                  # rowsum(I o I)
            elif eye[l]:
                prod = dA[k]
            elif eye[k]:
                prod = dA[l]
            else:
                prod = np.einsum("ij,ij->i", Al, Ak)
            T1 = prod.sum()
            CX = AX[k] if eye[l] else Al @ AX[k]          # (A_l A_k) X
            XCX = X.T @ CX
            T2 = np.trace(G @ XCX)
            T3 = np.trace(G @ PA[l] @ G @ PA[k])
            S[l, k] = S[k, l] = T1 - 2 * T2 + T3
            if not loo:
                continue
            # delete-one versions
            CtX = AX[l] if eye[k] else Ak @ AX[l]         # (A_k A_l) X
            T1i = T1 - 2 * prod + dA[l] * dA[k]
            M2 = (XCX[None] - np.einsum("ia,ib->iab", X, CX) - np.einsum("ia,ib->iab", CtX, X)
                  + prod[:, None, None] * np.einsum("ia,ib->iab", X, X)
                  - np.einsum("ia,ib->iab", V[l], V[k]))
            T2i = np.einsum("iab,iba->i", Gi, M2)
            T3i = np.einsum("iab,iba->i", W[l], W[k])
            S_loo[:, l, k] = S_loo[:, k, l] = T1i - 2 * T2i + T3i
    return S, S_loo, AX, PA, dA, Gu, h


def he_fit(Ks, Y, X, jackknife=True):
    """HE of every column of Y (n x P, no NaN) on kernels Ks with fixed effects X (with
    intercept). Returns dict with theta (m x P, m = len(Ks)+1, residual last) and, if
    jackknife, theta_loo (n x m x P)."""
    n, p = X.shape
    As = list(Ks) + [None]
    G = np.linalg.inv(X.T @ X)
    S, S_loo, AX, PA, dA, Gu, h = _pair_terms(As, X, G, loo=jackknife)
    E = Y - X @ (G @ (X.T @ Y))                  # full-sample residuals r = M y
    AE = [E if A is None else A @ E for A in As]
    b = np.array([np.einsum("ij,ij->j", E, ae) for ae in AE])     # (m, P)
    theta = np.linalg.solve(S, b)
    out = dict(theta=theta, S=S, a=np.array([d.mean() for d in dA]))
    if not jackknife:
        return out
    # fold residuals: z = u with entry i zeroed, u = y - X beta_-i = e + X w_i e_i
    w = Gu / (1 - h)[:, None]                                     # (n, p)
    ui = E / (1 - h)[:, None]                                     # u_i, (n, P)
    b_loo = np.empty((n, len(As), Y.shape[1]))
    for l, (ae, ax, P_, d) in enumerate(zip(AE, AX, PA, dA)):
        XAE = X.T @ ae                                            # (p, P)
        uAu = (b[l][None, :] + 2 * E * (w @ XAE)
               + E ** 2 * np.einsum("ia,ab,ib->i", w, P_, w)[:, None])
        Au_i = ae + E * np.einsum("ia,ia->i", ax, w)[:, None]
        b_loo[:, l, :] = uAu - 2 * ui * Au_i + ui ** 2 * d[:, None]
    out["theta_loo"] = np.linalg.solve(S_loo, b_loo)
    # fold mean diagonals a_k,-i = (sum diag - diag_i) / (n - 1)
    out["a_loo"] = np.stack([(d.sum() - d) / (n - 1) for d in dA], axis=1)   # (n, m)
    return out


def _h2(theta, a):
    """h2_k (scaled and unscaled) from theta (..., m, P) and a (..., m)."""
    g, e = theta[..., :-1, :], theta[..., -1:, :]
    ag = a[..., :-1, None] * g
    scaled = ag / (ag.sum(axis=-2, keepdims=True) + e)
    unscaled = g / (g.sum(axis=-2, keepdims=True) + e)
    return scaled, unscaled


def jk_var(x):
    """Delete-one jackknife variance: (n - 1) * np.var(x) over the fold axis 0."""
    return (x.shape[0] - 1) * np.var(x, axis=0)
