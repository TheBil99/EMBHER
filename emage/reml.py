"""Exact multi-component AI-REML heritability: genotype GRMs, covariate design matrix,
the unconstrained and non-negative AI-REML fits and the chi-bar-squared LRT helpers.

Model: y = X b + sum_k g_k + e, g_k ~ N(0, s_k K_k), e ~ N(0, s_e I); h2_k is each
kernel's share of the total variance. Fits are unconstrained unless nonneg=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

# valid = converged and lambda_min/lambda_max of the fitted V >= this
VALID_LAM_RATIO = 1e-5


# ── genotypes and GRMs ───────────────────────────────────────────────────── //

def build_grm(dosage):
    """Standardized GRM over the rows (donors) of `dosage` (n x m, NaN = missing).
    Allele frequency is estimated on exactly these rows; monomorphic variants are
    dropped; missing calls contribute z=0. Returns (K, n_variants_used)."""
    p = np.nanmean(dosage, axis=0, dtype=np.float64) / 2.0
    keep = (p > 0) & (p < 1)
    if not keep.any():
        raise ValueError("no polymorphic variant left to build a GRM from")
    x, p = dosage[:, keep].astype(np.float64), p[keep]
    z = (x - 2 * p) / np.sqrt(2 * p * (1 - p))
    z[np.isnan(z)] = 0.0
    return z @ z.T / keep.sum(), int(keep.sum())


def bin_masks(n_variants, maf_ld=None, breakpoints=None, bin_assignment=None):
    """Boolean variant mask per GRM component, in bin order. `breakpoints` bins the
    variants by the MAF column of the whitespace-delimited file `maf_ld` (bim order), bin
    i being (lo, hi]; `bin_assignment` is a file with one integer per bim variant, 0 = in
    no bin. Neither: one all-variant GRM."""
    if breakpoints is not None:
        maf = pd.read_csv(maf_ld, sep=r"\s+")["MAF"].to_numpy()
        if len(maf) != n_variants:
            raise ValueError(f"{maf_ld} has {len(maf)} SNPs but bim has {n_variants}")
        return [(maf > lo) & (maf <= hi) for lo, hi in zip(breakpoints[:-1], breakpoints[1:])]
    if bin_assignment is not None:
        a = np.loadtxt(bin_assignment, dtype=int)
        if len(a) != n_variants:
            raise ValueError(f"bin assignment has {len(a)} entries but bim has {n_variants}")
        return [a == i for i in range(1, int(a.max()) + 1)]
    return [np.ones(n_variants, dtype=bool)]


# ── covariates ───────────────────────────────────────────────────────────── //

def _drop_collinear_columns(df, thresh=10.0):
    """Iteratively drop the single highest-VIF column until every remaining column
    has VIF <= thresh."""
    cols = list(df.columns)
    clean = df.dropna()
    while len(cols) > 1:
        X = clean[cols].to_numpy(dtype=float)
        n = X.shape[0]
        worst_col, worst_vif = None, thresh
        for j in range(len(cols)):
            y = X[:, j]
            Xo = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
            beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
            resid = y - Xo @ beta
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 1.0
            vif = np.inf if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2)
            if vif > worst_vif:
                worst_col, worst_vif = cols[j], vif
        if worst_col is None:
            break
        cols.remove(worst_col)
    return df[cols]


def design_matrix(covariate_file, iids):
    """Intercept + VIF-filtered covariates, rows aligned to iids (NaN where a donor
    has no covariate row)."""
    X = pd.DataFrame({"Intercept": 1.0}, index=iids)
    if covariate_file:
        cov = pd.read_csv(covariate_file, sep="\t")
        cov = cov.set_index(cov.columns[0])
        cov.index = cov.index.astype(str)
        cov = _drop_collinear_columns(cov.reindex(iids))
        X = X.join(cov)
    return X


# ── AI-REML ──────────────────────────────────────────────────────────────── //

def _reml_state(theta, Ks, y, X):
    """REML log-likelihood (up to a constant), P and Py at theta = (s_1..s_K, s_e),
    or None if V is not positive definite."""
    n = len(y)
    V = theta[-1] * np.eye(n)
    for t, K in zip(theta[:-1], Ks):
        V += t * K
    try:
        cf = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return None
    Vi = cho_solve(cf, np.eye(n))
    ViX = Vi @ X
    try:
        cx = cho_factor(X.T @ ViX, lower=True)
    except np.linalg.LinAlgError:
        return None
    P = Vi - ViX @ cho_solve(cx, ViX.T)
    Py = P @ y
    ll = -0.5 * (2 * np.log(np.diag(cf[0])).sum() + 2 * np.log(np.diag(cx[0])).sum() + y @ Py)
    return ll, P, Py


def _score_ai(P, Py, comps):
    KPy = [K @ Py for K in comps]
    score = np.array([-0.5 * (np.einsum("ij,ji->", P, K) - Py @ kp) for K, kp in zip(comps, KPy)])
    PKPy = [P @ kp for kp in KPy]
    AI = 0.5 * np.array([[a @ b for b in PKPy] for a in KPy])
    return score, AI


def _at_boundary(theta, Ks, rel=1e-6):
    """True if V's smallest eigenvalue is ~0 relative to its average diagonal."""
    V = theta[-1] * np.eye(Ks[0].shape[0])
    for t, K in zip(theta[:-1], Ks):
        V += t * K
    return bool(np.linalg.eigvalsh(V)[0] < rel * np.mean(np.diag(V)))


def _ai_reml_from(theta0, Ks, y, X, max_iter, tol):
    comps = list(Ks) + [np.eye(len(y))]
    theta = np.asarray(theta0, dtype=float)
    state = _reml_state(theta, Ks, y, X)
    if state is None:
        return dict(theta=theta, logLL=-np.inf, n_iter=0, converged=False, AI=None, at_boundary=False)
    ll, P, Py = state
    for it in range(1, max_iter + 1):
        score, AI = _score_ai(P, Py, comps)
        try:
            step = np.linalg.solve(AI, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(AI, score, rcond=None)[0]
        if 0.5 * score @ step < tol:
            return dict(theta=theta, logLL=ll, n_iter=it, converged=True, AI=AI, at_boundary=False)
        lam = 1.0
        while lam > 1e-10:
            cand = theta + lam * step
            if cand[-1] > 0:
                cs = _reml_state(cand, Ks, y, X)
                if cs is not None and cs[0] >= ll:
                    break
            lam /= 2
        else:
            return dict(theta=theta, logLL=ll, n_iter=it, converged=False, AI=AI,
                        at_boundary=_at_boundary(theta, Ks))
        theta, (ll, P, Py) = cand, cs
    score, AI = _score_ai(P, Py, comps)
    return dict(theta=theta, logLL=ll, n_iter=max_iter, converged=False, AI=AI,
                at_boundary=_at_boundary(theta, Ks))


def _ai_reml_nonneg_from(theta0, Ks, y, X, max_iter, tol):
    """_ai_reml_from with every kernel's variance component constrained >= 0 (nonneg=True):
    projected AI-Newton with an active set -- a component at 0 whose score points
    negative is held at 0, the Newton step is solved over the remaining (free) ones,
    each candidate is projected back onto >= 0 and step-halved as in _ai_reml_from.
    Converged when the predicted gain over the free components falls below tol (the
    held ones then satisfy the KKT conditions). Since V >= s_e I, V cannot become
    singular, so at_boundary is always False. Also returns the final free mask."""
    comps = list(Ks) + [np.eye(len(y))]
    k = len(Ks)
    theta = np.asarray(theta0, dtype=float)
    state = _reml_state(theta, Ks, y, X)
    if state is None:
        return dict(theta=theta, logLL=-np.inf, n_iter=0, converged=False, AI=None,
                    at_boundary=False, free=np.ones(k + 1, dtype=bool))
    ll, P, Py = state
    for it in range(1, max_iter + 1):
        score, AI = _score_ai(P, Py, comps)
        free = np.r_[~((theta[:k] <= 0) & (score[:k] <= 0)), True]
        step = np.zeros_like(theta)
        try:
            step[free] = np.linalg.solve(AI[np.ix_(free, free)], score[free])
        except np.linalg.LinAlgError:
            step[free] = np.linalg.lstsq(AI[np.ix_(free, free)], score[free], rcond=None)[0]
        if 0.5 * score[free] @ step[free] < tol:
            return dict(theta=theta, logLL=ll, n_iter=it, converged=True, AI=AI,
                        at_boundary=False, free=free)
        lam = 1.0
        while lam > 1e-10:
            cand = theta + lam * step
            cand[:k] = np.maximum(cand[:k], 0.0)
            if cand[-1] > 0:
                cs = _reml_state(cand, Ks, y, X)
                if cs is not None and cs[0] >= ll:
                    break
            lam /= 2
        else:
            return dict(theta=theta, logLL=ll, n_iter=it, converged=False, AI=AI,
                        at_boundary=False, free=free)
        theta, (ll, P, Py) = cand, cs
    score, AI = _score_ai(P, Py, comps)
    free = np.r_[~((theta[:k] <= 0) & (score[:k] <= 0)), True]
    return dict(theta=theta, logLL=ll, n_iter=max_iter, converged=False, AI=AI,
                at_boundary=False, free=free)


def null_information(Ks, X):
    """Efficient expected information for (s_1..s_k) at s = 0, s_e = 1, s_e profiled
    out -- the covariance of the LRT's chi-bar-squared null (see chibar_weights)."""
    n = X.shape[0]
    P = np.eye(n) - X @ np.linalg.solve(X.T @ X, X.T)
    PK = [P @ K for K in Ks] + [P]
    info = 0.5 * np.array([[np.einsum("ij,ji->", A, B) for B in PK] for A in PK])
    k = len(Ks)
    return info[:k, :k] - np.outer(info[:k, k], info[k, :k]) / info[k, k]


def chibar_weights(I_eff, n_draws=20000, seed=1):
    """Mixture weights w_j = P(j components positive) of the non-negative LRT's null
    distribution sum_j w_j chi2_j (chi2_0 = point mass at 0): by Monte Carlo, the
    I_eff-metric projection of N(0, I_eff^-1) draws onto the non-negative orthant."""
    from scipy.optimize import nnls
    rng = np.random.default_rng(seed)
    k = I_eff.shape[0]
    L = np.linalg.cholesky(I_eff)
    Z = rng.multivariate_normal(np.zeros(k), np.linalg.inv(I_eff), size=n_draws)
    counts = np.zeros(k + 1)
    for z in Z:
        b, _ = nnls(L.T, L.T @ z)
        counts[int(np.sum(b > 1e-10))] += 1
    return counts / n_draws


def npz_information(theta, Ks, X, test):
    """Efficient expected information for the components `test` (indices into Ks) at
    theta = (s_1..s_k, s_e) with those components at 0, s_e and every other non-zero
    component profiled out -- the covariance of the chi-bar-squared null of the LRT
    of the `test` components = 0 (lrt_npz)."""
    n = X.shape[0]
    V = theta[-1] * np.eye(n)
    for t, K in zip(theta[:-1], Ks):
        V += t * K
    Vi = np.linalg.inv(V)
    ViX = Vi @ X
    P = Vi - ViX @ np.linalg.solve(X.T @ ViX, ViX.T)
    nuis = [j for j in range(len(Ks)) if j not in test and theta[j] > 0]
    PK = [P @ Ks[j] for j in list(test) + nuis] + [P]
    info = 0.5 * np.array([[np.einsum("ij,ji->", A, B) for B in PK] for A in PK])
    t = len(test)
    return info[:t, :t] - info[:t, t:] @ np.linalg.solve(info[t:, t:], info[t:, :t])


def chibar_p(T, w):
    """P(sum_j w_j chi2_j >= T). T below 1e-8 is taken as exactly 0 (p = 1): a fit with
    every component at 0 has T = 0 up to rounding, which would otherwise fall on either
    side of chi2_0's point mass."""
    from scipy.stats import chi2
    T = 0.0 if T < 1e-8 else T
    return float(sum(wj * (chi2.sf(T, j) if j > 0 else float(T <= 0)) for j, wj in enumerate(w)))


def fit_reml(Ks, y, X, max_iter=100, tol=1e-4, nonneg=False):
    """Multi-component REML of y on kernels Ks with fixed effects X (must already
    include an intercept column). Returns dict: h2, h2_se, h2_k (list), h2_k_se
    (list), h2_k_cov (k x k sampling covariance of h2_k, for SEs of sums of
    components), sigma (variance components incl. residual, on y's original scale),
    logLL, n_iter, converged, at_boundary, n_starts_converged, lam_ratio
    (lambda_min / lambda_max of V at the reported fit, see VALID_LAM_RATIO).
    nonneg=True: variance components constrained >= 0 (_ai_reml_nonneg_from); the
    AI-based SEs then cover only the components not held at 0, and lrt (2 x the REML
    log-likelihood gain over the no-kernel model) is also returned."""
    sd = y.std()
    ys = (y - y.mean()) / sd
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    vr = np.var(ys - X @ beta)
    k = len(Ks)
    starts = [np.r_[np.full(k, h * vr / k), (1 - h) * vr] for h in (0.5, 0.1, 0.9)]
    fit_from = _ai_reml_nonneg_from if nonneg else _ai_reml_from
    runs = [fit_from(s, Ks, ys, X, max_iter, tol) for s in starts]
    conv = [r for r in runs if r["converged"]]
    best = max(conv or runs, key=lambda r: r["logLL"])

    theta = best["theta"]
    a = np.r_[[np.mean(np.diag(K)) for K in Ks], 1.0]
    total = a @ theta
    h2_k = a[:-1] * theta[:-1] / total
    # delta method: d(h2_k)/d(theta_j) = a_k (delta_kj * total - theta_k a_j) / total^2
    J = (np.diag(a) * total - np.outer(a * theta, a)) / total ** 2
    J = J[:-1]
    try:
        if nonneg:
            # components held at 0 are fixed: covariance over the free ones only
            free = best["free"]
            cov = np.zeros((k + 1, k + 1))
            cov[np.ix_(free, free)] = np.linalg.inv(best["AI"][np.ix_(free, free)])
        else:
            cov = np.linalg.inv(best["AI"])
        cov_h2k = J @ cov @ J.T
        h2_k_se = np.sqrt(np.clip(np.diag(cov_h2k), 0, None))
        h2_se = float(np.sqrt(max(cov_h2k.sum(), 0.0)))
    except (np.linalg.LinAlgError, TypeError, ValueError):
        h2_k_se, h2_se = np.full(k, np.nan), np.nan
        cov_h2k = np.full((k, k), np.nan)
    V = theta[-1] * np.eye(len(ys))
    for t, K in zip(theta[:-1], Ks):
        V += t * K
    w = np.linalg.eigvalsh(V)
    out = dict(h2=float(h2_k.sum()), h2_se=h2_se, h2_k=h2_k.tolist(), h2_k_se=list(h2_k_se),
               h2_k_cov=cov_h2k,
               sigma=(theta * sd ** 2).tolist(), logLL=float(best["logLL"]),
               n_iter=int(best["n_iter"]), converged=bool(best["converged"]),
               at_boundary=bool(best["at_boundary"]), n_starts_converged=len(conv),
               lam_ratio=float(w[0] / w[-1]))
    if nonneg:
        # no-kernel model: residual variance at its REML estimate RSS / (n - p)
        null = np.r_[np.zeros(k), np.sum((ys - X @ beta) ** 2) / (len(ys) - X.shape[1])]
        out["lrt"] = float(2 * (best["logLL"] - _reml_state(null, Ks, ys, X)[0]))
    return out


def load_npz_kernels(paths):
    """(ids, [K, ...]) from .npz files holding a kernel `K` and its donor `ids`, each K reindexed onto
    the donors present in every file, in the first file's order."""
    loaded = [np.load(p, allow_pickle=True) for p in paths]
    id_lists = [[str(x) for x in d["ids"]] for d in loaded]
    common = set(id_lists[0]).intersection(*id_lists[1:])
    ids = [i for i in id_lists[0] if i in common]
    if not ids:
        raise ValueError(f"no donors shared across the kernel files: {paths}")
    Ks = []
    for d, lst in zip(loaded, id_lists):
        pos = {i: k for k, i in enumerate(lst)}
        idx = [pos[i] for i in ids]
        Ks.append(np.asarray(d["K"], dtype=np.float64)[np.ix_(idx, idx)])
    return np.array(ids), Ks
