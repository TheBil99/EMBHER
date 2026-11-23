"""Heritability of several phenotypes on a list of kernels: exact AI-REML or Haseman-Elston.

`estimate_h2` is the engine behind the `embher-h2` command; it takes kernels, phenotypes and a
design matrix already aligned on the same donors and returns one row per phenotype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .he import _h2, he_fit, jk_var
from .reml import VALID_LAM_RATIO, chibar_p, chibar_weights, fit_reml, null_information


def estimate_h2(Ks, labels, Y, X, estimator="reml", nonneg=False, jackknife=True,
                max_iter=100, tol=1e-4):
    """Ks: list of n x n kernels; labels: one name per kernel; Y: DataFrame donors x phenotypes
    (NaN allowed); X: DataFrame donors x covariates including an intercept (rows with NaN are
    dropped). All aligned on the same donors, in order. Returns a DataFrame with one row per
    phenotype: h2 (sum over kernels), h2_se, convergence flags, n_samples, and with several
    kernels h2_<label>, h2_<label>_se per kernel. REML adds logLL, n_iter, lam_ratio, valid (and
    with nonneg=True lrt, lrt_p against the no-kernel model); HE adds sigma_<label>, sigma_e and
    jackknife SEs (jackknife=False: no SEs)."""
    if estimator not in ("reml", "he"):
        raise ValueError("estimator must be 'reml' or 'he'")
    if len(Ks) != len(labels):
        raise ValueError("one label per kernel")
    if not (len(Y) == len(X) == Ks[0].shape[0]):
        raise ValueError("kernels, phenotypes and covariates must be aligned on the same donors")
    multi = len(Ks) > 1
    Xn = X.to_numpy(float)
    x_ok = X.notna().all(axis=1).to_numpy()
    cols = list(Y.columns)
    Yn = Y.to_numpy(float)
    ok_all = ~np.isnan(Yn) & x_ok[:, None]
    rows = [None] * len(cols)

    def comp_cols(row, h2k, h2k_se):
        if multi:
            for lab, h, s in zip(labels, h2k, h2k_se):
                row[f"h2_{lab}"], row[f"h2_{lab}_se"] = float(h), float(s)

    if estimator == "reml":
        cache = {}
        for j, col in enumerate(cols):
            ok = ok_all[:, j]
            sub = np.ix_(ok, ok)
            Kok = [K[sub] for K in Ks]
            r = fit_reml(Kok, Yn[ok, j], Xn[ok], max_iter, tol, nonneg=nonneg)
            row = dict(phenotype=col, estimator="reml_nonneg" if nonneg else "reml", h2=r["h2"],
                       h2_se=r["h2_se"], converged=r["converged"], at_boundary=r["at_boundary"],
                       logLL=r["logLL"], n_iter=r["n_iter"], n_samples=int(ok.sum()),
                       lam_ratio=r["lam_ratio"],
                       valid=bool(r["converged"] and r["lam_ratio"] >= VALID_LAM_RATIO))
            if nonneg:
                key = ok.tobytes()
                if key not in cache:
                    cache[key] = chibar_weights(null_information(Kok, Xn[ok]))
                row["lrt"], row["lrt_p"] = r["lrt"], chibar_p(r["lrt"], cache[key])
            comp_cols(row, r["h2_k"], r["h2_k_se"])
            rows[j] = row
    else:
        groups = {}
        for j in range(len(cols)):
            groups.setdefault(ok_all[:, j].tobytes(), []).append(j)
        for js in groups.values():
            ok = ok_all[:, js[0]]
            sub = np.ix_(ok, ok)
            r = he_fit([K[sub] for K in Ks], Yn[np.ix_(ok, js)], Xn[ok], jackknife)
            h2k, _ = _h2(r["theta"], r["a"])
            if jackknife:
                h2k_l, _ = _h2(r["theta_loo"], r["a_loo"])
            for c, j in enumerate(js):
                row = dict(phenotype=cols[j], estimator="he", h2=float(h2k[:, c].sum()), h2_se=np.nan,
                           converged=bool(np.all(np.isfinite(r["theta"][:, c]))), at_boundary=False,
                           n_samples=int(ok.sum()))
                if jackknife:
                    row["h2_se"] = float(np.sqrt(jk_var(h2k_l[:, :, c].sum(axis=1))))
                se = [float(np.sqrt(jk_var(h2k_l[:, i, c]))) if jackknife else np.nan
                      for i in range(h2k.shape[0])]
                comp_cols(row, h2k[:, c], se)
                row.update({f"sigma_{lab}": float(t) for lab, t in zip(labels, r["theta"][:-1, c])})
                row["sigma_e"] = float(r["theta"][-1, c])
                rows[j] = row
    return pd.DataFrame(rows)
