"""
Enrichment of TF motifs in attribution seqlets against sequence-matched null windows (numpy/scipy).

FIMO at a loose p-value finds many chance hits in any 36 bp window, so a motif hit in a seqlet
means nothing by itself. Per TF, each window counts once ("has >=1 hit of any of the TF's
motifs"); the null rate of that event is measured on a large pool of random windows of the same
sequence, and a one-sided binomial test asks whether more seqlet windows carry the TF than that
rate predicts. Benjamini-Hochberg over the tested TFs gives q-values; a TF is enriched at q < alpha
with at least `min_obs` seqlet windows.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom


def tf_name(motif: str) -> str:
    """HOCOMOCO v13 motif id -> TF symbol ('SPI1.H13CORE.0.P.B' -> 'SPI1')."""
    return motif.split(".")[0]


def window_tf_counts(hits: pd.DataFrame, n_windows: int, seq_col: str = "sequence",
                     motif_col: str = "motif") -> pd.Series:
    """Number of windows (out of `n_windows`) with >= 1 hit, per TF, from a FIMO hit table."""
    if len(hits) == 0:
        return pd.Series(dtype=float)
    tf = hits[motif_col].map(tf_name)
    return pd.Series(1, index=pd.MultiIndex.from_arrays([tf, hits[seq_col]])).groupby(
        level=0).apply(lambda s: s.index.get_level_values(1).nunique()).astype(float)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def enrichment(obs: pd.Series, n_obs: int, null: pd.Series, n_null: int,
               min_obs: int = 2, alpha: float = 0.05) -> pd.DataFrame:
    """Per-TF enrichment of `obs` (windows with the TF out of `n_obs` seqlet windows) against the
    null pool (`null` windows with the TF out of `n_null`). Returns one row per tested TF
    (obs >= min_obs) with columns tf, n_obs, null_rate, expected, fold, p, q, enriched."""
    cols = ["tf", "n_obs", "null_rate", "expected", "fold", "p", "q", "enriched"]
    tested = obs[obs >= min_obs]
    if len(tested) == 0 or n_obs == 0:
        return pd.DataFrame(columns=cols)
    rate = ((null.reindex(tested.index).fillna(0.0) + 0.5) / (n_null + 1.0)).to_numpy()
    k = tested.to_numpy().astype(int)
    p = binom.sf(k - 1, n_obs, rate)
    q = bh(p)
    df = pd.DataFrame(dict(tf=tested.index, n_obs=k, null_rate=rate, expected=rate * n_obs,
                           fold=k / (rate * n_obs), p=p, q=q))
    df["enriched"] = df["q"] < alpha
    return df.sort_values("p").reset_index(drop=True)
