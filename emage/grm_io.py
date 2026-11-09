"""GCTA-format binary GRM writer."""
from __future__ import annotations

import numpy as np


def write_gcta_grm(prefix: str, K: np.ndarray, ids: list, n_markers: int):
    """Write K in GCTA binary GRM format (lower triangle incl diagonal, float32)."""
    n = K.shape[0]
    tril = np.tril_indices(n)
    K[tril].astype(np.float32).tofile(f"{prefix}.grm.bin")
    np.full(tril[0].shape[0], float(n_markers), dtype=np.float32).tofile(f"{prefix}.grm.N.bin")
    with open(f"{prefix}.grm.id", "w") as fh:
        for i in ids:
            fh.write(f"{i}\t{i}\n")
