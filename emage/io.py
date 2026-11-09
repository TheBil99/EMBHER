"""Read a PLINK 1 .bed into a donor x variant dosage matrix (numpy, no plink call).

Genotype reader for per-donor allele counts from a PLINK bfile. Import it as
`from emage.io import read_bed` (or `from emage import read_bed`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def read_bed(bfile):
    """Return (iids, dosage) from a SNP-major PLINK .bed: dosage is n_samples x
    n_variants float32 counts of the bim's second allele (0/1/2), NaN = missing.
    The allele counted doesn't matter for a standardized GRM (sign cancels in ZZ')."""
    fam = pd.read_csv(f"{bfile}.fam", sep=r"\s+", header=None)
    n = len(fam)
    m = sum(1 for _ in open(f"{bfile}.bim"))
    nb = (n + 3) // 4
    raw = np.fromfile(f"{bfile}.bed", dtype=np.uint8)
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise ValueError(f"{bfile}.bed is not a SNP-major PLINK 1 bed file")
    raw = raw[3:].reshape(m, nb)
    # two bits per genotype, low bit first: 0=hom A1, 1=missing, 2=het, 3=hom A2
    bits = np.unpackbits(raw, axis=1, bitorder="little").reshape(m, nb * 4, 2)[:, :n, :]
    code = bits[..., 0] + 2 * bits[..., 1]
    lut = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)
    return fam[1].astype(str).to_numpy(), lut[code].T
