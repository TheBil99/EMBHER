"""
Results of a Borzoi attribution run: decima `Attribution` objects and files.

Layout follows decima's AttributionResult so its readers and seqlet calling apply:

    attrs: genome, model_name (+ target / method / precision / layer / readout ...)
    genes (n,)  gene_mask_start (n,)  gene_mask_end (n,)
    sequence (n, L) int8 base indices  attribution (n, 4, L) float32

with ONE gene per file (AttributionResult treats a list of files as an ensemble of models
and asserts they hold identical genes, so per-gene files cannot be passed as a list).

Coordinates: windows are in genomic + orientation, never strand-flipped, so `strand` is
always "+" for the `Attribution` object (the gene's true strand is stored as an attr), and
`from_tss` in its peak table is measured from `gene_mask_start`, the first base of the gene
body mask - the TSS for a + strand gene.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASES = "ACGT"


def make_window(fasta, chrom, tss, L):
    """(onehot (4, L), win_start): TSS-centred reference window of length L; N / off-end -> zeros."""
    ws = max(0, tss - L // 2)
    size = fasta.get_reference_length(chrom)
    seq = fasta.fetch(chrom, ws, min(ws + L, size)).upper()
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate(BASES):
        lut[ord(b), i] = 1
    onehot = np.zeros((4, L), np.float32)
    onehot[:, :len(seq)] = lut[np.frombuffer(seq.encode(), np.uint8)].T
    return onehot, ws


def onehot_to_indices(onehot: np.ndarray) -> np.ndarray:
    """(4, L) one-hot -> (L,) int8 indices; padding (all-zero column) becomes 4."""
    idx = onehot.argmax(axis=0).astype(np.int8)
    idx[onehot.sum(axis=0) == 0] = 4
    return idx


def to_decima_attribution(onehot: np.ndarray, attrs: np.ndarray, gene: str, chrom: str,
                          win_start: int, gene_start: int, gene_end: int, **kwargs):
    """Wrap one window as a decima `Attribution` (5-row input: one-hot + gene-body mask).

    gene_start/gene_end are genomic (0-based, half-open) and clipped to the window.
    """
    from decima.core.attribution import Attribution

    L = onehot.shape[1]
    mask = np.zeros((1, L), dtype=np.float32)
    lo, hi = max(gene_start - win_start, 0), min(gene_end - win_start, L)
    mask[0, lo:hi] = 1
    inputs = np.concatenate([onehot.astype(np.float32), mask], axis=0)
    return Attribution(inputs=inputs, attrs=attrs.astype(np.float32), gene=gene, chrom=chrom,
                       start=win_start, end=win_start + L, strand="+", **kwargs)


def write_attribution_h5(path: str, gene: str, onehot: np.ndarray, attrs: np.ndarray,
                         gene_mask_start: int, gene_mask_end: int,
                         meta: Dict[str, object]) -> None:
    """One gene, one attribution -> decima-compatible HDF5 (see module docstring)."""
    import h5py

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in meta.items():
            f.attrs[key] = value if value is not None else "none"
        f.create_dataset("genes", data=np.array([gene.encode()]))
        f.create_dataset("gene_mask_start", data=np.array([gene_mask_start]))
        f.create_dataset("gene_mask_end", data=np.array([gene_mask_end]))
        f.create_dataset("sequence", data=onehot_to_indices(onehot)[None], compression="gzip")
        f.create_dataset("attribution", data=attrs.astype(np.float32)[None],
                         compression="gzip")


def chrom_sizes_from_fai(fasta_path: str) -> Dict[str, int]:
    """Chromosome sizes from the FASTA's .fai (no genomepy download needed)."""
    fai = pd.read_csv(f"{fasta_path}.fai", sep="\t", header=None, usecols=[0, 1])
    return dict(zip(fai[0], fai[1]))


def save_bigwig(path: str, attrs: np.ndarray, chrom: str, win_start: int,
                sizes: Dict[str, int]) -> None:
    """Per-base attribution (summed over the 4 bases) as a bigwig, like
    Attribution.save_bigwig but with chromosome sizes taken from the local FASTA index."""
    import pyBigWig

    values = attrs.sum(axis=0).astype(np.float64)
    bw = pyBigWig.open(path, "w")
    bw.addHeader([(c, int(s)) for c, s in sizes.items()])
    bw.addEntries(chrom, int(win_start), values=values, span=1, step=1)
    bw.close()
