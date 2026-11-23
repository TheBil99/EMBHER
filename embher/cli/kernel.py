"""embher-kernel: build a relatedness kernel (.npz, optionally GCTA format) from an embedding or
from PLINK genotypes."""
from __future__ import annotations

import argparse

import numpy as np

from embher.cli.common import read_table
from embher.embedding.whiten import embedding_kernel
from embher.heritability.grm_io import write_gcta_grm
from embher.heritability.io import read_bed
from embher.heritability.reml import build_grm


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="embher-kernel",
        description="Build a kernel from --embeddings (whitened top PCs by default) or --bfile "
                    "(standardised genotype GRM). Writes <out>.npz (K, ids, n_markers).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--embeddings", help="donor x feature table (parquet, tsv or csv; first column = donor id)")
    src.add_argument("--bfile", help="PLINK 1 prefix (.bed/.bim/.fam)")
    ap.add_argument("--donors", default=None,
                    help="optional table whose index lists the donors to keep; features/allele "
                         "frequencies are then standardised over exactly these donors")
    ap.add_argument("--n-pcs", type=int, default=None, help="embeddings: keep this many whitened PCs")
    ap.add_argument("--pc-frac", type=float, default=0.7,
                    help="embeddings: keep this fraction of the numerical rank (default 0.7; "
                         "ignored if --n-pcs is given)")
    ap.add_argument("--no-whiten", action="store_true",
                    help="embeddings: plain standardised ZZ'/M instead of whitened PCs (not recommended)")
    ap.add_argument("--gcta", action="store_true", help="also write <out>.grm.bin/.N.bin/.id")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args(argv)

    if a.embeddings:
        df = read_table(a.embeddings)
        ids, M = df.index.to_numpy(), None
        if a.donors:
            keep = df.index.intersection(read_table(a.donors).index)
            df = df.loc[keep]
        K, M = embedding_kernel(df.to_numpy(float), n_pcs=a.n_pcs, pc_frac=a.pc_frac,
                                whiten=not a.no_whiten, verbose=True)
        ids = df.index.to_numpy()
    else:
        ids, dosage = read_bed(a.bfile)
        if a.donors:
            keep = np.isin(ids, read_table(a.donors).index.to_numpy())
            ids, dosage = ids[keep], dosage[keep]
        K, M = build_grm(dosage)
    np.savez(f"{a.out}.npz", K=K, ids=np.array(ids, dtype=str), n_markers=M)
    if a.gcta:
        write_gcta_grm(a.out, K.copy(), list(ids), M)
    print(f"[done] kernel {K.shape} from {M} {'components' if a.embeddings else 'variants'} -> {a.out}.npz")


if __name__ == "__main__":
    main()
