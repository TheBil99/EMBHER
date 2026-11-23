"""embher-h2: heritability of phenotypes on genotype and/or embedding kernels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from embher.cli.common import read_table
from embher.embedding.whiten import embedding_kernel
from embher.heritability.estimate import estimate_h2
from embher.heritability.io import read_bed
from embher.heritability.reml import bin_masks, build_grm, design_matrix, load_npz_kernels

DESCRIPTION = """Heritability of one or more phenotypes: exact multi-component AI-REML (default)
or Haseman-Elston regression with jackknife SEs. Kernels, in this output order: genotype GRM(s)
from --bfile (one, or one per bin), the embedding kernel from --embeddings, then any --kernel
.npz files. Every kernel is built or restricted to the donors shared by all inputs that have a
phenotype value, so a genotype and an embedding component are fitted jointly on the same donors."""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embher-h2", description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pheno", required=True,
                    help="donor x phenotype table (parquet, tsv or csv; first column = donor id)")
    ap.add_argument("--pheno-col", nargs="+", default=None,
                    help="phenotype column(s) to analyse (default: all columns)")
    ap.add_argument("--covar", default=None,
                    help="tab-separated covariates, first column = donor id; an intercept is always "
                         "fitted, covariates with VIF > 10 are dropped")
    ap.add_argument("--bfile", default=None, help="PLINK 1 prefix: genotype GRM(s)")
    ap.add_argument("--maf-bins", default=None,
                    help="with --bfile: comma-separated ascending MAF breakpoints, bins (lo, hi]; needs --maf-file")
    ap.add_argument("--maf-file", default=None, help="whitespace table with a MAF column in .bim order")
    ap.add_argument("--bin-assignment", default=None,
                    help="with --bfile: one integer per .bim variant (1..B), 0 = excluded; one GRM per bin")
    ap.add_argument("--embeddings", default=None, help="donor x feature table: embedding kernel")
    ap.add_argument("--n-pcs", type=int, default=None, help="embeddings: whitened PCs to keep")
    ap.add_argument("--pc-frac", type=float, default=0.7,
                    help="embeddings: fraction of the numerical rank to keep (default 0.7)")
    ap.add_argument("--kernel", nargs="+", default=None, help="prebuilt kernel .npz file(s) (embher-kernel)")
    ap.add_argument("--estimator", choices=["reml", "he"], default="reml")
    ap.add_argument("--nonneg", action="store_true",
                    help="REML: constrain variance components >= 0 and report the LRT (lrt, lrt_p)")
    ap.add_argument("--jackknife", choices=["loo", "none"], default="loo", help="HE: SEs (default loo)")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", required=True, help="results table (.tsv, or .parquet)")
    a = ap.parse_args(argv)

    if not (a.bfile or a.embeddings or a.kernel):
        ap.error("give at least one of --bfile, --embeddings, --kernel")
    if (a.maf_bins or a.bin_assignment) and not a.bfile:
        ap.error("--maf-bins/--bin-assignment need --bfile")
    if a.maf_bins and a.bin_assignment:
        ap.error("--maf-bins and --bin-assignment are exclusive")
    if a.maf_bins and not a.maf_file:
        ap.error("--maf-bins needs --maf-file")

    ph = read_table(a.pheno)
    if a.pheno_col:
        ph = ph[a.pheno_col]
    ph = ph.apply(pd.to_numeric)
    donors = ph.index[ph.notna().any(axis=1)]

    sources = {}
    if a.bfile:
        iids, dosage = read_bed(a.bfile)
        sources["bfile"] = pd.Index(iids)
    if a.embeddings:
        emb = read_table(a.embeddings)
        sources["embeddings"] = emb.index
    if a.kernel:
        npz_ids, npz_Ks = load_npz_kernels(a.kernel)
        sources["kernel"] = pd.Index(npz_ids)
    for idx in sources.values():
        donors = donors.intersection(idx)
    donors = ph.index[ph.index.isin(donors)]          # keep the phenotype file's order
    if len(donors) == 0:
        raise SystemExit("[error] no donor is shared by the phenotype and the kernel inputs")
    print(f"[info] {len(donors)} donors", flush=True)

    Ks, labels = [], []
    if a.bfile:
        sel = pd.Index(iids).get_indexer(donors)
        d = dosage[sel]
        bp = [float(x) for x in a.maf_bins.split(",")] if a.maf_bins else None
        masks = bin_masks(d.shape[1], a.maf_file, bp, a.bin_assignment)
        for i, mask in enumerate(masks, start=1):
            if not mask.any():
                raise SystemExit(f"[error] genotype bin {i} has no variants")
            K, m = build_grm(d[:, mask])
            print(f"[info] genotype kernel {i}/{len(masks)}: {m} variants", flush=True)
            Ks.append(K)
            labels.append("genotype" if len(masks) == 1 else f"genotype_bin{i}")
        del dosage, d
    if a.embeddings:
        K, k = embedding_kernel(emb.loc[donors].to_numpy(float), n_pcs=a.n_pcs, pc_frac=a.pc_frac, verbose=True)
        print(f"[info] embedding kernel from {k} whitened PCs", flush=True)
        Ks.append(K)
        labels.append("embedding")
    if a.kernel:
        pos = pd.Index(npz_ids).get_indexer(donors)
        for p, K in zip(a.kernel, npz_Ks):
            Ks.append(K[np.ix_(pos, pos)])
            labels.append(Path(p).stem)

    X = design_matrix(a.covar, donors.astype(str))
    res = estimate_h2(Ks, labels, ph.loc[donors], X, a.estimator, a.nonneg, a.jackknife == "loo",
                      a.max_iter, a.tol)
    if a.out.endswith(".parquet"):
        res.to_parquet(a.out, index=False)
    else:
        res.to_csv(a.out, sep="\t", index=False)
    bad = int((~res.converged).sum())
    if bad:
        print(f"[warn] {bad}/{len(res)} phenotypes did not converge", flush=True)
    print(f"[done] mean h2 = {res.h2.mean():.4f} over {len(res)} phenotype(s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
