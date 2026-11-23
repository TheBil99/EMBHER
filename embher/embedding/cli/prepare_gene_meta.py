#!/usr/bin/env python3
"""
Build a gene metadata TSV for capture_forward.py from the GENCODE v46 parquet.

capture_forward.py -> load_gene_metadata() expects this schema:
gene_id, chr_hg38, tss_hg38, start_hg38, end_hg38. The chromosome is written
bare ('22', not 'chr22') because PersonalDataset re-adds the prefix itself
via --contig_prefix (VCF) and genome_contig_prefix (FASTA).

Usage:
    python tmp/scripts/prepare_gene_meta.py --chrom 22 --out tmp/data/genes_chr22.tsv
"""

import argparse
from pathlib import Path

import pandas as pd

GENCODE_PARQUET = (
    'data/GENCODE/release_46/'
    'gencode.v46.annotation.gtf.parquet'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gencode', default=GENCODE_PARQUET)
    p.add_argument('--chrom', nargs='+', default=['22'],
                   help="Chromosomes to keep, bare names e.g. 22 21 (default: 22)")
    p.add_argument('--gene_type', default='protein_coding',
                   help="GENCODE gene_type to keep, or 'any' (default: protein_coding)")
    p.add_argument('--n_genes', type=int,
                   help='Keep only the first N genes after filtering (default: all)')
    p.add_argument('--min_span', type=int, default=0,
                   help='Minimum gene body length in bp (default: 0)')
    p.add_argument('--max_span', type=int,
                   help='Maximum gene body length in bp. Output size scales with '
                        'the cropped region, so keep this small for test runs.')
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(
        args.gencode,
        columns=['Chromosome', 'Feature', 'Start', 'End', 'Strand',
                 'gene_id', 'gene_type', 'gene_name'],
    )
    genes = df[df['Feature'] == 'gene'].copy()

    wanted = {f'chr{c}' for c in args.chrom}
    genes = genes[genes['Chromosome'].isin(wanted)]
    if args.gene_type != 'any':
        genes = genes[genes['gene_type'] == args.gene_type]

    # The parquet is pyranges-style: Start is already 0-based half-open, so it
    # needs no GTF-style -1. TSS is the 5' end, which is End on the minus strand.
    genes['tss_hg38'] = genes['Start'].where(genes['Strand'] == '+', genes['End'])
    genes['chr_hg38'] = genes['Chromosome'].str.removeprefix('chr')
    genes = genes.rename(columns={'Start': 'start_hg38', 'End': 'end_hg38'})

    # Strip the version suffix (ENSG00000128272.17 -> ENSG00000128272) so the
    # IDs match the plain ENSG form used everywhere else.
    genes['gene_id'] = genes['gene_id'].str.split('.').str[0]

    span = genes['end_hg38'] - genes['start_hg38']
    genes = genes[span >= args.min_span]
    if args.max_span:
        genes = genes[(genes['end_hg38'] - genes['start_hg38']) <= args.max_span]

    genes = genes.sort_values('gene_id').reset_index(drop=True)
    if args.n_genes:
        genes = genes.iloc[:args.n_genes].reset_index(drop=True)

    out_cols = ['gene_id', 'gene_name', 'chr_hg38', 'start_hg38', 'end_hg38',
                'tss_hg38', 'Strand', 'gene_type']
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genes[out_cols].to_csv(out_path, sep='\t', index=False)

    span = genes['end_hg38'] - genes['start_hg38']
    print(f"Wrote {len(genes)} genes to {out_path}")
    print(f"  chromosomes: {sorted(genes['chr_hg38'].unique())}")
    print(f"  gene body span: median {span.median():.0f} bp, max {span.max()} bp")


if __name__ == '__main__':
    main()
