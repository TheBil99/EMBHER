"""
Data loading utilities.

This module provides simple functions for loading peak and gene data,
creating genome datasets, and managing train/test splits.
"""

import pandas as pd
import numpy as np
import pysam
from pathlib import Path
from typing import List, Optional, Union


def load_gene_metadata(
    gene_meta_path: Union[str, Path],
    filter_chromosomes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load gene metadata and standardize its column names.

    Split out of load_genes so callers that don't need expression data can
    reuse the same TSV schema mapping.

    Args:
        gene_meta_path: Path to gene metadata TSV file
        filter_chromosomes: Chromosomes to keep (default: 1-22, X, Y)

    Returns:
        DataFrame with standardized chr, tss, start, end, ensg columns

    Example:
        >>> gene_meta = load_gene_metadata('gene-ids-and-positions.tsv')
        >>> gene_meta[['chr', 'tss', 'start', 'end', 'ensg']].head()
    """
    gene_meta = pd.read_csv(Path(gene_meta_path), sep="\t").copy()

    gene_meta['tss'] = gene_meta['tss_hg38']
    gene_meta['chr'] = gene_meta['chr_hg38'].astype(str)
    gene_meta['ensg'] = gene_meta['gene_id']

    # Fall back to the TSS when the file carries no gene body coordinates
    if 'start_hg38' in gene_meta.columns:
        gene_meta['start'] = gene_meta['start_hg38']
    elif 'start' not in gene_meta.columns:
        gene_meta['start'] = gene_meta['tss']

    if 'end_hg38' in gene_meta.columns:
        gene_meta['end'] = gene_meta['end_hg38']
    elif 'end' not in gene_meta.columns:
        gene_meta['end'] = gene_meta['tss']

    if filter_chromosomes is None:
        filter_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
    return gene_meta[gene_meta['chr'].isin(filter_chromosomes)].copy()


def load_vcf_samples(vcf_path: Union[str, Path]) -> List[str]:
    """
    Get sample names from VCF file header.

    Args:
        vcf_path: Path to VCF file

    Returns:
        List of sample IDs

    Example:
        >>> samples = load_vcf_samples('data.vcf.gz')
        >>> print(f"VCF contains {len(samples)} samples")
    """
    vcf_path = Path(vcf_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    with pysam.VariantFile(str(vcf_path), mode="r") as vcf:
        samples = list(vcf.header.samples)

    return samples


def create_genome_dataset(
    metadata: pd.DataFrame,
    vcf_file_path: Union[str, Path],
    genome_file_path: Union[str, Path],
    expression_data: pd.DataFrame,
    sample_list: List[str],
    window_size: int,
    verbose: bool = False,
    random_state: int = 42,
    return_idx: bool = False,
    contig_prefix: str = '',
    genome_contig_prefix: str = 'chr',
    onehot_encode: bool = False,
    min_maf: Optional[float] = None,
    max_maf: Optional[float] = None,
    maf_samples: Optional[List[str]] = None
) -> 'GenomeDataset':
    """
    Create GenomeDataset from per-gene metadata.

    This is a convenience wrapper around the GenomeDataset class that
    automatically validates inputs and handles common configuration.

    Args:
        metadata: DataFrame with chr, tss, ensg columns
        vcf_file_path: Path to VCF file with genetic variants
        genome_file_path: Path to reference genome FASTA file
        expression_data: DataFrame with expression/accessibility values
        sample_list: List of sample IDs to include
        window_size: Size of genomic window around TSS (bp)
        verbose: Print progress messages
        return_idx: Return sample index in dataset
        contig_prefix: Prefix for contig names ('' for hg19, 'chr' for hg38)
        onehot_encode: Return one-hot encoded sequences
        min_maf, max_maf, maf_samples: cohort-MAF variant filter, see GenomeDataset

    Returns:
        GenomeDataset instance

    Example:
        >>> dataset = create_genome_dataset(
        ...     metadata=peak_metadata,
        ...     vcf_file_path='genotypes.vcf.gz',
        ...     genome_file_path='hg19.fa',
        ...     expression_data=log_fpkm,
        ...     sample_list=samples,
        ...     window_size=100000
        ... )
        >>> seq, expr, _, idx = dataset[0]
    """
    from embher.embedding.dataset import GenomeDataset

    # Validate required columns
    required_cols = ['chr', 'tss', 'ensg']
    missing = [c for c in required_cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    # Validate sample list first (quick check before expensive file checks)
    if len(sample_list) == 0:
        raise ValueError("sample_list cannot be empty")

    # Validate files exist
    vcf_path = Path(vcf_file_path)
    genome_path = Path(genome_file_path)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome file not found: {genome_path}")

    # Create dataset
    dataset = GenomeDataset(
        gene_metadata=metadata,
        vcf_file_path=str(vcf_path),
        hg38_file_path=str(genome_path),
        expr_data=expression_data,
        sample_list=sample_list,
        window_size=window_size,
        verbose=verbose,
        return_idx=return_idx,
        contig_prefix=contig_prefix,
        genome_contig_prefix=genome_contig_prefix,
        onehot_encode=onehot_encode,
        min_maf=min_maf,
        max_maf=max_maf,
        maf_samples=maf_samples
    )

    return dataset

