#!/usr/bin/env python3
"""
Run an AlphaGenome or Borzoi/Flashzoi forward pass on personal genomes and dump
every intermediate and final tensor to an HDF5 file for later reuse, without
re-running the model.

Unlike a plain prediction run, which only gets final track predictions, this
captures the encoder/transformer/decoder activations and trunk embeddings
alongside them. That requires local weights for either backend: the hosted
AlphaGenome API returns predictions only and can never expose internals.

Two reductions keep the output tractable, both applied on device before any
tensor is transferred:

  * Embeddings are pooled with --pooling_axis/--pooling_func (mean/max over
    the sequence or channel axis). They are not cropped to the gene by default;
    --embedding_crop gene restores the gene-body crop.
  * Predictions are aggregated per gene with --output_type/--output_func/
    --output_crop: weighted sum/mean of the RNA tracks over the gene body, its
    exons, or its introns, with exon coordinates read from the GENCODE parquet
    and shifted into each haplotype's own frame.

Usage:
    python scripts/capture_forward.py --config my_config.yaml

    python scripts/capture_forward.py --model_type borzoi \
        --gene_meta_path genes.tsv \
        --vcf_file_path samples.vcf.gz \
        --hg38_file_path hg38.fa \
        --gencode_parquet gencode.annotation.gtf.parquet \
        --n_genes 5 --n_samples 10 \
        --pooling_axis channel_dim,seq_dim --pooling_func mean,max \
        --output_h5 out/borzoi_tensors.h5

    python scripts/capture_forward.py --model_type alphagenome \
        --weights_path /path/to/alphagenome/checkpoint \
        --gene_meta_path genes.tsv --vcf_file_path samples.vcf.gz \
        --hg38_file_path hg38.fa --gencode_parquet gencode.annotation.gtf.parquet \
        --n_genes 5 --n_samples 10 --output_h5 out/alphagenome_tensors.h5
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from embher.embedding.cli.cli_utils import load_yaml_config, merge_configs, setup_logging, print_config_summary

from embher.embedding.utils import create_genome_dataset, load_gene_metadata, load_vcf_samples
from embher.embedding.capture.regions import SEQUENCE_LENGTHS
from embher.embedding.capture import regions, reduce
from embher.embedding.capture.h5_writer import (
    ForwardH5Writer, StackedH5Writer, merge_stacked_delta,
)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Run an AlphaGenome forward pass and dump all intermediate/final tensors to HDF5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', '-c', type=str, help='Path to YAML configuration file')

    parser.add_argument('--model_type', type=str, choices=['alphagenome', 'borzoi'],
                        default='alphagenome',
                        help="Model backend to capture (default: 'alphagenome')")
    parser.add_argument('--weights_path', type=str,
                        help='Local AlphaGenome checkpoint, or local Borzoi checkpoint directory '
                             '(local weights are required - the hosted API has no intermediates)')
    parser.add_argument('--random_init', action='store_true', default=None,
                        help='Borzoi only: use a randomly initialised network (same architecture '
                             'as the checkpoint at --weights_path, whose config is read but whose '
                             'weights are not) instead of the trained weights')
    parser.add_argument('--seed', type=int, default=0,
                        help='Weights seed for --random_init (default: 0)')
    parser.add_argument('--device', type=str,
                        help="Device, e.g. 'cpu', 'gpu:0' (JAX) or 'cuda:0' (Borzoi). "
                             "Default: first available")
    parser.add_argument('--borzoi_model_name', type=str,
                        help='HuggingFace Borzoi model name (alternative to --weights_path)')
    parser.add_argument('--use_flashzoi', action='store_true', default=None,
                        help='Use Flashzoi instead of standard Borzoi')
    parser.add_argument('--borzoi_replicate', type=int,
                        help='Borzoi/Flashzoi replicate number 0-3 (default: 0)')

    parser.add_argument('--gene_meta_path', type=str, help='Gene metadata TSV path')
    parser.add_argument('--vcf_file_path', type=str, help='VCF file path')
    parser.add_argument('--hg38_file_path', type=str, help='Reference genome FASTA path')
    parser.add_argument('--contig_prefix', type=str, default='',
                        help="Prefix for VCF contig names, e.g. 'chr' (default: '')")

    parser.add_argument('--min_maf', type=float, default=None,
                        help='Only apply variants with cohort MAF > this (exclusive). MAF is '
                             'computed over all selected samples, before --seed_h5 narrows them')
    parser.add_argument('--max_maf', type=float, default=None,
                        help='Only apply variants with cohort MAF <= this (inclusive)')

    parser.add_argument('--sequence_length', type=str, default='1MB',
                        choices=list(SEQUENCE_LENGTHS.keys()),
                        help="AlphaGenome input window size (default: '1MB'). "
                             "Ignored for Borzoi, whose window is fixed at 524288 bp")
    parser.add_argument('--crop_flank_bp', type=int, default=5000,
                        help='Flank added on each side of the gene body before cropping (default: 5000)')

    parser.add_argument('--embedding_crop', type=str, choices=['none', 'gene'], default='none',
                        help="Crop embeddings to the gene body + flank, or keep the whole "
                             "window (default: 'none')")
    parser.add_argument('--pooling_axis', type=str,
                        help="Comma-separated axes to pool embeddings over: "
                             "channel_dim,seq_dim (default: no pooling)")
    parser.add_argument('--pooling_func', type=str,
                        help='Comma-separated pooling functions: mean,max (default: no pooling)')
    parser.add_argument('--embedding_layers', type=str, choices=['all', 'final'], default='all',
                        help="Which embedding layers to store: every captured one ('all', "
                             "default) or only those feeding the output head ('final': "
                             "embeddings/final for Borzoi, embeddings/1bp and /128bp for "
                             "AlphaGenome). Predictions are stored either way; --pooling_* "
                             "applies to whichever layers are kept")
    parser.add_argument('--allow_full_tensors', action='store_true',
                        help='Permit storing uncropped, unpooled embeddings - gigabytes per '
                             'haplotype, so it must be asked for explicitly')

    parser.add_argument('--output_type', type=str, default='rna_seq',
                        help="Comma-separated prediction heads to aggregate, or 'none' to "
                             "store the raw prediction tensors (default: 'rna_seq')")
    parser.add_argument('--output_func', type=str, default='sum,mean',
                        help="Comma-separated aggregation functions: sum,mean (default: 'sum,mean')")
    parser.add_argument('--output_crop', type=str, default='gene,exon,intron',
                        help="Comma-separated gene regions to aggregate over: "
                             "gene,exon,intron (default: 'gene,exon,intron')")
    parser.add_argument('--gencode_parquet', type=str,
                        help='GENCODE annotation parquet (exon coordinates). Required for '
                             'exon/intron output aggregation')

    parser.add_argument('--n_genes', type=int, help='Number of genes to process (default: all)')
    parser.add_argument('--select_genes', type=str, nargs='+',
                        help='Explicit ENSG IDs to process (overrides --n_genes)')
    parser.add_argument('--random_selection', action='store_true',
                        help='Randomly sample --n_genes genes instead of taking the first N')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--n_samples', type=int, help='Number of VCF samples to process (default: all)')
    parser.add_argument('--sample_ids', type=str, nargs='+',
                        help='Explicit sample IDs to process (overrides --n_samples)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers building personalised sequences while the '
                             'GPU runs the previous one (default: 0, no prefetch)')

    parser.add_argument('--dtype', type=str, choices=['bfloat16', 'float32'], default='bfloat16',
                        help='Storage dtype (default: bfloat16, the model\'s own compute dtype)')
    parser.add_argument('--compression', type=str, choices=['gzip', 'lzf', 'none'], default='gzip')
    parser.add_argument('--output_h5', type=str, required=False, help='Output HDF5 file path')
    parser.add_argument('--resume', action='store_true',
                        help="Skip genes already marked complete in an existing --output_h5 "
                             "and append the rest, instead of truncating it. Refuses if the "
                             "existing file was written with different settings.")
    parser.add_argument('--seed_h5', type=str,
                        help="An existing single-gene --output_h5 (e.g. a published shard) "
                             "whose samples should be treated as already captured. Only the "
                             "samples requested here that are NOT already in --seed_h5 are "
                             "run through the model; the two are then merged into --output_h5. "
                             "Requires exactly one gene selected (--select_genes with one id). "
                             "A mismatch in model settings or this gene's region between the "
                             "seed and this run raises immediately rather than mixing them.")

    parser.add_argument('--include_reference', action='store_true',
                        help="Also forward the plain reference sequence (no donor "
                             "variants applied) once per gene, written as one extra "
                             "'REFERENCE' entry alongside the real donors on the stacked "
                             "sample axis -- same (n_samples+1, 2, ...) shape every real "
                             "donor uses (both haplotype slots hold the identical "
                             "reference result, since it has no haplotype distinction), "
                             "so downstream readers that treat the sample axis generically "
                             "pick it up with no changes. "
                             "Only supported with the stacked layout (same requirement as "
                             "--seed_h5). With --seed_h5, skipped if the seed already has "
                             "'REFERENCE' (recomputing it would duplicate, not update, an "
                             "identical deterministic value).")

    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--log_file', type=str)

    return parser.parse_args()


def select_genes(gene_meta: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if config.get('select_genes'):
        return gene_meta[gene_meta['ensg'].isin(config['select_genes'])].reset_index(drop=True)

    n_genes = config.get('n_genes')
    if n_genes is None:
        return gene_meta

    if config.get('random_selection'):
        return gene_meta.sample(
            n=min(n_genes, len(gene_meta)), random_state=config.get('random_seed', 42)
        ).reset_index(drop=True)

    return gene_meta.iloc[:n_genes].reset_index(drop=True)


def borzoi_rna_tracks(tracks_df: pd.DataFrame, output_types: tuple):
    """
    Channel indices and un-squash parameters for Borzoi's RNA tracks.

    Borzoi fuses every assay into one 7611-track head, so aggregating 'rna_seq'
    means selecting a subset - and the aggregate's column j is then not
    channel j, which is why /meta/tracks carries track_index.
    """
    from embher.embedding.utils import get_borzoi_track_indices
    from embher.embedding.capture import borzoi

    if 'rna_seq' not in output_types:
        return None, None

    # An empty query matches every description; the assay filter does the work.
    index = get_borzoi_track_indices(tracks_df, '', assay_type='RNA')
    if not index:
        raise ValueError("No RNA tracks found in the Borzoi targets table")

    def column(name, default):
        if name in tracks_df.columns:
            return tracks_df.iloc[index][name].astype(float).to_numpy()
        return np.full(len(index), default, dtype=float)

    # Every parameter comes from the targets table where it exists - Borzoi's
    # RNA tracks use scale=0.3, far from baskerville's 0.01 default, so the
    # fallbacks matter only for a table that lacks the columns entirely.
    unsquash = {
        'scale': column('scale', borzoi.DEFAULT_TRACK_SCALE),
        'clip_soft': column('clip_soft', borzoi.DEFAULT_CLIP_SOFT),
        'transform': column('track_transform', borzoi.DEFAULT_TRACK_TRANSFORM),
    }
    return np.asarray(index, dtype=int), unsquash


def init_random_borzoi(weights_path: str, seed: int, device: str):
    """
    Borzoi with random weights: the checkpoint's architecture (its config.json only), a seeded
    torch init. Same construction as attribute_layers.py's random_init, so the two agree for a
    given seed. init_borzoi_model(use_random_init=True) is not used because it ignores the
    checkpoint's config (it builds the default one, with the mouse head) and takes no seed.
    """
    import torch
    from borzoi_pytorch import AnnotatedBorzoi
    from borzoi_pytorch.config_borzoi import BorzoiConfig
    from embher.embedding.utils.gpu import select_device

    if device == 'auto':
        device = select_device(prefer_cuda=True)
    torch.manual_seed(seed)
    model = AnnotatedBorzoi(BorzoiConfig.from_pretrained(weights_path)).eval().to(device)
    print(f"Initialised RANDOM Borzoi (seed {seed}, architecture of {weights_path})")
    return model, model.tracks_df.copy()


def embedding_layers(config: Dict, final_layers: tuple):
    """The embedding tensor names to keep, or None for every one (--embedding_layers)."""
    return final_layers if config.get('embedding_layers', 'all') == 'final' else None


def init_backend(config: Dict, output_types: tuple):
    """
    Load the requested model and return (forward_fn, window_bp, track_metadata, track_index).

    forward_fn has the same signature for both backends:
        forward_fn(sequence, window_bp, win_start, crop_start, crop_end,
                   embedding_crop, poolings, outputs)

    Imported lazily because the two backends pull in disjoint heavy stacks
    (jax/haiku/alphagenome_research vs torch/borzoi_pytorch).
    """
    if config.get('model_type', 'alphagenome') == 'borzoi':
        from embher.embedding.utils import init_borzoi_model, BORZOI_SEQ_LEN
        from embher.embedding.capture import borzoi

        if config.get('random_init'):
            model, tracks_df = init_random_borzoi(
                config['weights_path'], config.get('seed', 0), config.get('device') or 'auto')
        else:
            model, tracks_df = init_borzoi_model(
                model_name=config.get('borzoi_model_name') or config.get('weights_path'),
                device=config.get('device') or 'auto',
                use_flashzoi=config.get('use_flashzoi', False),
                replicate=config.get('borzoi_replicate', 0),
            )
        if tracks_df is None:
            raise ValueError(
                "Borzoi loaded without track metadata; an AnnotatedBorzoi checkpoint is required"
            )
        track_index, unsquash = borzoi_rna_tracks(tracks_df, output_types)
        forward_fn = partial(borzoi.run_forward, model, unsquash_params=unsquash,
                             embedding_layers=embedding_layers(config, borzoi.FINAL_EMBEDDINGS))
        return (
            forward_fn,
            BORZOI_SEQ_LEN,
            borzoi.get_track_metadata(tracks_df, track_index=track_index),
            track_index,
        )

    from embher.embedding.capture import alphagenome

    model = alphagenome.load_model(config['weights_path'], device=config.get('device'))
    window_bp = SEQUENCE_LENGTHS[config.get('sequence_length', '1MB')]
    forward_fn = partial(alphagenome.run_forward, model, output_heads=output_types,
                         embedding_layers=embedding_layers(config, alphagenome.FINAL_EMBEDDINGS))
    # AlphaGenome's heads are already per-assay, so no channel subsetting.
    return forward_fn, window_bp, alphagenome.get_track_metadata(model), None


def check_track_alignment(track_metadata: Dict, tensors: Dict) -> None:
    """
    Check each aggregate's width against the track table that describes it.

    Both backends take their metadata's row order as the channel order. That
    is almost certainly right, but a mismatch would mislabel every column of
    every aggregate in the file, so it fails here instead of silently.
    """
    for name, (array, _) in tensors.items():
        if not name.startswith('outputs/'):
            continue
        head = name.split('/')[1]
        # Borzoi's aggregates describe the RNA subset, not the fused head.
        table = track_metadata.get(head) or track_metadata.get('rna_seq')
        if table is None:
            continue
        n_rows = len(table['track_index'])
        if n_rows != array.shape[0]:
            raise ValueError(
                f"'{name}' has {array.shape[0]} tracks but /meta/tracks/{head} has "
                f"{n_rows} rows; channel order cannot be trusted"
            )


# Settings that determine the schema of every tensor in the file - what a
# resumed run must match exactly, since old and new genes would otherwise
# carry different tensor names/shapes in the same file. Cosmetic settings
# (compression, timestamp, the free-form config blob) are not included.
RESUME_SCHEMA_KEYS = (
    'model_type', 'sequence_length_bp', 'dtype', 'embedding_crop',
    'pooling_axis', 'pooling_func', 'output_type', 'output_func',
    'output_crop', 'layout', 'min_maf', 'max_maf', 'include_reference', 'embedding_layers',
)


def read_resume_state(output_h5: str) -> tuple:
    """
    What --resume needs from an existing capture: which genes are already
    done, and the settings it was written with.

    Returns ({}, set()) for a path that does not exist yet, since --resume
    on a fresh output is just an ordinary run.
    """
    path = Path(output_h5)
    if not path.exists():
        return {}, set()
    with h5py.File(path, 'r') as f:
        meta_attrs = dict(f['meta'].attrs) if 'meta' in f else {}
        completed = (
            {ensg for ensg in f['genes'] if f[f'genes/{ensg}'].attrs.get('complete', False)}
            if 'genes' in f else set()
        )
    return meta_attrs, completed


def check_resume_compatible(existing_attrs: Dict, new_attrs: Dict) -> None:
    """Refuse to resume into a file whose schema does not match this run's."""
    mismatches = [
        key for key in RESUME_SCHEMA_KEYS
        if str(existing_attrs.get(key, '')) != str(new_attrs.get(key, ''))
    ]
    if mismatches:
        detail = ', '.join(
            f"{key}: {existing_attrs.get(key)!r} -> {new_attrs.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(
            f"--resume target was written with different settings ({detail}); "
            f"continuing it would mix two schemas in one file"
        )


# A --seed_h5's data for this gene is only reusable if the region it was
# computed over is identical - the window, crop and TSS a stacked sample's
# row was written against. Anything else (e.g. a TSS coordinate fix) makes
# the seed's data for that gene simply wrong, not just differently shaped,
# so this must raise rather than fall back to a full recompute silently.
REGION_SCHEMA_KEYS = (
    'chr', 'tss', 'gene_start', 'gene_end',
    'crop_start_ref', 'crop_end_ref', 'window_start_ref',
)


def check_region_compatible(seed_attrs: Dict, new_attrs: Dict, ensg: str) -> None:
    """Refuse to seed from a file whose region for this gene does not match."""
    mismatches = [
        key for key in REGION_SCHEMA_KEYS
        if str(seed_attrs.get(key, '')) != str(new_attrs.get(key, ''))
    ]
    if mismatches:
        detail = ', '.join(
            f"{key}: {seed_attrs.get(key)!r} -> {new_attrs.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(
            f"--seed_h5's region for {ensg} does not match this run ({detail}); its "
            f"existing samples were captured against a different window/TSS and are not "
            f"safe to merge with new ones. Rerun that gene from scratch instead."
        )


def select_samples(all_samples: List[str], config: Dict) -> List[str]:
    if config.get('sample_ids'):
        return config['sample_ids']
    n_samples = config.get('n_samples')
    return all_samples[:n_samples] if n_samples is not None else all_samples


def gene_context(
    gene_row: pd.Series,
    sequence_length_bp: int,
    crop_flank_bp: int,
    output_types: tuple,
    output_crops: tuple,
    exons_by_gene: Dict,
) -> Dict:
    """
    Everything about one gene that does not depend on sample or haplotype.

    Pulled out of the main loop so it runs once per gene rather than once per
    (gene, sample) pair, whether genes are visited in a plain for-loop or as
    boundaries in a flat stream of DataLoader batches.
    """
    ensg = gene_row['ensg']
    tss = int(gene_row['tss'])
    gene_start = int(gene_row['start'])
    gene_end = int(gene_row['end'])

    # Mirrors GenomeDataset.process_gene_variants' own window construction
    # exactly, so win_start lines up with position 0 of the captured tensors.
    win_start = max(0, tss - sequence_length_bp // 2)
    crop_start_ref = max(win_start, gene_start - crop_flank_bp)
    crop_end_ref = min(win_start + sequence_length_bp, gene_end + crop_flank_bp)

    gene_exons = exons_by_gene.get(ensg)
    crops = [crop for crop in output_crops if crop == 'gene' or gene_exons is not None]
    regions_ref = (
        regions.gene_intervals(gene_start, gene_end, gene_exons, crops)
        if output_types and crops else {}
    )

    return {
        'ensg': ensg, 'tss': tss, 'gene_start': gene_start, 'gene_end': gene_end,
        'win_start': win_start, 'crop_start_ref': crop_start_ref,
        'crop_end_ref': crop_end_ref, 'gene_exons': gene_exons, 'regions_ref': regions_ref,
    }


def _worker_init(_worker_id: int) -> None:
    """
    Re-open each worker's VCF/FASTA handles after the fork that creates it.

    Looked up via get_worker_info() rather than closing over the parent's
    dataset object: fork happens to make that closure work too, but this is
    what actually documents *why* - a forked worker's pysam handles alias the
    parent's file offset until they're reopened.
    """
    get_worker_info().dataset.reopen_handles()


def main():
    args = parse_args()

    yaml_config = load_yaml_config(args.config) if args.config else None
    cli_args = {k: v for k, v in vars(args).items() if k not in ['config', 'verbose', 'log_file']}
    config = merge_configs(yaml_config, cli_args)

    setup_logging(verbose=config.get('verbose', False), log_file=config.get('log_file'))
    print_config_summary(config, title="Forward-Pass Capture Configuration")

    model_type = config.get('model_type', 'alphagenome')
    required = ['gene_meta_path', 'vcf_file_path', 'hg38_file_path', 'output_h5']
    # Borzoi can fall back to a HuggingFace model name; AlphaGenome cannot.
    if model_type == 'alphagenome' or config.get('random_init') or not config.get('borzoi_model_name'):
        required.append('weights_path')
    missing = [k for k in required if not config.get(k)]
    if missing:
        print(f"Error: missing required config keys: {missing}", file=sys.stderr)
        return 1

    crop_flank_bp = config.get('crop_flank_bp', 5000)
    compression = config.get('compression', 'gzip')
    compression = None if compression == 'none' else compression

    try:
        poolings = reduce.parse_poolings(config.get('pooling_axis'), config.get('pooling_func'))
        output_types = reduce.parse_list(
            config.get('output_type', 'rna_seq'), reduce.OUTPUT_TYPES + ('none',), '--output_type'
        )
        output_types = () if 'none' in output_types else output_types
        output_funcs = reduce.parse_list(
            config.get('output_func', 'sum,mean'), reduce.OUTPUT_FUNCS, '--output_func'
        )
        output_crops = reduce.parse_list(
            config.get('output_crop', 'gene,exon,intron'), reduce.OUTPUT_CROPS, '--output_crop'
        )
    except ValueError as error:
        # Validated before the model loads: a typo here would otherwise surface
        # minutes into a run, after the checkpoint is on the device.
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if output_types and not (output_funcs and output_crops):
        print("Error: --output_func and --output_crop cannot be empty unless "
              "--output_type none", file=sys.stderr)
        return 1

    embedding_crop = config.get('embedding_crop', 'none') == 'gene'
    if not embedding_crop and not poolings and not config.get('allow_full_tensors'):
        print("Error: storing embeddings uncropped and unpooled writes the full input "
              "window per tensor - gigabytes per haplotype for AlphaGenome's 1bp "
              "embeddings. Pass --pooling_axis/--pooling_func, or --embedding_crop gene, "
              "or --allow_full_tensors if that is really what you want.", file=sys.stderr)
        return 1

    # Every individual's tensors have the same shape as long as nothing is
    # cropped to a haplotype-shifted interval: the input window is padded to a
    # fixed length, so an uncropped embedding's bin count is the same for
    # everyone, and an aggregate is always (n_tracks,). --embedding_crop gene
    # and --output_type none both break that, and fall back to a group per
    # haplotype.
    stacked = not embedding_crop and bool(output_types)

    include_reference = bool(config.get('include_reference'))
    if include_reference and not stacked:
        print("Error: --include_reference only supports the stacked layout (pooled "
              "embeddings and/or output aggregates, no --embedding_crop gene / "
              "--output_type none) -- same requirement as --seed_h5.", file=sys.stderr)
        return 1

    print(f"Loading {model_type} model...")
    forward_fn, sequence_length_bp, track_metadata, track_index = init_backend(config, output_types)

    # Built now, not at write_meta time, so --resume can compare it against
    # an existing file's /meta before doing anything else with that file.
    meta_attrs = {
        'model_type': model_type,
        'weights_path': (f"random_init_seed{config.get('seed', 0)}" if config.get('random_init')
                         else config.get('weights_path') or config.get('borzoi_model_name', '')),
        'sequence_length_bp': sequence_length_bp,
        'crop_flank_bp': crop_flank_bp,
        'dtype': config.get('dtype', 'bfloat16'),
        # Recorded explicitly, not only inside the serialised config, so a
        # reader can tell a pooled file from a full one without inspecting shapes.
        'embedding_crop': config.get('embedding_crop', 'none'),
        # '' when 'all', like include_reference below, so files written before this
        # flag existed still pass check_resume_compatible
        'embedding_layers': '' if config.get('embedding_layers', 'all') == 'all' else config['embedding_layers'],
        'pooling_axis': config.get('pooling_axis') or '',
        'pooling_func': config.get('pooling_func') or '',
        'output_type': ','.join(output_types),
        'output_func': ','.join(output_funcs) if output_types else '',
        'output_crop': ','.join(output_crops) if output_types else '',
        # '' when unset, so files written before this filter existed still match
        'min_maf': '' if config.get('min_maf') is None else config['min_maf'],
        'max_maf': '' if config.get('max_maf') is None else config['max_maf'],
        # '' when off (not a bare False), matching min_maf/max_maf's own convention
        # just above: check_resume_compatible defaults a MISSING key to '' too, so a
        # file written before this flag existed (no 'include_reference' key at all)
        # still compares equal to a new off-by-default run instead of mismatching on
        # 'None -> False' (confirmed directly -- every genuinely-published gene from
        # before this flag existed failed --seed_h5/--resume with exactly that error
        # until this matched the same-convention fix).
        'include_reference': 'true' if include_reference else '',
        'unsquash': model_type == 'borzoi' and bool(output_types),
        # 'stacked': sample and haplotype are array axes; 'per_haplotype':
        # one group each. A reader must branch on this.
        'layout': 'stacked' if stacked else 'per_haplotype',
        'timestamp': datetime.now().isoformat(),
        'config': config,
    }

    gene_meta = load_gene_metadata(config['gene_meta_path']).reset_index(drop=True)
    gene_meta = select_genes(gene_meta, config)

    if model_type == 'borzoi':
        # Borzoi centre-crops to the middle 196,608bp before its head, so a gene
        # reaching past that would be aggregated over only the part that
        # survives - a truncated sum indistinguishable from a valid one.
        fits = gene_meta.apply(
            lambda row: regions.fits_borzoi_crop(
                int(row['tss']), int(row['start']), int(row['end'])
            ),
            axis=1,
        )
        if not fits.all():
            dropped = gene_meta.loc[~fits, 'ensg'].tolist()
            print(f"Dropping {len(dropped)} gene(s) longer than Borzoi's "
                  f"{regions.BORZOI_CROP_BP}bp cropped window: {', '.join(dropped[:10])}"
                  + (' ...' if len(dropped) > 10 else ''))
            gene_meta = gene_meta[fits].reset_index(drop=True)

    resume = bool(config.get('resume'))
    if resume:
        existing_meta, completed_genes = read_resume_state(config['output_h5'])
        if existing_meta:
            check_resume_compatible(existing_meta, meta_attrs)
        if completed_genes:
            before = len(gene_meta)
            gene_meta = gene_meta[~gene_meta['ensg'].isin(completed_genes)].reset_index(drop=True)
            print(f"--resume: {len(completed_genes)} gene(s) already complete in "
                  f"{config['output_h5']}, {len(gene_meta)}/{before} remaining")
            if gene_meta.empty:
                print("Nothing left to do.")
                return 0

    print(f"Processing {len(gene_meta)} genes")

    exons_by_gene = {}
    if output_types and {'exon', 'intron'} & set(output_crops):
        if not config.get('gencode_parquet'):
            print("Error: --gencode_parquet is required for exon/intron aggregation", file=sys.stderr)
            return 1
        print("Loading exon coordinates from GENCODE...")
        exons_by_gene = regions.load_exons(config['gencode_parquet'], gene_meta['ensg'].tolist())
        missing = set(gene_meta['ensg']) - set(exons_by_gene)
        if missing:
            print(f"Warning: no exon annotation for {len(missing)} gene(s); "
                  f"their exon/intron aggregates are skipped")

    all_samples = load_vcf_samples(config['vcf_file_path'])
    samples = select_samples(all_samples, config)
    cohort_samples = list(samples)  # MAF cohort; `samples` may shrink under --seed_h5
    print(f"Processing {len(samples)} samples")

    seed_h5_arg = config.get('seed_h5')
    do_seed_merge = False
    if seed_h5_arg:
        if config.get('resume'):
            print("Error: --seed_h5 and --resume are mutually exclusive", file=sys.stderr)
            return 1
        if len(gene_meta) != 1:
            print("Error: --seed_h5 requires exactly one gene selected via --select_genes",
                  file=sys.stderr)
            return 1
        if not stacked:
            print("Error: --seed_h5 only supports the stacked layout (pooled embeddings "
                  "and/or output aggregates, no --embedding_crop gene / --output_type none)",
                  file=sys.stderr)
            return 1
        seed_ensg = gene_meta.iloc[0]['ensg']
        if not Path(seed_h5_arg).exists():
            print(f"--seed_h5 '{seed_h5_arg}' does not exist yet; running {seed_ensg} fresh")
            compute_reference_now = include_reference
        else:
            seed_ctx = gene_context(
                gene_meta.iloc[0], sequence_length_bp, crop_flank_bp,
                output_types, output_crops, exons_by_gene,
            )
            seed_region_attrs = {
                'chr': gene_meta.iloc[0]['chr'], 'tss': seed_ctx['tss'],
                'gene_start': seed_ctx['gene_start'], 'gene_end': seed_ctx['gene_end'],
                'crop_start_ref': seed_ctx['crop_start_ref'],
                'crop_end_ref': seed_ctx['crop_end_ref'],
                'window_start_ref': seed_ctx['win_start'],
            }
            with h5py.File(seed_h5_arg, 'r') as sf:
                seed_meta = dict(sf['meta'].attrs) if 'meta' in sf else {}
                check_resume_compatible(seed_meta, meta_attrs)
                if f'genes/{seed_ensg}' in sf:
                    check_region_compatible(
                        dict(sf[f'genes/{seed_ensg}'].attrs), seed_region_attrs, seed_ensg
                    )
                seed_samples = (
                    [s.decode() for s in sf['samples'][...]] if 'samples' in sf else []
                )
            # 'REFERENCE' is never part of `samples` (the requested donor list) -- it's
            # an extra row the writer adds, not something --sample_ids/--n_samples asks
            # for -- so it needs its own presence check alongside the donor `missing`
            # list, or a seed missing only 'REFERENCE' would wrongly take the
            # copy-through-unchanged fast path below and silently drop it.
            seed_has_reference = 'REFERENCE' in set(seed_samples)
            compute_reference_now = include_reference and not seed_has_reference
            missing = [s for s in samples if s not in set(seed_samples)]
            if not missing and not compute_reference_now:
                print(f"--seed_h5: all {len(samples)} requested sample(s) of {seed_ensg} "
                      f"already in '{seed_h5_arg}'; copying through unchanged")
                shutil.copy(seed_h5_arg, config['output_h5'])
                return 0
            if not missing and compute_reference_now:
                # No donor is missing to drive a compute pass, but 'REFERENCE' is still
                # needed -- GenomeDataset's length is n_genes * n_samples, so an empty
                # `samples` means the main loop below never executes at all, and there
                # is no donor iteration to piggyback the one-off reference forward pass
                # on. Out of scope for now: ask the caller to force a real compute pass
                # instead of silently doing nothing.
                print(f"Error: --seed_h5 '{seed_h5_arg}' already has all {len(samples)} "
                      f"requested sample(s) of {seed_ensg} but lacks 'REFERENCE', and "
                      f"--include_reference was requested -- there are no missing donors "
                      f"left to drive a compute pass. Request at least one additional "
                      f"--sample_ids entry, or rerun this gene without --seed_h5, to "
                      f"force reference computation.", file=sys.stderr)
                return 1
            print(f"--seed_h5: {len(seed_samples)} sample(s) of {seed_ensg} already in "
                  f"'{seed_h5_arg}', computing {len(missing)} new one(s)"
                  + (" plus REFERENCE" if compute_reference_now else "") + ": "
                  + ', '.join(missing[:10]) + (' ...' if len(missing) > 10 else ''))
            samples = missing
            do_seed_merge = True
    else:
        compute_reference_now = include_reference

    # GenomeDataset only touches expr_data to compute a mean/diff we discard;
    # a dummy zero frame gets us its indel-aware CoordinateTransformer and
    # lazy per-gene variant caching for free, without needing expression data.
    dummy_expr = pd.DataFrame(0.0, index=gene_meta['ensg'].values, columns=samples)

    if compute_reference_now and not samples:
        # Reference piggybacks on the first donor iteration the main loop below
        # happens to see for each gene (cheapest way to get a gene-boundary hook
        # without a second dataset pass) -- with zero donors requested, that loop
        # never executes, so reference would silently never be computed either.
        print("Error: --include_reference has no donor iteration to piggyback on -- "
              "--sample_ids/--n_samples selected zero samples.", file=sys.stderr)
        return 1

    genome_dataset = create_genome_dataset(
        metadata=gene_meta,
        vcf_file_path=config['vcf_file_path'],
        genome_file_path=config['hg38_file_path'],
        expression_data=dummy_expr,
        sample_list=samples,
        window_size=sequence_length_bp,
        contig_prefix=config.get('contig_prefix', ''),
        genome_contig_prefix='chr',
        onehot_encode=False,
        return_idx=True,
        min_maf=config.get('min_maf'),
        max_maf=config.get('max_maf'),
        maf_samples=cohort_samples,
    )

    checked_alignment = False
    start_time = time.time()
    # 'a' only once we know the target exists and --resume was asked for;
    # otherwise this is an ordinary run and any existing file is truncated.
    mode = 'a' if (resume and Path(config['output_h5']).exists()) else 'w'
    writer_args = dict(dtype=config.get('dtype', 'bfloat16'), compression=compression, mode=mode)
    # 'REFERENCE' is appended to the WRITER's sample list only -- genome_dataset above
    # keeps the real donor list (`samples`), since its gene_idx*n_samples+sample_idx
    # indexing is donor iteration, not something 'REFERENCE' participates in. The
    # writer just needs one extra pre-allocated row for it (StackedH5Writer sizes every
    # dataset as (len(self.samples), 2, ...), generic over whatever name list it's given).
    writer_samples = samples + (['REFERENCE'] if compute_reference_now else [])
    reference_sample_idx = len(samples)  # the reserved row, one past every real donor
    writer = (
        StackedH5Writer(config['output_h5'], writer_samples, **writer_args) if stacked
        else ForwardH5Writer(config['output_h5'], **writer_args)
    )
    with writer:
        writer.write_meta(meta_attrs)
        writer.write_track_metadata(track_metadata)

        # batch_size=1: haplotypes cannot be batched through the model anyway
        # - each one has its own indel-shifted coordinate frame, so the bin
        # weights an aggregate needs differ per haplotype and there is no
        # shared (B, L) grid to batch against. num_workers only overlaps the
        # *CPU* side (VCF lookup, sequence construction) with the previous
        # item's GPU forward pass; every item still reaches the model alone.
        num_workers = config.get('num_workers', 0)
        loader_kwargs = dict(batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])
        if num_workers:
            # A DataLoader worker is a fork; see GenomeDataset.reopen_handles
            # for why its pysam handles must not be the inherited copies.
            loader_kwargs.update(
                num_workers=num_workers, worker_init_fn=_worker_init,
                persistent_workers=True, prefetch_factor=2,
            )
        loader = DataLoader(genome_dataset, **loader_kwargs)

        # GenomeDataset addresses items as gene_idx * n_samples + sample_idx,
        # and a plain (shuffle=False) DataLoader yields them in that order
        # regardless of worker count - so a change in gene_idx is exactly a
        # gene boundary, the same one the old nested for-loop used.
        current_gene_idx = None
        ctx = None
        progress = tqdm(loader, total=len(genome_dataset), desc="Genes")
        for seq_arr, _, transformers, gene_idx, sample_idx in progress:
            if gene_idx != current_gene_idx:
                if current_gene_idx is not None:
                    # Every sample/haplotype of the previous gene has now been
                    # written: this is the unit --resume skips re-doing.
                    writer.mark_gene_complete(ctx['ensg'])
                    writer.flush()
                current_gene_idx = gene_idx
                progress.set_description(f"Genes ({gene_idx + 1}/{len(gene_meta)})")

                gene_row = gene_meta.iloc[gene_idx]
                ctx = gene_context(
                    gene_row, sequence_length_bp, crop_flank_bp,
                    output_types, output_crops, exons_by_gene,
                )
                writer.write_gene(ctx['ensg'], {
                    'chr': gene_row['chr'],
                    'tss': ctx['tss'],
                    'gene_start': ctx['gene_start'],
                    'gene_end': ctx['gene_end'],
                    'crop_start_ref': ctx['crop_start_ref'],
                    'crop_end_ref': ctx['crop_end_ref'],
                    'window_start_ref': ctx['win_start'],
                    'n_exons': len(ctx['gene_exons']) if ctx['gene_exons'] else 0,
                    'exonic_bp': sum(e - s for s, e in ctx['regions_ref'].get('exon', [])),
                    'intronic_bp': sum(e - s for s, e in ctx['regions_ref'].get('intron', [])),
                })

                if compute_reference_now:
                    # seq_arr[0] is the reference stack (PersonalDataset.py:
                    # `reference = np.concatenate([ref_enc, ref_enc])`, same encoded
                    # sequence duplicated into both halves) -- identical for every
                    # donor of this gene, so grabbing it from whichever donor happens
                    # to be first in gene-id order costs nothing extra: one forward
                    # pass here, not one per donor. No transformer/to_personal() shift:
                    # the reference has no indel, so its own crop_start_ref/
                    # crop_end_ref/regions_ref (untransformed) are exactly its frame.
                    ref_seq, _ = seq_arr[0]
                    ref_outputs = None
                    if ctx['regions_ref']:
                        ref_outputs = reduce.OutputAggregation(
                            regions=ctx['regions_ref'],
                            funcs=output_funcs,
                            track_index=track_index,
                            unsquash=(model_type == 'borzoi'),
                        )
                    ref_tensors = forward_fn(
                        ref_seq,
                        window_bp=sequence_length_bp,
                        win_start=ctx['win_start'],
                        crop_start=ctx['crop_start_ref'],
                        crop_end=ctx['crop_end_ref'],
                        embedding_crop=embedding_crop,
                        poolings=poolings,
                        outputs=ref_outputs,
                    )
                    # Both haplotype slots get the identical result -- the reference
                    # has no haplotype distinction, but every other dataset in this
                    # writer is shaped (n_samples+1, 2, ...), so 'REFERENCE' needs
                    # both slots filled for a reader that unconditionally sums/reads
                    # across the haplotype axis to see
                    # the right value, not half of it.
                    for ref_hap_idx in (0, 1):
                        writer.write_haplotype(ctx['ensg'], reference_sample_idx, ref_hap_idx, ref_tensors)
                        writer.write_coords(ctx['ensg'], reference_sample_idx, ref_hap_idx, {
                            'net_indel_shift': 0,
                        })

            ensg = ctx['ensg']
            win_start = ctx['win_start']
            crop_start_ref, crop_end_ref = ctx['crop_start_ref'], ctx['crop_end_ref']
            regions_ref = ctx['regions_ref']
            sample_id = samples[sample_idx]

            maternal_seq, paternal_seq = seq_arr[1]
            mat_transformer, pat_transformer = transformers

            for hap_idx, (haplotype, seq, transformer) in enumerate((
                ('maternal', maternal_seq, mat_transformer),
                ('paternal', paternal_seq, pat_transformer),
            )):
                crop_start_p, crop_end_p = transformer.transform_interval(
                    crop_start_ref, crop_end_ref
                )
                # Every interval moves into this haplotype's own frame, so
                # an indel inside an exon resizes that exon rather than
                # sliding the whole gene. The tensors' genomic_start is
                # already personalised, so nothing is transformed twice.
                outputs = None
                if regions_ref:
                    outputs = reduce.OutputAggregation(
                        regions=regions.to_personal(regions_ref, transformer),
                        funcs=output_funcs,
                        track_index=track_index,
                        unsquash=(model_type == 'borzoi'),
                    )

                tensors = forward_fn(
                    seq,
                    window_bp=sequence_length_bp,
                    win_start=win_start,
                    crop_start=crop_start_p,
                    crop_end=crop_end_p,
                    embedding_crop=embedding_crop,
                    poolings=poolings,
                    outputs=outputs,
                )
                if not checked_alignment:
                    check_track_alignment(track_metadata, tensors)
                    checked_alignment = True

                if stacked:
                    writer.write_haplotype(ensg, sample_idx, hap_idx, tensors)
                    # Stacking puts every individual's bin i in the same
                    # column, but bin i is in that haplotype's own frame;
                    # indels accumulate along the window, so record how far
                    # the two frames have drifted apart by its end.
                    window_end = win_start + sequence_length_bp
                    writer.write_coords(ensg, sample_idx, hap_idx, {
                        'net_indel_shift': (
                            transformer.ref_to_personal(window_end)
                            - transformer.ref_to_personal(win_start)
                            - sequence_length_bp
                        ),
                    })
                    continue

                for name, (array, attrs) in tensors.items():
                    if name.startswith('outputs/'):
                        writer.write_aggregate(
                            ensg, sample_id, haplotype, name, array, attrs
                        )
                    else:
                        writer.write_tensor(
                            ensg, sample_id, haplotype, name, array, attrs
                        )

        if ctx is not None:
            writer.mark_gene_complete(ctx['ensg'])
        writer.flush()

    if do_seed_merge:
        # writer above just wrote the delta (missing samples only) straight to
        # --output_h5; move it aside and let merge_stacked_delta combine it
        # with the seed, writing the union back to --output_h5 atomically.
        delta_path = f"{config['output_h5']}.delta"
        Path(config['output_h5']).rename(delta_path)
        merge_stacked_delta(seed_h5_arg, delta_path, config['output_h5'], seed_ensg)
        Path(delta_path).unlink()
        print(f"--seed_h5: merged into '{config['output_h5']}' "
              f"({len(seed_samples) + len(missing)} sample(s) total)")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min). Output: {config['output_h5']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
