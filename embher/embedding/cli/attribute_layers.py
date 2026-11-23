#!/usr/bin/env python
"""
Population-free attribution of a Borzoi reference window along the architecture, for motif analysis.

For one gene the reference window (524,288 bp centred on the TSS) is attributed, with Captum input x
gradient, to one scalar per target:

  embedding_norm_<layer>   L2 norm of the position-pooled activation of each of the 24 checkpoints
                           capture_forward.py stores (--layers, default all; labels like
                           encoder_res_tower_0 / transformer_block_3 / embeddings_final)
  expression               the model's expression prediction over --tracks, one haplotype; with
                           --random_init the same readout through the randomly initialised network and
                           head (a null: the head has no trained meaning)

Results go to <out_dir>/<gene>/attribution/<role>/<target>/attribution.h5 (attribution (4, L), sequence (L,) as
0-3 / 4 = N, attrs: tss, win_start, layer, role), role being `reference` (pretrained weights) or
`random_init` (--random_init: same architecture and config, randomly initialised with --seed).

A target already done (summary.json present, same precision / seed / window / method and layer
definition) is skipped unless --force; attribution.h5 is written to a temporary name and renamed.

With --check_donor and --capture_h5 (trained weights, a validation run) the checkpoints are first
recomputed for that donor's two haplotypes and compared with capture_forward.py's stored pooled values
(relative max difference over channels after bf16 rounding, and correlation), written to
<out_dir>/<gene>/attribution/layer_check.tsv; layers over --tol are flagged ok = False.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gene_meta_path", required=True)
    p.add_argument("--gene", required=True)
    p.add_argument("--hg38_file_path", required=True)
    p.add_argument("--contig_prefix", default="chr")
    p.add_argument("--gencode_parquet", required=True, help="GENCODE annotation parquet (exon coordinates)")
    p.add_argument("--tracks", required=True, help="comma-separated Borzoi head track indices for the expression target")
    p.add_argument("--targets_path", default=None)
    p.add_argument("--layers", default=None, help="comma-separated capture layers (default: all 24)")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="random_init weights seed")
    p.add_argument("--check_donor", help="donor id for the layer check (trained weights)")
    p.add_argument("--vcf_file_path", help="needed with --check_donor")
    p.add_argument("--capture_h5", help="the gene's capture_forward.py H5, needed with --check_donor")
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--no_checkpoint", action="store_true")
    p.add_argument("--no_mask_padding", action="store_true")
    p.add_argument("--skip_expression", action="store_true",
                   help="norm targets only, no expression target (debugging on CPU)")
    p.add_argument("--window_size", type=int, default=524288)
    p.add_argument("--check_only", action="store_true", help="stop after the layer check")
    p.add_argument("--force", action="store_true",
                   help="recompute targets that are already done (same settings and layer definition)")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def bf16_round(a):
    import ml_dtypes
    return np.asarray(a, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def already_done(out, settings):
    """A target is done when its summary.json (written last) exists and its attribution.h5 was made
    with these settings; an h5 without `layer_def` predates it and counts as the "v1" definition."""
    import h5py
    if not (out / "summary.json").exists() or not (out / "attribution.h5").exists():
        return False
    try:
        with h5py.File(out / "attribution.h5", "r") as f:
            got = {k: f.attrs.get(k, "v1" if k == "layer_def" else None) for k in settings}
    except OSError:
        return False
    return all(str(got[k]) == str(v) for k, v in settings.items())


def layer_check(args, trunk, gene_row, gene, ensg, layers, ckpt):
    import h5py
    import ml_dtypes
    import torch

    from embher.embedding.attribution import donors as D
    from embher.embedding.attribution.borzoi_forward import pool, run_to_capture

    (w,) = D.build_donor_windows(gene_row, [args.check_donor], args.vcf_file_path,
                                 args.hg38_file_path, args.contig_prefix)
    with h5py.File(args.capture_h5, "r") as f:
        samples = [s.decode() if isinstance(s, bytes) else str(s) for s in f["samples"][...]]
        if args.check_donor not in samples:
            raise SystemExit(f"[error] {args.check_donor} not in {args.capture_h5}")
        row = samples.index(args.check_donor)
        grp = f[f"genes/{ensg}/pooled_embeddings"]
        stored = {}
        for layer in layers:
            raw = grp[f"{layer}__seq_mean"][row]                    # (2, d) bf16 bits
            stored[layer] = raw.view(ml_dtypes.bfloat16).astype(np.float32)
    rows = []
    for layer in layers:
        vals = []
        for oh in w.onehots:
            x = torch.from_numpy(oh)[None].to(args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=args.precision == "bf16"):
                t = run_to_capture(trunk, x, layer, False)
            vals.append(pool(t, "pos")[0].float().cpu().numpy())
        mine, ref = bf16_round(np.stack(vals)), stored[layer]
        rel = float(np.abs(mine - ref).max() / np.abs(ref).max())
        corr = float(np.corrcoef(mine.ravel(), ref.ravel())[0, 1])
        rows.append(dict(layer=layer, label=layer.replace("/", "_"), donor=args.check_donor,
                         max_rel_diff=rel, corr=corr, ok=bool(rel <= args.tol)))
        print(f"[check] {layer}: max rel diff {rel:.4f} corr {corr:.6f}", flush=True)
        torch.cuda.empty_cache()
    out = Path(args.out_dir) / gene / "attribution"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "layer_check.tsv", sep="\t", index=False)


def main():
    args = parse_args()
    import h5py
    import pysam
    import torch
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    from embher.embedding.attribution import donors as D
    from embher.embedding.attribution import io as aio
    from embher.embedding.attribution.attributer import BorzoiAttributer
    from embher.embedding.attribution.borzoi_forward import (
        CAPTURE_LAYERS, capture_label, layer_definition)
    from embher.embedding.attribution.donor_targets import (
        DonorExpressionTarget, exon_bin_weights)
    from embher.embedding.attribution.layer_targets import CaptureLayerNormTarget
    from embher.embedding.attribution.targets import read_track_params
    from embher.embedding.utils.data import load_gene_metadata

    layers = [l.strip() for l in args.layers.split(",")] if args.layers else list(CAPTURE_LAYERS)
    bad = [l for l in layers if l not in CAPTURE_LAYERS]
    if bad:
        raise SystemExit(f"[error] unknown layers {bad}; one of {list(CAPTURE_LAYERS)}")
    role = "random_init" if args.random_init else "reference"
    tracks = [int(t) for t in args.tracks.split(",")]

    meta = load_gene_metadata(args.gene_meta_path, filter_chromosomes=None)
    sym = meta["gene_symbol"] if "gene_symbol" in meta else meta["ensg"]
    hit = meta[(sym == args.gene) | (meta["ensg"] == args.gene)]
    if hit.empty:
        raise SystemExit(f"[error] {args.gene} not in {args.gene_meta_path}")
    gene_row = hit.iloc[0]
    gene = str(sym[hit.index[0]])
    ensg = str(gene_row["ensg"]).split(".")[0]
    chrom = f"{args.contig_prefix}{gene_row['chr']}"

    if args.random_init:
        torch.manual_seed(args.seed)
        trunk = Borzoi(BorzoiConfig.from_pretrained(args.weights_path)).eval().to(args.device)
        print(f"[info] random_init weights, seed {args.seed}", flush=True)
    else:
        trunk = Borzoi.from_pretrained(args.weights_path).eval().to(args.device)
    ckpt = not args.no_checkpoint

    if args.check_donor and not args.random_init:
        for name in ("vcf_file_path", "capture_h5"):
            if getattr(args, name) is None:
                raise SystemExit(f"[error] --check_donor needs --{name}")
        layer_check(args, trunk, gene_row, gene, ensg, layers, ckpt)
    if args.check_only:
        return

    fasta = pysam.FastaFile(args.hg38_file_path)
    onehot, ws = aio.make_window(fasta, chrom, int(gene_row["tss"]), args.window_size)
    x = torch.from_numpy(onehot)[None].to(args.device)
    seq_idx = onehot.argmax(0).astype(np.int8)
    seq_idx[onehot.sum(0) == 0] = 4
    tss = int(gene_row["tss"])

    targets = {}
    if not args.skip_expression:
        scale, clip = read_track_params(tracks, args.targets_path)
        exons = D.reference_exons(args.gencode_parquet, gene_row)
        bins = exon_bin_weights(exons, ws)
        targets["expression"] = (DonorExpressionTarget(trunk, tracks, scale, clip, [bins],
                                                       args.precision, ckpt).to(args.device), None)
    for layer in layers:
        targets[f"embedding_norm_{capture_label(layer)}"] = (
            CaptureLayerNormTarget(trunk, layer, args.precision, ckpt).to(args.device), layer)

    seed = args.seed if args.random_init else -1
    for name, (target, layer) in targets.items():
        out = Path(args.out_dir) / gene / "attribution" / role / name
        layer_def = layer_definition(layer) if layer else "v1"
        settings = dict(precision=args.precision, seed=seed, win_start=ws, tss=tss,
                        method="inputxgradient", layer_def=layer_def)
        if not args.force and already_done(out, settings):
            print(f"[skip] {gene} {role} {name}: done with the same settings", flush=True)
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            value = float(target(x).item())
        attr = BorzoiAttributer(target, "inputxgradient").attribute(
            x, mask_padding=not args.no_mask_padding)[0].numpy()               # (4, L)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").unlink(missing_ok=True)       # summary.json last = marks it done
        tmp = out / "attribution.tmp.h5"
        with h5py.File(tmp, "w") as f:
            f.attrs.update(dict(gene=gene, role=role, target=name, layer=str(layer), chrom=chrom,
                                value=value, **settings))
            f.create_dataset("attribution", data=attr.astype(np.float32), compression="gzip")
            f.create_dataset("sequence", data=seq_idx, compression="gzip")
        tmp.replace(out / "attribution.h5")
        (out / "summary.json").write_text(json.dumps(dict(
            gene=gene, role=role, target=name, layer=layer, value=value, win_start=ws, tss=tss,
            attribution_sum=float(attr.sum()), seconds=round(time.perf_counter() - t0, 1),
            random_init=args.random_init, seed=args.seed if args.random_init else None,
            layer_def=layer_def), indent=2))
        print(f"[done] {gene} {role} {name}: value {value:.5f} ({time.perf_counter() - t0:.1f}s)",
              flush=True)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
