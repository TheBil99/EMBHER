#!/usr/bin/env python3
"""
Inspect a capture_forward.py HDF5 file: layout, per-tensor geometry and a few
sanity checks that the stored coordinates actually describe the stored arrays.

Handles both layouts (see /meta layout): 'stacked', where sample and
haplotype are array axes, and 'per_haplotype', where each is a group. In both
it checks the partition identity - exon + intron == gene - which is what
proves the exon masks and the per-haplotype coordinate shift are both right.

Usage:
    python scripts/inspect_h5.py out/borzoi_tensors.h5
"""

import argparse
import json

import h5py
import ml_dtypes
import numpy as np


def read(dset):
    """bfloat16 is stored as raw uint16 bit patterns; h5py has no native bf16."""
    raw = dset[...]
    if dset.attrs.get('dtype') == 'bfloat16' or raw.dtype == np.uint16:
        return raw.view(ml_dtypes.bfloat16).astype(np.float32)
    return raw.astype(np.float32)


def inspect_stacked(f, problems):
    """
    Check a (n_samples, 2, ...) capture, whole cohort at a time.

    Stacking is itself the strongest check available: a dataset only exists
    because every haplotype agreed on its shape, so what is left to verify is
    the content and the per-haplotype coords stored beside it.
    """
    samples = [s.decode() for s in f['samples'][...]]
    print(f"\n=== genes === ({len(samples)} samples x 2 haplotypes)")

    for ensg in f['genes']:
        g = f[f'genes/{ensg}']
        print(f"\n  {ensg}: {dict(g.attrs)}")

        aggregates, stored = {}, 0
        for name in sorted(_datasets(g)):
            if name.startswith('coords/'):
                continue
            d = g[name]
            a = dict(d.attrs)
            stored += d.id.get_storage_size()
            res = int(a.get('resolution', -1))
            label = a.get('crop') or f"{a.get('pooled_axis')}/{a.get('pooled_func')}"
            print(f"    {name:52s} {str(d.shape):20s} res={res:<5d} {label}")

            if 'crop' in a:
                aggregates[name] = read(d)
                continue

            values = read(d)
            # An uncropped tensor spans the whole input window, so its row
            # count is fixed by the window and resolution - unless the
            # sequence axis is the one that was pooled away.
            if res > 0 and a.get('pooled_axis') != 'seq_dim':
                expect = (int(a['genomic_end']) - int(a['genomic_start'])) // res
                if expect != values.shape[-1]:
                    problems.append(f"{ensg}/{name}: {values.shape[-1]} positions, "
                                    f"span/res => {expect}")
            if not np.isfinite(values).all():
                problems.append(f"{ensg}/{name}: non-finite values")
            if np.all(values == 0):
                problems.append(f"{ensg}/{name}: all zeros")

        print(f"    [{len(aggregates)} aggregates, {stored/1e6:.1f} MB stored]")
        _check_partition(ensg, g, aggregates, problems)
        _report_coords(g)


def _datasets(group):
    names = []
    group.visit(lambda n: names.append(n) if isinstance(group[n], h5py.Dataset) else None)
    return names


def _check_partition(ensg, g, aggregates, problems):
    """exon + intron == gene, for every sample and haplotype at once."""
    for base in {n.rsplit('/', 1)[0] for n in aggregates}:
        for suffix in ('sum_squashed', 'sum_unsquashed', 'sum'):
            parts = [aggregates.get(f'{base}/{crop}_{suffix}')
                     for crop in ('gene', 'exon', 'intron')]
            if any(v is None for v in parts):
                continue
            gene_v, exon_v, intron_v = parts
            if np.allclose(gene_v, exon_v + intron_v, rtol=1e-3, atol=1e-3):
                print(f"    partition ok: {base}/{suffix} exon+intron == gene")
            else:
                worst = float(np.nanmax(np.abs(gene_v - exon_v - intron_v)))
                bad = int((~np.isclose(gene_v, exon_v + intron_v,
                                       rtol=1e-3, atol=1e-3)).any(axis=-1).sum())
                problems.append(f"{ensg}/{base} ({suffix}): exon+intron differs from "
                                f"gene by up to {worst:.4g} in {bad} haplotype(s)")


def _report_coords(g):
    """
    Per-haplotype placement, which is what stacking moves out of the attrs.

    net_indel_shift is how far this haplotype's coordinates have drifted from
    the reference by the end of the window: position i of a stacked tensor is
    in each haplotype's own frame, so a large spread here means columns are
    comparable only loosely.
    """
    if 'coords' not in g:
        return
    for name in sorted(_datasets(g['coords'])):
        v = g[f'coords/{name}'][...]
        print(f"    coords/{name:28s} min={np.nanmin(v):.6g} max={np.nanmax(v):.6g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('h5')
    p.add_argument('--max_tensors', type=int, default=100)
    args = p.parse_args()

    problems = []
    with h5py.File(args.h5, 'r') as f:
        print("=== /meta ===")
        for k, v in f['meta'].attrs.items():
            s = str(v)
            print(f"  {k}: {s[:120]}{'...' if len(s) > 120 else ''}")

        if 'tracks' in f['meta']:
            print("\n=== /meta/tracks ===")
            for head in f['meta/tracks']:
                grp = f[f'meta/tracks/{head}']
                n = {k: len(grp[k]) for k in grp}
                print(f"  {head}: {n}")

        if f['meta'].attrs.get('layout') == 'stacked':
            inspect_stacked(f, problems)
            _summary(problems)
            return

        cropped_embeddings = f['meta'].attrs.get('embedding_crop', 'gene') != 'none'

        print("\n=== genes ===")
        for ensg in f['genes']:
            g = f[f'genes/{ensg}']
            ga = dict(g.attrs)
            print(f"\n  {ensg}: {ga}")
            crop_lo = int(ga['crop_start_ref'])
            crop_hi = int(ga['crop_end_ref'])

            for sample in g:
                for hap in g[sample]:
                    node = g[f'{sample}/{hap}']
                    print(f"    --- {sample}/{hap} ---")
                    names = []
                    node.visit(lambda n, nm=names, nd=node:
                               nm.append(n) if isinstance(nd[n], h5py.Dataset) else None)
                    total = 0
                    aggregates = {}
                    for name in sorted(names)[:args.max_tensors]:
                        d = node[name]
                        a = dict(d.attrs)
                        total += d.id.get_storage_size()
                        res = int(a.get('resolution', -1))
                        gs, ge = int(a.get('genomic_start', -1)), int(a.get('genomic_end', -1))
                        crop = a.get('crop')
                        pooled = a.get('pooled_axis')
                        label = f" {crop}" if crop else (f" {pooled}/{a.get('pooled_func')}" if pooled else "")
                        print(f"      {name:38s} {str(d.shape):16s} res={res:<5d} "
                              f"[{gs},{ge}) span={ge-gs}{label}")

                        if crop is not None:
                            aggregates[name] = (read(d), a)
                            # An empty region (a single-exon gene's introns) is
                            # legitimately zero for sum and NaN for mean.
                            if float(a.get('n_bp', 0)) > 0:
                                v = aggregates[name][0]
                                if not np.isfinite(v).all():
                                    problems.append(f"{ensg}/{sample}/{hap}/{name}: non-finite")
                            continue

                        # The array must have exactly as many rows as the
                        # genomic span it claims, at its stated resolution -
                        # unless the sequence axis was pooled away.
                        if res > 0 and pooled != 'seq_dim':
                            expect = (ge - gs) // res
                            if expect != d.shape[0]:
                                problems.append(
                                    f"{ensg}/{sample}/{hap}/{name}: {d.shape[0]} rows "
                                    f"but span {ge-gs} at res {res} => {expect}")
                            # genomic_start/end are PERSONAL coordinates: the crop
                            # is put through CoordinateTransformer, so indels shift
                            # them off the reference crop. Only flag a drift far
                            # larger than any plausible indel load - and only when
                            # the file was cropped to the gene at all, since an
                            # uncropped tensor spans the whole input window by design.
                            drift = max(abs(gs - crop_lo), abs(ge - crop_hi))
                            if cropped_embeddings and drift > res + 10000:
                                problems.append(
                                    f"{ensg}/{sample}/{hap}/{name}: [{gs},{ge}) drifts "
                                    f"{drift} bp from reference crop [{crop_lo},{crop_hi})")

                        v = read(d)
                        if not np.isfinite(v).all():
                            problems.append(f"{ensg}/{sample}/{hap}/{name}: non-finite values")
                        if np.all(v == 0):
                            problems.append(f"{ensg}/{sample}/{hap}/{name}: all zeros")
                    print(f"      [{len(names)} datasets, {total/1e6:.1f} MB stored]")

                    # exon + intron must reconstruct the gene body. Exact at
                    # 1bp; at coarser resolutions the fractional bin weights
                    # still make it hold, so a failure means the masks or the
                    # haplotype shift disagree with the stored geometry.
                    for base in {n.rsplit('/', 1)[0] for n in aggregates}:
                        for suffix in ('sum', 'sum_unsquashed'):
                            parts = {
                                crop: aggregates.get(f'{base}/{crop}_{suffix}')
                                for crop in ('gene', 'exon', 'intron')
                            }
                            if any(v is None for v in parts.values()):
                                continue
                            gene_v = parts['gene'][0]
                            rebuilt = parts['exon'][0] + parts['intron'][0]
                            if not np.allclose(gene_v, rebuilt, rtol=1e-3, atol=1e-3):
                                worst = float(np.max(np.abs(gene_v - rebuilt)))
                                problems.append(
                                    f"{ensg}/{sample}/{hap}/{base} ({suffix}): exon+intron "
                                    f"differs from gene by up to {worst:.4g}")
                            else:
                                print(f"      partition ok: {base}/{suffix} "
                                      f"exon+intron == gene")
                    # Indels make the personal crop a different length per
                    # haplotype, so the same tensor has different row counts
                    # across haplotypes/samples. Surface that explicitly.
                    # Borzoi's finest tensor is 2 bp, AlphaGenome's is 1 bp, so
                    # pick whichever this file actually has.
                    ref_span = crop_hi - crop_lo
                    spatial = [
                        n for n in names
                        if 'crop' not in node[n].attrs
                        and node[n].attrs.get('pooled_axis') != 'seq_dim'
                    ]
                    if not spatial:
                        continue
                    finest = min(spatial, key=lambda n: int(node[n].attrs['resolution']))
                    fd = node[finest]
                    res = int(fd.attrs['resolution'])
                    span = int(fd.attrs['genomic_end']) - int(fd.attrs['genomic_start'])
                    if cropped_embeddings:
                        print(f"      finest={finest} res={res} span={span} vs reference crop "
                              f"span={ref_span} (indel shift {span - ref_span:+d} bp)")
                    else:
                        # Embeddings were stored uncropped, so their span is the
                        # whole input window and says nothing about indel load.
                        print(f"      finest={finest} res={res} span={span} (uncropped window)")

    _summary(problems)


def _summary(problems):
    print("\n=== checks ===")
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for x in problems:
            print("   -", x)
    else:
        print("  all geometry/finiteness checks passed")


if __name__ == '__main__':
    main()
