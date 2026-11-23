#!/usr/bin/env python3
"""
Exercise the pure parts of the capture reductions: interval algebra, bin
weighting, and the aggregation math both backends apply on device.

The partition identity (exon + intron == gene) is the real test here: it only
holds if the exon union, the intron complement, the per-haplotype coordinate
shift and the fractional bin weights are all consistent. Runs on CPU in a
second; no model, no weights, no GPU.

Usage:
    python tests/test_capture_reduce.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embher.embedding.capture import regions, reduce
from embher.embedding.utils.coordinates import CoordinateTransformer


def aggregate(values, intervals, resolution, tensor_start, func):
    """The reduction both backends run, in numpy."""
    weights = reduce.bin_weights(intervals, len(values), resolution, tensor_start)
    weighted = (values * weights[:, None]).sum(axis=0)
    return weighted if func == 'sum' else weighted / weights.sum()


def test_partition_identity():
    """exon + intron == gene, at 1bp and at 32bp resolution."""
    gene_start, gene_end = 1000, 5000
    exons = [(1000, 1200), (2350, 2400), (4700, 5000)]

    for resolution in (1, 32):
        length = (gene_end - gene_start + 2000) // resolution
        tensor_start = gene_start - 1000
        rng = np.random.default_rng(0)
        values = rng.random((length, 4)).astype(np.float64)

        parts = regions.gene_intervals(gene_start, gene_end, exons,
                                       ('gene', 'exon', 'intron'))
        sums = {
            crop: aggregate(values, intervals, resolution, tensor_start, 'sum')
            for crop, intervals in parts.items()
        }

        if resolution == 1:
            assert np.allclose(sums['exon'] + sums['intron'], sums['gene']), \
                'exon + intron must reconstruct the gene body at 1bp'
        else:
            # At 32bp a boundary bin is split between exon and intron by
            # fractional weight, so the identity still holds exactly - that is
            # the whole point of weighting rather than snapping each interval.
            assert np.allclose(sums['exon'] + sums['intron'], sums['gene']), \
                'fractional weights must partition boundary bins'
        print(f'  partition identity holds at {resolution}bp resolution')


def test_weights_are_fractional():
    """A 16bp exon inside a 32bp bin contributes half that bin."""
    weights = reduce.bin_weights([(1016, 1032)], length=4, resolution=32, tensor_start=1000)
    assert np.isclose(weights[0], 0.5), weights
    assert weights[1:].sum() == 0
    print('  sub-bin exon weighted as half a bin')


def test_exact_at_1bp():
    """At 1bp the weighted sum equals a plain slice sum."""
    values = np.arange(200, dtype=np.float64).reshape(200, 1)
    got = aggregate(values, [(1050, 1080)], 1, 1000, 'sum')
    assert np.isclose(got[0], values[50:80, 0].sum()), got
    got_mean = aggregate(values, [(1050, 1080)], 1, 1000, 'mean')
    assert np.isclose(got_mean[0], values[50:80, 0].mean())
    print('  1bp weighted sum/mean match plain slice sum/mean')


def test_haplotype_shift():
    """A deletion inside an exon shortens that exon, not the whole gene."""
    # 10bp deleted at 2000: everything past it shifts left by 10.
    transformer = CoordinateTransformer([(0, 0), (2000, -10)])
    parts = regions.gene_intervals(1000, 5000, [(1000, 1200), (1900, 2100)],
                                   ('gene', 'exon', 'intron'))
    shifted = regions.to_personal(parts, transformer)

    assert shifted['exon'][0] == (1000, 1200), 'exon before the indel is unmoved'
    # (1900, 2100) spans the deletion: its start stays, its end moves left.
    assert shifted['exon'][1] == (1900, 2090), shifted['exon'][1]
    assert shifted['gene'][0] == (1000, 4990), shifted['gene'][0]

    exonic = sum(e - s for s, e in shifted['exon'])
    intronic = sum(e - s for s, e in shifted['intron'])
    span = shifted['gene'][0][1] - shifted['gene'][0][0]
    assert exonic + intronic == span, (exonic, intronic, span)
    print('  indel inside an exon resizes it; partition survives the shift')


def test_empty_region():
    """A single-exon gene has no introns: an empty request, not a crash."""
    parts = regions.gene_intervals(1000, 2000, [(1000, 2000)], ('exon', 'intron'))
    assert parts['intron'] == []
    aggregation = reduce.OutputAggregation(regions=parts, funcs=('sum',))
    requests = {r.crop: r for r in reduce.build_requests(aggregation, 100, 32, 900)}
    assert requests['intron'].bin_end == requests['intron'].bin_start
    assert requests['intron'].n_bp == 0
    print('  single-exon gene yields an empty intron request')


def test_pooling_spec():
    """--pooling_axis/--pooling_func parse, cross, and reject typos."""
    poolings = reduce.parse_poolings('channel_dim,seq_dim', 'mean,max')
    assert [p.suffix for p in poolings] == [
        '__channel_mean', '__channel_max', '__seq_mean', '__seq_max'
    ]
    assert [p.dim for p in poolings] == [1, 1, 0, 0]

    for bad in (('seq_dim', None), (None, 'mean')):
        try:
            reduce.parse_poolings(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected rejection of {bad}')

    try:
        reduce.parse_poolings('seq_dim,bogus', 'mean')
    except ValueError as error:
        assert 'bogus' in str(error)
    else:
        raise AssertionError('expected rejection of an unknown axis')
    print('  pooling specs cross correctly and reject typos')


def test_borzoi_crop_filter():
    """Genes reaching past the centre crop are rejected, not truncated."""
    assert regions.fits_borzoi_crop(tss=1_000_000, gene_start=990_000, gene_end=1_010_000)
    assert not regions.fits_borzoi_crop(
        tss=1_000_000, gene_start=990_000, gene_end=1_200_000
    )
    print('  Borzoi centre-crop filter accepts short genes, rejects long ones')


def main():
    print('capture reduction checks:')
    test_partition_identity()
    test_weights_are_fractional()
    test_exact_at_1bp()
    test_haplotype_shift()
    test_empty_region()
    test_pooling_spec()
    test_borzoi_crop_filter()
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
