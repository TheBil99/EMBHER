"""
Reduction specs for capture_forward.py: pooled embeddings and aggregated outputs.

Two independent reductions share this module, because both must run on device -
before a tensor is transferred to the host - and both need the same parsing,
naming and validation:

  * Embedding pooling (--pooling_axis/--pooling_func): mean/max over the
    sequence or channel axis of every non-prediction tensor.
  * Output aggregation (--output_*): weighted sum/mean of the prediction
    tensors over a gene's exonic/intronic/whole-body coordinates.

Nothing here imports jax or torch. Each backend maps a Pooling onto its own
framework's ops, and applies the weight vectors this module computes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POOL_AXES = {'seq_dim': 0, 'channel_dim': 1}
POOL_FUNCS = ('mean', 'max')

OUTPUT_TYPES = ('rna_seq',)
OUTPUT_FUNCS = ('sum', 'mean')
OUTPUT_CROPS = ('gene', 'exon', 'intron')


def parse_list(value: Optional[str], valid: Sequence[str], flag: str) -> Tuple[str, ...]:
    """Parse a comma-separated flag value, rejecting unknown entries."""
    if value is None:
        return ()
    items = tuple(item.strip() for item in str(value).split(',') if item.strip())
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(
            f"{flag}: unknown value(s) {unknown}; choose from {list(valid)}"
        )
    # dict.fromkeys de-duplicates while keeping the order the user wrote.
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class Pooling:
    """One (axis, func) pooling of an embedding tensor."""

    axis: str
    func: str

    @property
    def dim(self) -> int:
        return POOL_AXES[self.axis]

    @property
    def suffix(self) -> str:
        short = 'seq' if self.axis == 'seq_dim' else 'channel'
        return f'__{short}_{self.func}'


def parse_poolings(axes: Optional[str], funcs: Optional[str]) -> Tuple[Pooling, ...]:
    """
    Build the cross product of --pooling_axis and --pooling_func.

    Both flags or neither: a single axis with no function (or vice versa)
    is a typo, not a meaningful request.
    """
    parsed_axes = parse_list(axes, tuple(POOL_AXES), '--pooling_axis')
    parsed_funcs = parse_list(funcs, POOL_FUNCS, '--pooling_func')

    if bool(parsed_axes) != bool(parsed_funcs):
        raise ValueError(
            "--pooling_axis and --pooling_func must be given together "
            f"(got axes={parsed_axes or None}, funcs={parsed_funcs or None})"
        )

    return tuple(Pooling(axis, func) for axis in parsed_axes for func in parsed_funcs)


@dataclass
class OutputAggregation:
    """
    How to reduce the prediction tensors for one (gene, haplotype).

    Attributes:
        regions: {'gene'|'exon'|'intron': [(start, end), ...]} in *personalised*
            coordinates, already merged and haplotype-shifted.
        funcs: which of 'sum'/'mean' to compute.
        track_index: channel indices to keep, for backends whose heads are
            fused (Borzoi's 7611 human tracks); None keeps every channel.
        unsquash: also store aggregates computed from un-squashed bins.
    """

    regions: Dict[str, List[Tuple[int, int]]]
    funcs: Tuple[str, ...]
    track_index: Optional[np.ndarray] = None
    unsquash: bool = False

    @property
    def span(self) -> Tuple[int, int]:
        """Smallest interval covering every region, for the initial crop."""
        starts = [s for intervals in self.regions.values() for s, _ in intervals]
        ends = [e for intervals in self.regions.values() for _, e in intervals]
        return min(starts), max(ends)


def bin_weights(
    intervals: Sequence[Tuple[int, int]],
    length: int,
    resolution: int,
    tensor_start: int,
) -> np.ndarray:
    """
    Fraction of each bin's base pairs covered by `intervals`.

    Weighting rather than per-interval bin slicing is what keeps exon and
    intron from both claiming a boundary bin: at Borzoi's 32bp resolution an
    exon edge falls mid-bin, and snapping each interval outward to whole bins
    would double-count it. At AlphaGenome's 1bp resolution the weights are 0/1
    and the aggregates are exact per-base sums and means.

    Args:
        intervals: Non-overlapping (start, end) pairs, personalised coordinates.
        length: Number of bins in the tensor.
        resolution: Base pairs per bin.
        tensor_start: Coordinate of the tensor's first bin.

    Returns:
        (length,) float64 weights in [0, 1].
    """
    weights = np.zeros(length, dtype=np.float64)

    for start, end in intervals:
        if end <= start:
            continue
        # Only touch the bins this interval can reach; tensors run to millions
        # of bins and a gene's exons cover a tiny part of them.
        first = max(0, (start - tensor_start) // resolution)
        last = min(length, -(-(end - tensor_start) // resolution))
        if last <= first:
            continue

        edges = tensor_start + np.arange(first, last + 1, dtype=np.int64) * resolution
        overlap = np.minimum(edges[1:], end) - np.maximum(edges[:-1], start)
        weights[first:last] += np.clip(overlap, 0, None)

    weights /= resolution
    # Merged intervals cannot overlap, so this only guards float error.
    return np.clip(weights, 0.0, 1.0)


@dataclass
class OutputRequest:
    """One weighted reduction of a prediction tensor, resolved onto its bin grid."""

    crop: str
    weights: np.ndarray
    bin_start: int
    bin_end: int
    n_bp: float = field(default=0.0)

    def attrs(self, resolution: int, tensor_start: int) -> Dict[str, object]:
        return {
            'crop': self.crop,
            'resolution': resolution,
            'bin_start': self.bin_start,
            'bin_end': self.bin_end,
            'n_bins': int(np.count_nonzero(self.weights)),
            'n_bp': float(self.n_bp),
            'genomic_start': tensor_start + self.bin_start * resolution,
            'genomic_end': tensor_start + self.bin_end * resolution,
        }


def build_requests(
    aggregation: OutputAggregation,
    length: int,
    resolution: int,
    tensor_start: int,
) -> List[OutputRequest]:
    """
    Resolve each requested region set onto one tensor's bin grid.

    `tensor_start` must already be a personalised coordinate - it comes from
    the crop the backend applied, which was itself haplotype-shifted - so the
    regions and the grid are in the same frame and neither is transformed twice.
    """
    requests = []
    for crop, intervals in aggregation.regions.items():
        weights = bin_weights(intervals, length, resolution, tensor_start)
        nonzero = np.nonzero(weights)[0]
        if nonzero.size == 0:
            # An empty region set (a single-exon gene has no introns) still
            # gets a row, so downstream tables keep one entry per crop.
            requests.append(OutputRequest(crop, weights[:0], 0, 0, 0.0))
            continue
        bin_start, bin_end = int(nonzero[0]), int(nonzero[-1]) + 1
        n_bp = float(sum(max(0, e - s) for s, e in intervals))
        requests.append(
            OutputRequest(crop, weights[bin_start:bin_end], bin_start, bin_end, n_bp)
        )
    return requests
