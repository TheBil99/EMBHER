"""Mapping genomic regions onto a captured tensor's own bin grid."""

from typing import Dict, Tuple


def crop_bins(
    length: int,
    resolution: int,
    tensor_start: int,
    crop_start: int,
    crop_end: int,
) -> Tuple[int, int, Dict[str, int]]:
    """
    Map a genomic sub-region onto a tensor's own bin grid, snapping outward to
    whole bins so no requested base pair is lost.

    Resolution and origin are passed in rather than derived from `length`,
    because a tensor does not necessarily span the whole model input window:
    Borzoi center-crops before its head, so its final tensors cover only the
    middle of the window.

    Args:
        length: Number of positions (bins) in the tensor.
        resolution: Base pairs per position.
        tensor_start: Coordinate of the tensor's first position.
        crop_start: Start coordinate to keep (inclusive).
        crop_end: End coordinate to keep (exclusive).

    Returns:
        (bin_start, bin_end, attrs). The slice is empty when the requested
        region lies outside what this tensor covers.
    """
    bin_start = max(0, (crop_start - tensor_start) // resolution)
    bin_end = max(bin_start, min(length, -(-(crop_end - tensor_start) // resolution)))

    return bin_start, bin_end, {
        'resolution': resolution,
        'bin_start': bin_start,
        'bin_end': bin_end,
        'genomic_start': tensor_start + bin_start * resolution,
        'genomic_end': tensor_start + bin_end * resolution,
    }
