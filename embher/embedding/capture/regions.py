"""
Gene-body, exonic and intronic intervals for output aggregation.

Exons come straight from the GENCODE GTF parquet that tmp/scripts/prepare_gene_meta.py
already reads, with the filter pushed down to pyarrow so only the selected
genes' exon rows are materialised out of its ~3.5M. Coordinates there are
pyranges-style 0-based half-open, so they need no GTF-style -1.

"Exonic" is the union of every transcript's exons, which makes exon and intron
a clean partition of the gene body: intron = gene_body - exon_union. Genes with
many transcripts therefore end up with little intron, by construction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Borzoi's TargetLengthCrop keeps the middle 6144 bins x 32bp of its window.
BORZOI_CROP_BP = 6144 * 32

# Named input window sizes accepted by capture_forward.py --sequence_length
# (the AlphaGenome API's sequence-length constants; Borzoi's window is fixed).
SEQUENCE_LENGTHS = {
    '2KB': 2048,
    '16KB': 16384,
    '100KB': 102400,
    '500KB': 524288,
    '1MB': 1048576,
}

Interval = Tuple[int, int]


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Sort and union overlapping/abutting intervals."""
    ordered = sorted((int(s), int(e)) for s, e in intervals if e > s)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(span: Interval, intervals: Sequence[Interval]) -> List[Interval]:
    """The parts of `span` not covered by `intervals` (assumed merged)."""
    start, end = span
    gaps = []
    cursor = start
    for interval_start, interval_end in intervals:
        if interval_end <= cursor:
            continue
        if interval_start >= end:
            break
        if interval_start > cursor:
            gaps.append((cursor, min(interval_start, end)))
        cursor = max(cursor, interval_end)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def load_exons(
    gencode_parquet: str,
    gene_ids: Sequence[str],
) -> Dict[str, List[Interval]]:
    """
    Load merged exon intervals per gene from the GENCODE parquet.

    The row filter is pushed down to the parquet reader, so entire row groups
    are skipped rather than read and discarded. Gene IDs are version-stripped
    to match load_gene_metadata's plain ENSG form; since the filter must run
    against the stored values, it matches on the bare ID after reading a
    version-agnostic superset.

    Args:
        gencode_parquet: Path to the GENCODE annotation parquet.
        gene_ids: Plain ENSG IDs (no version suffix) to keep.

    Returns:
        {ensg: [(start, end), ...]} with exons merged across all transcripts.
        Genes with no exon rows are absent.
    """
    import pyarrow.parquet as pq

    path = Path(gencode_parquet)
    if not path.exists():
        raise FileNotFoundError(f"GENCODE parquet not found: {path}")

    table = pq.read_table(
        path,
        columns=['Feature', 'Start', 'End', 'gene_id'],
        filters=[('Feature', '==', 'exon')],
    )
    exons = table.to_pandas()
    exons['gene_id'] = exons['gene_id'].str.split('.').str[0]
    exons = exons[exons['gene_id'].isin(set(gene_ids))]

    return {
        ensg: merge_intervals(list(zip(group['Start'], group['End'])))
        for ensg, group in exons.groupby('gene_id', sort=False)
    }


def gene_intervals(
    gene_start: int,
    gene_end: int,
    exons: Optional[Sequence[Interval]],
    crops: Sequence[str],
) -> Dict[str, List[Interval]]:
    """
    Build the requested {'gene'|'exon'|'intron': intervals} in reference coordinates.

    Exons are clipped to the gene body so an annotation whose transcripts run
    past the gene record cannot pull the aggregate outside it.
    """
    span = (int(gene_start), int(gene_end))
    clipped = merge_intervals(
        [(max(span[0], s), min(span[1], e)) for s, e in (exons or [])]
    )

    available = {
        'gene': [span],
        'exon': clipped,
        'intron': subtract_intervals(span, clipped),
    }
    return {crop: available[crop] for crop in crops}


def to_personal(
    regions: Dict[str, List[Interval]],
    transformer,
) -> Dict[str, List[Interval]]:
    """
    Shift every interval into one haplotype's coordinate frame.

    Each boundary transforms independently, so an indel inside an exon
    correctly lengthens or shortens that exon rather than sliding it.
    """
    return {
        crop: [transformer.transform_interval(start, end) for start, end in intervals]
        for crop, intervals in regions.items()
    }


def fits_borzoi_crop(tss: int, gene_start: int, gene_end: int, margin: int = 1000) -> bool:
    """
    Whether a TSS-centred Borzoi window keeps the whole gene body.

    Borzoi centre-crops to the middle 196,608bp before its head, so a gene
    reaching further than ~98kb from its TSS would be aggregated over only the
    part that survives the crop - a truncated sum that looks perfectly valid.
    Such genes are dropped instead. The margin absorbs indel shifts between
    reference and personalised coordinates.
    """
    half = BORZOI_CROP_BP // 2 - margin
    return (gene_start >= tss - half) and (gene_end <= tss + half)
