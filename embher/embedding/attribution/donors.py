"""
Personal-genome (donor) windows for attribution, built exactly as capture_forward.py builds them.

Cohort embeddings and predictions produced by capture_forward.py come from the personal-genome
windows it builds, so a donor's attribution must start from the same two haplotype sequences:
`create_genome_dataset` with the capture's arguments yields each donor's maternal and paternal
sequence for the TSS-centred window (right-padded with N / truncated to the window, so the length
is always `window_size`) together with one CoordinateTransformer per haplotype. An indel shifts
every later coordinate of that haplotype, so

  * the gene's exon intervals are moved into each haplotype's own frame before they weight the
    prediction bins (`personal_exons`), as the capture does, and
  * an attribution computed in a haplotype's frame is mapped back onto reference coordinates
    (`to_reference_frame`) to be compared across donors or drawn along the genome.

Heavy imports (pysam through the dataset, torch for the one-hot) happen inside the functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BORZOI_WINDOW = 524288
BIN_BP = 32
N_OUT_BINS = 6144


@dataclass
class DonorWindow:
    """One donor's two haplotypes for one gene's TSS-centred window."""

    donor_id: str
    win_start: int                       # reference coordinate of sequence position 0
    sequences: List[str]                 # [maternal, paternal], each `window_size` long
    transformers: List[object]           # CoordinateTransformer per haplotype
    onehots: List[np.ndarray] = field(default_factory=list)   # (4, L) float32, N -> zeros

    @property
    def net_indel_shift(self) -> List[int]:
        """How far each haplotype's frame has drifted from the reference by the window's end."""
        return [t.ref_to_personal(self.win_start + len(s)) - (self.win_start + len(s))
                for t, s in zip(self.transformers, self.sequences)]


def gene_window_start(tss: int, window_size: int = BORZOI_WINDOW) -> int:
    """Window start of the capture (GenomeDataset): max(0, tss - window_size // 2)."""
    return max(0, int(tss) - window_size // 2)


def output_start(win_start: int, window_size: int = BORZOI_WINDOW) -> int:
    """Coordinate of Borzoi's first output bin (the head covers only the central 196,608 bp)."""
    return win_start + (window_size - N_OUT_BINS * BIN_BP) // 2


def reference_exons(gencode_parquet: str, gene_row: pd.Series) -> List[Tuple[int, int]]:
    """Merged exon intervals of the gene (reference coordinates), clipped to the gene body,
    exactly as capture_forward.gene_context builds `regions_ref['exon']`."""
    from embher.embedding.capture import regions

    ensg = str(gene_row["ensg"]).split(".")[0]
    exons = regions.load_exons(gencode_parquet, [ensg]).get(ensg)
    if not exons:
        raise ValueError(f"no exons for {ensg} in {gencode_parquet}")
    return list(regions.gene_intervals(int(gene_row["start"]), int(gene_row["end"]), exons,
                                       ["exon"])["exon"])


def personal_exons(exons: Sequence[Tuple[int, int]], transformer) -> List[Tuple[int, int]]:
    """Exon intervals in one haplotype's frame; each boundary shifts independently, so an indel
    inside an exon resizes it (capture regions.to_personal)."""
    return [transformer.transform_interval(int(s), int(e)) for s, e in exons]


def build_donor_windows(gene_row: pd.Series, donor_ids: Sequence[str], vcf_file_path: str,
                        genome_file_path: str, contig_prefix: str = "chr",
                        window_size: int = BORZOI_WINDOW) -> List[DonorWindow]:
    """The two haplotypes of each donor over the gene's window, as the capture builds them.

    Same `create_genome_dataset` call as capture_forward.py (no MAF filter, i.e. every variant of
    the VCF; `genome_contig_prefix='chr'`, strings not one-hots), so sequences and transformers
    are the ones behind the cohort tables.
    """
    from embher.embedding.utils.borzoi_utils import dna_to_onehot
    from embher.embedding.utils.data import create_genome_dataset

    donor_ids = list(donor_ids)
    meta = pd.DataFrame([gene_row])
    dummy_expr = pd.DataFrame(0.0, index=meta["ensg"].values, columns=donor_ids)
    dataset = create_genome_dataset(
        metadata=meta, vcf_file_path=vcf_file_path, genome_file_path=genome_file_path,
        expression_data=dummy_expr, sample_list=donor_ids, window_size=window_size,
        contig_prefix=contig_prefix, genome_contig_prefix="chr", onehot_encode=False,
        return_idx=True)

    win_start = gene_window_start(gene_row["tss"], window_size)
    out = []
    for idx in range(len(dataset)):
        seq_arr, _, transformers, _, sample_idx = dataset[idx]
        mat, pat = seq_arr[1]
        win = DonorWindow(donor_ids[sample_idx], win_start, [mat, pat], list(transformers))
        win.onehots = [dna_to_onehot(s, window_size)[0].numpy().astype(np.float32)
                       for s in win.sequences]
        out.append(win)
    return out


def to_reference_frame(attr: np.ndarray, transformer, win_start: int) -> np.ndarray:
    """Map an attribution (C, L) from a haplotype's own frame onto reference coordinates.

    Index j of `attr` is personal coordinate win_start + j; reference position r sits at personal
    coordinate transformer.ref_to_personal(r). Reference positions whose personal coordinate falls
    outside the window get 0. With no indels this is the identity; across an indel a reference base
    inside a deletion repeats its neighbour's value (the haplotype has no base there), and inserted
    bases have no reference position and are dropped.
    """
    n = attr.shape[-1]
    ref = np.arange(win_start, win_start + n)
    j = np.fromiter((transformer.ref_to_personal(int(r)) for r in ref), dtype=np.int64,
                    count=n) - win_start
    ok = (j >= 0) & (j < n)
    out = np.zeros_like(attr)
    out[..., ok] = attr[..., j[ok]]
    return out
