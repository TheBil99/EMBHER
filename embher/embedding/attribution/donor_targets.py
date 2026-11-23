"""
Differentiable scalar targets for a donor's two haplotypes, taken jointly.

Both targets map the two haplotype windows (x1, x2), each (1, 4, L), to one scalar (1, 1), so one
Captum call attributes the scalar to BOTH sequences at once (the inputs are passed as a tuple).

  DonorExpressionTarget  the zero-shot expression prediction for a donor: per haplotype, the head's predictions at
                         the chosen tracks are un-squashed per track and summed over the gene's exon
                         bins with fractional-overlap weights, the exons being moved into that
                         haplotype's own coordinate frame; the two haplotypes are SUMMED per track
                         and the tracks averaged. Linear scale, no log (unlike ExpressionTarget, which
                         uses any-overlap bins and a log2).
  DonorEmbeddingTarget   the donor's pooled embedding (summed over haplotypes, as in the cohort
                         table) reduced to a number by a readout (e.g. a linear direction).

The trunk runs under bf16 autocast and the head in fp32, exactly as Borzoi.forward does (which
the capture calls), so values reproduce the stored cohort values; checked per haplotype and track
against the capture's own run_forward.
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from embher.embedding.attribution.borzoi_forward import pool, run_to_layer
from embher.embedding.attribution.donors import N_OUT_BINS, BIN_BP, output_start
from embher.embedding.attribution.targets import _Target, undo_squash
from embher.embedding.capture.reduce import OutputAggregation, build_requests


def exon_bin_weights(personal_exons: Sequence[Tuple[int, int]], win_start: int
                     ) -> Tuple[int, int, np.ndarray]:
    """(bin_start, bin_end, weights): the fraction of each output bin covered by the haplotype's
    exons, on Borzoi's 6144 x 32 bp grid (capture reduce.build_requests, 'exon' crop)."""
    agg = OutputAggregation(regions={"exon": list(personal_exons)}, funcs=("sum",),
                            track_index=None, unsquash=True)
    (request,) = build_requests(agg, N_OUT_BINS, BIN_BP, output_start(win_start))
    return request.bin_start, request.bin_end, request.weights.astype(np.float32)


class DonorExpressionTarget(_Target):
    """Zero-shot expression of a donor (see module docstring); forward(x1, x2) -> (1, 1)."""

    def __init__(self, trunk, tracks: Sequence[int], scale: np.ndarray, clip_soft: float,
                 haplotype_bins: Sequence[Tuple[int, int, np.ndarray]],
                 precision: str = "bf16", use_checkpoint: bool = False):
        super().__init__(trunk, precision, use_checkpoint)
        self.register_buffer("tracks", torch.as_tensor(list(tracks), dtype=torch.long))
        self.register_buffer("scale", torch.as_tensor(np.array(scale), dtype=torch.float32))
        self.clip_soft = float(clip_soft)
        self.slices: List[Tuple[int, int]] = []
        for i, (b0, b1, w) in enumerate(haplotype_bins):
            self.slices.append((int(b0), int(b1)))
            self.register_buffer(f"w{i}", torch.as_tensor(np.array(w), dtype=torch.float32))

    def haplotype_values(self, *xs: torch.Tensor) -> torch.Tensor:
        """Per-haplotype exon sums of the un-squashed prediction, (n_haplotypes, n_tracks)."""
        rows = []
        for i, x in enumerate(xs):
            b0, b1 = self.slices[i]
            with self._autocast(x):
                joined = run_to_layer(self.trunk, x, "joined", self.use_checkpoint)
            # head in fp32, as Borzoi.forward does ("disable autocast for more precision in final
            # layer"): a bf16 head rounds the large pre-activations of high-coverage tracks and the
            # soft-clip un-squash amplifies that into a systematic few-percent bias
            with torch.amp.autocast("cuda", enabled=False):
                pred = self.trunk.final_softplus(self.trunk.human_head(joined.float()))
            sl = pred.index_select(1, self.tracks)[:, :, b0:b1]               # (1, T, n_bins)
            lin = undo_squash(sl, self.scale, self.clip_soft)
            rows.append((lin * getattr(self, f"w{i}")).sum(-1))                # (1, T)
        return torch.cat(rows, dim=0)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        # haplotypes summed per track, then tracks averaged
        return self.haplotype_values(*xs).sum(0).mean().reshape(1, 1)


class DonorEmbeddingTarget(_Target):
    """Pooled embedding of `layer` summed over the donor's haplotypes, through `readout`."""

    def __init__(self, trunk, readout: nn.Module, layer: str = "joined", axis: str = "pos",
                 precision: str = "bf16", use_checkpoint: bool = False):
        super().__init__(trunk, precision, use_checkpoint)
        self.readout = readout
        self.layer = layer
        self.axis = axis

    def embedding(self, *xs: torch.Tensor) -> torch.Tensor:
        """(1, d): the donor's embedding, as the cohort table stores it."""
        total = 0
        for x in xs:
            with self._autocast(x):
                t = run_to_layer(self.trunk, x, self.layer, self.use_checkpoint)
            total = total + pool(t, self.axis)
        return total

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.readout(self.embedding(*xs)).unsqueeze(-1)
