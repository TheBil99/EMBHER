"""
Differentiable scalar targets for Borzoi attribution.

Every target is an nn.Module mapping a one-hot window (B, 4, L) to (B, 1), so Captum,
gReLU's `get_attributions` and Decima's attributer can all drive it unchanged.

Scalars follow the recipes used when embeddings and predictions are extracted, so that an
attribution explains the same number a downstream analysis uses:

  ExpressionTarget   scalar_prediction: head tracks -> unsquash each track with its own
                     `scale` -> sum over the gene's exon bins -> log2(x + 1) per track ->
                     mean over tracks. Haplotypes are averaged on the linear scale; for a
                     reference window both haplotypes are identical, so that is a no-op.
  EmbeddingTarget    pooled embedding of a layer (`joined:pos` by default) -> readout.
                     A donor's embedding is the SUM over its two haplotypes, so the
                     reference window (a homozygous-reference donor) is `n_haplotypes` x
                     the single-sequence embedding; `n_haplotypes=2` reproduces that and
                     is part of what makes the readouts line up with donor-space
                     quantities (BLUP directions, cohort means).
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from embher.embedding.attribution.borzoi_forward import pool, run_to_layer

BIN_SIZE = 32  # Borzoi output resolution (bp)
TRACK_TRANSFORM = 3 / 4  # exponent of Borzoi's coverage squashing


# --------------------------------------------------------------------------- #
# track / exon bookkeeping
# --------------------------------------------------------------------------- #
def default_targets_path() -> str:
    """targets.txt shipped with borzoi_pytorch (track scale / clip_soft / sum_stat)."""
    import borzoi_pytorch

    return str(Path(borzoi_pytorch.__file__).parent / "precomputed" / "targets.txt")


def read_track_params(tracks: Sequence[int], targets_path: Optional[str] = None
                      ) -> Tuple[np.ndarray, float]:
    """(per-track scale, shared clip_soft) read straight off targets.txt.

    Raises if clip_soft or sum_stat are non-uniform across `tracks`: the unsquash below
    assumes one clip_soft, and a mixed sum_stat would mean the tracks are not comparable.
    Scale alone may vary per track (RNA tracks range ~0.01-2.0).
    """
    table = pd.read_csv(targets_path or default_targets_path(), sep="\t", index_col=0)
    rows = table.loc[list(tracks)]
    if rows["clip_soft"].nunique() != 1:
        raise ValueError(f"non-uniform clip_soft among tracks {list(tracks)}")
    if rows["sum_stat"].nunique() != 1 or rows["sum_stat"].iloc[0] != "sum_sqrt":
        raise ValueError(f"tracks {list(tracks)} do not all use sum_stat=sum_sqrt")
    return rows["scale"].to_numpy(dtype=np.float64), float(rows["clip_soft"].iloc[0])


def exon_bin_indices(gtf_path: str, gene: str, out_start: int, n_bins: int) -> np.ndarray:
    """Output-bin indices overlapping any exon of `gene` (union across transcripts).

    `out_start` is the 0-based genomic start of the first output bin. Same mapping as
    the extraction pipeline's `exon_bins` (any overlap counts, no fractional weights), which is
    what ExpressionTarget must reproduce.
    """
    from borzoi_pytorch.gene_utils import Transcriptome

    tx = Transcriptome(gtf_path, use_geneid=gene.startswith("ENSG"))
    want = gene.split(".")[0]
    keys = [k for k in tx.genes if str(k).split(".")[0] == want]
    if not keys:
        raise ValueError(f"{gene} not found in {gtf_path}")
    sl = tx.genes[keys[0]].output_slice(out_start + 1, n_bins * BIN_SIZE, BIN_SIZE,
                                        majority_overlap=False)
    if len(sl) == 0:
        raise ValueError(f"{gene} has no exon overlapping the predicted interval")
    return np.asarray(sl, dtype=np.int64)


def undo_squash(x: torch.Tensor, scale: torch.Tensor, clip_soft: float,
                transform: float = TRACK_TRANSFORM) -> torch.Tensor:
    """Invert Borzoi's squashed coverage scaling per track; differentiable.

    x: (B, n_tracks, n_bins); scale: (n_tracks,). Same values as the extraction
    pipeline's `undo_squashed_scale_per_track` (which uses an in-place masked assignment), written
    with torch.where so the gradient flows through both branches cleanly.
    """
    x = x / scale.view(1, -1, 1)
    x = torch.where(x > clip_soft, (x - clip_soft) ** 2 + clip_soft, x)
    return x.clamp(min=0.0) ** (1.0 / transform)


# --------------------------------------------------------------------------- #
# readouts: pooled embedding (B, d) -> (B,)
# --------------------------------------------------------------------------- #
class LinearReadout(nn.Module):
    """s = (e - mean) . direction. With a BLUP/PC direction this is the cohort's own linear
    map from embedding to predicted genetic value / PC score, evaluated at one sequence."""

    def __init__(self, direction: np.ndarray, mean: Optional[np.ndarray] = None):
        super().__init__()
        d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        self.register_buffer("direction", d)
        m = torch.zeros_like(d) if mean is None else torch.as_tensor(
            np.asarray(mean), dtype=torch.float32)
        self.register_buffer("mean", m)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return (e - self.mean) @ self.direction


class NormReadout(nn.Module):
    """s = ||e||_2 : cohort-free; its gradient weights each channel by its own activation."""

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(e, dim=-1)


class MSEReadout(nn.Module):
    """s = mean_c (e_c - e_ref_c)^2. Zero, with zero gradient, at the reference itself, so
    it is only informative for path methods (integrated gradients from a baseline) or for
    non-reference inputs; the attributer refuses the degenerate combination."""

    def __init__(self, reference_embedding: np.ndarray):
        super().__init__()
        self.register_buffer("reference", torch.as_tensor(
            np.asarray(reference_embedding), dtype=torch.float32))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return ((e - self.reference) ** 2).mean(dim=-1)


# --------------------------------------------------------------------------- #
# targets: one-hot (B, 4, L) -> (B, 1)
# --------------------------------------------------------------------------- #
class _Target(nn.Module):
    def __init__(self, trunk, precision: str = "bf16", use_checkpoint: bool = False):
        super().__init__()
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be 'bf16' or 'fp32'")
        self.trunk = trunk
        self.precision = precision
        self.use_checkpoint = use_checkpoint
        for p in self.trunk.parameters():
            p.requires_grad = False

    def _autocast(self, x: torch.Tensor):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=(self.precision == "bf16" and x.is_cuda))


class ExpressionTarget(_Target):
    """Combined multi-track expression prediction for one gene (see module docstring)."""

    def __init__(self, trunk, tracks: Sequence[int], bins: np.ndarray, scale: np.ndarray,
                 clip_soft: float, precision: str = "bf16", use_checkpoint: bool = False):
        super().__init__(trunk, precision, use_checkpoint)
        self.register_buffer("tracks", torch.as_tensor(list(tracks), dtype=torch.long))
        self.register_buffer("bins", torch.as_tensor(np.array(bins), dtype=torch.long))
        self.register_buffer("scale", torch.as_tensor(np.array(scale), dtype=torch.float32))
        self.clip_soft = float(clip_soft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._autocast(x):
            joined = run_to_layer(self.trunk, x, "joined", self.use_checkpoint)
        # fp32 head, as in the extraction script, then slice tracks and exon bins
        with torch.amp.autocast("cuda", enabled=False):
            head = self.trunk.final_softplus(self.trunk.human_head(joined.float()))
        sl = head.index_select(1, self.tracks).index_select(-1, self.bins)
        lin = undo_squash(sl, self.scale, self.clip_soft)        # (B, n_tracks, n_exon_bins)
        per_track = torch.log2(lin.sum(-1).clamp(min=0.0) + 1.0)  # (B, n_tracks)
        return per_track.mean(dim=1, keepdim=True)


class EmbeddingTarget(_Target):
    """Pooled embedding of `layer` reduced to one number by `readout`."""

    def __init__(self, trunk, readout: nn.Module, layer: str = "joined", axis: str = "pos",
                 n_haplotypes: int = 2, precision: str = "bf16",
                 use_checkpoint: bool = False):
        super().__init__(trunk, precision, use_checkpoint)
        self.readout = readout
        self.layer = layer
        self.axis = axis
        self.n_haplotypes = int(n_haplotypes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """The pooled embedding as a donor would have it (summed over haplotypes), (B, d)."""
        with self._autocast(x):
            t = run_to_layer(self.trunk, x, self.layer, self.use_checkpoint)
        return self.n_haplotypes * pool(t, self.axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.embedding(x)).unsqueeze(-1)
