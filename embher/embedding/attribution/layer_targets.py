"""
Population-free scalar targets on the captured checkpoints, for motif analysis of a reference window.

`CaptureLayerNormTarget` is the L2 norm of a checkpoint's position-pooled activation (the same
pooling capture_forward.py stores as `<layer>__seq_mean`). Its gradient follows the activation's own
direction, so the attribution says which sequence drives the layer, with no cohort involved.
"""

import torch

from embher.embedding.attribution.borzoi_forward import pool, run_to_capture
from embher.embedding.attribution.targets import _Target


class CaptureLayerNormTarget(_Target):
    def __init__(self, trunk, layer: str, precision: str = "bf16", use_checkpoint: bool = False):
        super().__init__(trunk, precision, use_checkpoint)
        self.layer = layer

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C) position-pooled activation of the layer."""
        with self._autocast(x):
            t = run_to_capture(self.trunk, x, self.layer, self.use_checkpoint)
        return pool(t, "pos")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.embedding(x), dim=-1).unsqueeze(-1)
