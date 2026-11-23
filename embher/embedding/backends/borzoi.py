"""Borzoi / Flashzoi backend (needs embher[capture-borzoi])."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .base import EmbeddingBackend


class BorzoiBackend(EmbeddingBackend):
    """Borzoi (524,288 bp window). `model_name` is a HuggingFace name or a local weights folder
    (default: johahi/borzoi-replicate-0); pass a loaded `model` to skip loading. Layers are the
    names embher's capture uses, e.g. 'embeddings/final' (the default); layers='all' keeps every
    captured layer."""

    name = "borzoi"
    use_flashzoi = False

    def __init__(self, model_name: Optional[str] = None, device: str = "auto", model=None,
                 replicate: int = 0, default_layers: Optional[Sequence[str]] = None):
        from embher.embedding.capture import borzoi
        from embher.embedding.utils import BORZOI_SEQ_LEN, init_borzoi_model

        self._cap = borzoi
        self.window_bp = BORZOI_SEQ_LEN
        self.default_layers = tuple(default_layers or borzoi.FINAL_EMBEDDINGS)
        if model is None:
            model, _ = init_borzoi_model(model_name=model_name, device=device,
                                         use_flashzoi=self.use_flashzoi, replicate=replicate)
        self.model = model

    def embed(self, sequence: str, layers: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        from embher.embedding.capture.reduce import Pooling

        self.check_window(sequence)
        keep = None if layers == "all" else tuple(layers or self.default_layers)
        out = self._cap.run_forward(
            self.model, sequence, self.window_bp, 0, 0, self.window_bp, embedding_crop=False,
            poolings=[Pooling("seq_dim", "mean")], embedding_layers=keep)
        suffix = Pooling("seq_dim", "mean").suffix
        return {k[: -len(suffix)]: v[0].astype(np.float32) for k, v in out.items()
                if k.endswith(suffix)}


class FlashzoiBackend(BorzoiBackend):
    """Flashzoi: Borzoi with FlashAttention-2 (Ampere or newer GPU, flash-attn installed)."""

    name = "flashzoi"
    use_flashzoi = True
