"""AlphaGenome backend (needs the jax stack, embher[capture-alphagenome] + alphagenome_research)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .base import EmbeddingBackend


class AlphaGenomeBackend(EmbeddingBackend):
    """AlphaGenome from a local checkpoint (`weights_path`). `sequence_length` is one of
    capture.regions.SEQUENCE_LENGTHS ('2KB' ... '1MB', default '1MB'). Layers are
    'embeddings/1bp' and 'embeddings/128bp'. Not covered by the test suite (needs a GPU/TPU jax
    environment and the checkpoint)."""

    name = "alphagenome"

    def __init__(self, weights_path: str, sequence_length: str = "1MB", device: Optional[str] = None,
                 default_layers: Optional[Sequence[str]] = None, model=None):
        from embher.embedding.capture import alphagenome
        from embher.embedding.capture.regions import SEQUENCE_LENGTHS

        self._cap = alphagenome
        self.window_bp = SEQUENCE_LENGTHS[sequence_length]
        self.default_layers = tuple(default_layers or alphagenome.FINAL_EMBEDDINGS)
        self.model = model if model is not None else alphagenome.load_model(weights_path, device=device)

    def embed(self, sequence: str, layers: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        from embher.embedding.capture.reduce import OutputAggregation, Pooling

        self.check_window(sequence)
        keep = None if layers == "all" else tuple(layers or self.default_layers)
        pool = Pooling("seq_dim", "mean")
        out = self._cap.run_forward(
            self.model, sequence, self.window_bp, 0, 0, self.window_bp, embedding_crop=False,
            poolings=[pool], outputs=OutputAggregation(regions={}, funcs=()), output_heads=(),
            embedding_layers=keep)
        return {k[: -len(pool.suffix)]: np.asarray(v[0], dtype=np.float32) for k, v in out.items()
                if k.endswith(pool.suffix)}
