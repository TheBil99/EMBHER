"""The backend contract: a sequence-to-function model that turns a DNA window into embeddings."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


class EmbeddingBackend(ABC):
    """One model's embeddings, mean-pooled over sequence positions.

    A backend loads a model once and maps a DNA window of exactly `window_bp` bases
    (A/C/G/T/N, upper case) to {layer name: 1-D float32 vector}. Pooling over positions is what
    embher's heritability analyses use. Subclass this, set `name`, `window_bp` and
    `default_layers`, implement `embed`, and `register_backend` it to use it everywhere a
    backend name is accepted."""

    name: str = ""
    window_bp: int = 0
    default_layers: Sequence[str] = ()

    @abstractmethod
    def embed(self, sequence: str, layers: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Pooled embedding per layer for one window. `layers` defaults to `default_layers`."""

    def check_window(self, sequence: str) -> None:
        if len(sequence) != self.window_bp:
            raise ValueError(f"{self.name} reads windows of exactly {self.window_bp} bp, "
                             f"got {len(sequence)}")


def embed_sequences(backend: EmbeddingBackend, sequences: Sequence[str], ids: Optional[Sequence[str]] = None,
                    layer: Optional[str] = None, progress: bool = True) -> pd.DataFrame:
    """Embed many windows (one per donor, or averaged over haplotypes by the caller) into a
    donors x features DataFrame of one layer, ready for `embher-h2 --embeddings` or
    `embher.embedding.whiten.embedding_kernel`. `layer` defaults to the backend's first default
    layer."""
    layer = layer or backend.default_layers[0]
    it = sequences
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(sequences, desc=f"{backend.name} {layer}")
        except ImportError:
            pass
    rows = [backend.embed(s, [layer])[layer] for s in it]
    index = pd.Index(ids if ids is not None else range(len(rows)), name="IID").astype(str)
    return pd.DataFrame(np.vstack(rows), index=index)
