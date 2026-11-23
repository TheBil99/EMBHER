"""Enformer backend (needs torch and enformer-pytorch: embher[capture-enformer])."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .base import EmbeddingBackend

ENFORMER_WINDOW_BP = 196_608


class EnformerBackend(EmbeddingBackend):
    """Enformer (196,608 bp window). One layer, 'embeddings/final': the trunk output that feeds
    the heads, mean-pooled over its 896 bins (3072 features). `model_name` is a HuggingFace name
    for enformer_pytorch.from_pretrained (default EleutherAI/enformer-official-rough); pass a
    loaded `model` to skip loading. bf16=True runs the forward pass under bfloat16 autocast on GPU."""

    name = "enformer"
    window_bp = ENFORMER_WINDOW_BP
    default_layers = ("embeddings/final",)

    def __init__(self, model_name: str = "EleutherAI/enformer-official-rough", device: str = "auto",
                 model=None, bf16: bool = True):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            from enformer_pytorch import from_pretrained
            model = from_pretrained(model_name)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.bf16 = bf16 and self.device.type == "cuda"

    def embed(self, sequence: str, layers: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        import torch
        from enformer_pytorch import str_to_one_hot

        self.check_window(sequence)
        wanted = set(self.default_layers if layers in (None, "all") else layers)
        if not wanted <= set(self.default_layers):
            raise ValueError(f"enformer layers are {list(self.default_layers)}, got {sorted(wanted)}")
        x = str_to_one_hot(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            emb = self.model(x, return_only_embeddings=True)          # (1, bins, channels)
        return {"embeddings/final": emb[0].float().mean(dim=0).cpu().numpy().astype(np.float32)}
