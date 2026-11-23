"""Model backends: one interface over Borzoi, Flashzoi, Enformer and AlphaGenome embeddings.

    from embher.embedding.backends import get_backend, embed_sequences
    backend = get_backend("enformer")                 # or "borzoi", "flashzoi", "alphagenome"
    Z = embed_sequences(backend, sequences, ids=donors)   # donors x features, one layer

Importing this package imports no model library; each backend imports its own on construction.
"""
from .base import EmbeddingBackend, embed_sequences
from .registry import available_backends, get_backend, register_backend

__all__ = ["EmbeddingBackend", "embed_sequences", "available_backends", "get_backend", "register_backend"]
