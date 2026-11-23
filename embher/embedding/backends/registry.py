"""Backend registry: names to lazily-imported factories, so a backend's heavy imports (torch, jax)
only happen when it is asked for."""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List

from .base import EmbeddingBackend

_FACTORIES: Dict[str, Callable[..., EmbeddingBackend]] = {}
_BUILTIN = {
    "borzoi": ("embher.embedding.backends.borzoi", "BorzoiBackend"),
    "flashzoi": ("embher.embedding.backends.borzoi", "FlashzoiBackend"),
    "enformer": ("embher.embedding.backends.enformer", "EnformerBackend"),
    "alphagenome": ("embher.embedding.backends.alphagenome", "AlphaGenomeBackend"),
}


def register_backend(name: str, factory: Callable[..., EmbeddingBackend]) -> None:
    """Make `get_backend(name, **kwargs)` call `factory(**kwargs)`."""
    _FACTORIES[name] = factory


def available_backends() -> List[str]:
    return sorted(set(_BUILTIN) | set(_FACTORIES))


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    """Instantiate a backend by name: 'borzoi', 'flashzoi', 'enformer', 'alphagenome', or any
    name added with `register_backend`. kwargs go to the backend's constructor."""
    if name in _FACTORIES:
        return _FACTORIES[name](**kwargs)
    if name in _BUILTIN:
        module, cls = _BUILTIN[name]
        return getattr(importlib.import_module(module), cls)(**kwargs)
    raise KeyError(f"unknown backend {name!r}; available: {available_backends()}")
