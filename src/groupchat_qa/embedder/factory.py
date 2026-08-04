"""Embedder factory - creates an instance from the backend name in config."""
from __future__ import annotations

from typing import Any

from .base import Embedder


def make_embedder(backend: str, model: str, **kwargs: Any) -> Embedder:
    """backend name -> Embedder instance.

    backend:
      - "openai" : OpenAI embedding API
      - "st"     : sentence-transformers (local)
    """
    backend = backend.lower()
    if backend in ("openai", "openai-embedding"):
        from .openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model=model, **kwargs)
    if backend in ("st", "sentence-transformer", "sentence_transformer"):
        from .sentence_transformer_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model=model, **kwargs)
    raise ValueError(f"Unknown embedder backend: {backend!r}. Expected: openai / st.")
