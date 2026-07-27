"""Embedder ABC — text -> vector (unified provider interface)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Provider-agnostic embedding interface.

    Implementations:
      - `OpenAIEmbedder` — text-embedding-3-small, etc.
      - `SentenceTransformerEmbedder` — local ST models

    Design: for a fair baseline comparison, every method must be injected with the *same embedder
    instance* — created only once based on `embedder.name` in config.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier (used in cache keys, etc.). e.g. 'openai:text-embedding-3-small'."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """list[str] -> (N, D) numpy array."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience: single text -> (D,) vector."""
        vecs = self.embed([text])
        # If a provider degenerately returns an empty result, surface a meaningful error instead of
        #   an [0] IndexError.
        if len(vecs) == 0:
            raise RuntimeError(f"embedder returned no vector for text: {text[:80]!r}")
        return vecs[0]
