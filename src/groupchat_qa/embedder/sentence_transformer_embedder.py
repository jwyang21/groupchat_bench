"""sentence-transformers based local embedder.

A fallback backend. Runs on a local GPU with no API cost.
"""
from __future__ import annotations

import numpy as np

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model
        self._device = device
        from sentence_transformers import SentenceTransformer

        full_name = model if "/" in model else f"sentence-transformers/{model}"
        self._model = SentenceTransformer(full_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        # empty input -> encode may return shape (0,), violating the (N,D)
        # contract -> explicit (0,D).
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)
