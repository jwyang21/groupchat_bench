"""OpenAI Embedding API wrapper.

Default model: `text-embedding-3-small`. Uses langchain's `OpenAIEmbeddings`.
"""
from __future__ import annotations

import numpy as np

from .base import Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI embedding model wrapper.

    Defaults to `text-embedding-3-small`. dim is inferred from the model:
      - text-embedding-3-small : 1536
      - text-embedding-3-large : 3072
      - text-embedding-ada-002 : 1536
    """

    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        # If OPENAI_API_KEY is unset, the client is created silently and fails with an opaque 401 on
        # the first embed call, aborting mid-ablation. Validate up front to fail fast.
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. Put the key in a .env file or "
                "export it, then retry. (Check that load_project_env() was called.)"
            )
        self.model = model
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=model)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    @property
    def dim(self) -> int:
        return self._DIM_MAP.get(self.model, 1536)

    def embed(self, texts: list[str]) -> np.ndarray:
        # empty input → explicit (0,D) (keeps the (N,D) contract, avoids an unnecessary API call).
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._client.embed_documents(texts)
        return np.asarray(vecs, dtype=np.float32)
