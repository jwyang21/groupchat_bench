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
        # If the key is unset, the client is created silently and fails with an
        # opaque 401 on the first embed call, aborting mid-ablation. Validate up
        # front to fail fast.
        # OpenAI-direct vs AI-gateway toggle (GROUPCHAT_API_MODE): in letsur mode
        # route to the gateway (base_url = gw.letsur.ai/v1 + LETSUR_API_KEY); the
        # embedding model id (text-embedding-3-small) is the same code.
        import os
        from ..client.api_mode import is_letsur, resolve_openai_credentials
        if is_letsur():
            if not os.environ.get("LETSUR_API_KEY"):
                raise RuntimeError(
                    "LETSUR_API_KEY environment variable is not set "
                    "(GROUPCHAT_API_MODE=letsur). Put the key in a .env file "
                    "(LETSUR_API_KEY=...) or export it, then retry."
                )
        elif not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. Put the "
                "key in a .env file or export it, then retry. (Check that "
                "load_project_env() was called.)"
            )
        self.model = model
        # base_url=None (OpenAI default) input -> openai mode keeps
        # (None, OPENAI_API_KEY) unchanged; letsur mode substitutes
        # (gw.letsur.ai/v1, LETSUR_API_KEY).
        base_url, api_key = resolve_openai_credentials(
            None, os.environ.get("OPENAI_API_KEY")
        )
        from langchain_openai import OpenAIEmbeddings

        _emb_kwargs: dict = {}
        if base_url:
            _emb_kwargs["base_url"] = base_url
        if api_key:
            _emb_kwargs["api_key"] = api_key
        self._client = OpenAIEmbeddings(model=model, **_emb_kwargs)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    @property
    def dim(self) -> int:
        return self._DIM_MAP.get(self.model, 1536)

    def embed(self, texts: list[str]) -> np.ndarray:
        # empty input -> explicit (0,D) (keeps the (N,D) contract, avoids an
        # unnecessary API call).
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._client.embed_documents(texts)
        return np.asarray(vecs, dtype=np.float32)
