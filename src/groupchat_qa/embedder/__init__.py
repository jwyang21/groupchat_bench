"""Embedder — text -> vector."""
from .base import Embedder
from .cache import EmbeddingCache
from .cached_embedder import CachedEmbedder
from .factory import make_embedder

__all__ = ["Embedder", "EmbeddingCache", "CachedEmbedder", "make_embedder"]
