"""CachedEmbedder — a transparent wrapper around a base Embedder using a file-backed EmbeddingCache.

Purpose: whenever a baseline (a_mem/hipporag2/mem0) calls `self.embedder.embed` / `embed_one`:
  (a) first check whether the vector for that text exists in the local cache file,
  (b) if it does, use it (hit),
  (c) if not (or the cache file itself does not exist), live-embed → add to the cache + save
      immediately (autoflush) → the in-memory dict is updated so later sessions/calls reuse it.
The same text is embedded once and reused regardless of LLM/session/run (text-keyed).

The baseline code is not changed at all — the runner injects the base embedder wrapped by this
wrapper, so `embed`/`embed_one` automatically go through the cache. tkg_hybrid uses its own quad
cache separately, so this wrapper applies only to baselines enabled via config (`embedder.cache: true`).

Concurrency: even when multiple LLM threads share one CachedEmbedder instance, EmbeddingCache's
per-path RLock + fcntl.flock keeps it safe (same path → same lock). Therefore the runner must wrap
**once** and pass the same instance to all threads, so the in-memory dict is shared and duplicate
embedding is avoided.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Embedder
from .cache import EmbeddingCache


class CachedEmbedder(Embedder):
    """base Embedder + EmbeddingCache wrapper. Same Embedder interface (name/dim/embed/embed_one).

    Args:
        base: the Embedder that performs the actual embedding (OpenAIEmbedder / SentenceTransformerEmbedder).
        cache_path: cache file path (.pickle recommended). The parent directory is created on first flush.
        autoflush_every: disk-flush every N misses in embed_one.
            Default 64. A default of 1 would rewrite the *entire dict* as pickle on every miss →
            O(N^2) disk I/O + lock held during flush blocking all threads → slower for larger caches
            (especially phrase/fact embeddings in hippo/mem0). Intra-session reuse is served immediately
            by the in-memory dict (set updates _data right away), so the flush period affects only *file
            persistence*. The runner calls flush() on exit (try/finally), so a normal shutdown loses
            nothing. On a crash, at most N-1 are re-embedded (no correctness loss).
        fsync_on_flush: whether to fsync on flush. False recommended for the runtime hot path (reduces latency).
        flush_every_n_batches: flush period for embed(batch) (passed to EmbeddingCache). Exposed so the
            caller can raise it during large-batch precompute, where 1 would be O(N^2).
    """

    def __init__(
        self,
        base: Embedder,
        cache_path: str | Path,
        autoflush_every: int = 64,
        fsync_on_flush: bool = False,
        flush_every_n_batches: int = 10,
    ) -> None:
        self._base = base
        # EmbeddingCache validates the cache file by (model_name, dim) → never mixes with a cache of
        # a different model/dim (mismatch → readonly, preserving the old file).
        self._cache = EmbeddingCache(
            path=cache_path,
            model_name=base.name,
            dim=int(base.dim),
            autoflush_every=autoflush_every,
            flush_every_n_batches=flush_every_n_batches,
            fsync_on_flush=fsync_on_flush,
        )

    @property
    def name(self) -> str:
        return self._base.name

    @property
    def dim(self) -> int:
        return self._base.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """list[str] → (N, D). Only cache misses call base.embed() + store the result in the cache."""
        return self._cache.embed_with_cache(self._base, list(texts))

    def embed_one(self, text: str) -> np.ndarray:
        """single text → (D,). hit O(1), miss → live embed + autoflush (save)."""
        return self._cache.embed_one_cached(self._base, text)

    def flush(self) -> None:
        """Immediately save pending cache to disk (explicit flush — recommended at runner shutdown)."""
        self._cache.flush()
