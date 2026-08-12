"""EmbeddingCache - a file-backed key->vec dict that cumulatively reduces
OpenAI API calls.

Two uses:
  1) precompute: run once so that all embeddings (entire/partial dialogue +
     question + record) are batch-embedded and stored.
  2) experiment runtime (`memory/groupchat_mem.py`, `CachedEmbedder`):
     cache lookup -> on miss, live embed -> add the result to the cache +
     flush immediately.
     From the next cell/session within the same run, it is a cache hit.

Pickle security: these cache files are **produced/consumed only by this
module** (trusted source). Never load pickles from an external source -
pickle.load can execute arbitrary code. When the cache path is user-supplied,
validate that it lives inside the project (e.g. data/) before use.

File format (incremental/delta approach - avoids saturating shared storage):

A naive flush() that rewrites the entire _data (hundreds of MB) atomically on
every miss does not scale: when multiple processes share one cache and flush
on every miss, this causes (a) hundreds of MB x frequent flushes = a burst of
writes, and (b) one process's atomic rename turns another process's open read
fd into ESTALE on network filesystems, so reads contend indefinitely and the
whole run starves on I/O.

-> Replaced with an **incremental (delta) approach**:
  - **base file** `{path}` : the full dict pickle (written once by
        build_embeddings or compact).
        runtime flush **never touches it** (read-only baseline).
  - **delta file** `{path}.delta.{pid}_{uuid}.pkl` : **only items added since
        the last flush**.
        Each flush writes a new file with a **unique name** -> two processes
        never touch the same file -> no rename-over-open conflict/ESTALE, write
        size = only new items (small).
  - **load** : merge base + all `{path}.delta.*.pkl` (later values win - the
        same text has the same embedding, so it is idempotent).
  - **compact()** : merge base + all deltas -> rewrite base (atomic) + delete
        the merged deltas.
        Under an exclusive lock (`{path}.compact.lock`), **only one process**
        runs it. When deltas exceed a threshold, load attempts it automatically
        (try-lock; skip if held) - replacing N processes each rewriting the
        whole file with **a single compaction**.

Format of each file (base/delta):
    {
      "__meta__": {"model_name": "openai:text-embedding-3-small", "dim": 1536,
                   "version": 2, "created_at": "...", "kind": "base"|"delta"},
      "<text>": np.ndarray(dim, dtype=float32),
      ...
    }

Invariants:
  - **unique-name delta + atomic rename**: no partial write/conflict/ESTALE.
  - **base is immutable to runtime flush** - concurrent reads are safe.
  - **fcntl.flock**: best-effort on network filesystems. Correctness is
    guaranteed by unique file names + atomic rename (not dependent on the
    lock).
  - **threading.RLock**: safe across multiple threads of the same process.
  - **(model_name, dim) validation**: if a file's metadata differs from the
    current embedder, that file is skipped / writing is blocked.
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# read base embeddings local-first (avoid re-reading from shared storage)
from groupchat_qa.utils.filesys_utils import local_first
from typing import Any, Iterable

import numpy as np

# module-level threading lock (prevents in-process thread conflicts)
# One RLock is shared per cache file path - even if the same path is created
# via multiple instances,
# it is the same lock object. Multiple threads in ablation updating the same
# cache file concurrently
# do not corrupt the in-memory dict.
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()

# When the number of delta files exceeds this, load attempts an automatic
# compact (try-lock, skip if held).
# Too small -> frequent compaction increases base rewrites; too large -> delta
# glob/load increases on load.
_DEFAULT_COMPACT_THRESHOLD = 64


def _get_path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lk = _PATH_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _PATH_LOCKS[key] = lk
        return lk


class EmbeddingCache:
    """text->vec dict, file-backed, supports **incremental (delta) update +
    flush**.

    Args:
        path: base cache file path (`.pickle` extension recommended). deltas
            are `{path}.delta.*.pkl`.
        model_name: embedder.name. Must match the file metadata to load. On
            mismatch, that file is skipped and this instance blocks writes
            (readonly) - permanently protecting old data.
        dim: vector dimension (cache validation + sanity check).
        autoflush_every: auto-flush (= write 1 delta) after N sets. 0 to
            disable. runtime 1, build 100 recommended. With the delta approach,
            flush cost = only new items.
        flush_every_n_batches: batch-flush period for embed_with_cache.
        fsync_on_flush: whether to call os.fsync on flush.
        compact_threshold: if the number of delta files exceeds this, load
            attempts an automatic compact.
    """

    def __init__(
        self,
        path: str | Path,
        model_name: str,
        dim: int,
        autoflush_every: int = 1,
        flush_every_n_batches: int = 1,
        fsync_on_flush: bool = True,
        compact_threshold: int = _DEFAULT_COMPACT_THRESHOLD,
    ) -> None:
        self.path = Path(path)
        self.model_name = str(model_name)
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        self.autoflush_every = int(autoflush_every)
        self.flush_every_n_batches = max(1, int(flush_every_n_batches))
        self.fsync_on_flush = bool(fsync_on_flush)
        self.compact_threshold = max(1, int(compact_threshold))
        self._lock = _get_path_lock(self.path)
        self._pending = 0
        self._data: dict[str, np.ndarray] = {}
        # _dirty: only items added since the last flush (the next flush writes
        # these as a delta).
        self._dirty: dict[str, np.ndarray] = {}
        self._meta: dict[str, Any] = {
            "model_name": self.model_name,
            "dim": self.dim,
            "version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # flag to block writes on metadata mismatch (permanently preserves the
        # old cache).
        self._readonly = False
        self._load_if_exists()

    # path helpers
    def _delta_glob_pat(self) -> str:
        # base file name followed by `.delta.*.pkl`. (never overlaps the base
        # itself.)
        return self.path.name + ".delta.*.pkl"

    def _delta_paths(self) -> list[Path]:
        parent = self.path.parent
        if not parent.exists():
            return []
        try:
            return sorted(parent.glob(self._delta_glob_pat()))
        except OSError:
            return []

    def _new_delta_path(self) -> Path:
        # unique name: pid + uuid4 -> two processes/flushes never collide.
        token = f"{os.getpid()}_{uuid.uuid4().hex}"
        return self.path.with_name(f"{self.path.name}.delta.{token}.pkl")

    def _compact_lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.compact.lock")

    # I/O - load
    def _validate_meta(self, meta: dict, src: str) -> bool:
        """Whether the file meta's model_name/dim matches this instance. False
        (skip) on mismatch."""
        try:
            cached_dim = int(meta.get("dim") or -1)
        except (TypeError, ValueError):
            cached_dim = -1
        if meta.get("model_name") != self.model_name or cached_dim != self.dim:
            logging.warning(
                f"[EmbeddingCache] {src}: metadata mismatch "
                f"(file: model={meta.get('model_name')!r}"
                f"/dim={meta.get('dim')}, "
                f"requested: model={self.model_name!r}/dim={self.dim})"
                f" - skipping this file."
            )
            return False
        return True

    def _load_one_file(self, p: Path, *, is_base: bool) -> int:
        """Merge one pickle file (base or delta) into _data. Returns the number
        of accepted entries.

        Missing/corrupt/mismatch -> warn and return 0 (does not kill the whole
        load). If is_base=True and there is a mismatch, set readonly (preserve
        the old base).
        """
        if not p.exists():
            return 0
        try:
            with p.open("rb") as f:
                self._flock(f.fileno(), exclusive=False)
                obj = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
                ImportError, ModuleNotFoundError, TypeError, ValueError) as e:
            logging.warning(
                f"[EmbeddingCache] load failed ({p.name}): "
                f"{type(e).__name__}: {e}. "
                f"{'base->readonly. ' if is_base else ''}skipping this file."
            )
            if is_base:
                self._readonly = True
            return 0
        if not isinstance(obj, dict):
            logging.warning(f"[EmbeddingCache] {p.name}: not a dict - skip.")
            if is_base:
                self._readonly = True
            return 0
        meta = obj.get("__meta__") or {}
        if not self._validate_meta(meta, p.name):
            if is_base:
                # base is a different model/dim -> block writes (permanently
                # preserve the old base).
                self._readonly = True
            return 0
        if is_base:
            # preserve the base's old meta (created_at, etc.).
            for k, v in meta.items():
                if k not in ("model_name", "dim"):
                    self._meta[k] = v
        loaded = 0
        for k, v in obj.items():
            if k == "__meta__":
                continue
            if isinstance(v, np.ndarray) and v.shape == (self.dim,):
                self._data[k] = v.astype(np.float32, copy=False)
                loaded += 1
        return loaded

    def _load_if_exists(self) -> None:
        """Merge base + all deltas into _data. If there are many deltas,
        attempt an automatic compact."""
        with self._lock:
            base_n = self._load_one_file(
                Path(local_first(self.path)), is_base=True
            )
            deltas = self._delta_paths()
            delta_n = 0
            for dp in deltas:
                delta_n += self._load_one_file(dp, is_base=False)
            logging.info(
                f"[EmbeddingCache] {self.path.name}: loaded base={base_n} "
                f"+ delta={delta_n} ({len(deltas)} files) "
                f"-> {len(self._data)} total."
            )
        # if there are many deltas, auto-compact (one process only; skip if
        # readonly).
        if not self._readonly and len(deltas) > self.compact_threshold:
            try:
                self.compact(_auto=True)
            # a compaction failure must not kill the load.
            except Exception as e:
                logging.warning(
                    f"[EmbeddingCache] auto-compact failed (ignored): {e}"
                )

    @staticmethod
    def _flock(fd: int, *, exclusive: bool, nonblock: bool = False) -> bool:
        """fcntl.flock - assumes POSIX. Returns whether the lock was acquired
        (meaningful with nonblock).

        On NFS, flock is best-effort. Correctness is guaranteed by unique file
        names + atomic rename, so data is safe even if the lock fails. With
        nonblock=True (compaction try-lock), returns False if not acquired.
        """
        try:
            import fcntl  # POSIX only
        except ImportError:
            return True  # Windows - safe via threading.RLock + atomic rename.
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblock:
            flag |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flag)
            return True
        except OSError as e:
            if nonblock:
                # another process holds it - normal (compaction skip).
                return False
            logging.debug(
                f"[EmbeddingCache] fcntl.flock failed (best-effort): {e}"
            )
            return True  # the blocking path proceeds best-effort.

    # I/O - flush (write delta) / compact
    def _atomic_write_pickle(self, obj: dict, dest: Path) -> None:
        """Write obj to dest atomically (mkstemp + os.replace). dest must be a
        unique name to be safe."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: str | None = None
        tmp_fd: int | None = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent)
            )
            try:
                f = os.fdopen(tmp_fd, "wb")
            except Exception:
                os.close(tmp_fd)
                raise
            tmp_fd = None  # ownership -> f
            try:
                pickle.dump(obj, f, protocol=5)
                f.flush()
                if self.fsync_on_flush:
                    os.fsync(f.fileno())
            finally:
                f.close()
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass

    def flush(self) -> None:
        """**Incremental flush** - write only new items (_dirty) since the last
        flush as a new delta file.

        The base (full dict) is never touched (avoids the shared-storage
        bandwidth burst + ESTALE). No-op if _dirty is empty. No-op if
        readonly.
        """
        if self._readonly:
            return
        with self._lock:
            if not self._dirty:
                self._pending = 0
                return
            payload: dict[str, Any] = {
                "__meta__": {
                    "model_name": self.model_name,
                    "dim": self.dim,
                    "version": 2,
                    "kind": "delta",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            payload.update(self._dirty)
            self._atomic_write_pickle(payload, self._new_delta_path())
            self._dirty = {}
            self._pending = 0

    def compact(self, _auto: bool = False) -> bool:
        """Merge base + all deltas -> rewrite base (atomic) + delete the merged
        deltas.

        Runs in **only one process** under an exclusive lock
        (`{path}.compact.lock`). With _auto=True (auto-called on load), it uses
        a try-lock and skips immediately if held. Returns whether compaction
        ran.

        Order: (1) snapshot the current delta list -> (2) rewrite base (merged,
                 atomic) -> (3) delete only the snapshotted deltas. Since base
                 is atomically renamed before the deltas are deleted, a
                 concurrent loader reading at any point loses no data
                 (duplicate merges are idempotent). New deltas created by other
                 processes after the snapshot are not touched.
        """
        if self._readonly:
            return False
        lock_path = self._compact_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lf = None
        try:
            lf = open(lock_path, "w")
            got = self._flock(lf.fileno(), exclusive=True, nonblock=_auto)
            if not got:
                return False  # another process is compacting - skip.
            # (1) snapshot - target only the deltas that exist now.
            deltas = self._delta_paths()
            with self._lock:
                # _data is already the base+delta merge + this process's new
                # additions.
                # base rewrite uses the full _data (this becomes the new base).
                self._meta["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                self._meta["kind"] = "base"
                payload: dict[str, Any] = {"__meta__": dict(self._meta)}
                payload.update(self._data)
                # this process's _dirty is included in the new base too, so
                # clear it (avoids duplicate deltas).
                self._dirty = {}
                self._pending = 0
            # (2) atomic rewrite of base.
            self._atomic_write_pickle(payload, self.path)
            # (3) delete the snapshotted deltas (preserve any created
            # afterward).
            for dp in deltas:
                try:
                    dp.unlink()
                except OSError:
                    pass
            logging.info(
                f"[EmbeddingCache] {self.path.name}: compacted "
                f"{len(deltas)} delta files "
                f"-> base ({len(self._data)} entries)."
            )
            return True
        except OSError as e:
            logging.warning(
                f"[EmbeddingCache] compact I/O failed (ignored): {e}"
            )
            return False
        finally:
            if lf is not None:
                try:
                    lf.close()  # also releases the flock.
                except OSError:
                    pass

    # In-memory ops
    def get(self, text: str) -> np.ndarray | None:
        with self._lock:
            v = self._data.get(text)
        return v

    def get_many(self, texts: Iterable[str]) -> list[np.ndarray | None]:
        with self._lock:
            return [self._data.get(t) for t in texts]

    def set(self, text: str, vec: np.ndarray) -> None:
        """Add the vector in-memory + register it in _dirty. Flush when
        autoflush_every is reached. No-op if readonly."""
        if self._readonly:
            return
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(
                f"vec shape {vec.shape} != expected ({self.dim},)"
            )
        vec = np.array(vec, dtype=np.float32, copy=True)  # defensive copy.
        with self._lock:
            self._data[text] = vec
            self._dirty[text] = vec
            self._pending += 1
            af = self.autoflush_every
            if af > 0 and self._pending >= af:
                self.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # High-level: batch embed missing only
    def embed_with_cache(
        self,
        embedder: Any,
        texts: list[str] | Iterable[str],
        batch_size: int = 100,
    ) -> np.ndarray:
        """Return the embeddings (n, dim) of `texts`. Only cache misses call
        embedder.embed() + write a delta.

        - preserves order (out[i]=texts[i]), dedups duplicates,
          empty/None->zero vec (not stored in cache), flushes a delta every
          flush_every_n_batches.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not isinstance(texts, list):
            texts = list(texts)
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        if n == 0:
            return out
        text_to_indices: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            if not t:
                continue
            text_to_indices.setdefault(t, []).append(i)
        miss_texts: list[str] = []
        with self._lock:
            for t, idxs in text_to_indices.items():
                cached = self._data.get(t)
                if cached is not None:
                    for i in idxs:
                        out[i] = cached
                else:
                    miss_texts.append(t)
        if not miss_texts:
            return out
        if not hasattr(embedder, "embed"):
            raise AttributeError(
                f"embedder {type(embedder).__name__} does not implement "
                f"embed(list[str])"
            )
        batch_idx = 0
        for start in range(0, len(miss_texts), batch_size):
            batch_idx += 1
            batch = miss_texts[start : start + batch_size]
            t0 = time.perf_counter()
            vecs = embedder.embed(batch)
            dt = time.perf_counter() - t0
            try:
                vecs_arr = np.asarray(vecs, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise RuntimeError(
                    f"failed to convert embedder.embed return to a float32 "
                    f"array: {e}. suspected jagged list."
                ) from e
            if vecs_arr.shape != (len(batch), self.dim):
                raise ValueError(
                    f"embedder.embed returned shape {vecs_arr.shape}, "
                    f"expected ({len(batch)}, {self.dim})"
                )
            if not np.isfinite(vecs_arr).all():
                raise ValueError(
                    f"embedder.embed returned non-finite vector (NaN/inf). "
                    f"batch[0]={batch[0][:80]!r}..."
                )
            with self._lock:
                for j, t in enumerate(batch):
                    v = np.array(vecs_arr[j], dtype=np.float32, copy=True)
                    self._data[t] = v
                    if not self._readonly:
                        self._dirty[t] = v
                    for idx in text_to_indices[t]:
                        out[idx] = v
                    self._pending += 1
            logging.info(
                f"[EmbeddingCache] {self.path.name}: batch {batch_idx} "
                f"({len(batch)} new, {len(self)} total, {dt:.2f}s)"
            )
            fnb = self.flush_every_n_batches
            if not self._readonly and batch_idx % fnb == 0:
                self.flush()
        if not self._readonly and self._dirty:
            self.flush()
        return out

    def embed_one_cached(self, embedder: Any, text: str) -> np.ndarray:
        """Embed a single text. hit O(1), miss -> embed + delta autoflush."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        with self._lock:
            cached = self._data.get(text)
        if cached is not None:
            return cached.copy()  # copy to avoid aliasing.
        if hasattr(embedder, "embed_one"):
            vec = embedder.embed_one(text)
        elif hasattr(embedder, "embed"):
            vec = embedder.embed([text])[0]
        else:
            raise AttributeError(
                f"embedder {type(embedder).__name__} implements neither "
                f"embed_one nor embed"
            )
        try:
            vec_arr = np.asarray(vec, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise RuntimeError(
                f"failed to convert embedder return: {e}"
            ) from e
        if vec_arr.shape != (self.dim,):
            raise ValueError(
                f"embedder.embed_one returned shape {vec_arr.shape}, "
                f"expected ({self.dim},)"
            )
        if not np.isfinite(vec_arr).all():
            raise ValueError(
                f"embedder returned non-finite vector. "
                f"text[:80]={text[:80]!r}"
            )
        # defensive copy + _dirty register + autoflush. no-op if readonly.
        self.set(text, vec_arr)
        with self._lock:
            cached = self._data.get(text)
        return cached.copy() if cached is not None else vec_arr
