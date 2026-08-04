"""LoggingLLM - wraps a base LLM to record the (prompt, response) of *every* generate /
generate_structured call to a JSONL log file without omission.

Used for baseline experiments (a_mem/hipporag2/mem0) to log the prompts and responses of all LLM
calls - note analysis, evolution, OpenIE, filter, fact extraction, update, QA, and so on.

Design: the baseline code is not changed at all. Wrapping the base llm with this wrapper in the
runner makes `self.llm.generate(...)` / `self.llm.generate_structured(...)` log automatically.
There is one LLM instance per LLM (thread), so there is one log file per LLM -> almost no file
contention (still thread-safe via a lock). Other LLM attributes (model_name, etc.) are delegated
to the base via __getattr__.

One log line = one LLM call = a JSON object:
  {call_idx, kind("generate"|"generate_structured"), schema(structured schema name|null),
   prompt(full), response_text, reasoning_text, parsed(structured result dict|str|null),
   latency_sec, input_tokens, output_tokens}
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

# Block silent bypass - if these method names are forwarded via __getattr__ they would not be
# logged, violating the "log all LLM inference" policy. Emit an explicit warning when called.
_BYPASS_WATCH = frozenset({
    "invoke", "ainvoke", "stream", "astream",
    "with_structured_output", "with_retry", "with_fallbacks",
    "_chat", "client",
})
_BYPASS_LOGGER = logging.getLogger(__name__ + ".bypass_warn")
_WARNED_BYPASS: set[str] = set()

# Within one process, a given log path is truncated (overwrite start) only once, on the first
# LoggingLLM creation; afterwards (e.g. re-invocation of run_one_llm per ablation grid cell) it
# appends -> all prompts of one run accumulate in one file. A new process (re-run) has an empty
# set and truncates again (avoiding stale logs).
# _PATH_COUNTERS: per-path monotonic call_idx (even when a new LoggingLLM appends to the same file
#   per ablation grid cell, call_idx keeps increasing rather than resetting to 1 -> in-file order
#   can be analyzed).
# _PATH_LOCKS: per-path shared lock - an instance-local `self._lock` cannot protect other instances
#   on the same path (e.g. same LLM x different cell in multi-grid ablation). A path-level lock
#   prevents concurrent-append races.
_TRUNCATED_PATHS: set[str] = set()
_PATH_COUNTERS: dict[str, int] = {}
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATHS_GUARD = threading.Lock()


class LoggingLLM:
    """base LLM wrapper. Logs generate / generate_structured calls as JSONL + delegates to base."""

    def __init__(self, base: Any, log_path: str | Path,
                 extra_metadata: dict | None = None,
                 fsync: bool = False) -> None:
        """
        extra_metadata: extra metadata merged into every _log entry (e.g. {"grid_idx": 3, "trial": 1}),
            to enable per-ablation-grid-cell analysis. On collision with a reserved key
            (call_idx/kind/schema/prompt/...), a `meta_` prefix is applied automatically.
        fsync: when True, os.fsync on every call - prevents loss on kill -9 / OOM. Default False to
            avoid NFS I/O bottlenecks (the thousands of calls in one ablation cell accumulate 5-15min
            of latency). Set True only for windows where it is truly needed.

        Multi-process note: `_TRUNCATED_PATHS` / `_PATH_LOCKS` / `_PATH_COUNTERS` are all
            process-local. Two processes writing the same log_path concurrently may cause a truncate
            race. This wrapper assumes single-process; if multi-process is needed, fcntl.flock is required.
        """
        self._base = base
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._key = str(self._log_path.resolve())
        self._extra_metadata = dict(extra_metadata or {})
        self._fsync = fsync
        # path-level shared lock - multi-instances on the same log_path (LLM x multi-grid in ablation,
        #   or re-invocation of the same LLM) share the same lock.
        with _PATHS_GUARD:
            if self._key not in _PATH_LOCKS:
                _PATH_LOCKS[self._key] = threading.Lock()
            self._lock = _PATH_LOCKS[self._key]
            first = self._key not in _TRUNCATED_PATHS
            if first:
                _TRUNCATED_PATHS.add(self._key)
                _PATH_COUNTERS[self._key] = 0
                # truncate inside the lock - so truncation finishes before another instance appends
                #   to the same path (prevents losing the first line under shared-path contention).
                try:
                    self._log_path.write_text("", encoding="utf-8")
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # the two intercepted methods (everything the baseline calls)
    # ------------------------------------------------------------------
    def generate(self, user_prompt: str):
        rec = self._base.generate(user_prompt)
        self._log("generate", None, user_prompt, rec, None)
        return rec

    def generate_structured(self, user_prompt: str, schema, *args, **kwargs):
        parsed, rec = self._base.generate_structured(user_prompt, schema, *args, **kwargs)
        self._log(
            "generate_structured",
            getattr(schema, "__name__", str(schema)),
            user_prompt, rec, parsed,
        )
        return parsed, rec

    # ------------------------------------------------------------------
    # all other attributes/methods delegate to base (model_name, system_prompt, _extract_usage, etc.)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails. Prevent infinite recursion when
        # _base is unset (partial init / unpickle).
        if name == "_base":
            raise AttributeError(name)
        # Block silent bypass - on bypassing calls (invoke/stream/with_structured_output, etc.),
        # warn only on the first occurrence. key = {path}:{name} (each multi-instance in ablation
        # can warn once -> per-cell diagnosis).
        if name in _BYPASS_WATCH:
            warn_key = f"{self._key}:{name}"
            if warn_key not in _WARNED_BYPASS:
                _WARNED_BYPASS.add(warn_key)
                _BYPASS_LOGGER.warning(
                    f"[LoggingLLM] BYPASS DETECTED: '{name}' called directly on base LLM - "
                    f"this call will not be recorded in the JSONL log (forwarded to base.{name}). "
                    f"May violate the 'log all LLM inference' policy. "
                    f"Fix the memory module code to use only self.llm.generate(_structured). "
                    f"(log_path={self._key})"
                )
        return getattr(self._base, name)

    # ------------------------------------------------------------------
    def _log(self, kind: str, schema: str | None, prompt: str, rec: Any, parsed: Any) -> None:
        parsed_dump: Any = None
        if parsed is not None:
            try:
                parsed_dump = parsed.model_dump() if hasattr(parsed, "model_dump") else str(parsed)
            except Exception:
                parsed_dump = str(parsed)
        # call_idx is per-path monotonic (instance-independent) - even when a new instance appends
        #   to the same file per ablation grid cell, it does not reset to 1.
        with _PATHS_GUARD:
            idx = _PATH_COUNTERS.get(self._key, 0) + 1
            _PATH_COUNTERS[self._key] = idx
        entry = {
            "call_idx": idx,
            "kind": kind,
            "schema": schema,
            "prompt": prompt,
            "response_text": getattr(rec, "text", ""),
            "reasoning_text": getattr(rec, "reasoning_text", ""),
            "parsed": parsed_dump,
            "latency_sec": getattr(rec, "latency_sec", 0.0),
            "input_tokens": getattr(rec, "input_tokens", 0),
            "output_tokens": getattr(rec, "output_tokens", 0),
        }
        # Merge extra_metadata - for per-ablation-grid-cell analysis. On a reserved-key collision,
        # apply a `meta_` prefix automatically to avoid a silent collision with entry's default keys
        # (e.g. input_tokens with default 0/"").
        if self._extra_metadata:
            _RESERVED = {"call_idx", "kind", "schema", "prompt", "response_text",
                         "reasoning_text", "parsed", "latency_sec",
                         "input_tokens", "output_tokens"}
            for k, v in self._extra_metadata.items():
                key = f"meta_{k}" if k in _RESERVED else k
                entry[key] = v
        # (1) serialize outside open() with default=str for robustness (stringifies arbitrary objects),
        #   (2) except Exception - no serialization/IO exception should kill the experiment (best-effort logging).
        #   On parsed serialize failure, narrow the failure to parsed only (repr it) - otherwise the
        #   whole entry would dump as `{error:serialize_failed}`, losing prompt/response too.
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            try:
                entry["parsed"] = repr(parsed_dump)
                line = json.dumps(entry, ensure_ascii=False, default=str)
            except Exception:
                line = json.dumps({"call_idx": idx, "kind": kind, "error": "serialize_failed"})
        # flush by default + fsync optional.
        # fsync is a round-trip per call on NFS -> accumulates 5-15min over thousands of calls x LLM x grid.
        # default fsync=False (flush only, to OS buffer). Set True only when kill -9 protection is truly needed.
        try:
            with self._lock:
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    if self._fsync:
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass  # ignore on environments without fsync support (e.g. NFS)
        except Exception:
            pass  # a logging failure (disk/permission/other) must not kill the experiment (best-effort).
