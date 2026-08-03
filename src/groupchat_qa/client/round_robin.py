"""Load-aware wrapper + failover over multiple LLM clients (different vLLM endpoints).

The old round-robin (distributing in index order) starved fast endpoints and piled requests on
slow ones when endpoint speeds differed. Load-aware routing instead sends each request to the
endpoint with the fewest in-flight requests (least-loaded), so faster endpoints naturally receive
more. When all endpoints are equally fast this is effectively round-robin (even distribution).

For in-cell question parallelism (ThreadPool, answer_max_workers), each question is spread across
multiple endpoints → distributes load off a single GPU and utilizes multiple server GPUs.

[failover] If the chosen endpoint call raises (dead endpoint connection refused / timeout / 5xx),
retry on the next least-loaded endpoint. The last exception is raised only when all endpoints fail.
Reason: vllm_client uses max_retries=0 and the answer-phase per-session try/except excludes failed
sessions from n_scored. Failover drives question failures to zero to avoid re-runs.

LoggingLLM-style pattern (does not inherit LLM, duck-typed): only generate/generate_structured are
wrapped; other attributes (name, model_name, etc.) are delegated to the first client via __getattr__.
"""
from __future__ import annotations

import logging
import threading
from typing import Any


class RoundRobinLLM:
    """Calls clients in load-aware (least in-flight) order, and on call failure fails over to the
    next least-loaded endpoint. (The class name is kept for call-site compatibility — the actual
    behavior is load-aware.)

    Each self.llm.generate(_structured) call goes to the currently idlest endpoint; if that
    endpoint is down, it moves to the next live least-loaded endpoint."""

    def __init__(self, clients: list[Any]):
        if not clients:
            raise ValueError("RoundRobinLLM requires at least 1 client")
        # set directly on __dict__ → avoids __getattr__ (prevents infinite recursion).
        self._clients = list(clients)
        self._lock = threading.Lock()
        # in-flight request count currently being processed by each client. Basis for least-loaded selection.
        self._inflight = [0] * len(self._clients)

    def _acquire_least(self, tried: "set[int]") -> "int | None":
        """Pick the idx with the minimum in-flight among clients not in `tried`, increment its
        in-flight, and return it. Selection + increment are atomic within one lock, so concurrent
        calls do not pile onto the same idx and naturally distribute (on concurrent acquire, the
        first thread takes idx0→1, the next thread sees idx0 at 1 and picks idx1…). Returns None if
        no candidate. tie-break: on equal in-flight, prefer the smaller idx (strict '<' comparison)."""
        with self._lock:
            best: "int | None" = None
            best_load = 0
            for i in range(len(self._clients)):
                if i in tried:
                    continue
                load = self._inflight[i]
                if best is None or load < best_load:
                    best, best_load = i, load
            if best is not None:
                self._inflight[best] += 1
            return best

    def _release(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] -= 1

    def _call_with_failover(self, method: str, *args: Any, fail_pred=None, **kwargs: Any) -> Any:
        # fail_pred: a callback that takes the return value (no exception raised) and decides whether
        #   to treat it as a failure (True = fail over to the next endpoint). None means fail over
        #   only when an exception is raised (default behavior). Some clients (e.g. the invoke wrapper
        #   in _structured_with_fallback) do not raise on dead/timeout but swallow it into a
        #   dict(parsing_error=...), so we inspect that dict to decide failover.
        tried: "set[int]" = set()
        last_exc: "Exception | None" = None
        last_result: Any = None
        have_result = False
        for _ in range(len(self._clients)):     # at most one attempt per endpoint
            idx = self._acquire_least(tried)
            if idx is None:                     # no more endpoints to try
                break
            tried.add(idx)
            c = self._clients[idx]
            try:
                res = getattr(c, method)(*args, **kwargs)
                # no exception, but if fail_pred judges it a failure, fail over to the next endpoint.
                if fail_pred is not None and fail_pred(res):
                    last_result = res
                    have_result = True
                    ep = getattr(c, "base_url", getattr(c, "name", "?"))
                    logging.warning(
                        f"[RoundRobinLLM/load-aware] {method} judged failed (endpoint={ep}) → failover to next least-loaded"
                    )
                    continue
                return res
            except Exception as e:              # noqa: BLE001 — dead/timeout/5xx all fail over
                last_exc = e
                ep = getattr(c, "base_url", getattr(c, "name", "?"))
                logging.warning(
                    f"[RoundRobinLLM/load-aware] {method} failed (endpoint={ep}) → failover to next least-loaded: {e}"
                )
            finally:
                # regardless of success/failure, request ends → restore in-flight (after return on success, before next attempt on failure).
                self._release(idx)
        # All endpoints failed. If there is a result judged failed by fail_pred, return the last
        #   result (no exception — the caller handles that result, e.g. a parsing_error dict,
        #   gracefully). If neither exists, raise the exception.
        if have_result:
            return last_result
        assert last_exc is not None  # n>=1 guaranteed (__init__), so at least one attempt was made
        raise last_exc

    def generate(self, *args: Any, **kwargs: Any):
        return self._call_with_failover("generate", *args, **kwargs)

    def generate_structured(self, *args: Any, **kwargs: Any):
        return self._call_with_failover("generate_structured", *args, **kwargs)

    @staticmethod
    def _invoke_failed(res: Any) -> bool:
        # Decide whether the invoke-wrapper dict returned by _structured_with_fallback is a "failure".
        #   parsing_error not None = dead/timeout/all methods failed → fail over to the next endpoint.
        #   parsing_error None (even if parsed=None) means the endpoint responded normally, so treat
        #   as success (an empty response is the LLM's own result → the caller handles it with
        #   count-preserving logic).
        return isinstance(res, dict) and res.get("parsing_error") is not None

    def invoke(self, *args: Any, **kwargs: Any):
        # langchain-style `invoke` interface. A batch-oriented caller can wrap the invoke
        #   wrappers of _structured_with_fallback (multi-endpoint ChatOpenAI-based) with this pool
        #   to send one batch call to the least-in-flight endpoint.
        #   Note: if a caller's batches are strictly sequential (each batch depends on the previous
        #     result), there are no concurrent calls, so in-flight is always 0→1→0 and every batch
        #     effectively goes to idx0 (low distribution effect). The real value of this pool is
        #     "fail over to the next endpoint if one endpoint dies (dead/timeout) or parse fails".
        #   The wrapper does not raise but swallows into a dict(parsing_error=...), so pass
        #     _invoke_failed as fail_pred to trigger failover based on that dict. If all endpoints
        #     fail, return the last result dict (including parsing_error) as-is → the caller
        #     handles it gracefully.
        return self._call_with_failover("invoke", *args, fail_pred=self._invoke_failed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # attributes other than generate/generate_structured (name/model_name, etc.) delegate to the first client.
        # _clients/_lock/_inflight are set on __dict__ in __init__, so they don't reach here.
        return getattr(self._clients[0], name)
