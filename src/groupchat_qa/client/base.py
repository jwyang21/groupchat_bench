"""LLM client ABC + GenerationRecord.

Design intent:
- Wrap langchain's `BaseChatModel` to provide a *provider-agnostic* interface.
- Every call is record-bearing - returns text + latency + input/output tokens together.
- Structured responses are enforced via `generate_structured(prompt, schema=BaseModel)`
  (using langchain `with_structured_output`).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from groupchat_qa.utils.types import GenerationRecord

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


class LLM(ABC):
    """Unified interface for LLM providers.

    Implementers only need to write:
      1. `_make_chat_model()` to build a langchain `BaseChatModel` instance
      2. `_extract_usage(response)` to extract token usage from the provider response
    The `generate` / `generate_structured` bodies are handled commonly by the ABC.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 1000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # reproducibility seed - passed to vLLM/OpenAI per request. Even with temperature>0,
        # sampling is reproducible given the same seed + same HW + same vLLM version
        # (vLLM Reproducibility docs). This is separate from the client-side seed_everything -
        # that seeds python/numpy/torch RNG, this seeds the server sampling RNG.
        # None means seed is not passed (default behavior). Whether the seed is actually
        # attached to the request is each client's responsibility
        # (vLLM/OpenAI support it, other providers may ignore it).
        self.seed = seed
        self.extra_kwargs = kwargs
        self._chat = self._make_chat_model()

    # hooks the implementer must fill in
    @abstractmethod
    def _make_chat_model(self) -> "BaseChatModel":
        """Build a langchain `BaseChatModel` instance."""
        ...

    @abstractmethod
    def _extract_usage(self, response: Any) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from the provider response.

        Returns (0, 0) if the provider does not supply usage metadata (safe fallback).
        """
        ...

    # common API (record-bearing)
    def generate(self, user_prompt: str) -> GenerationRecord:
        """Free-form text generation. Measures latency + tokens per call automatically."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        t0 = time.perf_counter()
        response = self._chat.invoke(messages)
        latency = time.perf_counter() - t0
        in_tok, out_tok = self._extract_usage(response)
        # Prevent str(None)="None" from leaking when content is None (e.g. tool-only responses) -
        # use an empty string when content is None/missing.
        _content = getattr(response, "content", None)
        text = _content if _content is not None else str(response)
        return GenerationRecord(
            text=str(text),
            latency_sec=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    def generate_structured(
        self,
        user_prompt: str,
        schema: type[T],
        methods: tuple[str, ...] = ("json_schema", "function_calling", "json_mode"),
    ) -> tuple[T | None, GenerationRecord]:
        """Return a structured response (pydantic schema) + record together.

        Uses langchain `with_structured_output(schema, method=..., include_raw=True)`.
        - `include_raw=True` also returns the raw AIMessage to extract tokens from usage_metadata.
        - Since vLLM's OpenAI-compat method support varies by model, fall back through
          `json_schema -> function_calling -> json_mode`.
        - On parse failure (parsing_error present), returns `parsed = None` - the caller can
          then apply a free-form fallback.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        # The construction step (with_structured_output) and the invoke step are both wrapped in
        # the per-method try, so a failure at invoke time (e.g. vLLM 400) does not prevent the
        # fallback from firing.
        last_err: Exception | None = None
        resp = None
        latency = 0.0
        t_loop = time.perf_counter()
        for m in methods:
            try:
                structured_chat = self._chat.with_structured_output(
                    schema, method=m, include_raw=True
                )
                t0 = time.perf_counter()
                resp = structured_chat.invoke(messages)
                latency = time.perf_counter() - t0
                break
            except Exception as e:
                last_err = e
                resp = None
        if resp is None:
            # When all methods fail, return (None, record) instead of raising.
            #   Returning a record that carries the whole loop latency ensures the latency of the
            #   failed structured attempt is not lost from metrics; the caller accumulates that
            #   record normally, and parsed=None triggers the free-form fallback.
            #   tokens are 0 because these failed calls received no server usage. last_err is for
            #   debugging only.
            failed_latency = time.perf_counter() - t_loop
            return None, GenerationRecord(
                text="", latency_sec=failed_latency, input_tokens=0, output_tokens=0
            )
        # include_raw=True -> resp = {"raw": AIMessage, "parsed": T | None, "parsing_error": ...}
        parsed = resp.get("parsed") if isinstance(resp, dict) else resp
        raw_msg = resp.get("raw") if isinstance(resp, dict) else None
        in_tok, out_tok = self._extract_usage(raw_msg) if raw_msg is not None else (0, 0)
        text = (
            str(raw_msg.content) if raw_msg is not None and hasattr(raw_msg, "content")
            else str(parsed)
        )
        # Capture vLLM thinking-on reasoning traces on the structured path as well.
        #   langchain ChatOpenAI may carry vLLM's reasoning_content in additional_kwargs /
        #   response_metadata rather than standard content, so check both best-effort.
        reasoning = ""
        if raw_msg is not None:
            _ak = getattr(raw_msg, "additional_kwargs", {}) or {}
            _rm = getattr(raw_msg, "response_metadata", {}) or {}
            reasoning = (
                _ak.get("reasoning_content")
                or _ak.get("reasoning")
                or _rm.get("reasoning_content")
                or ""
            )
        record = GenerationRecord(
            text=text,
            reasoning_text=str(reasoning) if reasoning else "",
            latency_sec=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        return parsed, record
