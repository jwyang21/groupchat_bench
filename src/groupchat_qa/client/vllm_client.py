"""vLLM client - via an OpenAI-compatible endpoint.

Open-source LLM inference goes exclusively through this client (no direct
transformers calls).

By default this communicates with an external vLLM serve process
(`vllm serve --port 8000`).
- `generate(prompt)` : calls the OpenAI HTTP API directly -> extracts both
  content and reasoning_content. vLLM's `--reasoning-parser qwen3 / gemma4`
  sends the split-out `reasoning` field, which langchain ChatOpenAI ignores,
  so we receive it via raw HTTP and store it in
  `GenerationRecord.reasoning_text`.
- `_make_chat_model()` : a langchain `ChatOpenAI` instance - used only on the
  `generate_structured` (with_structured_output) path. That path needs
  content only, so losing reasoning is irrelevant.
"""
from __future__ import annotations

import os
import time
from typing import Any

from groupchat_qa.utils.types import GenerationRecord

from .base import LLM


class VLLMClient(LLM):
    """Client that communicates with a vLLM serve endpoint.

    OpenAI-compatible.
    """

    def __init__(
        self,
        model_name: str,                  # = --served-model-name
        # vLLM endpoint, e.g. http://localhost:8000/v1
        base_url: str | None = None,
        api_key: str = "EMPTY",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 1000,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        # if base_url is not given, read from env
        self.base_url = base_url or os.getenv(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
        # raw OpenAI client - used only on the generate() path (preserves
        # reasoning_content).
        # We set max_retries=0 so the client does not retry on its own; the
        # caller decides retries
        # explicitly. Long timeouts (thinking-ON models with lengthy reasoning
        # traces) can exceed
        # the default 60s, so a larger timeout is used.
        from openai import OpenAI
        # timeout can be overridden via the VLLM_CLIENT_TIMEOUT env var when a
        # specific experiment
        # needs a larger client timeout (e.g. thinking-ON with long reasoning).
        _client_timeout = float(
            os.environ.get("VLLM_CLIENT_TIMEOUT", "1200.0")
        )
        self._oa = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=_client_timeout,
            max_retries=0,
        )

    def _make_chat_model(self):
        from langchain_openai import ChatOpenAI

        # top_k / min_p / repetition_penalty / presence_penalty are not
        # accepted as top-level
        # kwargs by the OpenAI SDK and trigger a vLLM 400 BadRequest, so they
        # must be lifted into
        # extra_body. The generate() path already sends these via extra_body,
        # but the ChatOpenAI
        # path of generate_structured() was missing them, causing structured
        # calls to silently
        # fail for ON variants.
        extra = dict(self.extra_kwargs) if self.extra_kwargs else {}
        extra_body = dict(extra.pop("extra_body", {}) or {})
        for k in ("top_k", "min_p", "repetition_penalty", "presence_penalty"):
            if k in extra:
                extra_body[k] = extra.pop(k)
        return ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # reproducibility - pass seed per request (None is equivalent to
            # not passing)
            seed=self.seed,
            extra_body=(extra_body or None),
            **extra,
        )

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        # The raw AIMessage from generate_structured may have an empty
        # usage_metadata (in some
        # langchain/vLLM versions token usage only comes via
        # response_metadata.token_usage),
        # which would record structured calls as input/output 0. Fall back only
        # when
        #   usage_metadata is 0.
        if in_tok == 0 and out_tok == 0:
            _rm = getattr(response, "response_metadata", None) or {}
            tu = _rm.get("token_usage", {}) or {}
            in_tok = int(tu.get("prompt_tokens", 0) or 0)
            out_tok = int(tu.get("completion_tokens", 0) or 0)
        return in_tok, out_tok

    # override : preserve reasoning_content via raw OpenAI HTTP
    def generate(self, user_prompt: str) -> GenerationRecord:
        """Free-form text generation.

        Preserves content + reasoning_content + latency + tokens.

        When `--reasoning-parser` is enabled, the vLLM OpenAI-compat response
        contains:
          choices[0].message.content          -> final answer (refined by
              the parser)
          choices[0].message.reasoning_content -> thinking trace (split out
              by the parser)
        Both are stored in GenerationRecord - non-reasoning models have no
        reasoning_content -> "".

        extra_kwargs' `extra_body` (e.g. chat_template_kwargs.enable_thinking)
        is passed through.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        # reproducibility - pass seed in the request (OpenAI standard field,
        # honored by vLLM). None means not passed.
        if self.seed is not None:
            kwargs["seed"] = self.seed
        # extra_body (may be present in extra_kwargs - e.g. yaml's
        # chat_template_kwargs)
        extra_body = (
            dict(self.extra_kwargs.get("extra_body", {}))
            if self.extra_kwargs
            else {}
        )
        # other sampling keys in extra_kwargs (top_k, min_p,
        # repetition_penalty, presence_penalty, ...)
        # are not accepted directly by the OpenAI client, so pass them via
        # extra_body.
        for k in ("top_k", "min_p", "repetition_penalty", "presence_penalty"):
            if self.extra_kwargs and k in self.extra_kwargs:
                extra_body[k] = self.extra_kwargs[k]
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.perf_counter()
        resp = self._oa.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        text = msg.content or ""
        # The vLLM reasoning-parser result key is either "reasoning" or
        # "reasoning_content"
        # depending on the vLLM version - try both.
        _rc = getattr(msg, "reasoning_content", None)
        reasoning = _rc or getattr(msg, "reasoning", None) or ""

        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        return GenerationRecord(
            text=str(text),
            latency_sec=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning_text=str(reasoning),
        )

    # override : structured responses via raw OpenAI client too - preserve
    # reasoning_content
    def generate_structured(
        self,
        user_prompt: str,
        schema,
        methods: tuple[str, ...] = (
            "json_schema", "function_calling", "json_mode",
        ),
    ):
        """Structured responses via the raw OpenAI client too - preserve
        vLLM reasoning_content.

        langchain ChatOpenAI(with_structured_output) discards vLLM's
        reasoning_content (langchain-openai 1.2.x does not preserve
        third-party reasoning fields), so the reasoning of
        note/link/evolution structured calls in thinking-on models was not
        being logged.
        -> Like generate(), use the raw client + vLLM guided decoding
        (`guided_json`) to enforce the schema and capture reasoning_content.
        Only on a guided_json exception (e.g. unsupported vLLM version) do we
        safely fall back to the base (langchain) path (reasoning is lost
        there).
        The return matches base: `(parsed | None, GenerationRecord)` -
        parsed=None lets the caller apply its usual free-form fallback.
        """
        import json as _json

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        extra_body = (
            dict(self.extra_kwargs.get("extra_body", {}))
            if self.extra_kwargs
            else {}
        )
        for k in ("top_k", "min_p", "repetition_penalty", "presence_penalty"):
            if self.extra_kwargs and k in self.extra_kwargs:
                extra_body[k] = self.extra_kwargs[k]
        if extra_body:
            kwargs["extra_body"] = extra_body
        # Enforce the schema via the OpenAI-standard
        # response_format(json_schema) - the same
        # mechanism langchain with_structured_output uses, so it actually
        # works. (extra_body.guided_json
        # is ignored by some vLLM versions, leaving content as free text.)
        # reasoning is split out as reasoning_content.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(schema, "__name__", "Schema"),
                "schema": schema.model_json_schema(),
            },
        }

        t0 = time.perf_counter()
        try:
            resp = self._oa.chat.completions.create(**kwargs)
        except Exception:
            # guided_json unsupported/error -> safely fall back to the
            # langchain path (accepting reasoning loss).
            return super().generate_structured(user_prompt, schema, methods)
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        text = msg.content or ""
        _rc = getattr(msg, "reasoning_content", None)
        reasoning = _rc or getattr(msg, "reasoning", None) or ""
        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        # The guided_json result is schema-conforming JSON -> parse with
        # pydantic. On failure,
        # parsed=None (caller applies free-form fallback). The reasoning-parser
        # splits out any
        # thinking trace that leaks before content, so content is pure JSON; if
        # it still mixes in,
        #   retry with json.loads.
        parsed = None
        if text:
            try:
                parsed = schema.model_validate_json(text)
            except Exception:
                try:
                    parsed = schema.model_validate(_json.loads(text))
                except Exception:
                    parsed = None

        return parsed, GenerationRecord(
            text=str(text),
            reasoning_text=str(reasoning),
            latency_sec=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
