"""OpenAI client - langchain `ChatOpenAI` wrapper."""
from __future__ import annotations

from typing import Any

from .base import LLM


class OpenAIClient(LLM):
    def _make_chat_model(self):
        import os

        from langchain_openai import ChatOpenAI

        # api_key resolution: if config passes the vLLM convention "EMPTY" (or
        # an empty value),
        # authentication against the official OpenAI endpoint fails with 401.
        # To work without
        # committing a secret key to config (git), fall back to the
        # OPENAI_API_KEY env var when
        # the key is "EMPTY"/empty. (openai provider only - no effect on the
        # vLLM path.)
        kwargs = dict(self.extra_kwargs)
        ak = kwargs.get("api_key")
        if not ak or str(ak).upper() == "EMPTY":
            # Gemini's OpenAI-compatible endpoint (generativelanguage)
            # authenticates with GOOGLE_API_KEY; the official OpenAI endpoint
            # uses OPENAI_API_KEY. Pick by base_url.
            _burl = str(kwargs.get("base_url", "") or "")
            env_key = (
                os.getenv("GOOGLE_API_KEY")
                if "generativelanguage" in _burl else None
            ) or os.getenv("OPENAI_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key
            else:
                # if neither is set, defer to langchain's default (env)
                # behavior
                kwargs.pop("api_key", None)

        params = dict(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # gpt-5.x reasoning models (e.g. gpt-5.6 family) reject top_p
        # ("Unsupported parameter: 'top_p' is not supported with this model").
        # When top_p is None (config `top_p: null` for reasoning models) omit
        # it entirely; otherwise pass it (unchanged behavior for other models).
        if self.top_p is not None:
            params["top_p"] = self.top_p
        # reproducibility - OpenAI/vLLM support seed. Gemini's OpenAI-compat
        # endpoint rejects it ("Unknown name 'seed'"), so skip there.
        # None means not passed.
        _burl = str(kwargs.get("base_url", "") or "")
        if self.seed is not None and "generativelanguage" not in _burl:
            params["seed"] = self.seed
        # Any extra provider params (e.g. extra_body={"reasoning_effort": ...}
        # for reasoning models) flow through kwargs unchanged.
        return ChatOpenAI(**params, **kwargs)

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        # reasoning tokens are also folded into output_tokens as exposed by
        # langchain
        return (
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

    def _invoke_raw_fallback(
        self, messages: Any, orig_exc: Exception,
    ) -> tuple[str, int, int]:
        """Fallback when base.generate()'s langchain invoke fails to parse.

        Re-sends the same request via the OpenAI SDK to inspect the raw
        response. Some providers (notably Gemini's OpenAI-compatible endpoint)
        return HTTP 200 with ``message=None`` / empty choices when the prompt
        is blocked by a non-disableable content filter (e.g.
        PROHIBITED_CONTENT); langchain then raises
        ``'NoneType' object has no attribute 'get'`` while converting the
        response. Such a prompt is unanswerable, so return an empty string
        (the caller scores it as wrong) rather than crashing the whole run.
        If valid content is present, return it (recovering from the langchain
        parsing quirk). On a genuine API error (4xx/5xx, network), re-raise the
        original exception so normal error handling applies.
        """
        import os

        import openai as _openai

        kwargs = dict(self.extra_kwargs)
        ak = kwargs.get("api_key")
        burl = str(kwargs.get("base_url", "") or "")
        if not ak or str(ak).upper() == "EMPTY":
            ak = (
                os.getenv("GOOGLE_API_KEY")
                if "generativelanguage" in burl else None
            ) or os.getenv("OPENAI_API_KEY")
        client = _openai.OpenAI(base_url=(burl or None), api_key=ak)

        # langchain messages (SystemMessage/HumanMessage) -> OpenAI chat dicts.
        _role = {"system": "system", "human": "user", "ai": "assistant"}
        msgs = [
            {
                "role": _role.get(getattr(m, "type", "human"), "user"),
                "content": getattr(m, "content", "") or "",
            }
            for m in messages
        ]
        params: dict[str, Any] = dict(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=msgs,
        )
        if self.top_p is not None:
            params["top_p"] = self.top_p
        # Pass extra_body (e.g. reasoning_effort) exactly as the chat model does.
        eb = kwargs.get("extra_body")
        if isinstance(eb, dict) and eb:
            params["extra_body"] = dict(eb)
        if self.seed is not None and "generativelanguage" not in burl:
            params["seed"] = self.seed

        try:
            resp = client.chat.completions.create(**params)
        except Exception:
            raise orig_exc  # genuine API error -> keep original handling.

        d = resp.model_dump()
        usage = d.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        ch = d.get("choices") or []
        if not ch:
            return "", in_tok, out_tok  # prompt-level block (no candidates).
        msg = ch[0].get("message")
        if msg is None:
            return "", in_tok, out_tok  # content filter -> message is None.
        content = msg.get("content")
        return (content if content is not None else ""), in_tok, out_tok
