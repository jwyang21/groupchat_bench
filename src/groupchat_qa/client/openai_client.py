"""OpenAI client — langchain `ChatOpenAI` wrapper."""
from __future__ import annotations

from typing import Any

from .base import LLM


class OpenAIClient(LLM):
    def _make_chat_model(self):
        import os

        from langchain_openai import ChatOpenAI

        # api_key resolution: if config passes the vLLM convention "EMPTY" (or an empty value),
        #   authentication against the official OpenAI endpoint fails with 401. To work without
        #   committing a secret key to config (git), fall back to the OPENAI_API_KEY env var when
        #   the key is "EMPTY"/empty. (openai provider only — no effect on the vLLM path.)
        kwargs = dict(self.extra_kwargs)
        ak = kwargs.get("api_key")
        if not ak or str(ak).upper() == "EMPTY":
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key
            else:
                kwargs.pop("api_key", None)  # if neither is set, defer to langchain's default (env) behavior

        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,   # reproducibility — OpenAI also supports seed (None means not passed)
            **kwargs,
        )

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        # reasoning tokens are also folded into output_tokens as exposed by langchain
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
