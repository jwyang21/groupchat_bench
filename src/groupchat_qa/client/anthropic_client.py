"""Anthropic client — langchain `ChatAnthropic` wrapper."""
from __future__ import annotations

from typing import Any

from .base import LLM


class AnthropicClient(LLM):
    def _make_chat_model(self):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            **self.extra_kwargs,
        )

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
