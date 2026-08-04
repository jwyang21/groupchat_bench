"""LLM factory - creates a client instance from the provider name in config."""
from __future__ import annotations

from typing import Any

from .base import LLM


def make_llm(provider: str, model_name: str, **kwargs: Any) -> LLM:
    """provider name -> LLM instance.

    provider:
      - "vllm"      : open-source LLM (vLLM serve endpoint).
      - "openai"    : OpenAI API
      - "anthropic" : Anthropic API
      - "google"    : Google Gemini API

    Example:
      >>> llm = make_llm("vllm", "qwen2.5-72b-unsloth",
      ...                base_url="http://localhost:8000/v1", temperature=0.0)
    """
    provider = provider.lower()
    if provider == "vllm":
        from .vllm_client import VLLMClient
        return VLLMClient(model_name=model_name, **kwargs)
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model_name=model_name, **kwargs)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model_name=model_name, **kwargs)
    if provider == "google":
        from .google_client import GoogleClient
        return GoogleClient(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown provider: {provider!r}. Expected one of: vllm/openai/anthropic/google.")
