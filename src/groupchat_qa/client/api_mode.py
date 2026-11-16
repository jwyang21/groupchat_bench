"""Toggle between calling OpenAI directly vs. an OpenAI-compatible gateway.

A single switch that lets the experiment code run either (1) against the
official OpenAI API or (2) through any OpenAI-compatible gateway, selected by
the `GROUPCHAT_API_MODE` environment variable:
  - "openai" (default) : official OpenAI endpoint + OPENAI_API_KEY.
  - "gateway"          : OpenAI-compatible gateway at GATEWAY_BASE_URL,
    authenticated with GATEWAY_API_KEY.

The gateway must expose the same model ids as OpenAI (e.g.
text-embedding-3-small); model names are passed through unchanged.

Scope: the OpenAI path only (OpenAIClient, OpenAIEmbedder). vLLM
(localhost base_url), Gemini (generativelanguage base_url) and any other
explicitly-set endpoint keep their base_url (see the resolve rule below).
"""
from __future__ import annotations

import os


def api_mode() -> str:
    """Return the current API mode: "openai" (default) or "gateway".

    Case- and whitespace-insensitive.
    """
    return (os.getenv("GROUPCHAT_API_MODE") or "openai").strip().lower()


def is_gateway() -> bool:
    return api_mode() == "gateway"


def gateway_base_url() -> str:
    """Return GATEWAY_BASE_URL; raise if gateway mode is on but it is unset."""
    url = (os.getenv("GATEWAY_BASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "GROUPCHAT_API_MODE=gateway requires GATEWAY_BASE_URL "
            "(e.g. https://<your-gateway>/v1)."
        )
    return url


def _is_openai_default_base(base_url: str | None) -> bool:
    """Whether base_url is the OpenAI official endpoint (or unset).

    In gateway mode we reroute *only* in this case. vLLM, Gemini and any
    other explicit endpoint return False and are kept as-is.
    """
    b = str(base_url or "").strip()
    if b == "":
        return True  # unset = OpenAI SDK default (api.openai.com)
    return "api.openai.com" in b


def resolve_openai_credentials(
    base_url: str | None, api_key: str | None
) -> tuple[str | None, str | None]:
    """Convert (base_url, api_key) to the current mode.

    - openai mode: return the inputs unchanged.
    - gateway mode: substitute (GATEWAY_BASE_URL, GATEWAY_API_KEY) only when
      base_url is the OpenAI default (empty / api.openai.com). Everything
      else is returned unchanged. If GATEWAY_API_KEY is unset, the input
      api_key is kept.
    """
    if not is_gateway():
        return base_url, api_key
    if _is_openai_default_base(base_url):
        key = os.getenv("GATEWAY_API_KEY") or api_key
        return gateway_base_url(), key
    return base_url, api_key
