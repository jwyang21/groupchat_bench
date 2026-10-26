"""Toggle between calling OpenAI directly vs. an OpenAI-compatible AI gateway.

A single switch that lets the experiment code run either (1) against the
official OpenAI API or (2) through an OpenAI-compatible gateway, selected by the
`GROUPCHAT_API_MODE` environment variable:
  - "openai" (default) : official OpenAI endpoint + OPENAI_API_KEY (unchanged
    behavior).
  - "letsur"           : OpenAI-compatible gateway + LETSUR_API_KEY.

Gateway facts (OpenAI-compatible):
  - base_url  = "https://gw.letsur.ai/v1"
  - auth      = Authorization: Bearer <LETSUR_API_KEY>  (= openai SDK api_key)
  - model ids = same codes as OpenAI (the catalog exposes identical names such
    as text-embedding-3-small / gpt-4o / etc.), so no model-name mapping is
    needed (pass-through).

Scope: the OpenAI path only (OpenAIClient, OpenAIEmbedder). vLLM
(VLLMClient, localhost base_url), Gemini (generativelanguage base_url) and any
other explicitly-set endpoint keep their base_url even when routed through this
helper (see the resolve rule below).
"""
from __future__ import annotations

import os

# OpenAI-compatible gateway endpoint.
LETSUR_BASE_URL = "https://gw.letsur.ai/v1"


def api_mode() -> str:
    """Return the current API mode: "openai" (default) or "letsur".

    Case- and whitespace-insensitive.
    """
    return (os.getenv("GROUPCHAT_API_MODE") or "openai").strip().lower()


def is_letsur() -> bool:
    return api_mode() == "letsur"


def _is_openai_default_base(base_url: str | None) -> bool:
    """Whether base_url is the OpenAI official endpoint (or unset = official
    default).

    In letsur mode we route to the gateway *only* in this case. vLLM
    (localhost/127.0.0.1), Gemini (generativelanguage) and any other explicit
    endpoint return False -> kept as-is (protected).
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
    - letsur mode: substitute (LETSUR_BASE_URL, LETSUR_API_KEY) only when
      base_url is the OpenAI default (empty / api.openai.com). Everything else
      (vLLM / gemini / explicit endpoint) is returned unchanged.

    If LETSUR_API_KEY is unset, fall back to the input api_key (the caller / SDK
    surfaces a clear error).
    """
    if not is_letsur():
        return base_url, api_key
    if _is_openai_default_base(base_url):
        letsur_key = os.getenv("LETSUR_API_KEY") or api_key
        return LETSUR_BASE_URL, letsur_key
    return base_url, api_key
