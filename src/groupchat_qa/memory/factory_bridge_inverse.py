"""Memory factory (BRIDGE-INVERSE variant) — dedicated copy for
experiment 1 (enhanced_neutral_iter_nodia) + experiment 2 (iter_inverse_tkg_first).

This file dispatches only the name `tkg_hybrid_bridge_inverse` to the new class
`TKGHybridBridgeInverseMemory`, delegating all other names to the original
`make_memory` (behavior unchanged).

Both experiments use the same class (`TKGHybridBridgeInverseMemory`) and are
distinguished only by `retrieval_method` in config:
  - Exp 1 `enhanced_neutral_iter_nodia` : same as nbridge but drops 1-hop dialogue accumulation during iteration (quad-only).
  - Exp 2 `iter_inverse_tkg_first`       : TKG seednode-2hop first → TKG-only inverse iteration (early stop on a~d)
                                          → RAG retrieve + enhanced_full only on UNKNOWN/no-progress/max_tries.
"""
from __future__ import annotations

from typing import Any

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder

from .base import Memory
from .factory import make_memory as _orig_make_memory


def make_memory(name: str, llm: LLM, embedder: Embedder, cfg: dict[str, Any]) -> Memory:
    """memory baseline name → Memory instance (adds bridge-inverse variant dispatch).

    - "tkg_hybrid_bridge_inverse" : shared class for exp 1/exp 2 (method selected by retrieval_method).
    - all other names : delegated to the original factory.make_memory (identical behavior).
    """
    if name.lower() in ("tkg_hybrid_bridge_inverse", "tkg_hybrid-bridge-inverse", "tkg_hybrid.bridge_inverse"):
        from .tkg_hybrid_bridge_inverse import TKGHybridBridgeInverseMemory
        return TKGHybridBridgeInverseMemory(llm=llm, embedder=embedder, cfg=cfg)
    return _orig_make_memory(name, llm, embedder, cfg)
