"""Memory factory — instantiate from `memory.name` in config.

To add a baseline: add one line here + write `memory/<name>.py`.
"""
from __future__ import annotations

from typing import Any

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder

from .base import Memory


def make_memory(name: str, llm: LLM, embedder: Embedder, cfg: dict[str, Any]) -> Memory:
    """memory baseline name → Memory instance.

    Currently supported:
      - "tkg_hybrid" : our method (q_clf routing + 2-hop TKG + RAG + (E)-fallback)
      - "a_mem"     : A-MEM (arXiv 2502.12110) — note + link/evolution, dense retrieval
      - "hipporag2" : HippoRAG 2 (arXiv 2502.14802) — OpenIE KG + Personalized PageRank
      - "mem0"      : Mem0 (arXiv 2504.19413 §2.1) — fact extraction + ADD/UPDATE/DELETE, dense retrieval
      - "oracle"    : retrieval-ceiling upper bound (perfect retrieval + routing)
    """
    name = name.lower()
    if name == "tkg_hybrid":
        from .tkg_hybrid import TKGHybridMemory
        return TKGHybridMemory(llm=llm, embedder=embedder, cfg=cfg)
    # baseline memory methods (implemented from papers + official repos)
    if name in ("a_mem", "a-mem", "amem"):
        from .a_mem import AMemMemory
        return AMemMemory(llm=llm, embedder=embedder, cfg=cfg)
    if name in ("hipporag2", "hipporag_2", "hipporag-2"):
        from .hipporag2 import HippoRAG2Memory
        return HippoRAG2Memory(llm=llm, embedder=embedder, cfg=cfg)
    if name == "mem0":
        from .mem0 import Mem0Memory
        return Mem0Memory(llm=llm, embedder=embedder, cfg=cfg)
    # mem0 B-side QA-only variant (retrieve/render/QA-prompt lens experiment). commit is identical to mem0 (snapshot reuse).
    if name in ("mem0_bside", "mem0-bside"):
        from .mem0_bside import Mem0BsideMemory
        return Mem0BsideMemory(llm=llm, embedder=embedder, cfg=cfg)
    # oracle — retrieval ceiling (perfect retrieval + routing). Inject the build_oracle bundle as retrieved context.
    if name == "oracle":
        from .oracle import OracleMemory
        return OracleMemory(llm=llm, embedder=embedder, cfg=cfg)
    raise ValueError(
        f"Unknown memory: {name!r}. "
        f"Currently supported: tkg_hybrid / a_mem / hipporag2 / mem0 / mem0_bside / oracle."
    )
