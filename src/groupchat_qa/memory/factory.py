"""Memory factory — instantiate a memory from `memory.name` in the config.

To add a baseline: add one branch here and write `memory/<name>.py`.
"""
from __future__ import annotations

from typing import Any

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder

from .base import Memory


def make_memory(name: str, llm: LLM, embedder: Embedder, cfg: dict[str, Any]) -> Memory:
    """Map a memory name to a Memory instance.

    Supported:
      - "groupchat_mem" : the proposed method (dual-view relational memory R + raw session repository S).
      - "simple_rag"    : Simple RAG / Direct QA reference baselines (BM25 or embedding retrieval; no-history).
      - "a_mem"         : A-MEM — interlinked memory notes with dynamic linking/evolution.
      - "hipporag2"     : HippoRAG 2 — OpenIE knowledge graph + Personalized PageRank retrieval.
      - "mem0"          : Mem0 — fact extraction + add/update/delete consolidation, dense retrieval.
      - "oracle"        : oracle upper bound (perfect retrieval + routing).
    """
    name = name.lower()
    if name in ("groupchat_mem", "groupchat-mem", "groupchatmem"):
        from .groupchat_mem import GroupChatMem
        return GroupChatMem(llm=llm, embedder=embedder, cfg=cfg)
    if name in ("simple_rag", "simple-rag", "direct_qa", "direct-qa"):
        from .simple_rag import SimpleRAGMemory
        return SimpleRAGMemory(llm=llm, embedder=embedder, cfg=cfg)
    if name in ("a_mem", "a-mem", "amem"):
        from .a_mem import AMemMemory
        return AMemMemory(llm=llm, embedder=embedder, cfg=cfg)
    if name in ("hipporag2", "hipporag_2", "hipporag-2"):
        from .hipporag2 import HippoRAG2Memory
        return HippoRAG2Memory(llm=llm, embedder=embedder, cfg=cfg)
    if name == "mem0":
        from .mem0 import Mem0Memory
        return Mem0Memory(llm=llm, embedder=embedder, cfg=cfg)
    if name == "oracle":
        from .oracle import OracleMemory
        return OracleMemory(llm=llm, embedder=embedder, cfg=cfg)
    raise ValueError(
        f"Unknown memory: {name!r}. Supported: "
        f"groupchat_mem / simple_rag / a_mem / hipporag2 / mem0 / oracle."
    )
