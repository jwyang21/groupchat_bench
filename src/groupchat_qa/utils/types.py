"""dataclasses / common types shared across files.

Design intent — collect only *data shapes* that depend on no directory under
`src/groupchat_qa/`. ABCs (LLM, Memory, Embedder) live in their own module's `base.py`, not here.
"""
from __future__ import annotations  # avoid runtime eval of tuple[X,...] | None (Python 3.9)

from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------------------
# Quadruple — the basic unit of the TKG
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Quad:
    head: str
    relation: str
    tail: str
    start_date: str

    def as_dict(self) -> dict[str, str]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "start_date": self.start_date,
        }


# ----------------------------------------------------------------------------
# Session — one DialSim session (dialogue + metadata)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    idx: int                          # 0-based session index
    date: str                         # "Month Day, Year"
    questioner: str                   # the character asking the question
    dialogue: str                     # the full session dialogue
    partial_dialogue: str             # dialogue up to just before the question
    # TKG path kept symmetric to the RAG path — store entire/partial quads separately.
    #   - entire_quadruples : quads extracted from the full session dialogue (accumulated into
    #     KG(A) at commit). Symmetric to RAG's dialogue (entire) add.
    #   - partial_quadruples: quads extracted from the dialogue up to just before the question
    #     (used as the current session B at retrieval). Symmetric to RAG's partial_dialogue candidate add.
    #   Both are tkg_hybrid-only (baselines a_mem/hipporag2/mem0 use only session.dialogue).
    entire_quadruples: tuple[Quad, ...] | None = None
    partial_quadruples: tuple[Quad, ...] | None = None
    # Precomputed dialogue embedding (injected by data_loader from entire_session_embedding.npy).
    # Avoids recomputation at commit() (e.g. 6304 calls -> 0 with 8 LLMs in parallel).
    # numpy arrays are unhashable, so exclude from the frozen dataclass's auto __hash__/__eq__.
    dialogue_embedding: Any | None = field(default=None, hash=False, compare=False, repr=False)
    # Precomputed partial-dialogue embedding (injected by data_loader from partial_dialogues_embedding.npy).
    # Used for dense ranking when _retrieve_rag adds "the current session's partial" as a candidate.
    partial_embedding: Any | None = field(default=None, hash=False, compare=False, repr=False)


# ----------------------------------------------------------------------------
# Question — the multiple-choice question asked at the end of a session
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    text: str
    questioner: str
    options: tuple[str, ...]          # usually 5: (A) ~ (E)
    timestamp: str                    # when the question is asked (= session date)
    true_answer: str | None = None    # ground-truth label for evaluation

    # ------------------------------------------------------------------
    # Optional per-question artifacts. The `prebuilt_prompt_*` fields are unused by
    # the live-retrieval pipeline (kept `None`); `q_clf_label` / `tkg_hop` feed the
    # question-classifier routing when precomputed.
    # ------------------------------------------------------------------
    prebuilt_prompt_rag: str | None = None         # optional prebuilt RAG QA prompt
    prebuilt_prompt_tkg: str | None = None         # optional prebuilt TKG QA prompt
    prebuilt_prompt_no_history: str | None = None  # optional prebuilt no-history prompt
    q_clf_label: str | None = None                 # 'RAG' / 'TKG' (question classifier)
    tkg_hop: int | None = None                     # 1 / 2 / -1 (fanquiz)


# ----------------------------------------------------------------------------
# Item — the generalized unit returned by Memory.retrieve
# ----------------------------------------------------------------------------
@dataclass
class Item:
    text: str                         # canonical representation
    meta: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


# ----------------------------------------------------------------------------
# CallMetrics — metrics for LLM/embedder calls (collected on every call)
# ----------------------------------------------------------------------------
@dataclass
class CallMetrics:
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_llm_calls: int = 0
    num_embedding_calls: int = 0

    def merge(self, other: "CallMetrics") -> "CallMetrics":
        return CallMetrics(
            latency_sec=self.latency_sec + other.latency_sec,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            num_llm_calls=self.num_llm_calls + other.num_llm_calls,
            num_embedding_calls=self.num_embedding_calls + other.num_embedding_calls,
        )


# ----------------------------------------------------------------------------
# RetrievalResult — wrapper returned by Memory.retrieve
# ----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    items: list[Item] = field(default_factory=list)
    rendered: str = ""                # string inserted into the {triples} / {history} slot of the QA prompt
    metrics: CallMetrics = field(default_factory=CallMetrics)
    extra: dict[str, Any] = field(default_factory=dict)  # baseline-specific (q_clf, q_subj, etc.)


# ----------------------------------------------------------------------------
# GenerationRecord — record-bearing wrapper returned by LLM.generate
# ----------------------------------------------------------------------------
@dataclass
class GenerationRecord:
    text: str
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0           # includes reasoning tokens
    # thinking trace separated out by vLLM's --reasoning-parser (qwen3/gemma4).
    # "" for non-reasoning models or endpoints without a reasoning-parser.
    # Saved alongside the raw output in result files.
    reasoning_text: str = ""

    def to_metrics(self) -> CallMetrics:
        return CallMetrics(
            latency_sec=self.latency_sec,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            num_llm_calls=1,
        )
