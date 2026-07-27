"""TKG-hybrid memory (RERANKED variant) — adds the `enhanced_reranked` retrieval method.

This file is a variant of `tkg_hybrid.py` (class name TKGHybridRerankedMemory) that
preserves all behavior of the original while adding one thing:

  retrieval_method = "enhanced_reranked"
    Keeps the enhanced skeleton (RAG top-k + TKG top-k together, 1 LLM call,
    prompt=qa_enhanced.txt). But enhanced puts ~110k chars directly into final_qa_prompt,
    burying the 2-hop answer so the model replies "(E) I don't know" 53% of the time
    (diagnosis: recall is 100% but the answer is not found due to long-context /
    lost-in-the-middle. Liu et al. 2023 "Lost in the Middle", arXiv 2307.03172).
    → Take the enhanced retrieval result (same recall) and (1) path-aware rerank by the
       question's 2-hop relevance (question entity match + q_subj seed-node connection +
       cos_sim rank), (2) trim irrelevant distractors into a short context
       (compress_tkg quad + compress_rag dialogue).
    Recall is preserved (same retrieval set) while giving the model a usable context in
    which it can actually solve the 2-hop question.

All 12 original axes (1-2 retrieval method, 1-3 reranker, 1-4 num_retrieved, 1-5 RAG method,
1-6 context size, 1-9 prompt differentiation, 1-10 fallback toggle, etc.) are preserved.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder, EmbeddingCache
from groupchat_qa.utils import (
    CallMetrics,
    Item,
    Question,
    RetrievalResult,
    Session,
    classify_timestamp,
    cos_sim,
    apply_token_limit,
    count_tokens,
    extract_timestamps,
    get_clean_entity,
    is_valid_quad,
    load_pickle,
    load_prompt,
    process_answer,
    remove_brackets,
    remove_quotes,
    render_question,
    save_pickle,
    trim_text_to_tokens,
)

from .base import Memory

# Module-level q_clf cache — avoids re-invoking q_clf for the same (LLM, question)
# across ablation cells. Time-saving fix.
# key = (id(llm), question_text), value = "TKG" or "RAG"
_Q_CLF_CACHE: dict[tuple, str] = {}
_Q_CLF_CACHE_LOCK = threading.Lock()

# Process-level EmbeddingCache instance pool.
# Recreating an EmbeddingCache for the same path per ablation grid / LLM triggers a
# pickle load every time. Sharing path → instance removes the main ablation throughput
# bottleneck.
# key = Path.resolve() string
_CACHE_POOL: dict[str, "EmbeddingCache"] = {}
_CACHE_POOL_LOCK = threading.Lock()

# For extracting "Month DD, YYYY" dates from the question text. Identifies the first/second
# date of a 2-hop question to preserve the hop-1 seed quad / hop-2 answer quad via
# date-match rerank.
_RR_DATE_RE = re.compile(r"[A-Z][a-z]+ \d{1,2},? \d{4}")


def _get_or_create_cache(
    path: Path,
    model_name: str,
    dim: int,
    autoflush_every: int = 50,
    fsync_on_flush: bool = False,
) -> "EmbeddingCache":
    """Return a path-keyed singleton EmbeddingCache.

    A second (or later) call with the same (path, model_name, dim) returns the same
    instance immediately. Attempting reuse with a different (model_name, dim) raises
    ValueError — a caller bug.
    """
    key = str(path.resolve())
    with _CACHE_POOL_LOCK:
        existing = _CACHE_POOL.get(key)
        if existing is not None:
            # Same path but different model/dim → caller-side error.
            if existing.model_name != model_name or existing.dim != dim:
                raise ValueError(
                    f"[_get_or_create_cache] path={key!r} is already "
                    f"registered with model={existing.model_name!r}/dim={existing.dim}. "
                    f"Reuse request model={model_name!r}/dim={dim} does not match."
                )
            return existing
        instance = EmbeddingCache(
            path, model_name, dim,
            autoflush_every=autoflush_every,
            fsync_on_flush=fsync_on_flush,
        )
        _CACHE_POOL[key] = instance
        logging.debug(f"[_get_or_create_cache] registered new instance: {key!r}")
        return instance


# ============================================================================
# pydantic schema — structured LLM response
# ============================================================================
class QClfLabel(BaseModel):
    """q_clf classification response schema. label ∈ {"TKG","RAG"}."""
    label: Literal["TKG", "RAG"] = Field(description="TKG (knowledge graph) or RAG (raw dialogue)")


class QSubjList(BaseModel):
    """Response schema for q_subj_idf. A list of candidate entities."""
    subjects: list[str] = Field(description="question subject candidate names")


def _normalize_q_clf_label(raw: str) -> str:
    """Normalize the q_clf label to a canonical form.

    v1_2 baseline: `q_clf.upper(); "TKG" if "KG" in q_clf else "RAG"`.
    Ensures structured / free-form / precomputed labels all go through the same path.
    """
    if raw is None:
        return "RAG"
    label = remove_quotes(str(raw)).strip().upper()
    return "TKG" if "KG" in label else "RAG"


# ============================================================================
# Helpers — KG / dialogue index
# ============================================================================
@dataclass
class KGStore:
    tkg: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)
    # Future-leakage guard: maps each quad to the session_idx where it first appeared.
    #   key=(h,r,t,ts), val=min sess_idx. At retrieve time keep only quads below current_session_idx
    #   (applies the temporal no-future-leakage principle to the tkg_hybrid TKG path).
    quad_sess: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def insert_quad(self, head: str, relation: str, tail: str, ts: str,
                    sess_idx: int | None = None) -> None:
        # BUG FIX: .capitalize() lowercased everything after the first word of a multi-word entity
        # (e.g. "Lincoln High" → "Lincoln high", "Helen Willick-Bunch" → "Helen willick-bunch"),
        # corrupting the node form. Result: inaccurate entity spelling in the rendered prompt + dedup mismatch.
        # Fix: preserve the raw form. Case-insensitive matching is handled on the query side (`get_clean_entity` exact match).
        h = remove_quotes(remove_brackets(head.strip()))
        r = remove_quotes(remove_brackets(relation.strip()))
        t = remove_quotes(remove_brackets(tail.strip()))
        if h.strip().lower() == t.strip().lower():
            return
        if not is_valid_quad({"head": h, "relation": r, "tail": t, "start_date": ts}):
            return
        node = self.tkg.setdefault(h, {})
        rel = node.setdefault(r, {})
        bucket = rel.setdefault(ts, [])
        if t not in bucket:
            bucket.append(t)
        # Record sess_idx — if the same quad appears in multiple sessions, keep the earliest (min).
        #   Update even when t is already in the bucket, to reflect an earlier session. Skip if sess_idx is None (legacy).
        if sess_idx is not None:
            key = (h, r, t, ts)
            prev = self.quad_sess.get(key)
            self.quad_sess[key] = sess_idx if prev is None else min(prev, sess_idx)

    def nodes(self) -> list[str]:
        ns: set[str] = set(self.tkg)
        for h in self.tkg:
            for r in self.tkg[h]:
                for ts in self.tkg[h][r]:
                    for t in self.tkg[h][r][ts]:
                        ns.add(t)
        return list(ns)

    def nodes_until(self, current_session_idx: "int | None") -> list[str]:
        """Return only head/tail of quads that appeared in sessions below current_session_idx (future-leakage guard).

        - current_session_idx is None → the full nodes() (snapshot path — already a point-in-time-correct KG, no future quad).
        - Entities of quads without a recorded quad_sess (sess_idx None) are kept (live commit always records,
          so this only happens on the snapshot path). The comparison `sess is None or sess < idx` matches the
          quad-filter semantics of `_filter_quads_by_timestamp`, ensuring temporal alignment between q_subj options
          and retrieved quads.
        - Unlike nodes(), the head `h` is not added unconditionally; it is added only when the (h,r,t,ts) quad is
          in the past → blocks exposure of entities that come only from future-only quads.
        """
        if current_session_idx is None:
            return self.nodes()
        ns: set[str] = set()
        for h, rel_map in self.tkg.items():
            for r, ts_map in rel_map.items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        sess = self.quad_sess.get((h, r, t, ts))
                        if sess is None or sess < current_session_idx:
                            ns.add(h)
                            ns.add(t)
        return list(ns)

    def combined_until(
        self,
        current_session_idx: "int | None",
        extra_quads: "tuple | list | None" = None,
    ) -> "KGStore":
        """TKG(C) at retrieve time = A (entire quads accumulated up to the previous session) + B (current-session partial quads).

        This is exactly symmetric to the RAG path's
          "pool = (entire dialogue with s_idx < idx) + (current-session partial_dialogue)".
        Builds a new ephemeral KGStore:
          - A: copies only quads of self.tkg with sess_idx < current_session_idx (or sess_idx None = legacy/snapshot).
               Same temporal semantics as nodes_until / _filter_quads_by_timestamp.
          - B: inserts extra_quads (= current_session.partial_quadruples) with sess_idx = current_session_idx.
               Quads of future sessions (>idx) enter through no path → future-leakage guard.
        current_session_idx is None (snapshot path) → A is all of self.tkg (already point-in-time-correct) + B.

        The returned KGStore is the target C for q_subj candidate nodes, 2-hop one_hop traversal, and
        _retrieve_by_timestamp. self.tkg (permanent commit accumulation) is not mutated (read-only copy).
        """
        out = KGStore()
        # A — copy only quads with sess_idx < idx (or None)
        for h, rel_map in self.tkg.items():
            for r, ts_map in rel_map.items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        sess = self.quad_sess.get((h, r, t, ts))
                        if current_session_idx is None or sess is None or sess < current_session_idx:
                            out.insert_quad(h, r, t, ts, sess)
        # B — current-session partial quads (sess_idx = current_session_idx)
        if extra_quads:
            for q in extra_quads:
                out.insert_quad(q.head, q.relation, q.tail, q.start_date, current_session_idx)
        return out

    def one_hop(self, node: str) -> list[dict]:
        out: list[dict] = []
        if node in self.tkg:
            for r, ts_map in self.tkg[node].items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        q = {"head": node, "relation": r, "tail": t, "start_date": ts,
                             "sess_idx": self.quad_sess.get((node, r, t, ts))}
                        if is_valid_quad(q):
                            out.append(q)
        for h in self.tkg:
            for r, ts_map in self.tkg[h].items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        if t == node and h != node:
                            q = {"head": h, "relation": r, "tail": t, "start_date": ts,
                                 "sess_idx": self.quad_sess.get((h, r, t, ts))}
                            if is_valid_quad(q):
                                out.append(q)
        return out


@dataclass
class DialogueIndex:
    dialogues: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    sess_idxs: list[int] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    _bm25: Any = None

    def add(self, dialogue: str, date: str, idx: int, vec: np.ndarray | None = None) -> None:
        self.dialogues.append(dialogue)
        self.dates.append(date)
        self.sess_idxs.append(idx)
        if vec is not None:
            self.embeddings.append(np.asarray(vec, dtype=np.float32))
        self._bm25 = None

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            from nltk.tokenize import word_tokenize
            from rank_bm25 import BM25Okapi

            tokenized = [word_tokenize(d.lower()) for d in self.dialogues]
            self._bm25 = BM25Okapi(tokenized)

    def bm25_rank(self, question: str, candidate_ids: list[int]) -> list[int]:
        from nltk.tokenize import word_tokenize

        self._ensure_bm25()
        scores = self._bm25.get_scores(word_tokenize(question.lower()))
        return sorted(candidate_ids, key=lambda i: -scores[i])

    def dense_rank(self, q_vec: np.ndarray, candidate_ids: list[int]) -> list[int]:
        scored = [(i, cos_sim(q_vec, self.embeddings[i])) for i in candidate_ids]
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored]


# ============================================================================
# TKGHybridMemory — live retrieval, 12 tunable axes
# ============================================================================
class TKGHybridBridgeInverseMemory(Memory):
    """Tunable settings for ablation — keys of the config dict:

    retrieval_method   : 'tkg_only' / 'rag_only' / 'tkg_rag' / 'enhanced'
                         / 'enhanced_fallback' / 'enhanced_fallback_inverse'   (1-2)
                         enhanced = no router, concat RAG retrieval top k1 + TKG retrieval top k2.
                         enhanced_fallback = 1st tkg_rag (q_clf single path) → on (E), 2nd enhanced.
                         enhanced_fallback_inverse = 1st enhanced → on (E), 2nd q_clf single path.
    fallback           : bool — (E)-fallback (only meaningful in tkg_rag / enhanced_fallback /
                         enhanced_fallback_inverse modes)                       (1-10)
    num_retrieved_rag  : int — top-k1 of the RAG path.                         (1-4a)
    num_retrieved_tkg  : int — top-k2 of the TKG path.                         (1-4b)
    num_retrieved      : (deprecated, backward compat) — old single top-k. If set, k1=k2=num_retrieved.
                         unset or 0 → fill up to the context_size budget with no top-k cut (default).
                         positive (>0) → top-k cut at that value, then fill up to context_size.
                         negative → ValueError.
    context_size       : int — final prompt token limit (default 8000)         (1-6)

    rag_method         : 'bm25_only' / 'emb_only' / 'alternating'             (1-5)
    bm25_first         : bool — prefer BM25 in alternating
    rag_time_filter    : bool — narrow candidate sessions by question timestamp

    hop                : int — number of TKG traversal hops (default 2)
    triple_normalization: 'en' / 'raw' — which prebuilt TKG variant to use     (1-1)
                          (the actual snapshot path is a separate cfg 'tkg_snapshot')

    reranker           : 'none' / 'bge' — reranking after retrieval            (1-3)

    prompt_rag/quad/enhanced/no_history/q_clf/q_subj_idf
                       : each prompt filename (relative to assets/prompt/)      (1-9)

    tkg_snapshot       : str | None — prebuilt KG pickle path.
                         If present, overwrite with that session's KG at commit time.
                         If absent, use commit-based accumulation.

    question_format    : 'multi_structured' / 'multi_unstructured' / 'open_ended'  (1-12)

    precomputed_q_clf  : dict[int -> 'RAG'/'TKG'] (optional — saves LLM calls)
    """

    NAME = "tkg_hybrid"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        self._kg = KGStore()
        self._idx = DialogueIndex()

        # retrieval policy
        self.retrieval_method = cfg.get("retrieval_method", "tkg_rag")
        self.fallback = bool(cfg.get("fallback", True))
        # no_history: when True, skip all retrieval and use only prompt_no_history.
        self.no_history = bool(cfg.get("no_history", False))
        # Split top-k separately for the RAG path and the TKG path.
        # - num_retrieved_rag (k1): top-k1 of RAG retrieval results
        # - num_retrieved_tkg (k2): top-k2 of TKG retrieval results
        # Behavior:
        #   - retrieval_method != "enhanced": (k1==0 AND k2==0) → fill up to context_size (existing behavior)
        #                                     (k1>0 AND k2>0) → cut each path at its top-k
        #                                     mixed (only one is 0) → ValueError
        #   - retrieval_method == "enhanced": k1>0 AND k2>0 required (no router, RAG top k1 + TKG top k2)
        # Backward compat: if only the old `num_retrieved` is set, fall back to k1=k2=num_retrieved.
        _nr_legacy = cfg.get("num_retrieved", None)
        _nr_rag = cfg.get("num_retrieved_rag", None)
        _nr_tkg = cfg.get("num_retrieved_tkg", None)
        if _nr_rag is None and _nr_tkg is None and _nr_legacy is not None:
            # legacy: use the old single num_retrieved (k1=k2)
            _legacy_val = 0 if _nr_legacy is None else int(_nr_legacy)
            if _legacy_val < 0:
                raise ValueError(f"num_retrieved must be non-negative, got {_legacy_val}")
            self.num_retrieved_rag = _legacy_val
            self.num_retrieved_tkg = _legacy_val
        else:
            self.num_retrieved_rag = 0 if _nr_rag is None else int(_nr_rag)
            self.num_retrieved_tkg = 0 if _nr_tkg is None else int(_nr_tkg)
            if self.num_retrieved_rag < 0:
                raise ValueError(
                    f"num_retrieved_rag must be non-negative, got {self.num_retrieved_rag}"
                )
            if self.num_retrieved_tkg < 0:
                raise ValueError(
                    f"num_retrieved_tkg must be non-negative, got {self.num_retrieved_tkg}"
                )
        # Validation per retrieval_method
        _rm = cfg.get("retrieval_method", "tkg_rag")
        if _rm in ("enhanced", "enhanced_reranked"):
            # enhanced top-k: quad(tkg) = 100 after cos_sim rerank,
            #   dialogue(rag) = 10 after bm25-cos_sim alternating. Enhanced-specific config keys
            #   (enhanced_num_retrieved_tkg/rag) take priority, else default 100/10.
            #   The num_retrieved=0 budget policy of non-enhanced (tkg_only/rag_only/tkg_rag) is unchanged.
            # enhanced_reranked uses the same top-k policy (top-k cut on both paths). Compression is
            #   applied as a separate step after retrieval (top-k), so num_retrieved_* stays the same as enhanced
            #   (same recall = same retrieval set, only rerank+compress added).
            self.num_retrieved_tkg = int(cfg.get("enhanced_num_retrieved_tkg", 100))
            self.num_retrieved_rag = int(cfg.get("enhanced_num_retrieved_rag", 10))
            if self.num_retrieved_tkg <= 0 or self.num_retrieved_rag <= 0:
                raise ValueError(
                    f"enhanced num_retrieved must be positive, got "
                    f"tkg={self.num_retrieved_tkg}, rag={self.num_retrieved_rag}"
                )
        else:
            # otherwise: both 0 or both positive (mixing not allowed)
            _rag_zero = (self.num_retrieved_rag == 0)
            _tkg_zero = (self.num_retrieved_tkg == 0)
            if _rag_zero != _tkg_zero:
                raise ValueError(
                    f"For retrieval_method='{_rm}', num_retrieved_rag and num_retrieved_tkg "
                    f"must both be 0 (context_size budget) OR both positive, "
                    f"got rag={self.num_retrieved_rag}, tkg={self.num_retrieved_tkg}"
                )
        # Backward compat: keep the old self.num_retrieved too (for external code).
        # It is meaningful in non-enhanced cases where k1==k2. In enhanced, k1 and k2 can differ, so
        # self.num_retrieved cannot be used — but all code in this class is split into _rag/_tkg.
        self.num_retrieved = self.num_retrieved_rag if self.num_retrieved_rag == self.num_retrieved_tkg else -1
        self.context_size = int(cfg.get("context_size", 0))  # 0/negative = token limit off (unlimited)
        # If k>0 (top-k specified), the context_size (8K upper limit) is released automatically.
        #   When num_retrieved_rag/tkg are positive (both positive per validation), override context_size to unlimited →
        #   the top-k count becomes the actual retrieve limit and bypasses the 8K prompt trim (in _render_quads_in_order).
        #   The prompt size is bounded by top-k (top_tkg 250 ≈ 12K tok) and the serving max-model-len (40960) is the real cap.
        #   If k=0 (both 0), fill the pool up to the context_size (default 8000) budget as before.
        if self.num_retrieved_rag > 0 or self.num_retrieved_tkg > 0:
            self.context_size = 10 ** 9

        # RAG path
        self.rag_method = cfg.get("rag_method", "alternating")
        self.bm25_first = bool(cfg.get("bm25_first", True))
        self.rag_time_filter = bool(cfg.get("rag_time_filter", False))
        # Unified time-filter toggle — controls both the RAG path (narrow candidates by question timestamp)
        #   and the TKG path (_filter_quads_by_timestamp filters 2-hop results by question timestamp).
        #   default = False (time-filtering off by default).
        #   False → bypass both temporal filters (question timestamp does not restrict retrieval). Set True for the old behavior.
        #   Note: the TKG _retrieve_by_timestamp (the only retrieval for no-subject questions) is retrieval, not a filter,
        #   so it is kept as a separate toggle (retrieve_by_timestamp); turning it off means subject-less questions get nothing.
        self.time_filter = bool(cfg.get("time_filter", False))
        # retrieve_by_timestamp toggle — timestamp-only retrieval for no-subject questions.
        #   default True = existing behavior (retrieve quads by question timestamp for subject-less questions).
        #   False → disables that path → subject-less questions retrieve nothing even by timestamp (empty result).
        #   Part of "turn off all time-filtering" (a retrieval path, not a filter, but timestamp-based so turned OFF together).
        self.retrieve_by_timestamp = bool(cfg.get("retrieve_by_timestamp", True))

        # TKG path
        self.hop = int(cfg.get("hop", 2))
        # quad entity normalization type default = "raw".
        # The entity-consistency loss from max_candidates=200 truncation affects only partial_en, not raw.
        self.triple_normalization = cfg.get("triple_normalization", "raw")
        self.quad_format = cfg.get("quad_format", "quad")  # quad / sentence
        self.num_q_subj_candidates = int(cfg.get("num_q_subj_candidates", 3))  # TKG seed node default = 3

        # Reranker (lazy)
        self.reranker_kind = cfg.get("reranker", "none")
        self._reranker = None

        # Prompts
        # question_format → automatic prompt mapping.
        # If cfg.prompt_X is specified it takes priority; otherwise use the question_format default.
        _FMT_PROMPT_MAP = {
            "multi_structured": {
                "quad": "qa_quad.multi-unstructured.txt",
                "rag":  "qa_rag.txt",
                "enhanced": "qa_enhanced.txt",
                "no_history": "qa_no-history-eager.multi-structured.txt",
            },
            "multi_unstructured": {
                "quad": "qa_quad.multi-unstructured.txt",
                "rag":  "qa_rag.multi-unstructured.txt",
                "enhanced": "qa_enhanced.multi-unstructured.txt",
                "no_history": "qa_no-history-eager.multi-unstructured.txt",
            },
            "open_ended": {
                "quad": "qa_quad.open-ended.txt",
                "rag":  "qa_rag.open-ended.txt",
                "enhanced": "qa_enhanced.open-ended.txt",
                "no_history": "qa_no-history-eager.open-ended.txt",
            },
        }
        # CRITICAL fix: self.question_format is set later, so read directly from cfg here.
        _fmt_key = str(cfg.get("question_format", "multi_structured")).lower().replace("-", "_")
        _defaults = _FMT_PROMPT_MAP.get(_fmt_key, _FMT_PROMPT_MAP["multi_structured"])
        self._p_rag = load_prompt(cfg.get("prompt_rag") or _defaults["rag"])
        self._p_quad = load_prompt(cfg.get("prompt_quad") or _defaults["quad"])
        self._p_enhanced = load_prompt(cfg.get("prompt_enhanced") or _defaults["enhanced"])
        # The prompt for iter answers / final enhanced_full follows config prompt_enhanced (= self._p_enhanced).
        #   base (multi-structured) uses prompt_enhanced=qa_enhanced.txt, so behavior is unchanged;
        #   qformat (multi-unstructured/open-ended) uses prompt_enhanced=qa_enhanced.{...}.txt for format-specific prompts.
        #   Injects both RAG retrieved + TKG retrieved (placeholders {triples}, {history}).
        self._p_enhanced_fallback = self._p_enhanced
        # enhanced_bridge_fallback: prompt for extracting the 2-hop bridge entity (2nd-try step 1).
        #   After bridge resolve, re-query qa_enhanced.txt with the focused context of that entity (2nd-try step 2).
        self._p_bridge_extract = load_prompt("qa_bridge_extract.txt")
        # enhanced_iter_bridge_fallback: "next unresolved entity" extraction prompt for n-hop iterative bridge resolution.
        self._p_bridge_iter_extract = load_prompt("qa_bridge_iter_extract.txt")
        # enhanced_neutral_iter_fallback: neutral prompt with inductive bias removed.
        #   No mention of "multi-hop"/"bridge"/"intermediate"/"hop"/"property" — the model only decides whether
        #   there is a further named entity to look up (entity / UNKNOWN / I don't know). No multi-hop assumption injected.
        self._p_next_lookup = load_prompt("qa_next_lookup.txt")
        # nodia-specific lookup prompt — same slots as qa_next_lookup.txt but with updated IDK/UNKNOWN criteria
        #   (IDK = unsolvable + no entity to look up; UNKNOWN = an entity to inspect seems to exist but cannot be
        #   pinned down among the collected triples' entities). Kept separate from nbridge (qa_next_lookup.txt).
        self._p_next_lookup_nodia = load_prompt("qa_next_lookup_nodia.txt")
        # bridge_inverse: TKG-only lookup prompt (no {history} / [DIALOGUES]).
        #   Used only in iter_inverse_tkg_first's lookup (deciding whether to look up more entities). RAG (dialogue) not exposed.
        self._p_next_lookup_inverse = load_prompt("qa_next_lookup_inverse.txt")
        # bridge_inverse redefinition: copy of qa_next_lookup_inverse.txt — same slots (TKG-only),
        #   IDK/UNKNOWN criteria updated to match qa_next_lookup_nodia.txt. Used by iter_inverse_tkg_first.
        self._p_next_lookup_inverse_v2 = load_prompt("qa_next_lookup_inverse_v2.txt")
        # K_hop = an untuned, generous safety cap (to prevent cycles/pathological cases). Not tuned to the dataset.
        #   Actual termination is decided by the progress guard (UNKNOWN / revisiting a visited node / 0 new nodes) —
        #   the question's hop structure decides. That K is not load-bearing is shown by the record's bridge_iters
        #   distribution (mostly terminates naturally in ≤2 iterations).
        self._iter_bridge_k_hop = int(cfg.get("iter_bridge_k_hop", 6))
        # bridge-first variant: max iterations of the iter-first seednode subgraph search.
        #   If unspecified, uses iter_bridge_k_hop (same upper bound as the existing K).
        self._max_tries = int(cfg.get("max_tries", self._iter_bridge_k_hop))
        # Input token cap for the bridge-extract/final prompt. Stacking the bridge-extract template on top of the
        #   full enhanced context exceeds the serving max-model-len (65536) in late sessions (huge KG), causing
        #   400 BadRequest (input 49153 + output 16384 = 65537) and 4 unscored sessions (measured on a 2-hop bridge run).
        #   Cap the input at this value to guarantee input + output (thinking budget) ≤ max-model-len.
        #   default 40000 (gpt-tok). The server limit is qwen-token 49152 (=65536−16384 output), but the cap is
        #   computed in gpt-token (tiktoken) → assuming qwen expansion up to ~1.15x, 40000×1.15=46000<49152 is safe.
        #   Measured: a 49000 cap becomes 49153 in qwen tokens (1 token over → 400); 44000 also risks 49280 worst-case;
        #   so 40000 is chosen. Independent of context_size (this is over-length protection only).
        self._bridge_input_token_cap = int(cfg.get("bridge_input_token_cap", 40000))
        self._p_no_hist = load_prompt(cfg.get("prompt_no_history") or _defaults["no_history"])
        self._p_q_clf = load_prompt(cfg.get("prompt_q_clf", "q_clf.txt"))
        self._p_q_subj_idf = load_prompt(cfg.get("prompt_q_subj_idf", "q_subj_idf.txt"))
        # q_subj 2-pass stage2 prompt — after merging the stage1 batch candidates,
        #   feed the merged set back in as options and select the final k only from those options (= a subset of the current TKG nodes).
        self._p_q_subj_select = load_prompt(cfg.get("prompt_q_subj_select", "q_subj_idf_select.txt"))

        # TKG snapshot (optional — use a prebuilt KG)
        snap_path = cfg.get("tkg_snapshot")
        self._tkg_snapshot = load_pickle(snap_path) if snap_path else None

        # question format
        self.question_format = cfg.get("question_format", "multi_structured")

        # misc
        self.delimiter = cfg.get("delimiter", "=" * 40)
        self.precomputed_q_clf: dict[int, str] = cfg.get("precomputed_q_clf") or {}
        # precomputed q_subj (per-llm q_subj.json) — {question_idx: [candidate,...]}.
        #   If absent, empty dict → live _identify_q_subj (same pattern as q_clf).
        self.precomputed_q_subj: dict[int, list[str]] = cfg.get("precomputed_q_subj") or {}
        # q_subj options batch size (100). Split nodes into batch_size chunks per call → merge.
        self.q_subj_batch_size: int = int(cfg.get("q_subj_batch_size", 100))
        # q_subj stage1 batch parallelism. default 1 = sequential (result unchanged).
        #   >1 → call the LLM concurrently over batches with an index-order-preserving ThreadPool
        #   (result bit-identical → fair gemma comparison). For precompute speedup — not applied to ablation, which
        #   early-returns on precomputed_q_subj. Total concurrent LLM calls = (question parallel max-workers) × (this value),
        #   so tune to endpoint capacity at launch.
        self.q_subj_batch_workers: int = max(1, int(cfg.get("q_subj_batch_workers", 1)))

        # ====================================================================
        # enhanced_reranked: path-aware rerank + context compression hyperparameters.
        #   enhanced puts the retrieval (top-k) results (num_retrieved_tkg quads + num_retrieved_rag dialogues)
        #   directly into final_qa_prompt (~110k chars, context_size=0=unlimited). enhanced_reranked takes those
        #   results and (1) does a path-aware rerank → (2) keeps only the top compress_* items to compress into a short context.
        #   Recall is preserved since the retrieval set is identical (compression only reduces the count put into the prompt).
        #   - rr_compress_tkg : number of top quads to put in the final prompt after rerank (default 80).
        #   - rr_compress_rag : number of top dialogues to put in the final prompt after rerank (default 12).
        #   - rr_entity_boost : entity-overlap weight of the path-aware rerank (default 0.5).
        #       boosted_score = (1 - position_norm) + rr_entity_boost * entity_overlap_score
        #       (position_norm = [0,1] normalization of the retrieval rank. i.e. take the existing cos_sim/alternating
        #        rank as base, then pull items overlapping with question entity / q_subj seed-node higher.)
        #   - rr_seed_boost   : extra weight for quads containing a q_subj seed-node (2-hop anchor) (default 0.5).
        self.rr_compress_tkg: int = int(cfg.get("rr_compress_tkg", 80))
        self.rr_compress_rag: int = int(cfg.get("rr_compress_rag", 12))
        self.rr_entity_boost: float = float(cfg.get("rr_entity_boost", 0.5))
        self.rr_seed_boost: float = float(cfg.get("rr_seed_boost", 0.5))
        # 2-hop path-aware bridge boost. Core weakness of enhanced_reranked:
        #   in a 2-hop question ("Julie went out with X on DATE → how old was X"), the hop-1 bridge entity (X=Nate)
        #   is a neighbor of the q_subj seed (Julie) and does not surface via token-overlap rerank → the hop-2 answer quad
        #   ({Nate, age, 26, ...}) is cut in top-k compression → (E) wrong answer. (Measured 19 hop2 losses, e.g. sess174.)
        #   fix: extract the bridge entity = the *opposite endpoint* of a retrieved quad whose head/tail is a seed entity,
        #   then strongly boost quads whose head/tail is that bridge entity (preserve the hop-2 target).
        self.rr_bridge_boost: float = float(cfg.get("rr_bridge_boost", 1.0))
        # Boost when a quad's start_date matches the question date (the 2-hop answer is usually on the question's second date).
        self.rr_date_boost: float = float(cfg.get("rr_date_boost", 0.7))
        # bridge entity AND date-match together = a strong identifying signal for the hop-2 answer quad → highest priority to preserve.
        self.rr_combo_boost: float = float(cfg.get("rr_combo_boost", 1.5))
        if self.rr_compress_tkg <= 0 or self.rr_compress_rag <= 0:
            raise ValueError(
                f"enhanced_reranked compress sizes must be positive, got "
                f"tkg={self.rr_compress_tkg}, rag={self.rr_compress_rag}"
            )

        # Set question_embedding_type before the q_embed_path block (which references it); placing it later caused AttributeError.
        # Question embedding type axis (separate from quad embedding).
        # temporal     = embed the whole question
        # non_temporal = strip_question_timestamp(q) — remove timestamp + 1 time-preposition word before embedding
        # default = "temporal".
        _qet_q = cfg.get("question_embedding_type", "temporal")
        if _qet_q not in ("temporal", "non_temporal"):
            raise ValueError(
                f"question_embedding_type invalid: {_qet_q!r}; expected 'temporal' or 'non_temporal'"
            )
        self.question_embedding_type: str = _qet_q

        # Precomputed question embeddings (automatic lookup — shared across all ablation cells).
        # path = {dataset_root}/{data_type}/precomputed/per-trial/t{trial}/question_embeddings_{model_alias}.npz
        # embed_one fallback if absent.
        self._q_embed_map: dict[str, np.ndarray] = {}
        q_embed_path = cfg.get("precomputed_q_embed_path")  # explicit path takes priority
        if not q_embed_path:
            ds_root = cfg.get("dataset_root")
            data_type = cfg.get("data_type")
            trial = cfg.get("trial")
            model = cfg.get("embedder_model") or getattr(embedder, "model", None) or "text-embedding-3-small"
            if ds_root and data_type is not None and trial is not None:
                alias = str(model).replace("/", "_").replace(":", "_")
                # question_embedding_type suffix path
                # (build_embeddings.py stores separate _temporal / _non_temporal npz).
                candidate = (
                    f"{ds_root}/{data_type}/precomputed/per-trial/t{int(trial)}/"
                    f"question_embeddings_{alias}_{self.question_embedding_type}.npz"
                )
                if Path(candidate).exists():
                    q_embed_path = candidate
        if q_embed_path:
            try:
                data = np.load(q_embed_path, allow_pickle=True)
                qs = list(data["questions"])
                embs = data["embeddings"]
                self._q_embed_map = {str(q): embs[i] for i, q in enumerate(qs) if q is not None}
            except (OSError, ValueError, KeyError, EOFError) as e:
                # On load failure, fall back (embed_one). Keep empty dict.
                logging.warning("[TKGHybridMemory.__init__] q_embed_path load failed → fallback: %s", e)
                self._q_embed_map = {}

        # Precomputed quadruple embeddings (injected by the entry point via cfg["quad_embeddings"]).
        # Format: {quad_string_key: vector}
        #   temporal     = f"{head} {relation} {tail} on {start_date}"
        #   non_temporal = f"{head} {relation} {tail}"
        # Empty dict if absent (the cos_sim re-rank in the retrieve step becomes a no-op automatically).
        self._quad_emb_map: dict[str, Any] = cfg.get("quad_embeddings") or {}

        # cos_sim re-rank toggle:
        # True  → re-sort retrieved quads by cosine similarity to the question, descending.
        # False → skip re-sorting (keep BM25/TKG order). default = False.
        self.cos_sim_rerank: bool = bool(cfg.get("cos_sim_rerank", False))

        # quad embedding type toggle.
        # "temporal"     → key = f"{h} {r} {t} on {sd}"  (includes start_date, build filename _temporal)
        # "non_temporal" → key = f"{h} {r} {t}"          (build filename _non_temporal)
        # default = "non_temporal". simulate yaml overrides to "temporal".
        # Can be added as an ablation grid axis (config's grid memory.quad_embedding_type).
        _qet = cfg.get("quad_embedding_type", "non_temporal")
        if _qet not in ("temporal", "non_temporal"):
            raise ValueError(f"quad_embedding_type invalid: {_qet!r}; expected 'temporal' or 'non_temporal'")
        self.quad_embedding_type: str = _qet
        # question_embedding_type is set earlier (must run before the q_embed_path block references it).

        # ====================================================================
        # 4 EmbeddingCache instances.
        # None if the cfg key is absent — live embed fallback (existing behavior).
        # ====================================================================
        self._init_embedding_caches(cfg, embedder)

    # ====================================================================
    # Embedding cache init — 4 kinds (entire_dialogue/partial_dialogue/question/quad)
    # Required cfg keys: dataset_root, data_type, trial — cache disabled if absent (live embed fallback).
    # The quad cache additionally needs llm_name + ie_type + normalization.
    # ====================================================================
    def _init_embedding_caches(self, cfg: dict[str, Any], embedder: "Embedder") -> None:
        # Use module-level logging / Path (no function-local import).
        self._entire_dialogue_cache: EmbeddingCache | None = None
        self._partial_dialogue_cache: EmbeddingCache | None = None
        self._question_cache: EmbeddingCache | None = None
        self._partial_quad_cache: EmbeddingCache | None = None
        self._entire_quad_cache: EmbeddingCache | None = None

        ds_root = cfg.get("dataset_root")
        dt = cfg.get("data_type")
        trial = cfg.get("trial")

        # Prevent a bug where dt=0 wrongly passes — isinstance + valid-value check.
        if not isinstance(dt, str) or dt not in {"normal", "newname", "shuffled"}:
            # Required key absent or invalid dt — disable cache. live embed fallback.
            return
        if not (ds_root and trial is not None):
            return

        ds_root = Path(ds_root)
        dt = str(dt)
        trial = int(trial)

        # Use _get_or_create_cache to build EmbeddingCache.
        # Reuse the instance for the same path within the process (improves ablation throughput).
        # autoflush_every=50 + fsync_on_flush=False — eases runtime hot-path latency
        # (the build script fills the cache in advance, so runtime misses are rare).
        _kw = dict(autoflush_every=50, fsync_on_flush=False)

        # Catch specific exceptions only, and emit an explicit warning instead of pass.
        try:
            # entire dialogue — once per ds × dt (trial-independent)
            self._entire_dialogue_cache = _get_or_create_cache(
                ds_root / dt / "entire_session_embedding_cache.pickle",
                embedder.name, embedder.dim, **_kw,
            )
            # partial dialogue — ds × dt × trial
            self._partial_dialogue_cache = _get_or_create_cache(
                ds_root / dt / "precomputed" / "per-trial" / f"t{trial}"
                       / "partial_dialogues_embedding_cache.pickle",
                embedder.name, embedder.dim, **_kw,
            )
            # question — ds × dt × trial × type (temporal/non_temporal)
            # question_embedding_type suffix.
            _q_suffix = self.question_embedding_type
            self._question_cache = _get_or_create_cache(
                ds_root / dt / "precomputed" / "per-trial" / f"t{trial}"
                       / f"question_embeddings_{_q_suffix}_cache.pickle",
                embedder.name, embedder.dim, **_kw,
            )
        except (OSError, ValueError, ImportError, RuntimeError) as e:
            logging.warning(f"[tkg_hybrid] dialogue/question cache init failed: {e}. live embed fallback.")
            self._entire_dialogue_cache = None
            self._partial_dialogue_cache = None
            self._question_cache = None

        # quad cache — per LLM × ie × normalization (only when LLM info is present)
        llm_name = cfg.get("llm_name")
        ie_type = cfg.get("ie_type", "openie")
        norm = cfg.get("normalization", "raw")
        # cache_budget segment branch.
        #   data_loader reads the quad json under `bNNNNN` in the LLM folder, so the quad's embedding cache
        #   must be read under the same segment for path consistency (absent → no segment = existing behavior).
        from groupchat_qa.pipeline.data_loader import budget_segment as _budget_segment
        _bseg = _budget_segment(cfg.get("cache_budget"))
        if llm_name:
            # Add embedding_type suffix to the quad cache path
            # (build_embeddings.py stores separate _temporal / _non_temporal pickles).
            suffix = self.quad_embedding_type  # "temporal" or "non_temporal"
            try:
                _entire_quad_dir = ds_root / dt / "precomputed" / "common" / ie_type / llm_name
                if _bseg:
                    _entire_quad_dir = _entire_quad_dir / _bseg
                self._entire_quad_cache = _get_or_create_cache(
                    _entire_quad_dir
                           / f"entire_session_quadruples_embedding_{norm}_{suffix}.pickle",
                    embedder.name, embedder.dim, **_kw,
                )
                _partial_quad_dir = (ds_root / dt / "precomputed" / "per-trial" / f"t{trial}"
                                     / "per-llm" / llm_name)
                if _bseg:
                    _partial_quad_dir = _partial_quad_dir / _bseg
                self._partial_quad_cache = _get_or_create_cache(
                    _partial_quad_dir / ie_type
                           / f"partial_session_quadruples_embedding_{norm}_{suffix}.pickle",
                    embedder.name, embedder.dim, **_kw,
                )
            except (OSError, ValueError, ImportError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] quad cache init failed: {e}. live embed fallback.")
                self._entire_quad_cache = None
                self._partial_quad_cache = None

        # Lazy-prime the cache with the in-memory map's precomputed values.
        # If _q_embed_map (npz) is filled, also register in the question cache so lookup goes through a single cache.get path.
        # The quad map can be large, so it is set at first lookup (handled inside _cached_quad_embed below).
        if self._question_cache is not None:
            for _text, _vec in self._q_embed_map.items():
                if self._question_cache.get(_text) is None:
                    try:
                        self._question_cache.set(_text, _vec)
                    except (OSError, ValueError, RuntimeError):
                        pass

    # ------------------------------------------------------------------
    # Common handling of empty text.
    # Return a uniform unit vec instead of a zero-vec → avoids cos_sim ZeroDivisionError.
    # ------------------------------------------------------------------
    def _empty_vec(self) -> np.ndarray:
        dim = self.embedder.dim
        v = np.ones(dim, dtype=np.float32)
        v /= np.sqrt(dim)
        return v

    def _cached_entire_dialogue_embed(self, text: str) -> np.ndarray:
        """entire dialogue embedding used by commit().

        Separates the partial pool of _rag_order_adhoc from the cache-write path.
        lookup: entire → partial chain. On miss, set in the entire cache.
        Convert live embed result to np.asarray(float32) before returning.
        """
        if not text:
            return self._empty_vec()
        # entire first, on miss check partial
        for c in (self._entire_dialogue_cache, self._partial_dialogue_cache):
            if c is None:
                continue
            v = c.get(text)
            if v is not None:
                return v
        # miss → live embed + set in entire cache
        vec = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        if self._entire_dialogue_cache is not None:
            try:
                self._entire_dialogue_cache.set(text, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] entire_dialogue cache set failed: {e}")
        return vec

    def _cached_partial_dialogue_embed(self, text: str) -> np.ndarray:
        """partial dialogue embedding used by the pool dialogue of _rag_order_adhoc.

        Fully separated from commit()'s entire-cache write path → prevents trial contamination.
        lookup: partial → entire chain. On miss, set in the partial cache.
        Convert live embed result to np.asarray(float32) before returning.
        """
        if not text:
            return self._empty_vec()
        # partial first, on miss check entire
        for c in (self._partial_dialogue_cache, self._entire_dialogue_cache):
            if c is None:
                continue
            v = c.get(text)
            if v is not None:
                return v
        # miss → live embed + set in partial cache
        vec = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        if self._partial_dialogue_cache is not None:
            try:
                self._partial_dialogue_cache.set(text, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] partial_dialogue cache set failed: {e}")
        return vec

    def _cached_question_embed(self, text: str) -> np.ndarray:
        """question embedding — _q_embed_map (npz) first, cache as secondary, on miss live + cache.

        When question_embedding_type=non_temporal, transform text with strip_question_timestamp before
        lookup/embed. When temporal, use text as-is.
        The cache key is the transformed text → different cache files per temporal/non_temporal.

        Convert live embed result to np.asarray(float32).
        num_embedding_calls is a combined count including cache hits (not the OpenAI billable count).
        """
        if not text:
            return self._empty_vec()
        if self.question_embedding_type == "non_temporal":
            from groupchat_qa.utils.timestamp_utils import strip_question_timestamp
            text = strip_question_timestamp(text)
            if not text:
                return self._empty_vec()
        v = self._q_embed_map.get(text)
        if v is not None:
            return v
        if self._question_cache is not None:
            v = self._question_cache.get(text)
            if v is not None:
                return v
        vec = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        if self._question_cache is not None:
            try:
                self._question_cache.set(text, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] question cache set failed: {e}")
        return vec

    def _cached_quad_embed(self, key: str) -> np.ndarray:
        """quad embedding — _quad_emb_map (build script output) first, cache as secondary, on miss live + cache.

        On _quad_emb_map hit, also lazy-prime the partial cache (subsequent lookups go through a single cache.get path).
        Convert live embed result to np.asarray(float32).
        """
        if not key:
            return self._empty_vec()
        v = self._quad_emb_map.get(key)
        if v is not None:
            # lazy prime: on quad map hit, also register in the partial cache (subsequent lookups converge to the cache-hit path)
            if self._partial_quad_cache is not None and self._partial_quad_cache.get(key) is None:
                try:
                    self._partial_quad_cache.set(key, np.asarray(v, dtype=np.float32))
                except (OSError, ValueError, RuntimeError):
                    pass
            return v
        for c in (self._entire_quad_cache, self._partial_quad_cache):
            if c is None:
                continue
            v = c.get(key)
            if v is not None:
                return v
        vec = np.asarray(self.embedder.embed_one(key), dtype=np.float32)
        # Store the runtime live embed result in the partial cache (the build script manages entire).
        if self._partial_quad_cache is not None:
            try:
                self._partial_quad_cache.set(key, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] quad cache set failed: {e}")
        return vec

    @property
    def reranker(self):
        if self.reranker_kind == "none":
            return None
        if self._reranker is None:
            from groupchat_qa.utils.reranker import BGEReranker

            self._reranker = BGEReranker()
        return self._reranker

    # ====================================================================
    # commit
    # ====================================================================
    def commit(self, session: Session) -> CallMetrics:
        metrics = CallMetrics()
        # 1) update KG
        if self._tkg_snapshot is not None:
            # overwrite with the prebuilt KG at that session's point in time
            snap_tkg = self._tkg_snapshot.get("tkg", {})
            if session.idx in snap_tkg:
                self._kg.tkg = snap_tkg[session.idx]
        elif session.entire_quadruples:
            # commit accumulates the session's *entire* quads into KG(A).
            #   Symmetric to RAG's dialogue(entire) add. Sequential evolve: after session s ends, its entire quads
            #   become the A of the next session. (At retrieve time, the current session's B=partial is merged by retrieve, not commit.)
            for q in session.entire_quadruples:
                # Pass session.idx — records sess_idx to block future quads at retrieve time.
                self._kg.insert_quad(q.head, q.relation, q.tail, q.start_date, session.idx)

        # 2) add to the dialogue index (for RAG retrieval)
        if session.dialogue:
            # Avoid a re-call if a precomputed embedding exists (injected in advance by data_loader).
            # Otherwise (e.g. no dialogue or no emb npy), fall back to a live call.
            if session.dialogue_embedding is not None:
                vec = session.dialogue_embedding
            else:
                t0 = time.perf_counter()
                # commit() uses the entire-dialogue cache path only.
                vec = self._cached_entire_dialogue_embed(session.dialogue)
                metrics.latency_sec += time.perf_counter() - t0
                # num_embedding_calls is a combined count including cache hits (not the OpenAI billable count).
            self._idx.add(session.dialogue, session.date, session.idx, vec)
        return metrics

    # ====================================================================
    # retrieve — dispatcher
    # current_session: the Session object of the current session. Needed so _retrieve_rag can add
    # partial_dialogue + partial_embedding to the candidates. None → entire only (legacy).
    # ====================================================================
    def retrieve(self, question: Question, current_session_idx: int,
                 current_session: "Session | None" = None) -> RetrievalResult:
        m = self.retrieval_method
        if m in ("tkg_only", "iter_inverse_tkg_first"):
            # bridge_inverse: iter_inverse_tkg_first's 1st try = seednode 2-hop TKG-only (no RAG retrieve).
            #   Only on (E) does answer() run the TKG-only inverse iter, and only in the final fallback
            #   (UNKNOWN / no progress / max_tries) does it retrieve RAG and finish with enhanced_full.
            return self._retrieve_tkg(question, current_session_idx, route="TKG",
                                       current_session=current_session)
        if m == "rag_only":
            return self._retrieve_rag(question, current_session_idx, route="RAG",
                                       current_session=current_session)
        if m in ("tkg_rag", "enhanced_fallback"):
            # enhanced_fallback's 1st try is exactly the same as tkg_rag
            #   (q_clf routing → RAG or TKG). Only on fallback (the (E) branch of answer())
            #   does the 2nd try inject both RAG+TKG results into qa_enhanced.txt (see answer() below).
            q_clf = self._classify_question(question, current_session_idx)
            route = "RAG" if q_clf == "RAG" else "TKG"
            return (
                self._retrieve_rag(question, current_session_idx, route=route,
                                    current_session=current_session)
                if route == "RAG"
                else self._retrieve_tkg(question, current_session_idx, route=route,
                                         current_session=current_session)
            )
        if m in ("enhanced", "enhanced_bridge_fallback", "enhanced_iter_bridge_fallback",
                 "enhanced_neutral_iter_fallback", "enhanced_neutral_iter_fallback_unkE",
                 "iter_first_enhanced_fallback", "enhanced_neutral_iter_nodia"):
            # enhanced_neutral_iter_nodia = same 1st try (enhanced full) and final (enhanced full) as
            #   nbridge (enhanced_neutral_iter_fallback). The difference is dropping the 1-hop dialogue accumulation
            #   during iter (subgraph answers are quad-only). retrieve is the same as enhanced (full RAG+TKG).
            # iter_first_enhanced_fallback = the order-reversal of neutral.
            #   retrieve itself is the same as enhanced (secures full RAG+TKG context → rag_rendered/tkg_rendered).
            #   answer() skips the enhanced-first generate and runs the seednode subgraph iter first;
            #   if a~d appears it terminates, else after max_tries with (E) it finishes with one enhanced full on that context.
            # enhanced_neutral_iter_fallback_unkE = an A/B variant of neutral:
            #   on UNKNOWN(STOP), terminate immediately with (E) instead of re-querying enhanced full. Everything else identical.
            # enhanced_neutral_iter_fallback = the inductive-bias-removed version of iter (neutral prompt).
            #   Same 1st try (enhanced full). On (E), answer() runs the neutral loop (the model decides progress).
            # enhanced_iter_bridge_fallback = the n-hop general form of bridge (iterative bridge resolution).
            #   1st try is the same as enhanced (full context). Only on (E) does answer() run the iterative loop.
            #   enhanced_bridge_fallback's 1st try is exactly the same as enhanced
            #   (full context, qa_enhanced.txt, 1 call). Only on (E) "I don't know" (answer() branch) does the
            #   2nd try extract a bridge entity → re-query on that entity's one-hop focused context.
            #   (Not a compression route — it directly attacks 2-hop linking failure: resolve hop1 bridge, then focus hop2.)
            return self._retrieve_enhanced(question, current_session_idx,
                                            current_session=current_session)
        if m in ("enhanced_reranked", "enhanced_reranked_fallback"):
            # retrieve with the same enhanced skeleton (same recall) → path-aware rerank + compression.
            #   enhanced_reranked_fallback's 1st try is identical (compressed short context). Only on (E)
            #   (answer() branch) does the 2nd try re-query with enhanced (uncompressed full context).
            return self._retrieve_enhanced_reranked(question, current_session_idx,
                                                    current_session=current_session)
        if m == "enhanced_fallback_inverse":
            # The exact opposite order of enhanced_fallback.
            #   1st try = always enhanced (both RAG k_rag + TKG k_tkg, prompt=qa_enhanced.txt).
            #   Only on (E) (answer() branch) does the 2nd try re-query on a single path per the "original q_clf" value.
            #   → precompute q_clf here and store it in extra["q_clf"] (used by answer()'s 2nd branch).
            r_enh = self._retrieve_enhanced(question, current_session_idx,
                                            current_session=current_session)
            q_clf = self._classify_question(question, current_session_idx,
                                            metrics=r_enh.metrics)
            r_enh.extra["q_clf"] = "RAG" if q_clf == "RAG" else "TKG"
            return r_enh
        raise ValueError(f"Unknown retrieval_method: {m!r}")

    # ====================================================================
    # build_qa_prompt — branch by retrieval_method × question_format
    # ====================================================================
    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        # creates properly formatted question: e.g.,) "Joey: So, what does Mrs. bing do?\n\t(A) Butt double\n\t(B) ..."
        q_str = render_question(question, self.question_format)
        route = retrieval.extra.get("route", "RAG")

        # 1) no-history shortcut — only for the intentional no-history/prior-knowledge ablation (route == 'NO_HISTORY').
        #   Removed the old condition `or (not retrieval.items and not retrieval.rendered.strip())` that treated a
        #   normal method's "retrieval happens to be completely empty" case as no_hist (qa_no-history-eager.*, for
        #   prior-knowledge evaluation). Even when retrieval is empty, the per-method branches below (enhanced/RAG/TKG)
        #   handle it normally with {triples}=No relevant triple!, {history}=No relevant history!.
        #   (no-history-eager is a prompt dedicated to the "prior-knowledge ceiling" experiment that uses no retrieval at all,
        #    so it must not be used for the empty case of a normal retrieval method.)
        if route == "NO_HISTORY":
            return self._p_no_hist.replace("{question}", q_str)

        # 2) enhanced / enhanced_reranked always use qa_enhanced.txt (RAG + TKG together).
        #   enhanced_reranked uses the same prompt (qa_enhanced.txt) — the only difference is that extra's
        #   rag_rendered/tkg_rendered is a path-aware reranked + compressed short context.
        #   enhanced_reranked_fallback's 1st try (route==ENHANCED_RERANKED) is handled by this same branch
        #   (uses the compressed rag_rendered/tkg_rendered). The 2nd try (route==ENHANCED, full enhanced) is
        #   handled by the (2c) branch below.
        #   enhanced_bridge_fallback's 1st try (route==ENHANCED) is also this same branch (full enhanced).
        #   The 2nd try (bridge-focused) is built directly by answer() as prompt2 (does not enter this branch).
        if self.retrieval_method in ("enhanced", "enhanced_reranked",
                                     "enhanced_reranked_fallback",
                                     "enhanced_bridge_fallback",
                                     "enhanced_iter_bridge_fallback",
                                     "enhanced_neutral_iter_fallback",
                                     "enhanced_neutral_iter_fallback_unkE",
                                     "enhanced_neutral_iter_nodia"):
            rag_rendered = retrieval.extra.get("rag_rendered") or "No relevant history!"
            tkg_rendered = retrieval.extra.get("tkg_rendered") or "No relevant triple!"
            return (
                self._p_enhanced.replace("{question}", q_str)
                .replace("{triples}", tkg_rendered)
                .replace("{history}", rag_rendered)
            )

        # 2b) enhanced_fallback_inverse's 1st try (route == "ENHANCED") = always qa_enhanced.txt
        #   (RAG + TKG together). prompt is self._p_enhanced_fallback (=qa_enhanced.txt, regardless of override).
        #   The 2nd try (single path) has route RAG/TKG, so it is handled by the (3) branch below.
        if self.retrieval_method == "enhanced_fallback_inverse" and route == "ENHANCED":
            rag_rendered = retrieval.extra.get("rag_rendered") or "No relevant history!"
            tkg_rendered = retrieval.extra.get("tkg_rendered") or "No relevant triple!"
            return (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", tkg_rendered)
                .replace("{history}", rag_rendered)
            )

        # 3) RAG prompt or TKG prompt depending on route
        if route == "RAG":
            return self._p_rag.replace("{question}", q_str).replace(
                "{history}", retrieval.rendered or "No relevant history!"
            )
        return self._p_quad.replace("{question}", q_str).replace(
            "{triples}", retrieval.rendered or "No relevant triple!"
        )

    # ====================================================================
    # answer — supports the (E)-fallback toggle
    # ====================================================================
    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        # When no_history=True, skip retrieval and use only the no-history prompt.
        if self.no_history:
            # create properly formatted question: e.g., "Joey: So, what does Mrs. bing do?\n\t(A) Butt double\n\t(B) ..." for multi-structured setting
            q_str = render_question(question, self.question_format)
            prompt = self._p_no_hist.replace("{question}", q_str)
            prompt = apply_token_limit(prompt, self.context_size)
            gen = self.llm.generate(prompt)
            r0 = RetrievalResult(
                items=[], rendered="",
                extra={
                    "route": "NO_HISTORY",
                    "qa_raw_text": gen.text,
                    "qa_reasoning_text": getattr(gen, "reasoning_text", ""),
                    "qa_final_prompt": prompt,
                    "qa_retrieval_rendered": "",
                },
                metrics=gen.to_metrics(),
            )
            return gen.text, r0, gen.to_metrics()
        # iter_first_enhanced_fallback: skip the enhanced-first generate and run the seednode subgraph iter first.
        #   retrieve is the same as enhanced (secures full context) but does not do the 1st answer generate
        #   (the exact opposite order of neutral). If a~d appears in iter it terminates; else after max_tries with (E),
        #   finish with one enhanced full on that context. early return.
        if self.retrieval_method == "iter_first_enhanced_fallback":
            r1 = self.retrieve(question, current_session_idx, current_session=current_session)
            return self._iter_first_enhanced_fallback(
                question, current_session_idx, r1, r1.metrics, "ITER_FIRST",
                current_session=current_session,
            )
        # 1st retrieve + generate
        r1 = self.retrieve(question, current_session_idx, current_session=current_session)
        prompt1 = apply_token_limit(self.build_qa_prompt(question, r1), self.context_size)
        gen1 = self.llm.generate(prompt1)
        metrics = r1.metrics.merge(gen1.to_metrics())
        # Preserve the last QA generation's raw + reasoning + final prompt + retrieval rendering
        # — used by main.simulate's record (for post-hoc retrieval/prompt analysis).
        r1.extra["qa_raw_text"] = gen1.text
        r1.extra["qa_reasoning_text"] = getattr(gen1, "reasoning_text", "")
        r1.extra["qa_final_prompt"] = prompt1
        r1.extra["qa_retrieval_rendered"] = r1.rendered

        # Terminate if fallback is off, the mode does not support fallback, or the 1st answer is not (E)
        if (
            not self.fallback
            or self.retrieval_method not in ("tkg_rag", "enhanced_fallback",
                                             "enhanced_fallback_inverse",
                                             "enhanced_reranked_fallback",
                                             "enhanced_bridge_fallback",
                                             "enhanced_iter_bridge_fallback",
                                             "enhanced_neutral_iter_fallback",
                                             "enhanced_neutral_iter_fallback_unkE",
                                             "enhanced_neutral_iter_nodia",
                                             "iter_inverse_tkg_first")
            or process_answer(gen1.text) != "(E)"
        ):
            return gen1.text, r1, metrics

        first_route = r1.extra.get("route", "RAG")

        if self.retrieval_method == "enhanced_fallback_inverse":
            # The exact opposite order of enhanced_fallback.
            #   1st try = enhanced (already run above). (E) → 2nd try is a single path per the "original q_clf" value:
            #   q_clf==TKG → TKG-path (quads only, default tkg prompt),
            #   q_clf==RAG → RAG-path (dialogue only, default rag prompt). The 2nd output is final.
            q_clf = r1.extra.get("q_clf", "RAG")  # precomputed and stored in retrieve().
            if q_clf == "TKG":
                r2 = self._retrieve_tkg(question, current_session_idx, route="TKG",
                                        current_session=current_session)
            else:
                r2 = self._retrieve_rag(question, current_session_idx, route="RAG",
                                        current_session=current_session)
            r2.extra["route"] = "ENHANCED_FALLBACK_INVERSE"
            # build_qa_prompt cannot pick the RAG/TKG prompt by route (route was overwritten above),
            #   so select the single-path prompt directly by q_clf and render.
            q_str = render_question(question, self.question_format)
            if q_clf == "TKG":
                prompt2 = self._p_quad.replace("{question}", q_str).replace(
                    "{triples}", r2.rendered or "No relevant triple!"
                )
            else:
                prompt2 = self._p_rag.replace("{question}", q_str).replace(
                    "{history}", r2.rendered or "No relevant history!"
                )
            prompt2 = apply_token_limit(prompt2, self.context_size)
            r2.extra["fallback_from"] = q_clf
        elif self.retrieval_method == "enhanced_fallback":
            # (E) → in the 2nd try, inject both RAG-path (k_rag=40) + TKG-path (k_tkg=350) results into
            #   qa_enhanced.txt and re-query. Retrieve both paths together as in enhanced mode,
            #   but the prompt is always self._p_enhanced_fallback (=qa_enhanced.txt).
            rag2 = self._retrieve_rag(question, current_session_idx, route="RAG_in_enhanced",
                                      current_session=current_session)
            tkg2 = self._retrieve_tkg(question, current_session_idx, route="TKG_in_enhanced",
                                      current_session=current_session)
            r2 = RetrievalResult(
                items=rag2.items + tkg2.items,
                rendered="",  # rendered directly with the enhanced prompt below
                extra={
                    "route": "ENHANCED_FALLBACK",
                    "fallback_from": first_route,
                    "rag_rendered": rag2.rendered,
                    "tkg_rendered": tkg2.rendered,
                    "rag_items": rag2.items,
                    "tkg_items": tkg2.items,
                    # For retrieval recall analysis (stored separately per path, as in enhanced).
                    "retrieval_b_rag": rag2.extra.get("retrieval_b", []),
                    "retrieval_c_rag": rag2.extra.get("retrieval_c", []),
                    "retrieval_b_tkg": tkg2.extra.get("retrieval_b", []),
                    "retrieval_c_tkg": tkg2.extra.get("retrieval_c", []),
                    "tkg_path_info": tkg2.extra.get("tkg_path_info"),
                },
                metrics=rag2.metrics.merge(tkg2.metrics),
            )
            q_str = render_question(question, self.question_format)
            rag_rendered = rag2.rendered or "No relevant history!"
            tkg_rendered = tkg2.rendered or "No relevant triple!"
            prompt2 = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", tkg_rendered)
                .replace("{history}", rag_rendered)
            )
            prompt2 = apply_token_limit(prompt2, self.context_size)
        elif self.retrieval_method == "enhanced_reranked_fallback":
            # (fallback is the core contribution) 1st try = reranked (compressed short context, run above).
            #   (E) "I don't know" → in the 2nd try, re-query with enhanced (uncompressed full context).
            #   i.e. "efficient compression first, reinforce with full context if the model gives up". Both paths (RAG k_rag +
            #   Retrieve everything (TKG k_tkg) as in enhanced, but inject it into qa_enhanced.txt uncompressed.
            r2 = self._retrieve_enhanced(question, current_session_idx,
                                         current_session=current_session)
            r2.extra["route"] = "ENHANCED_RERANKED_FALLBACK"
            r2.extra["fallback_from"] = first_route  # "ENHANCED_RERANKED"
            q_str = render_question(question, self.question_format)
            rag_rendered = r2.extra.get("rag_rendered") or "No relevant history!"
            tkg_rendered = r2.extra.get("tkg_rendered") or "No relevant triple!"
            prompt2 = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", tkg_rendered)
                .replace("{history}", rag_rendered)
            )
            prompt2 = apply_token_limit(prompt2, self.context_size)
        elif self.retrieval_method == "enhanced_bridge_fallback":
            # Targets 2-hop bridge-linking failures (not a compression route).
            #   Diagnosis (data): of 47 enhanced hop2 errors, 31 are (E)"I don't know", and 29 of those
            #   give up saying "found the bridge entity but couldn't find its property". Recall is 100%
            #   (gold is inside the 356-quad/110k-char prompt), i.e. hop2 linking fails in long context.
            #   2nd try = (1) extract bridge entity (1 LLM call over full context) → (2) re-ask over a
            #   *focused* short context containing only that entity's one-hop quads + related dialogue
            #   (1 LLM call). This gathers hop2 evidence into a small context so the model actually links.
            r2, prompt2 = self._bridge_fallback(
                question, current_session_idx, r1, metrics,
                current_session=current_session,
            )
            r2.extra["fallback_from"] = first_route  # "ENHANCED"
        elif self.retrieval_method == "enhanced_iter_bridge_fallback":
            # Generalized n-hop iterative bridge resolution (dataset-agnostic, no hardcoded hop count).
            #   On (E), a loop that resolves the "next unresolved intermediate entity" one at a time,
            #   accumulating (2-hop=1 iter, 3-hop=2 iter, …).
            #   Termination/progress guards: K_hop cap + UNKNOWN early exit + visited (break on revisiting
            #   the same node) + break when 0 new nodes. On failure, finish with one enhanced-full call
            #   (≥ enhanced guaranteed). This method makes its own answer call per hop, so it returns the
            #   final answer/metrics directly (early return).
            ans_text, r2, metrics = self._iter_bridge_fallback(
                question, current_session_idx, r1, metrics, first_route,
                current_session=current_session,
            )
            return ans_text, r2, metrics
        elif self.retrieval_method == "enhanced_neutral_iter_fallback":
            # Inductive-bias removed: same loop as iter but uses a neutral prompt; the model decides how to
            #   proceed at each step (entity / UNKNOWN / I don't know). "I don't know" → confirm (E) and stop
            #   immediately. No multi-hop/bridge assumption injected into the model. early return.
            ans_text, r2, metrics = self._neutral_iter_fallback(
                question, current_session_idx, r1, metrics, first_route,
                current_session=current_session,
            )
            return ans_text, r2, metrics
        elif self.retrieval_method == "enhanced_neutral_iter_nodia":
            # Same as nbridge but removes 1-hop dialogue accumulation during iter (quad-only). early return.
            ans_text, r2, metrics = self._neutral_iter_fallback_nodia(
                question, current_session_idx, r1, metrics, first_route,
                current_session=current_session,
            )
            return ans_text, r2, metrics
        elif self.retrieval_method == "iter_inverse_tkg_first":
            # bridge_inverse: 1st try = seednode 2-hop TKG-only (run in the normal flow above).
            #   (E) → TKG-only inverse iter (lookup=inverse prompt, answer quad-only, early termination a~d).
            #   UNKNOWN/no-progress/max_tries → finish with RAG retrieve + enhanced_full. early return.
            ans_text, r2, metrics = self._iter_inverse_tkg_first(
                question, current_session_idx, r1, metrics, first_route,
                current_session=current_session,
            )
            return ans_text, r2, metrics
        elif self.retrieval_method == "enhanced_neutral_iter_fallback_unkE":
            # A/B variant. Only difference from neutral: on UNKNOWN(STOP), terminate with (E) immediately
            #   instead of re-asking with enhanced full. Everything else (IDK→(E), ENTITY→1-hop accumulate
            #   re-ask, guards/finish) is identical.
            ans_text, r2, metrics = self._neutral_iter_fallback_unkE(
                question, current_session_idx, r1, metrics, first_route,
                current_session=current_session,
            )
            return ans_text, r2, metrics
        else:
            # tkg_rag: original behavior — one more pass on the opposite path (single path).
            fb_route = "TKG" if first_route == "RAG" else "RAG"
            r2 = (
                self._retrieve_tkg(question, current_session_idx, route=fb_route,
                                   current_session=current_session)
                if fb_route == "TKG"
                else self._retrieve_rag(question, current_session_idx, route=fb_route,
                                         current_session=current_session)
            )
            r2.extra["fallback_from"] = first_route
            prompt2 = apply_token_limit(self.build_qa_prompt(question, r2), self.context_size)

        gen2 = self.llm.generate(prompt2)
        metrics = metrics.merge(r2.metrics).merge(gen2.to_metrics())
        # fallback branch — overwrite with the final-answer raw + reasoning + prompt + rendering.
        r2.extra["qa_raw_text"] = gen2.text
        r2.extra["qa_reasoning_text"] = getattr(gen2, "reasoning_text", "")
        r2.extra["qa_final_prompt"] = prompt2
        r2.extra["qa_retrieval_rendered"] = r2.rendered
        return gen2.text, r2, metrics

    # ====================================================================
    # snapshot / restore
    # ====================================================================
    def snapshot(self, path: str) -> None:
        save_pickle(
            {
                "kg": self._kg.tkg,
                # quad→sess_idx map for future-leakage blocking. tuple keys are picklable.
                "quad_sess": self._kg.quad_sess,
                "dialogues": self._idx.dialogues,
                "dates": self._idx.dates,
                "sess_idxs": self._idx.sess_idxs,
                "embeddings": [v.tolist() for v in self._idx.embeddings],
            },
            path,
        )

    def restore(self, path: str) -> None:
        d = load_pickle(path)
        self._kg.tkg = d["kg"]
        # Old-snapshot compat — empty dict if quad_sess is absent (sess_idx None→keep, legacy behavior).
        self._kg.quad_sess = d.get("quad_sess", {})
        self._idx.dialogues = d["dialogues"]
        self._idx.dates = d["dates"]
        self._idx.sess_idxs = d["sess_idxs"]
        self._idx.embeddings = [np.asarray(v, dtype=np.float32) for v in d["embeddings"]]
        self._idx._bm25 = None

    # ====================================================================
    # internal — q_clf
    # ====================================================================
    def _classify_question(self, question: Question, idx: int,
                           metrics: "CallMetrics | None" = None) -> str:
        """q_clf classification. Review #1/#3/#4 fixes:
        - Include model_name + prompt fingerprint in the cache key (guards against id(llm) collision + prompt-grid).
        - Accumulate _rec metrics on structured failure too (Review #3 token usage drop fix).
        - Unified label normalization (precomputed + structured + free-form all share the same path).
        """
        # precomputed first (q_clf.json in per-llm cache) — same path as v1_2 baseline.
        if idx in self.precomputed_q_clf:
            return _normalize_q_clf_label(self.precomputed_q_clf[idx])
        # Module-level q_clf cache — avoids re-calling for the same (model, prompt, question).
        model_name = getattr(self.llm, "model_name", str(id(self.llm)))
        prompt_fingerprint = hash(self._p_q_clf)
        cache_key = (model_name, prompt_fingerprint, question.text.strip())
        with _Q_CLF_CACHE_LOCK:
            if cache_key in _Q_CLF_CACHE:
                return _Q_CLF_CACHE[cache_key]
        prompt = self._p_q_clf.replace("{question}", question.text.strip())
        result = None
        # Try structured — free-form fallback on failure / parse miss.
        try:
            parsed, rec = self.llm.generate_structured(prompt, QClfLabel)
            if metrics is not None and rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, QClfLabel):
                result = _normalize_q_clf_label(parsed.label)
        except Exception:
            pass
        if result is None:
            # free-form fallback (legacy behavior — protects v1_2 baseline reproducibility).
            gen = self.llm.generate(prompt)
            if metrics is not None:
                metrics.latency_sec += gen.latency_sec
                metrics.input_tokens += gen.input_tokens
                metrics.output_tokens += gen.output_tokens
                metrics.num_llm_calls += 1
            result = _normalize_q_clf_label(gen.text)
        with _Q_CLF_CACHE_LOCK:
            _Q_CLF_CACHE[cache_key] = result
        return result

    # ====================================================================
    # internal — RAG path
    # ====================================================================
    def _retrieve_rag(self, question: Question, idx: int, route: str = "RAG",
                       current_session: "Session | None" = None) -> RetrievalResult:
        """By design, scope = entire dialogues of the previous sessions + partial dialogue of the current session.

        - Among dialogue_index (committed entire dialogue), only s_idx < idx are candidates.
        - The current_session's partial_dialogue + partial_embedding are also appended as candidates.
        - Extra filtering within that scope only when timestamp_filter is ON (scope is the same regardless of timestamps).
        - Ad-hoc BM25/dense/alternating ranking (corpus is the pool only).
        """
        metrics = CallMetrics()
        timestamps = {t: {"type": classify_timestamp(t)} for t in extract_timestamps(question.text)}

        # 1) pool = (dialogue, date, sess_idx, embedding)
        pool: list[tuple[str, str, int, Any]] = []
        for dia, date, s_idx, emb in zip(
            self._idx.dialogues, self._idx.dates, self._idx.sess_idxs, self._idx.embeddings
        ):
            if s_idx < idx:  # only up to the previous session (do NOT use the current session's entire)
                pool.append((dia, date, s_idx, emb))
        # add the current session's partial
        if current_session is not None and current_session.partial_dialogue:
            pool.append((
                current_session.partial_dialogue,
                current_session.date,
                current_session.idx,
                current_session.partial_embedding,
            ))

        # 2) timestamp filter (optional) — only when rag_time_filter ON + time_filter ON + timestamps present
        if self.rag_time_filter and self.time_filter and timestamps:
            pool = [t for t in pool if _date_matches_any(t[1], timestamps)]
        if not pool:
            return RetrievalResult(
                items=[], rendered="No relevant history!", extra={"route": route}, metrics=metrics
            )

        # 3) ranking (ad-hoc corpus)
        order = self._rag_order_adhoc(question, pool, metrics)

        # 4) top-k cut (when num_retrieved_rag > 0) — RAG path
        # BUG FIX: variable swap — the RAG path cuts by num_retrieved_rag (previously misused _tkg).
        if self.num_retrieved_rag > 0:
            order = order[: self.num_retrieved_rag]

        # 5) (optional) reranker
        if self.reranker is not None and len(order) > 1:
            cands = [pool[i][0] for i in order]
            reranked = self.reranker.rerank(question.text, cands)
            order = [order[i] for i in reranked]

        # 6) fill up to context_size (full QA prompt = base + question + dialogue)
        items: list[Item] = []
        rendered = ""
        q_str = render_question(question, self.question_format)
        approx_tokens = count_tokens(self._p_rag) + count_tokens(q_str)
        for i in order:
            dia, date, s_idx, _ = pool[i]
            chunk = _format_dialogue_chunk(dia, date, s_idx)
            tk = count_tokens(chunk)
            if self.context_size > 0 and approx_tokens + tk > self.context_size:
                break
            rendered += chunk
            items.append(Item(text=dia, meta={"date": date, "session_idx": s_idx}))
            approx_tokens += tk
        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant history!",
            extra={
                "route": route,
                # retrieval recall: B = the whole pool set (before ranking/top-k/reranker),
                #   C = items (what the QA prompt actually includes after context_size). pool elem = (dia, date, s_idx, emb).
                "retrieval_b": _dedup_dialogue_items([(p[2], p[1]) for p in pool]),
                "retrieval_c": _dedup_dialogue_items(
                    [(it.meta["session_idx"], it.meta["date"]) for it in items]
                ),
            },
            metrics=metrics,
        )

    def _rag_order_adhoc(self, question: Question,
                          pool: list[tuple[str, str, int, Any]],
                          metrics: CallMetrics) -> list[int]:
        """Ranking using the pool only (temporary BM25 build + dense embedding lookup).

        Prefer the precomputed question embedding. If a pool embedding is None, live embed_one.
        """
        method = self.rag_method
        n = len(pool)
        ids = list(range(n))
        bm25_order: list[int] = []
        dense_order: list[int] = []

        if method in ("bm25_only", "alternating"):
            from nltk.tokenize import word_tokenize
            from rank_bm25 import BM25Okapi
            tokenized = [word_tokenize(p[0].lower()) for p in pool]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(word_tokenize(question.text.lower()))
            bm25_order = sorted(ids, key=lambda i: -scores[i])
            if method == "bm25_only":
                return bm25_order

        if method in ("emb_only", "alternating"):
            t0 = time.perf_counter()
            q_vec = self._cached_question_embed(question.text)
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += 1
            embs: list[Any] = []
            for p in pool:
                e = p[3]
                if e is None:
                    t0 = time.perf_counter()
                    # L1+S4 fix: _rag_order_adhoc's pool uses the partial-dialogue-only path.
                    # Separated from commit()'s entire-cache write → prevents trial contamination.
                    e = self._cached_partial_dialogue_embed(p[0])
                    metrics.latency_sec += time.perf_counter() - t0
                    metrics.num_embedding_calls += 1
                embs.append(e)
            scored = [(i, cos_sim(q_vec, embs[i])) for i in ids]
            scored.sort(key=lambda x: -x[1])
            dense_order = [i for i, _ in scored]
            if method == "emb_only":
                return dense_order

        if method == "alternating":
            return _alternate_merge(bm25_order, dense_order, bm25_first=self.bm25_first)
        raise ValueError(f"Unknown rag_method: {method!r}")

    # ====================================================================
    # internal — obtain question embedding (precomputed first, else live embed)
    # ====================================================================
    def _question_embedding(self, question: Question, metrics: "CallMetrics | None" = None):
        """Return the question's embedding vector. Prefer the precomputed map, fall back to embed_one if absent.

        Same pattern as the emb_only path in `_rag_order_adhoc` (keeps them consistent).
        """
        t0 = time.perf_counter()
        q_vec = self._cached_question_embed(question.text)
        if metrics is not None:
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += 1
        return q_vec

    # ====================================================================
    # internal — TKG path (2-hop traversal)
    # ====================================================================
    def _retrieve_tkg(self, question: Question, idx: int, route: str = "TKG",
                      current_session: "Session | None" = None) -> RetrievalResult:
        timestamps = {t: {"type": classify_timestamp(t)} for t in extract_timestamps(question.text)}
        metrics = CallMetrics()

        # TKG/RAG symmetry: TKG(C) at retrieve time = A (committed accumulation, sess_idx < idx) + B (current
        #   session's partial quad). Exactly symmetric to RAG's "previous entire dialogue + current-session
        #   partial dialogue" pool. q_subj candidate nodes, 2-hop traversal, and timestamp retrieval all
        #   operate over C. Future-leakage blocking: A only sess<idx, B only the current session's partial →
        #   0 future-session quads exposed.
        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)

        # q_subj options = nodes of C (A+B). All nodes of the combined KG instead of nodes_until (already time-accurate).
        nodes = kg.nodes()
        if not nodes:
            return RetrievalResult(
                items=[], rendered="No relevant triple!", extra={"route": route}, metrics=metrics
            )

        # 1) identify q_subj (precomputed_q_subj[idx] first → else 100-batch live over nodes)
        # _identify_q_subj returns a (kg_nodes, raw_subjects, prompt_text) tuple. Precompute is looked up by question_idx=idx.
        #   ablation uses kg_nodes only → raw_subjects ignored.
        q_subj_candidates, _, q_subj_idf_prompt = self._identify_q_subj(
            question, nodes, metrics, question_idx=idx,
        )
        q_subj_candidates = [
            c if c.lower() not in {"i", "me"} else question.questioner for c in q_subj_candidates
        ]
        if any(p in f" {question.text.lower()} " for p in (" i ", " my ", " me ")):
            q_subj_candidates.append(question.questioner)
        q_subj_candidates = list({c for c in q_subj_candidates if c})
        # BUG FIX: .capitalize() lowercased everything after the second word of multi-word entities
        # (e.g. "Paul the wine guy", "Ross Geller"), causing nodes matching to fail. Also, without
        # case-insensitive matching, "andrea" / "ANDREA" / "Andrea" were inconsistent.
        # Fix: case-insensitive lookup (lower → original mapping).

        # get_clean_entity is a pure helper in utils/common_utils.py (module-level import).
        # BUG FIX: the old dict comprehension silently overwrote on collision of two nodes with the same
        # clean form (depended on dict insertion order, hurting reproducibility). setdefault preserves first-seen.
        _node_lookup: dict[str, str] = {}
        for n in nodes:
            _node_lookup.setdefault(get_clean_entity(n), n)
        # fuzzy fallback removed by design. q_subj→TKG node matching uses exact match (_node_lookup) only.
        # q_subj candidates without an exact match are discarded (not included in q_subj_nodes).
        # The old fuzzy fallback (rapidfuzz score>=60) is removed. The subsequent timestamp retrieval
        # (below, `if not q_subj_nodes and ts_retrieval:`, controlled by the retrieve_by_timestamp toggle)
        # is a separate mechanism and is kept (default ON).
        q_subj_nodes = [
            _node_lookup[get_clean_entity(c)]
            for c in q_subj_candidates
            if get_clean_entity(c) in _node_lookup
        ]
        # Enforce num_q_subj_candidates (K = number of TKG seed nodes) cut — top K after exact-match.
        #   The precompute cache (`q_subj_{norm}_nq{K}.json`) already holds K, but (a) questioner append
        #   (first person) could make it K+1, and (b) there was a bug where the precompute-first early-return
        #   returned the entire cache and ignored K. Cutting by K here makes the seed-node count exactly
        #   num_q_subj_candidates for both precompute and live (no-op if the cache is exactly K, harmless).
        q_subj_nodes = q_subj_nodes[: self.num_q_subj_candidates]

        # 2) 2-hop traversal (case branching)
        retrieved: list[dict] = []
        # When retrieve_by_timestamp is OFF, timestamp-only retrieval (subject-less questions) is disabled.
        ts_retrieval = self.retrieve_by_timestamp and bool(timestamps)
        if not q_subj_nodes and not ts_retrieval:
            return RetrievalResult(
                items=[], rendered="No relevant triple!",
                extra={
                    "route": route,
                    "tkg_path_info": {
                        "question_text": question.text,
                        "q_subj_candidates": list(q_subj_candidates),
                        "tkg_nodes": list(nodes),
                        "matched_subjs": [],
                        "q_subj_idf_prompt": q_subj_idf_prompt,
                    },
                },
                metrics=metrics,
            )
        if not q_subj_nodes and ts_retrieval:
            # timestamp retrieval over the combined KG (C=A+B), not self._kg.
            retrieved = _retrieve_by_timestamp(kg, timestamps)
        else:
            seeds = list(q_subj_nodes)
            seen: set[str] = set()
            for _ in range(self.hop):
                new_seeds: list[str] = []
                for s in seeds:
                    if s in seen:
                        continue
                    seen.add(s)
                    # 2-hop traversal over the combined KG (C=A+B), not self._kg.
                    for q in kg.one_hop(s):
                        if q not in retrieved:
                            retrieved.append(q)
                            for ent in (q["head"], q["tail"]):
                                if ent not in seen and ent not in new_seeds:
                                    new_seeds.append(ent)
                seeds = new_seeds
                if not seeds:
                    break
            if timestamps and self.time_filter:
                # When time_filter is OFF, this whole timestamp-filter block is bypassed → 2-hop results as is.
                # Behavior is decided by the tkg_timestamp_fallback cfg flag.
                # True (default): if filtered is empty, keep unfiltered → prevents degeneration (future-proof).
                # False: if filtered is empty, empty list → for legacy/v1_2_2 reproduction.
                # v1.3.1 guarantees absolute dates via _force_session_date, so False is fine too.
                _ts_fallback = self.cfg.get("tkg_timestamp_fallback", True)
                filtered = _filter_quads_by_timestamp(retrieved, timestamps)
                if filtered:
                    retrieved = filtered
                elif _ts_fallback:
                    pass  # keep unfiltered retrieved (prevents degeneration)
                else:
                    retrieved = []  # legacy behavior: empty when filtered is empty

        # 2.5) Future-leakage blocking is already guaranteed by combined_until(C=A+B):
        #   A copies only sess<idx, B only the current session's partial (sess==idx) — no future (>idx) quads in C.
        #   So the old post-filter `sess_idx < idx` wrongly removed B (sess==idx) and is removed.
        #   (The old filter was a safety net from when we traversed all of self._kg.)

        # retrieval recall: B = the whole set of retrieved raw evidence (before top-k/reranker).
        #   reranker/cos_sim only reorder while top-k (num_retrieved_tkg) cuts, so we snapshot here
        #   (right after 2-hop/timestamp retrieval), i.e. the raw set before any transformation.
        b_retrieved_snapshot = list(retrieved)

        # 4) (optional) reranker
        if self.reranker is not None and len(retrieved) > 1:
            cands = [_quad_to_string(q) for q in retrieved]
            order = self.reranker.rerank(question.text, cands)
            retrieved = [retrieved[i] for i in order]

        # 4.5) cos_sim re-rank — right before render, sort DESCENDING by cosine similarity between the
        #      question embedding and each quad embedding (highest similarity first → top-first render).
        # This whole block is skipped when self.cos_sim_rerank=False (default).
        # BUG FIX: works even when _quad_emb_map is empty, via partial_quad_cache (file cache) or the
        # live embed fallback → no silent skip in the cache-only case.
        if self.cos_sim_rerank and len(retrieved) > 1 and (
            self._quad_emb_map or self._partial_quad_cache or self._entire_quad_cache
        ):
            q_vec = self._question_embedding(question, metrics)
            scored: list[tuple[dict, float]] = []
            for q in retrieved:
                head, rel, tail = q["head"], q["relation"], q["tail"]
                sd = q.get("start_date")
                # Split the key by quad_embedding_type:
                # - "temporal" → f"{h} {r} {t} on {sd}" (non_temporal fallback if sd is absent)
                # - "non_temporal" → f"{h} {r} {t}" only (sd ignored)
                if self.quad_embedding_type == "temporal":
                    key = f"{head} {rel} {tail} on {sd}" if sd else f"{head} {rel} {tail}"
                else:  # non_temporal
                    key = f"{head} {rel} {tail}"
                emb = self._quad_emb_map.get(key)
                if emb is None and self.quad_embedding_type == "temporal" and sd:
                    # non_temporal fallback on temporal match miss (in-memory map only).
                    emb = self._quad_emb_map.get(f"{head} {rel} {tail}")
                if emb is None:
                    # in-memory map miss → cache lookup chain + live embed + cache update.
                    t0 = time.perf_counter()
                    emb = self._cached_quad_embed(key)
                    metrics.latency_sec += time.perf_counter() - t0
                    metrics.num_embedding_calls += 1
                sim = cos_sim(q_vec, emb)
                scored.append((q, sim))
            # Descending by similarity. Quads without an embedding were live-embedded above, so they
            # always have a valid sim — no special last-placement needed.
            scored.sort(key=lambda x: -x[1])
            retrieved = [q for q, _ in scored]

        # 3') num_retrieved_tkg top-k cut — moved after reranker + cos_sim rerank.
        #   enhanced: 2-hop → (reranker) → cos_sim rerank → top-k(=100) cut → render.
        #   In the old order, the cut preceded the rerank, so the first k in 2-hop order were kept
        #   rather than the cos_sim top ones. With num_retrieved_tkg=0 (tkg_only budget mode) no cut is
        #   done → fill up to context_size.
        if self.num_retrieved_tkg > 0:
            retrieved = retrieved[: self.num_retrieved_tkg]

        # 5) render + cut to context_size
        # BUG FIX: the old _render_quads_by_date re-grouped the cos_sim-sorted `retrieved` by date,
        # destroying the similarity order. Highest-sim quads then didn't come first in the prompt, so the
        # most important quads could be dropped when cutting to context_size.
        # Fix: render preserving the cos_sim-descending order (retrieved's current order) without date-grouping.
        # _render_quads_by_date is still defined (may be used by other paths) — only unused on this reranked path.
        # full QA prompt = base + question + quads (consistent with RAG)
        q_str = render_question(question, self.question_format)
        base_p = self._p_quad + q_str
        rendered, c_quads = _render_quads_in_order(retrieved, self.quad_format, self.context_size, base_p)
        items = [Item(text=_quad_to_string(q), meta=q) for q in retrieved]
        return RetrievalResult(
            items=items,
            rendered=rendered,
            extra={
                "route": route,
                "q_subj_nodes": q_subj_nodes,
                "retrieval_b": _dedup_quad_items(b_retrieved_snapshot),  # B = the whole set of retrieved quads (before top-k/reranker)
                "retrieval_c": _dedup_quad_items(c_quads),               # C = set included in the QA prompt after context_size
                "tkg_path_info": {
                    "question_text": question.text,
                    "q_subj_candidates": list(q_subj_candidates),
                    "tkg_nodes": list(nodes),
                    "matched_subjs": list(q_subj_nodes),
                    "q_subj_idf_prompt": q_subj_idf_prompt,
                },
            },
            metrics=metrics,
        )

    # ====================================================================
    # internal — enhanced retrieval (RAG top-k + TKG top-k simultaneously)
    # ====================================================================
    def _retrieve_enhanced(self, question: Question, idx: int,
                            current_session: "Session | None" = None) -> RetrievalResult:
        rag = self._retrieve_rag(question, idx, route="RAG_in_enhanced",
                                  current_session=current_session)
        tkg = self._retrieve_tkg(question, idx, route="TKG_in_enhanced",
                                 current_session=current_session)
        metrics = rag.metrics.merge(tkg.metrics)
        return RetrievalResult(
            items=rag.items + tkg.items,
            rendered="",  # build_qa_prompt uses rag_rendered/tkg_rendered from extra
            extra={
                "route": "ENHANCED",
                "rag_rendered": rag.rendered,
                "tkg_rendered": tkg.rendered,
                "rag_items": rag.items,
                "tkg_items": tkg.items,
                # retrieval recall: enhanced = RAG+TKG simultaneously → store B/C separately per path.
                #   For recall measurement, RAG-path = dialogue-level, TKG-path = quad-level.
                "retrieval_b_rag": rag.extra.get("retrieval_b", []),
                "retrieval_c_rag": rag.extra.get("retrieval_c", []),
                "retrieval_b_tkg": tkg.extra.get("retrieval_b", []),
                "retrieval_c_tkg": tkg.extra.get("retrieval_c", []),
                # The TKG-path is used in the ENHANCED route too, so propagate tkg_path_info.
                "tkg_path_info": tkg.extra.get("tkg_path_info"),
            },
            metrics=metrics,
        )

    # ====================================================================
    # internal — enhanced_bridge_fallback 2nd try (2-hop bridge linking)
    # ====================================================================
    def _bridge_fallback(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, current_session: "Session | None" = None,
    ) -> "tuple[RetrievalResult, str]":
        """(E) fallback: extract bridge entity → build a re-ask prompt from that entity's one-hop focused context.

        Procedure:
          (1) Extract 1 bridge entity from the 1st enhanced full context (r1's rag_rendered/tkg_rendered) (LLM 1call).
          (2) Resolve bridge to a combined-KG (C=A+B) node (exact get_clean_entity, else substring).
          (3) kg.one_hop(bridge) → bridge quads (where hop2 evidence gathers) + dialogue where bridge appears.
          (4) Return a short prompt2 injecting only focused triples/history into qa_enhanced.txt.
        Returns (r2 RetrievalResult, prompt2). On failure (bridge UNKNOWN/unresolved), re-ask once with enhanced
              full (one more chance for the model without info loss — guarantees at least enhanced parity).
        """
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"

        # (1) extract bridge entity — 1 LLM call over full context.
        bridge_prompt = (
            self._p_bridge_extract.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        bridge_prompt = self._cap_bridge_prompt(bridge_prompt)
        gen_b = self.llm.generate(bridge_prompt)
        # The bridge-extract LLM call's accounting (tokens/latency/num_llm_calls) must flow into r2.metrics
        # so it is summed in answer()'s `metrics.merge(r2.metrics)` (both returns below use this metrics).
        bridge_metrics = gen_b.to_metrics()
        bridge_raw = (gen_b.text or "").strip().splitlines()[0].strip() if (gen_b.text or "").strip() else ""
        bridge_raw = bridge_raw.strip().strip('"').strip("'").strip()

        # (2) combined KG(C) + node resolve
        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        bridge_node = None
        if bridge_raw and bridge_raw.upper() != "UNKNOWN":
            lookup: dict[str, str] = {}
            for n in nodes:
                lookup.setdefault(get_clean_entity(n), n)
            ce = get_clean_entity(bridge_raw)
            if ce in lookup:
                bridge_node = lookup[ce]
            else:
                # substring fallback: bridge_raw is contained in some node (or vice versa) — shortest match.
                br_low = bridge_raw.lower()
                cands = [n for n in nodes if br_low and (br_low in n.lower() or n.lower() in br_low)]
                if cands:
                    bridge_node = min(cands, key=len)

        # (3) bridge resolution failed → re-ask once with enhanced full (at least enhanced parity).
        if bridge_node is None:
            prompt2 = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            prompt2 = self._cap_bridge_prompt(prompt2)
            r2 = RetrievalResult(
                items=list(r1.items), rendered="",
                extra={"route": "ENHANCED_BRIDGE_FALLBACK", "bridge_entity": bridge_raw or "",
                       "bridge_resolved": False,
                       "rag_rendered": full_rag, "tkg_rendered": full_tkg},
                metrics=bridge_metrics,
            )
            return r2, prompt2

        # (4) focused context from bridge's one-hop quads (hop2 evidence) + bridge dialogue.
        bq = kg.one_hop(bridge_node)
        # date-descending stable (preserve original order) — latest facts first, but include all (few in count).
        base_p_tkg = self._p_quad + q_str
        tkg_focused, _c = _render_quads_in_order(bq, self.quad_format, self.context_size, base_p_tkg)
        if not bq:
            tkg_focused = "No relevant triple!"
        # dialogue where bridge appears (sess<idx + current-session partial) — substring match, up to 8.
        rag_focused = self._bridge_dialogues(bridge_node, idx, current_session, max_n=8)

        prompt2 = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", tkg_focused)
            .replace("{history}", rag_focused or "No relevant history!")
        )
        prompt2 = self._cap_bridge_prompt(prompt2)
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ENHANCED_BRIDGE_FALLBACK",
                "bridge_entity": bridge_node, "bridge_resolved": True,
                "bridge_n_quads": len(bq),
                "rag_rendered": rag_focused or "No relevant history!",
                "tkg_rendered": tkg_focused,
            },
            metrics=bridge_metrics,
        )
        return r2, prompt2

    def _bridge_dialogues(self, bridge: str, idx: int,
                          current_session: "Session | None", max_n: int = 8) -> str:
        """Dialogue chunks where the bridge entity appears (future-leakage blocking: sess<idx + current-session partial)."""
        out: list[str] = []
        bl = bridge.lower()
        if not bl:
            return ""
        # entire dialogue index (sess_idx < idx only)
        dias = getattr(self._idx, "dialogues", []) or []
        dates = getattr(self._idx, "dates", []) or []
        sidxs = getattr(self._idx, "sess_idxs", []) or []
        for dia, date, s_idx in zip(dias, dates, sidxs):
            if s_idx is not None and idx is not None and s_idx >= idx:
                continue
            if bl in (dia or "").lower():
                out.append(_format_dialogue_chunk(dia, date, s_idx))
                if len(out) >= max_n:
                    break
        # current-session partial dialogue
        if len(out) < max_n and current_session is not None:
            pd = getattr(current_session, "partial_dialogue", None)
            if pd and bl in pd.lower():
                out.append(_format_dialogue_chunk(pd, getattr(current_session, "date", ""), idx))
        return "".join(out)

    def _node_one_hop_dialogues(self, node: str, idx: int,
                                current_session: "Session | None", max_n: int = 4) -> list[str]:
        """List of dialogue chunks where node appears (future-leakage blocking: sess<idx + current-session partial). dedup at the caller."""
        out: list[str] = []
        bl = (node or "").lower()
        if not bl:
            return out
        dias = getattr(self._idx, "dialogues", []) or []
        dates = getattr(self._idx, "dates", []) or []
        sidxs = getattr(self._idx, "sess_idxs", []) or []
        for dia, date, s_idx in zip(dias, dates, sidxs):
            if s_idx is not None and idx is not None and s_idx >= idx:
                continue
            if bl in (dia or "").lower():
                out.append(_format_dialogue_chunk(dia, date, s_idx))
                if len(out) >= max_n:
                    break
        if len(out) < max_n and current_session is not None:
            pd = getattr(current_session, "partial_dialogue", None)
            if pd and bl in pd.lower():
                out.append(_format_dialogue_chunk(pd, getattr(current_session, "date", ""), idx))
        return out

    def _cap_bridge_prompt(self, prompt: str) -> str:
        """Over-length guard for bridge-family prompts. Independent of context_size, cap the input to
        _bridge_input_token_cap tokens to avoid exceeding the serving max-model-len (on overflow keep the
        front = instruction+question+triples first; dialogue is at the back so it is cut first). If
        context_size>0, apply the smaller of the two."""
        cap = self._bridge_input_token_cap
        if self.context_size and self.context_size > 0:
            cap = min(cap, self.context_size) if cap > 0 else self.context_size
        return trim_text_to_tokens(prompt, cap) if cap > 0 else prompt

    def _resolve_bridge_node(self, bridge_raw: str, nodes: "list[str]",
                             node_lookup: "dict[str, str]") -> "str | None":
        """Resolve bridge_raw → KG node. exact get_clean_entity first, else substring (shortest match)."""
        if not bridge_raw or bridge_raw.upper() == "UNKNOWN":
            return None
        ce = get_clean_entity(bridge_raw)
        if ce in node_lookup:
            return node_lookup[ce]
        br_low = bridge_raw.lower()
        cands = [n for n in nodes if br_low and (br_low in n.lower() or n.lower() in br_low)]
        return min(cands, key=len) if cands else None

    # ====================================================================
    # internal — enhanced_iter_bridge_fallback (n-hop iterative bridge resolution, dataset-agnostic)
    # ====================================================================
    def _iter_bridge_fallback(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """General n-hop form of the (E) fallback. A loop that resolves intermediate entities one at a time,
        *accumulating sequentially*.

        2-hop = 1 iter, 3-hop = 2 iter … (no hardcoded hop count; K is just a non-tuning safety cap).
        Each iter:
          ① Extract 1 "next unresolved intermediate entity" from accumulated focused + full context (LLM 1call).
             UNKNOWN → break (early exit). Already-visited node/unresolved → break (no progress).
          ② Add that entity's one-hop quads + dialogue to the accumulated context. If 0 new nodes(quads), break (idle loop).
          ③ Try answering over the accumulated focused context (qa_enhanced.txt, LLM 1call). Return that answer unless (E).
        After the loop ends (UNKNOWN/no-progress/K-reached) → re-ask once with enhanced full (guarantees at least enhanced parity).
        Returns (answer_text, r2, total_metrics). r2.extra records bridge_iters/bridge_chain (to prove K-independence).
        """
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"

        # combined KG(C=A+B) — future-leakage blocking (sess<idx + current-session partial). Build node lookup once.
        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []            # resolved bridge nodes (order preserved) — break on revisit
        visited_set: set[str] = set()
        acc_quad_keys: set[tuple] = set()  # accumulated focused quad dedup
        acc_quads: list[dict] = []
        acc_dia_set: set[str] = set()      # accumulated dialogue dedup
        acc_dias: list[str] = []
        base_p_tkg = self._p_quad + q_str

        iters = 0
        for _hop in range(max(1, self._iter_bridge_k_hop)):
            # ① extract next unresolved intermediate entity (accumulated focused + full context).
            focused_tkg_now, _c = _render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", [])
            ex_prompt = (
                self._p_bridge_iter_extract
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_tkg_now)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            raw = (gen_ex.text or "").strip()
            bridge_raw = raw.splitlines()[0].strip().strip('"').strip("'").strip() if raw else ""

            # progress guard: UNKNOWN / unresolved / already visited → break.
            if not bridge_raw or bridge_raw.upper() == "UNKNOWN":
                break
            node = self._resolve_bridge_node(bridge_raw, nodes, node_lookup)
            if node is None or node in visited_set:
                break  # couldn't resolve a new entity or repeating the same node = no progress

            # ② add one-hop neighbors to the accumulated context. If 0 new quads, break (idle loop).
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k)
                    acc_quads.append(q)
                    added += 1
            for dchunk in self._node_one_hop_dialogues(node, idx, current_session, max_n=4):
                if dchunk not in acc_dia_set:
                    acc_dia_set.add(dchunk)
                    acc_dias.append(dchunk)
            if added == 0:
                break  # context didn't grow = no point iterating further

            visited.append(node)
            visited_set.add(node)
            iters += 1

            # ③ try answering over the accumulated focused context.
            focused_tkg, _c2 = _render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            )
            if not acc_quads:
                focused_tkg = "No relevant triple!"
            focused_rag = "".join(acc_dias) if acc_dias else "No relevant history!"
            ans_prompt = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", focused_tkg)
                .replace("{history}", focused_rag)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                # success — return this answer.
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_ITER_BRIDGE_FALLBACK", "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "rag_rendered": focused_rag, "tkg_rendered": focused_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        # loop ended (no-progress/UNKNOWN/K-reached) → finish with one enhanced-full call (≥ enhanced guaranteed).
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ENHANCED_ITER_BRIDGE_FALLBACK", "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    # ====================================================================
    # internal — enhanced_neutral_iter_fallback (inductive bias removed: no multi-hop/bridge assumption)
    # ====================================================================
    @staticmethod
    def _classify_next_lookup(text: str) -> "tuple[str, str]":
        """Classify the neutral prompt (qa_next_lookup.txt) output into 3. Returns (kind, entity).

        kind ∈ {"IDK", "STOP", "ENTITY"}:
          - "IDK"    : "I don't know"-like → model declares it can't answer → caller confirms (E) and stops.
          - "STOP"   : "UNKNOWN"-like → no more entity to find → stop the cycle (finish with enhanced full).
          - "ENTITY" : otherwise → entity name (first line). Caller tries KG resolve.
        Based on the first non-blank line, ignoring case/punctuation/quotes.
        """
        raw = (text or "").strip()
        if not raw:
            return "STOP", ""
        first = raw.splitlines()[0].strip().strip('"').strip("'").strip()
        low = first.lower().strip().rstrip(".!?,").strip()
        # "I don't know" / "i dont know" / "none of the above"-like → give up.
        if low in ("i don't know", "i dont know", "idk", "none", "none of the above") or low.startswith("i don't know") or low.startswith("i dont know"):
            return "IDK", ""
        if low == "unknown" or low.startswith("unknown"):
            return "STOP", ""
        return "ENTITY", first

    def _neutral_iter_fallback(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """Inductive-bias-removed version of _iter_bridge_fallback. Without injecting any multi-hop/bridge
        assumption into the prompt, the model itself decides at each step whether "there is a further named
        entity to look up" (entity / UNKNOWN / I don't know).

        Entered after the 1st enhanced (E). Each iter:
          ① Ask the model with the neutral prompt (_p_next_lookup, accumulated facts + full context) (LLM 1call).
             - "I don't know" → declares it can't answer → confirm (E) and stop immediately (no re-ask).
             - "UNKNOWN"      → no more entity to find → break the loop → finish with one enhanced-full call.
             - "{entity}"     → KG resolve. unresolved/revisiting visited → no progress → break (finish).
          ② Accumulate the resolved entity's one-hop quads + dialogue. 0 new nodes(quads) → break (idle loop).
          ③ Try answering over the accumulated focused context (qa_enhanced.txt, LLM 1call). Return that answer unless (E).
        Keeps progress guards (visited / new-node-0) + a K_hop non-tuning safety cap. bridge_iters logging (proves K-independence).
        Returns (answer_text, r2, total_metrics). route="ENHANCED_NEUTRAL_ITER_FALLBACK".
        """
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"

        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        acc_quad_keys: set[tuple] = set()
        acc_quads: list[dict] = []
        acc_dia_set: set[str] = set()
        acc_dias: list[str] = []
        base_p_tkg = self._p_quad + q_str

        iters = 0
        for _hop in range(max(1, self._iter_bridge_k_hop)):
            # ① neutral query: is there another entity to look up? (no multi-hop/bridge assumption injected)
            focused_now, _c = (_render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", []))
            ex_prompt = (
                self._p_next_lookup
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            # By-design termination: "I don't know" → model declares it can't answer → confirm (E) and stop immediately (no re-ask).
            if kind == "IDK":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_NEUTRAL_ITER_FALLBACK", "fallback_from": first_route,
                        "bridge_resolved": False, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "IDK",
                        "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                        "qa_raw_text": "(E)", "qa_reasoning_text": "",
                        "qa_final_prompt": ex_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return "(E)", r2, metrics  # final answer (E) confirmed

            if kind == "STOP":
                break  # no more entity to find → finish with enhanced full

            node = self._resolve_bridge_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break  # resolve failed / revisit = no progress

            # ② accumulate one-hop
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k); acc_quads.append(q); added += 1
            for dchunk in self._node_one_hop_dialogues(node, idx, current_session, max_n=4):
                if dchunk not in acc_dia_set:
                    acc_dia_set.add(dchunk); acc_dias.append(dchunk)
            if added == 0:
                break  # context didn't grow

            visited.append(node); visited_set.add(node); iters += 1

            # ③ try answering over the accumulated focused context
            focused_tkg, _c2 = _render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            )
            if not acc_quads:
                focused_tkg = "No relevant triple!"
            focused_rag = "".join(acc_dias) if acc_dias else "No relevant history!"
            ans_prompt = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", focused_tkg)
                .replace("{history}", focused_rag)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_NEUTRAL_ITER_FALLBACK", "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "ANSWERED",
                        "rag_rendered": focused_rag, "tkg_rendered": focused_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        # loop ended (UNKNOWN/no-progress/K-reached) → finish with one enhanced-full call (≥ enhanced guaranteed).
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ENHANCED_NEUTRAL_ITER_FALLBACK", "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "neutral_stop": "FINAL_ENHANCED",
                "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    def _neutral_iter_fallback_nodia(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """nodia redefinition: the subgraph-exploration iter answer prompt = **initial enhanced triples +
        quads newly discovered by exploration, accumulated** (qa_enhanced.txt). Dialogue **keeps the initial
        full_rag as is** in every QA prompt (do NOT clear dialogue during iter — initial dialogue is always
        included). lookup uses the nodia-only prompt (qa_next_lookup_nodia.txt: IDK=can't solve + no entity
        to look up / UNKNOWN=there is an entity to check but it cannot be pinpointed among the gathered triple
        entities). Each iter:
          ① lookup → IDK=(E) terminate immediately, UNKNOWN=break→initial enhanced_full, entity=resolve.
          resolve failed/revisit/0 new quads → break→initial enhanced_full.
          if resolve+new quads exist, ③ try answering over (initial triples + accumulated new quads, initial
          dialogue) → terminate unless (E); if (E), next iter. If still (E) after exhausting max_tries
          (iter_bridge_k_hop), finish with initial enhanced_full.
        route="ENHANCED_NEUTRAL_ITER_NODIA".
        """
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"
        # Initial enhanced triples (list of quad dicts) — iter answer {triples} = these initial quads + accumulated new quads.
        #   ★ r1.extra["tkg_items"] is a list of Item objects (_retrieve_tkg wraps as Item(meta=quad)) → extract .meta (quad dict).
        #   (Item has no .get — calling q.get(...) directly crashes with "'Item' object has no attribute 'get'".)
        initial_items = [it.meta for it in (r1.extra.get("tkg_items") or [])
                         if isinstance(getattr(it, "meta", None), dict)]

        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        # dedup key seed = initial triple keys → quads already present initially don't count as "new quads" (avoids duplicates).
        acc_quad_keys: set[tuple] = {
            (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
            for q in initial_items
        }
        acc_quads: list[dict] = []   # only quads newly discovered by exploration (accumulated for {focused} + iter answer {triples}).
        base_p_tkg = self._p_quad + q_str

        iters = 0
        for _hop in range(max(1, self._iter_bridge_k_hop)):
            # ① nodia-only lookup: any more entity to look up? {focused}=new quads, {triples}=initial full_tkg, {history}=initial full_rag.
            focused_now, _c = (_render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", []))
            ex_prompt = (
                self._p_next_lookup_nodia
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            if kind == "IDK":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_NEUTRAL_ITER_NODIA", "fallback_from": first_route,
                        "bridge_resolved": False, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "IDK",
                        "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                        "qa_raw_text": "(E)", "qa_reasoning_text": "",
                        "qa_final_prompt": ex_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return "(E)", r2, metrics

            if kind == "STOP":
                # UNKNOWN (further exploration needed but cannot be pinpointed) → cannot explore further →
                #   treat the 1st enhanced answer (answer()'s gen1 = r1.extra["qa_raw_text"], which is (E) by
                #   the entry condition) as the final answer and terminate. (old: break → FINAL_ENHANCED re-ask)
                first_text = r1.extra.get("qa_raw_text", "(E)")
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_NEUTRAL_ITER_NODIA", "fallback_from": first_route,
                        "bridge_resolved": bool(visited), "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "STOP_FIRST_ANSWER",
                        "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                        "qa_raw_text": first_text,
                        "qa_reasoning_text": r1.extra.get("qa_reasoning_text", ""),
                        "qa_final_prompt": r1.extra.get("qa_final_prompt", ""),
                        "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return first_text, r2, metrics

            node = self._resolve_bridge_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break

            # ② among one-hop quads, accumulate as new quads only those "not in the previous QA prompt (initial triples + existing accumulation)".
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k); acc_quads.append(q); added += 1
            if added == 0:
                break

            visited.append(node); visited_set.add(node); iters += 1

            # ③ try answering over (initial triples + accumulated new quads) + initial dialogue.
            combined_items = initial_items + acc_quads
            combined_tkg, _c2 = _render_quads_in_order(
                combined_items, self.quad_format, self.context_size, base_p_tkg
            )
            if not combined_items:
                combined_tkg = "No relevant triple!"
            ans_prompt = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", combined_tkg)
                .replace("{history}", full_rag)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ENHANCED_NEUTRAL_ITER_NODIA", "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "ANSWERED",
                        "rag_rendered": full_rag, "tkg_rendered": combined_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        else:
            # The for loop exhausted all 6 iters (iter_bridge_k_hop) without a break →
            #   still no answer even with further exploration → terminate with "I don't know (E)" as the final
            #   answer (no extra LLM call).
            #   ⚠ for-else: else runs only when the loop ran to completion without any break.
            #      If it exited via break (resolve failed/revisit/added0), else is skipped and it falls through
            #      to FINAL_ENHANCED below.
            r2 = RetrievalResult(
                items=list(r1.items), rendered="",
                extra={
                    "route": "ENHANCED_NEUTRAL_ITER_NODIA", "fallback_from": first_route,
                    "bridge_resolved": bool(visited), "bridge_iters": iters,
                    "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                    "neutral_stop": "MAX_ITER_IDK",
                    "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                    "qa_raw_text": "(E)", "qa_reasoning_text": "",
                    "qa_final_prompt": "", "qa_retrieval_rendered": "",
                },
                metrics=CallMetrics(),
            )
            return "(E)", r2, metrics

        # Loop exited via break (resolve failed / revisit / added0) → finalize with one enhanced full call
        #   (FINAL_ENHANCED). (6-hop exhaustion is already terminated with (E) in the for-else above, so it
        #   never reaches here.)
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ENHANCED_NEUTRAL_ITER_NODIA", "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "neutral_stop": "FINAL_ENHANCED",
                "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    def _iter_inverse_tkg_first(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """bridge_inverse: TKG-first / RAG-last.
        First try = seed-node 2-hop TKG-only (run in the normal answer() flow, r1=_retrieve_tkg). Entered after (E).
        Each iteration:
          ① inverse lookup prompt (_p_next_lookup_inverse_v2, no {history} = TKG-only) decides the next entity.
             - "I don't know" (IDK) → break loop → RAG retrieve + enhanced_full finalize (treated like UNKNOWN).
             - "UNKNOWN" / unresolved / revisit of visited → break loop → RAG retrieve + enhanced_full finalize.
          ② accumulate one-hop quads of the resolved entity (no dialogue). 0 new quads → break.
          ③ try answering with accumulated focused quad-only context. If a~d (not (E)) → terminate immediately
             (enhanced bypass allowed).
        Loop end (UNKNOWN / no progress / max_tries) → **retrieve RAG for the first time here** and finalize with
        enhanced_full (RAG k_rag + initial TKG). route="ITER_INVERSE_TKG_FIRST".
        """
        q_str = render_question(question, self.question_format)
        # r1 = TKG-only 2-hop retrieve → TKG text is in rendered (or extra.tkg_rendered). RAG not retrieved.
        full_tkg = r1.extra.get("tkg_rendered") or r1.rendered or "No relevant triple!"
        # Stage1 (2-hop TKG) quad list — iter re-answer {triples} = initial quads + accumulated new quads.
        stage1_quads = [it.meta for it in r1.items if isinstance(getattr(it, "meta", None), dict)]

        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        # dedup seed = Stage1 quad keys → quads already present initially are not counted as "new" (avoid dup).
        acc_quad_keys: set[tuple] = {
            (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
            for q in stage1_quads
        }
        acc_quads: list[dict] = []   # only newly-discovered quads (for {focused} + iter re-answer {triples} accumulation).
        base_p_tkg = self._p_quad + q_str

        iters = 0
        fell_from_idk = False  # IDK falls through to enhanced_full fallback instead of immediate (E) termination.
        for _hop in range(max(1, self._max_tries)):
            # ① inverse lookup: TKG-only (initial 2-hop + accumulated quads + resolved entities), RAG not exposed.
            focused_now, _c = (_render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", []))
            ex_prompt = (
                self._p_next_lookup_inverse_v2
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_tkg)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            if kind == "IDK":
                # IDK does not terminate immediately with (E); it falls through to the enhanced_full fallback
                #   (treated like UNKNOWN). A question that TKG-only said it didn't know is retried once with full
                #   RAG+TKG context via the enhanced safety net. Only an IDK-origin flag is kept for analysis.
                fell_from_idk = True
                break

            if kind == "STOP":
                break  # UNKNOWN → RAG retrieve + enhanced_full finalize

            node = self._resolve_bridge_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break  # no progress → enhanced_full finalize

            # ② accumulate one-hop quads only (no dialogue — inverse = RAG-last).
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k); acc_quads.append(q); added += 1
            if added == 0:
                break

            visited.append(node); visited_set.add(node); iters += 1

            # ③ try answering with (Stage1 triples + accumulated new quads) via qa_quad.txt (TKG-only, no dialogue)
            #    → if a~d, terminate immediately.
            combined_items = stage1_quads + acc_quads
            combined_tkg, _c2 = _render_quads_in_order(
                combined_items, self.quad_format, self.context_size, base_p_tkg
            )
            if not combined_items:
                combined_tkg = "No relevant triple!"
            ans_prompt = (
                self._p_quad.replace("{question}", q_str)
                .replace("{triples}", combined_tkg)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ITER_INVERSE_TKG_FIRST", "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "ANSWERED",
                        "rag_rendered": "No relevant history!", "tkg_rendered": combined_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        # Loop end (UNKNOWN / no progress / max_tries) → retrieve RAG for the first time + enhanced_full finalize.
        rag_r = self._retrieve_rag(question, idx, route="RAG", current_session=current_session)
        metrics = metrics.merge(rag_r.metrics)
        final_rag = rag_r.rendered or "No relevant history!"
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", final_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ITER_INVERSE_TKG_FIRST", "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "neutral_stop": "FINAL_ENHANCED_FROM_IDK" if fell_from_idk else "FINAL_ENHANCED",
                "rag_rendered": final_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    def _iter_first_enhanced_fallback(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """bridge-first: order-reversed variant of neutral.
        No enhanced-first generate; iterate the seed-node subgraph from the start (up to _max_tries).
        Each iteration: neutral prompt decides the next entity → resolve → accumulate one-hop → try answering
        with the accumulated focused context. If a~d (not (E)) → terminate immediately. IDK / UNKNOWN / no
        progress / max_tries reached → after loop end, finalize with one enhanced full call (≥ enhanced guarantee).
        Differences from neutral: (1) no enhanced-first (iter first), (2) IDK does not terminate immediately with
        (E) but finalizes via enhanced full (always a final enhanced). route="ITER_FIRST_ENHANCED_FALLBACK".
        """
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"

        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        acc_quad_keys: set[tuple] = set()
        acc_quads: list[dict] = []
        acc_dia_set: set[str] = set()
        acc_dias: list[str] = []
        base_p_tkg = self._p_quad + q_str

        iters = 0
        for _hop in range(max(1, self._max_tries)):
            # ① neutral query: any more entities to look up?
            focused_now, _c = (_render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", []))
            ex_prompt = (
                self._p_next_lookup
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            # bridge-first: both IDK/UNKNOWN break the loop → enhanced full finalize (always a final enhanced).
            if kind in ("IDK", "STOP"):
                break

            node = self._resolve_bridge_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break  # resolve failed / revisit = no progress

            # ② accumulate one-hop
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k); acc_quads.append(q); added += 1
            for dchunk in self._node_one_hop_dialogues(node, idx, current_session, max_n=4):
                if dchunk not in acc_dia_set:
                    acc_dia_set.add(dchunk); acc_dias.append(dchunk)
            if added == 0:
                break  # context did not grow

            visited.append(node); visited_set.add(node); iters += 1

            # ③ try answering with accumulated focused context → if a~d, terminate
            focused_tkg, _c2 = _render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            )
            if not acc_quads:
                focused_tkg = "No relevant triple!"
            focused_rag = "".join(acc_dias) if acc_dias else "No relevant history!"
            ans_prompt = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", focused_tkg)
                .replace("{history}", focused_rag)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "ITER_FIRST_ENHANCED_FALLBACK", "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "ANSWERED",
                        "rag_rendered": focused_rag, "tkg_rendered": focused_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        # Loop end (IDK / UNKNOWN / no progress / max_tries reached) → finalize with one enhanced full call
        # (≥ enhanced guarantee).
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "ITER_FIRST_ENHANCED_FALLBACK", "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "neutral_stop": "FINAL_ENHANCED",
                "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    def _neutral_iter_fallback_unkE(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """A/B variant of _neutral_iter_fallback. **Only difference**: on UNKNOWN(STOP), instead of an enhanced
        full re-query (break → finalize), terminate immediately with (E) (neutral_stop="UNK_E").

        Everything else is identical to _neutral_iter_fallback:
          - "I don't know" → confirm (E) and terminate immediately (neutral_stop="IDK").
          - "{entity}"     → KG resolve + one-hop accumulation + re-query (neutral_stop="ANSWERED").
          - resolve failed / revisit of visited / 0 new nodes / K reached → finalize with one enhanced full call
            (neutral_stop="FINAL_ENHANCED"). Keeps the progress guard (visited / new-node-0) + non-tuned K_hop cap.
        Returns (answer_text, r2, total_metrics). route="ENHANCED_NEUTRAL_ITER_FALLBACK_UNKE".
        """
        ROUTE = "ENHANCED_NEUTRAL_ITER_FALLBACK_UNKE"
        q_str = render_question(question, self.question_format)
        full_rag = r1.extra.get("rag_rendered") or "No relevant history!"
        full_tkg = r1.extra.get("tkg_rendered") or "No relevant triple!"

        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        acc_quad_keys: set[tuple] = set()
        acc_quads: list[dict] = []
        acc_dia_set: set[str] = set()
        acc_dias: list[str] = []
        base_p_tkg = self._p_quad + q_str

        iters = 0
        for _hop in range(max(1, self._iter_bridge_k_hop)):
            focused_now, _c = (_render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            ) if acc_quads else ("(none yet)", []))
            ex_prompt = (
                self._p_next_lookup
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_tkg)
                .replace("{history}", full_rag)
            )
            ex_prompt = self._cap_bridge_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            # "I don't know" → confirm (E) and terminate immediately (no re-query). Same as neutral.
            if kind == "IDK":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": ROUTE, "fallback_from": first_route,
                        "bridge_resolved": False, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "IDK",
                        "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                        "qa_raw_text": "(E)", "qa_reasoning_text": "",
                        "qa_final_prompt": ex_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return "(E)", r2, metrics

            # ★ Only difference: UNKNOWN(STOP) → terminate immediately with (E), no enhanced full re-query.
            if kind == "STOP":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": ROUTE, "fallback_from": first_route,
                        "bridge_resolved": False, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "UNK_E",
                        "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                        "qa_raw_text": "(E)", "qa_reasoning_text": "",
                        "qa_final_prompt": ex_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return "(E)", r2, metrics

            node = self._resolve_bridge_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break  # resolve failed / revisit = no progress → enhanced full finalize

            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_quad_keys:
                    acc_quad_keys.add(k); acc_quads.append(q); added += 1
            for dchunk in self._node_one_hop_dialogues(node, idx, current_session, max_n=4):
                if dchunk not in acc_dia_set:
                    acc_dia_set.add(dchunk); acc_dias.append(dchunk)
            if added == 0:
                break

            visited.append(node); visited_set.add(node); iters += 1

            focused_tkg, _c2 = _render_quads_in_order(
                acc_quads, self.quad_format, self.context_size, base_p_tkg
            )
            if not acc_quads:
                focused_tkg = "No relevant triple!"
            focused_rag = "".join(acc_dias) if acc_dias else "No relevant history!"
            ans_prompt = (
                self._p_enhanced_fallback.replace("{question}", q_str)
                .replace("{triples}", focused_tkg)
                .replace("{history}", focused_rag)
            )
            ans_prompt = self._cap_bridge_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": ROUTE, "fallback_from": first_route,
                        "bridge_resolved": True, "bridge_iters": iters,
                        "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                        "neutral_stop": "ANSWERED",
                        "rag_rendered": focused_rag, "tkg_rendered": focused_tkg,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        # Loop end (resolve failed / revisit / 0 new nodes / K reached) → finalize with one enhanced full call
        # (≥ enhanced guarantee). (UNKNOWN/IDK already early-returned with (E) above — never reaches here.)
        final_prompt = (
            self._p_enhanced_fallback.replace("{question}", q_str)
            .replace("{triples}", full_tkg)
            .replace("{history}", full_rag)
        )
        final_prompt = self._cap_bridge_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": ROUTE, "fallback_from": first_route,
                "bridge_resolved": bool(visited), "bridge_iters": iters,
                "bridge_chain": list(visited), "bridge_n_quads": len(acc_quads),
                "neutral_stop": "FINAL_ENHANCED",
                "rag_rendered": full_rag, "tkg_rendered": full_tkg,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    # ====================================================================
    # internal — enhanced_reranked retrieval (enhanced + path-aware rerank + compression)
    # ====================================================================
    @staticmethod
    def _rr_question_tokens(question: "Question") -> set[str]:
        """Extract the content-token (lowercase word) set from the question text for path-aware rerank.

        ⚠ get_clean_entity removes *even whitespace* (for single-entity normalization), so it must not be used
        for sentence tokenization. First split on whitespace → strip only punctuation from each word (lowercase)
        → keep only alphabetic tokens of length >= 3.
        (Length < 3 and pure stopwords are matching noise, so they are excluded — prevents boosting on common words.)
        """
        text = (question.text or "").lower()
        toks: set[str] = set()
        for w in text.split():
            w = w.strip(".,!?;:\"'()[]{}*").strip()
            if len(w) >= 3 and any(c.isalpha() for c in w):
                toks.add(w)
        return toks

    @staticmethod
    def _rr_question_dates(question: "Question") -> set[str]:
        """Extract the set of "Month DD, YYYY" dates from the question text, normalized (lowercase, comma removed).

        A 2-hop question usually has two dates (e.g. "Julie went out on June 11, 1995 → how old as of
        September 22, 1994"). Both the hop-1 seed quad's date (first date) and the hop-2 answer quad's date
        (second date) appear in the question → date-match is a strong signal for identifying the answer quad.
        (None → safe: quad has no date.)
        """
        toks = set()
        for m in _RR_DATE_RE.findall(question.text or ""):
            toks.add(m.replace(",", "").lower().strip())
        return toks

    def _rr_score_items(
        self,
        items: list["Item"],
        text_of: "Any",
        q_tokens: set[str],
        seed_tokens: set[str],
        bridge_tokens: "set[str] | None" = None,
        q_dates: "set[str] | None" = None,
    ) -> list[tuple[int, float]]:
        """Score items with a path-aware boosted score. Returns: [(orig_idx, score), ...] (input order preserved).

        base   = (1 - position_norm): preserve the existing retrieval rank (cos_sim/alternating) in [0,1].
        entity = overlap ratio between the item text and question content tokens (0~1).
        seed   = whether a q_subj seed-node (2-hop anchor) token appears in the item (1/0).
        bridge = whether a bridge entity (hop-2 target candidate discovered at hop-1) token appears in the item (1/0).
        date   = (quad only) 1/0 if the quad's start_date is one of the question dates.
        combo  = (quad only) bridge AND date both 1 → a strong identification signal for the hop-2 answer quad
                 ({Nate,age,...,Sep22 1994}) (second date + bridge entity). Preserves answers that token-overlap misses.
        boosted = base + eb·entity + sb·seed + brb·bridge + dm·date + combo·(bridge&date).
        """
        bridge_tokens = bridge_tokens or set()
        q_dates = q_dates or set()
        n = len(items)
        out: list[tuple[int, float]] = []
        for i, it in enumerate(items):
            base = 1.0 - (i / n if n > 1 else 0.0)
            txt = (text_of(it) or "").lower()
            if q_tokens:
                hit = sum(1 for t in q_tokens if t in txt)
                ent = hit / len(q_tokens)
            else:
                ent = 0.0
            seed = 1.0 if (seed_tokens and any(s in txt for s in seed_tokens)) else 0.0
            # bridge: for a quad, whether head/tail is a bridge entity (exact); for dialogue, text substring match.
            meta = it.meta if isinstance(it.meta, dict) else {}
            if meta.get("head") is not None and meta.get("tail") is not None:
                ht = (str(meta.get("head", "")) + " " + str(meta.get("tail", ""))).lower()
                bridge = 1.0 if (bridge_tokens and any(b in ht for b in bridge_tokens)) else 0.0
                qd = str(meta.get("start_date", "")).replace(",", "").lower().strip()
                date = 1.0 if (q_dates and qd in q_dates) else 0.0
            else:
                bridge = 1.0 if (bridge_tokens and any(b in txt for b in bridge_tokens)) else 0.0
                date = 0.0
            combo = 1.0 if (bridge > 0 and date > 0) else 0.0
            boosted = (base + self.rr_entity_boost * ent + self.rr_seed_boost * seed
                       + self.rr_bridge_boost * bridge + self.rr_date_boost * date
                       + self.rr_combo_boost * combo)
            out.append((i, boosted))
        return out

    @staticmethod
    def _rr_bridge_entities(
        tkg_items: list["Item"], seed_tokens: set[str],
        q_dates: "set[str] | None" = None, max_bridges: int = 6,
    ) -> set[str]:
        """Among the retrieved TKG quads, extract the *opposite endpoint* of quads that have the seed entity
        (q_subj) as head/tail as the bridge entity → identifies the middle node of the 2-hop path (e.g. Nate in
        Julie→Nate).

        Returns: lowercase token set of bridge entities (full string + each word of len >= 3). _rr_score_items
        uses these tokens to boost bridge-entity-related quads (hop-2 target: {Nate, age, ...}).
        - Priority: (1) seed quads whose date matches the question date (first date) first (exact hop-1),
          (2) if insufficient, supplement with date-agnostic seed quads. → minimizes noisy bridges (unrelated quad
          endpoints).
        - Empty set if seed_tokens is empty. max_bridges caps the count (prevents over-expansion).
        - Lightweight: O(n) over tkg_items, no embeddings/graph traversal. Uses meta(head/tail) directly.
        """
        if not seed_tokens:
            return set()
        q_dates = q_dates or set()
        bridges: list[str] = []
        seen: set[str] = set()

        def _collect(date_anchored: bool) -> None:
            for it in tkg_items:
                if len(bridges) >= max_bridges:
                    return
                meta = it.meta if isinstance(it.meta, dict) else {}
                head = str(meta.get("head", "")).strip()
                tail = str(meta.get("tail", "")).strip()
                if not head or not tail:
                    continue
                if date_anchored:
                    qd = str(meta.get("start_date", "")).replace(",", "").lower().strip()
                    if not q_dates or qd not in q_dates:
                        continue
                hl, tl = head.lower(), tail.lower()
                head_is_seed = any(s in hl for s in seed_tokens)
                tail_is_seed = any(s in tl for s in seed_tokens)
                cand = None
                if head_is_seed and not tail_is_seed:
                    cand = tail
                elif tail_is_seed and not head_is_seed:
                    cand = head
                if cand and cand.lower() not in seen:
                    seen.add(cand.lower())
                    bridges.append(cand)

        # pass1: date-anchored seed quads (exact hop-1) → pass2: date-agnostic supplement.
        _collect(date_anchored=True)
        if len(bridges) < max_bridges:
            _collect(date_anchored=False)

        bridge_tokens: set[str] = set()
        for b in bridges:
            cb = b.lower().strip()
            if not cb:
                continue
            bridge_tokens.add(cb)
            for w in cb.split():
                w = w.strip(".,!?;:\"'()[]{}")
                if len(w) >= 3 and any(c.isalpha() for c in w):
                    bridge_tokens.add(w)
        return bridge_tokens

    def _retrieve_enhanced_reranked(self, question: Question, idx: int,
                                    current_session: "Session | None" = None) -> RetrievalResult:
        """enhanced skeleton (RAG top-k + TKG top-k simultaneously) → path-aware rerank → top compress_* compression.

        recall: the retrieval set (rag.items + tkg.items) is received the same as enhanced, so it is preserved.
        Difference: the quads/dialogues fed into final_qa_prompt are (1) reranked by relevance to the question's
              2-hop path, then (2) kept to only the top rr_compress_tkg / rr_compress_rag, compressing them
              (110k → ~20-30k chars).
        The prompt is the same qa_enhanced.txt as enhanced (the enhanced_reranked branch of build_qa_prompt).
        """
        rag = self._retrieve_rag(question, idx, route="RAG_in_enhanced",
                                  current_session=current_session)
        tkg = self._retrieve_tkg(question, idx, route="TKG_in_enhanced",
                                 current_session=current_session)
        metrics = rag.metrics.merge(tkg.metrics)

        # path-aware rerank material: question content tokens + q_subj seed-node (2-hop anchor) tokens.
        q_tokens = self._rr_question_tokens(question)
        # seed_tokens: lowercase of the q_subj seed-node (e.g. "Ross Geller") as (a) the full string (whitespace
        #   preserved, for multi-word substring match) + (b) each word (len>=3). ⚠ get_clean_entity removes even
        #   whitespace, so it won't substring-match the item text ("Ross joined ...") → use whitespace-preserved
        #   lowercase only here.
        seed_tokens: set[str] = set()
        _tpi = tkg.extra.get("tkg_path_info") or {}
        for s in (_tpi.get("matched_subjs") or []):
            cs = str(s).lower().strip()
            if not cs:
                continue
            seed_tokens.add(cs)  # full (whitespace preserved) — multi-word substring like "ross geller"
            for w in cs.split():
                w = w.strip(".,!?;:\"'()[]{}")
                if len(w) >= 3 and any(c.isalpha() for c in w):
                    seed_tokens.add(w)  # individual words — "ross", "geller"

        # --- bridge entity extraction (2-hop path) ---
        # The opposite endpoint of retrieved quads that have the seed entity (q_subj) as head/tail = bridge entity.
        #   (e.g. from {Julie, going with, Nate, ...}, seed=Julie → bridge=Nate.)
        #   The hop-2 question's answer quad ({Nate, age, ...}) doesn't surface via token-overlap and gets cut in
        #   compression — this defect is preserved by boosting with bridge_tokens. Applied identically to both RAG/TKG rerank.
        q_dates = self._rr_question_dates(question)
        tkg_items = list(tkg.items)
        bridge_tokens = self._rr_bridge_entities(tkg_items, seed_tokens, q_dates)

        # --- TKG path rerank + compression ---
        tkg_scored = self._rr_score_items(
            tkg_items, lambda it: it.text, q_tokens, seed_tokens, bridge_tokens, q_dates
        )
        # stable sort: score descending, ties keep the original (=cos_sim) order (orig_idx ascending).
        tkg_order = sorted(tkg_scored, key=lambda x: (-x[1], x[0]))
        tkg_keep_items = [tkg_items[i] for i, _ in tkg_order[: self.rr_compress_tkg]]
        tkg_quads = [it.meta for it in tkg_keep_items if isinstance(it.meta, dict)]
        # render (same helper as the cos_sim_rerank path; context_size is unlimited, so compression is by count only).
        q_str = render_question(question, self.question_format)
        base_p_tkg = self._p_quad + q_str
        tkg_rendered, tkg_c = _render_quads_in_order(
            tkg_quads, self.quad_format, self.context_size, base_p_tkg
        )
        if not tkg_quads:
            tkg_rendered = "No relevant triple!"

        # --- RAG path rerank + compression ---
        rag_items = list(rag.items)
        rag_scored = self._rr_score_items(
            rag_items, lambda it: it.text, q_tokens, seed_tokens, bridge_tokens, q_dates
        )
        rag_order = sorted(rag_scored, key=lambda x: (-x[1], x[0]))
        rag_keep_items = [rag_items[i] for i, _ in rag_order[: self.rr_compress_rag]]
        rag_rendered = ""
        for it in rag_keep_items:
            meta = it.meta if isinstance(it.meta, dict) else {}
            rag_rendered += _format_dialogue_chunk(
                it.text, meta.get("date", ""), meta.get("session_idx", -1)
            )
        if not rag_rendered:
            rag_rendered = "No relevant history!"

        return RetrievalResult(
            items=rag_keep_items + tkg_keep_items,
            rendered="",  # build_qa_prompt uses rag_rendered/tkg_rendered from extra
            extra={
                "route": "ENHANCED_RERANKED",
                "rag_rendered": rag_rendered,
                "tkg_rendered": tkg_rendered,
                "rag_items": rag_keep_items,
                "tkg_items": tkg_keep_items,
                # [retrieval recall] B = before compression (same retrieval set as enhanced), C = the compressed part included in the prompt.
                "retrieval_b_rag": rag.extra.get("retrieval_b", []),
                "retrieval_c_rag": _dedup_dialogue_items(
                    [(it.meta.get("session_idx"), it.meta.get("date"))
                     for it in rag_keep_items if isinstance(it.meta, dict)]
                ),
                "retrieval_b_tkg": tkg.extra.get("retrieval_b", []),
                "retrieval_c_tkg": _dedup_quad_items(tkg_c),
                "tkg_path_info": tkg.extra.get("tkg_path_info"),
                # compression stats (for analysis): counts before/after compression.
                "rr_n_tkg_before": len(tkg_items),
                "rr_n_tkg_after": len(tkg_keep_items),
                "rr_n_rag_before": len(rag_items),
                "rr_n_rag_after": len(rag_keep_items),
                "rr_n_bridge_tokens": len(bridge_tokens),
            },
            metrics=metrics,
        )

    # ====================================================================
    # internal — q_subj identification
    # ====================================================================
    def _identify_q_subj(
        self, question: Question, nodes: list[str], metrics: CallMetrics,
        question_idx: "int | None" = None,
        debug_out: "dict | None" = None,
    ) -> tuple[list[str], list[str], str]:
        """Extract q_subj candidates (batch + precompute support).

        - If precomputed_q_subj[question_idx] exists, skip the LLM (precompute first, same pattern as q_clf).
        - Otherwise split nodes into q_subj_batch_size (default 100) chunks and call _q_subj_one_batch per batch →
          merge candidates (dedup, first-seen order). By design: 100-batch.
        Returns: (kg_nodes: list[str], raw_subjects: list[str], last_prompt_text: str)
        - kg_nodes: the final q_subj after mapping the stage2 LLM output to KG nodes and applying the _allowed filter
          (used by ablation).
        - raw_subjects: the raw question subjects emitted by the stage2 LLM (*before* KG mapping/_allowed filter, for
          analysis). By design, both raw + kg_nodes are stored.
        - prompt_text: the full prompt of the last batch (empty string if no LLM call/precompute).
        """
        # precompute first (per-llm q_subj.json) — 0 LLM calls.
        if question_idx is not None and question_idx in self.precomputed_q_subj:
            return list(self.precomputed_q_subj[question_idx]), [], ""
        # Review #2 MEDIUM-E11: trivial return when nodes < k (sparse KG) — LLM skip.
        if not nodes:
            return [], [], ""
        k = min(self.num_q_subj_candidates, len(nodes))
        # Removed the k<=1 (single node) early return — even with 1 node, it must go through q_subj_one_batch
        #   (subject identification) to be accurate. (The early return unconditionally returned nodes[0] regardless
        #   of subject = inaccurate.)
        bs = max(1, self.q_subj_batch_size)
        merged: list[str] = []
        seen: set[str] = set()
        last_prompt = ""
        # stage1 batch parallelization. Batches are independent (each sees only nodes[s:s+bs]) → concurrent LLM calls.
        #   Results are merged in index order, so it is bit-identical to sequential (same merged order, same last_prompt).
        #   metrics are received as independent CallMetrics per batch and summed in a single-threaded merge (avoids
        #   shared += race). q_subj_batch_workers=1 is fully identical to the original sequential path.
        _starts = list(range(0, len(nodes), bs))
        _bw = max(1, min(self.q_subj_batch_workers, len(_starts) or 1))

        def _run_batch(_bi: int) -> "tuple[int, list[str], str, CallMetrics]":
            _s = _starts[_bi]
            _bm = CallMetrics()
            _cands, _prompt = _q_subj_one_batch(
                self.llm, self._p_q_subj_idf, question.text, question.questioner,
                k, nodes[_s:_s + bs], _bm,
            )
            return _bi, _cands, _prompt, _bm

        _results: "list[tuple[list[str], str, CallMetrics] | None]" = [None] * len(_starts)
        if _bw <= 1:
            for _bi in range(len(_starts)):
                _, _c, _p, _bm = _run_batch(_bi)
                _results[_bi] = (_c, _p, _bm)
        else:
            with ThreadPoolExecutor(max_workers=_bw) as _bex:
                for _bi, _c, _p, _bm in _bex.map(_run_batch, range(len(_starts))):
                    _results[_bi] = (_c, _p, _bm)
        # index-order merge (same first-seen order as sequential) + metrics summation (single-threaded, no race).
        for _entry in _results:
            if _entry is None:
                continue
            cands, prompt, _bm = _entry
            last_prompt = prompt
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    merged.append(c)
            if metrics is not None:
                metrics.latency_sec += _bm.latency_sec
                metrics.input_tokens += _bm.input_tokens
                metrics.output_tokens += _bm.output_tokens
                metrics.num_llm_calls += _bm.num_llm_calls
                metrics.num_embedding_calls += _bm.num_embedding_calls
        # q_subj 2-pass stage2: feed stage1's merged (union of each batch's q_subj_idf candidates) back into the
        #   q_subj_idf_select prompt as options, and select the final k q_subj only from the options (= a subset of
        #   current TKG nodes). If merged is empty, skip stage2 (return []).
        #   q_subj_check (TKG node existence verification) was removed — node matching (exact, no fuzzy) is done by
        #   the caller _retrieve_tkg. Since the LLM may hallucinate entities outside the options, filter final once
        #   more against the merged set.
        if merged:
            final, sel_prompt = _q_subj_one_batch(
                self.llm, self._p_q_subj_select, question.text, question.questioner, k, merged, metrics
            )
            last_prompt = sel_prompt
            _merged_set = set(merged)
            # The final q_subj must always be within (current TKG node set + questioner).
            #   Even if the stage1/stage2 LLM hallucinates entities outside the options, they are removed here.
            #   The questioner is allowed for first-person questions.
            _allowed = set(nodes)
            if question.questioner:
                _allowed.add(question.questioner)
            filtered = [c for c in final if c in _merged_set and c in _allowed]
            # raw_subjects = the stage2 LLM output final (before KG mapping/_allowed filter).
            raw_subjects = list(final)
            # Prevent 0-candidate collapse from stage2 LLM caprice: if stage1 has candidates but stage2 discards
            #   them all, keep the top k of stage1 (within allowed) (avoids wiping out q_subj-based TKG retrieval).
            #   The first-person questioner is additionally guaranteed by the caller _retrieve_tkg. A genuine 'no
            #   subject' already yields merged=[] at stage1.
            result = filtered if filtered else [c for c in merged if c in _allowed][:k]
        else:
            result = merged
            raw_subjects = []
        if debug_out is not None and question_idx is not None:
            debug_out[question_idx] = {
                "merged": list(merged),
                "raw_subjects": list(raw_subjects),
                "kg_nodes": list(result),
            }
        return result, raw_subjects, last_prompt


# ============================================================================
# Module-level helpers
# ============================================================================
def _q_subj_one_batch(llm, prompt_template: str, question_text: str, questioner: str,
                      k: int, nodes_batch: list[str],
                      metrics: "CallMetrics | None" = None) -> tuple[list[str], str]:
    """One q_subj_idf LLM call with nodes_batch as options. Returns (candidates, prompt).

    structured (QSubjList) first → free-form (ast.literal_eval) fallback on failure — **identical** to the parse
    semantics of the original _identify_q_subj (protects Qwen2.5 reproducibility). If metrics is None, skip accumulation.
    llm is an object with generate_structured/generate (returning GenerationRecord) (tkg_hybrid's self.llm).
    """
    nodes_str = "[" + ", ".join(nodes_batch) + "]"
    prompt = (
        prompt_template.replace("{question}", question_text)
        .replace("{questioner}", questioner)
        .replace("{k}", str(k))
        .replace("{options}", nodes_str)
    )
    try:
        parsed, rec = llm.generate_structured(prompt, QSubjList)
        if metrics is not None and rec is not None:
            metrics.latency_sec += rec.latency_sec
            metrics.input_tokens += rec.input_tokens
            metrics.output_tokens += rec.output_tokens
            metrics.num_llm_calls += 1
        if isinstance(parsed, QSubjList) and parsed.subjects:
            return [str(x) for x in parsed.subjects], prompt
    except Exception:
        pass
    # free-form fallback (legacy — protects Qwen2.5 reproducibility)
    gen = llm.generate(prompt)
    if metrics is not None:
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
    import ast
    try:
        val = ast.literal_eval(gen.text.strip())
        if isinstance(val, list):
            return [str(x) for x in val], prompt
        if isinstance(val, str):
            return [val], prompt
    except (ValueError, SyntaxError):
        pass
    return [], prompt


def _format_dialogue_chunk(text: str, date: str, sess_idx: int) -> str:
    return f"\n[Date: {date} (Session {sess_idx})]\n{text}\n"


def _date_matches_any(session_date: str, q_timestamps: dict) -> bool:
    for t, meta in q_timestamps.items():
        tp = meta["type"]
        if tp == "month_day_year" and session_date == t:
            return True
        if tp == "month_year":
            parts = session_date.split()
            if len(parts) >= 2 and " ".join([parts[0], parts[-1]]) == t:
                return True
        if tp == "year_only" and session_date.split()[-1] == t:
            return True
    return False


def _retrieve_by_timestamp(kg: KGStore, timestamps: dict) -> list[dict]:
    out: list[dict] = []
    for t, meta in timestamps.items():
        tp = meta["type"]
        for h in kg.tkg:
            for r in kg.tkg[h]:
                for ts in kg.tkg[h][r]:
                    match = False
                    if tp == "month_day_year":
                        match = ts == t
                    elif tp == "month_year":
                        parts = ts.split()
                        match = len(parts) >= 2 and " ".join([parts[0], parts[-1]]) == t
                    elif tp == "year_only":
                        parts = ts.split() if ts else []
                        match = bool(parts) and parts[-1] == t
                    if match:
                        for tl in kg.tkg[h][r][ts]:
                            q = {"head": h, "relation": r, "tail": tl, "start_date": ts,
                                 "sess_idx": kg.quad_sess.get((h, r, tl, ts))}
                            if q not in out and is_valid_quad(q):
                                out.append(q)
    return out


def _filter_quads_by_timestamp(quads: list[dict], timestamps: dict) -> list[dict]:
    out: list[dict] = []
    for q in quads:
        qs = q.get("start_date") or ""
        qs_parts = qs.split() if qs else []
        for t, meta in timestamps.items():
            tp = meta["type"]
            if (
                (tp == "month_day_year" and qs == t)
                or (tp == "month_year" and len(qs_parts) >= 2 and " ".join([qs_parts[0], qs_parts[-1]]) == t)
                or (tp == "year_only" and bool(qs_parts) and qs_parts[-1] == t)
            ):
                if q not in out:
                    out.append(q)
                break
    return out


def _alternate_merge(order_a: list[int], order_b: list[int], bm25_first: bool = True) -> list[int]:
    """Merge two rankings into one list by alternating (dedup, preserve appearance order)."""
    out: list[int] = []
    seen: set[int] = set()
    a, b = (order_a, order_b) if bm25_first else (order_b, order_a)
    ai = bi = 0
    while ai < len(a) or bi < len(b):
        while ai < len(a) and a[ai] in seen:
            ai += 1
        if ai < len(a):
            out.append(a[ai])
            seen.add(a[ai])
            ai += 1
        while bi < len(b) and b[bi] in seen:
            bi += 1
        if bi < len(b):
            out.append(b[bi])
            seen.add(b[bi])
            bi += 1
    return out


def _quad_to_string(q: dict) -> str:
    # quad list element format: {head, relation, tail, date}
    return f'{{{q["head"]}, {q["relation"].lower()}, {q["tail"]}, {q["start_date"]}}}'


def _render_quads_by_date(quads: list[dict], quad_format: str, context_size: int, base_prompt: str) -> str:
    # render quadruples as a list: [{head, relation, tail, date}, ...]
    # (cos_sim_rerank=false path. quads are in the caller's sort order. No grouping by date.)
    if not quads:
        return "No relevant triple!"
    base_tokens = count_tokens(base_prompt)
    cur = base_tokens + 2  # allowance for the enclosing "[" "]" tokens
    items: list[str] = []
    for q in quads:
        date = q.get("start_date") or ""
        if quad_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        st = count_tokens(s) + 2  # allowance for the ", " separator tokens
        if cur + st > context_size:
            break
        items.append(s)
        cur += st
    return "[" + ", ".join(items) + "]"


def _dedup_quad_items(quads: "list[dict]") -> "list[dict]":
    """quad set for retrieval recall B/C — normalize (head,relation,tail,start_date) + dedup (order-agnostic set)."""
    seen: set = set()
    out: list = []
    for q in quads or []:
        if not isinstance(q, dict):
            continue
        sig = (q.get("head") or "", q.get("relation") or "", q.get("tail") or "", q.get("start_date") or "")
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"head": sig[0], "relation": sig[1], "tail": sig[2], "start_date": sig[3]})
    return out


def _dedup_dialogue_items(pairs: "list[tuple]") -> "list[dict]":
    """dialogue set for retrieval recall B/C — dedup by session_idx. pairs=[(session_idx, date), ...]."""
    seen: set = set()
    out: list = []
    for s_idx, date in pairs or []:
        if s_idx in seen:
            continue
        seen.add(s_idx)
        out.append({"session_idx": s_idx, "date": date})
    return out


def _render_quads_in_order(quads: list[dict], quad_format: str, context_size: int,
                           base_prompt: str) -> "tuple[str, list[dict]]":
    """Render quads sorted in descending cos_sim in that exact order (no date-grouping).

    Returns (retrieval recall B/C measurement): (rendered_text, included_quads).
      included_quads = the list of quad dicts actually fit within context_size (= C, the part included in the QA prompt).
      Previously only the rendered text was returned; the fit quads are also returned to extract C (the set) for retrieval recall.


    BUG FIX: an alternative renderer to fix the problem where _render_quads_by_date re-groups by date and destroys
    the similarity order. quads must already be sorted by the caller so the highest-sim comes first. Here, preserving
    that order, quads are added one at a time from the top (top-first), filling only until just before base_prompt +
    accumulated quad tokens exceed context_size.
    → The highest-similarity quad sits near the front of the prompt and survives even under truncation.

    render quadruples as a list: [{head, relation, tail, date}, ...] (date is included as the quad's 4th element,
    the whole wrapped in "[" ... "]"). No grouping by date. Token budget accounting starts from base_prompt tokens,
    same as _render_quads_by_date.
    """
    if not quads:
        return "No relevant triple!", []
    base_tokens = count_tokens(base_prompt)
    cur = base_tokens + 2  # allowance for the enclosing "[" "]" tokens
    items: list[str] = []
    included: list[dict] = []  # quads actually fit within context_size (= C, for retrieval recall measurement)
    for q in quads:
        date = q.get("start_date") or ""
        if quad_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        st = count_tokens(s) + 2  # allowance for the ", " separator tokens
        # stop if base_prompt + accumulated so far + this quad exceeds context_size (preserves top-first).
        # context_size<=0 = token limit off → fill all without cutting (unlimited).
        if context_size > 0 and cur + st > context_size:
            break
        items.append(s)
        included.append(q)
        cur += st
    # If not even the first quad fits and items is empty, keep at least one quad, trimmed to context_size.
    if not items and quads:
        q = quads[0]
        date = q.get("start_date") or ""
        if quad_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        items = [trim_text_to_tokens(s, max(1, context_size - 2))]
        included = [quads[0]]
    return "[" + ", ".join(items) + "]", included
