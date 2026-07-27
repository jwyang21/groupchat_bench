"""TKG-hybrid memory — live retrieval mode with configurable settings for ablation.

Covers all 12 ablation axes (1-2 retrieval method, 1-3 reranker, 1-4 num_retrieved,
1-5 RAG method, 1-6 context size, 1-9 per-prompt variation, 1-10 fallback toggle, etc.).

This class takes a broad config dict and branches per axis, so every axis can be
ablated via a single line change in the yaml config.
"""
from __future__ import annotations

import logging
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
    local_first,  # local-first read for embeddings (reduce shared-storage load)
    process_answer,
    remove_brackets,
    remove_quotes,
    render_question,
    save_pickle,
    trim_text_to_tokens,
)

from .base import Memory

# Module-level q_clf cache — avoids re-calling q_clf for the same (LLM, question)
# across ablation cells (time-saving fix).
# key = (id(llm), question_text), value = "TKG" or "RAG"
_Q_CLF_CACHE: dict[tuple, str] = {}
_Q_CLF_CACHE_LOCK = threading.Lock()

# Process-level pool of EmbeddingCache instances.
# Recreating the EmbeddingCache for the same path per ablation grid / LLM triggers a
# pickle load every time. Sharing path -> instance removes a major ablation-throughput bottleneck.
# key = Path.resolve() string
_CACHE_POOL: dict[str, "EmbeddingCache"] = {}
_CACHE_POOL_LOCK = threading.Lock()


def _get_or_create_cache(
    path: Path,
    model_name: str,
    dim: int,
    autoflush_every: int = 50,
    fsync_on_flush: bool = False,
) -> "EmbeddingCache":
    """Return a path-keyed singleton EmbeddingCache.

    The second and later calls for the same (path, model_name, dim) return the same
    instance immediately. Reusing a path with a different (model_name, dim) raises
    ValueError — a caller bug.
    """
    key = str(path.resolve())
    with _CACHE_POOL_LOCK:
        existing = _CACHE_POOL.get(key)
        if existing is not None:
            # Same path but different model/dim = caller error.
            if existing.model_name != model_name or existing.dim != dim:
                raise ValueError(
                    f"[_get_or_create_cache] path={key!r} is already registered with "
                    f"model={existing.model_name!r}/dim={existing.dim}, "
                    f"but reuse requested model={model_name!r}/dim={dim} — mismatch."
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
    """q_clf classification response schema. label in {"TKG","RAG"}."""
    label: Literal["TKG", "RAG"] = Field(description="TKG (knowledge graph) or RAG (raw dialogue)")


class QSubjList(BaseModel):
    """q_subj_idf response schema. A list of candidate entities."""
    subjects: list[str] = Field(description="question subject candidate names")


def _normalize_q_clf_label(raw: str) -> str:
    """Unify q_clf label normalization.

    v1_2 baseline: `q_clf.upper(); "TKG" if "KG" in q_clf else "RAG"`.
    Ensures structured / free-form / precomputed all go through the same path.
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
    # Future-leakage prevention: quad -> the session_idx where the quad first appeared.
    #   key=(h,r,t,ts), val=min sess_idx. At retrieve time, keep only quads before
    #   current_session_idx (applies the no-future-leakage principle to the TKG path).
    quad_sess: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def insert_quad(self, head: str, relation: str, tail: str, ts: str,
                    sess_idx: int | None = None) -> None:
        # BUG FIX: .capitalize() lowercases everything after the first word of a
        # multi-word entity (e.g. "Lincoln High" -> "Lincoln high", "Helen Willick-Bunch"
        # -> "Helen willick-bunch"), corrupting the node form. This produced inaccurate
        # entity strings in the rendered prompt and dedup mismatches.
        # Fix: preserve the raw form. Case-insensitive matching is handled on the query
        # side (`get_clean_entity` exact match).
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
        #   Update even when t is already in bucket, to reflect an earlier session. Skip if sess_idx is None (legacy).
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
        """Return head/tail only for quads that appeared before current_session_idx (future-leakage prevention).

        - current_session_idx is None -> full nodes() (snapshot path — already time-accurate KG, no future quads).
        - Entities of quads with no quad_sess record (sess_idx None) are kept (live commit always
          records, so this only happens on the snapshot path). The comparison
          `sess is None or sess < idx` matches the quad-filter semantics of
          `_filter_quads_by_timestamp`, keeping q_subj options and retrieved quads
          temporally consistent.
        - Head `h` is not added unconditionally (unlike nodes()); it is added only when that
          (h,r,t,ts) quad is in the past, preventing exposure of entities from future-only quads.
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
        """TKG at retrieve time (C) = A (entire quads accumulated up to the previous session) + B (current-session partial quads).

        TKG/RAG symmetry: exactly mirrors the RAG path's
          "pool = (entire dialogue with s_idx < idx) + (current-session partial_dialogue)".
        Builds a new ephemeral KGStore:
          - A: copy only quads of self.tkg with sess_idx < current_session_idx (or sess_idx None =
               legacy/snapshot). Same temporal semantics as nodes_until / _filter_quads_by_timestamp.
          - B: insert extra_quads (= current_session.partial_quadruples) with
               sess_idx = current_session_idx. Quads from future sessions (>idx) never enter through
               any path -> future-leakage prevented.
        current_session_idx is None (snapshot path) -> A is all of self.tkg (already time-accurate) + B.

        The returned KGStore is C, the target of q_subj candidate nodes, 2-hop one_hop traversal,
        and _retrieve_by_timestamp. self.tkg (the permanent committed accumulation) is not mutated (read-only copy).
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
# TKGHybridMemory — live retrieval, 12 configurable axes
# ============================================================================
class TKGHybridMemory(Memory):
    """Configurable settings for ablation — keys of the config dict:

    retrieval_method   : 'tkg_only' / 'rag_only' / 'tkg_rag' / 'enhanced'
                         / 'enhanced_fallback' / 'enhanced_fallback_inverse'   (1-2)
                         enhanced = no router; concat of RAG retrieval top k1 + TKG retrieval top k2.
                         enhanced_fallback = 1st tkg_rag (q_clf single path) -> if (E), 2nd enhanced.
                         enhanced_fallback_inverse = 1st enhanced -> if (E), 2nd q_clf single path.
    fallback           : bool — (E)-fallback (only meaningful in tkg_rag / enhanced_fallback /
                         enhanced_fallback_inverse modes)                       (1-10)
    num_retrieved_rag  : int — top-k1 of the RAG path.                          (1-4a)
    num_retrieved_tkg  : int — top-k2 of the TKG path.                          (1-4b)
    num_retrieved      : (deprecated, backward compat) — old single top-k. If set, k1=k2=num_retrieved.
                         Unset or 0 -> no top-k cut; fill up to the context_size budget (default).
                         Positive int (>0) -> cut to that top-k, then fill up to context_size.
                         Negative -> ValueError.
    context_size       : int — final prompt token limit (default 8000)         (1-6)

    rag_method         : 'bm25_only' / 'emb_only' / 'alternating'             (1-5)
    bm25_first         : bool — BM25 first in alternating
    rag_time_filter    : bool — narrow candidate sessions by the question timestamp

    hop                : int — number of TKG traversal hops (default 2)
    triple_normalization: 'en' / 'raw' — which prebuilt TKG variant to use      (1-1)
                          (the actual snapshot path is a separate cfg 'tkg_snapshot')

    reranker           : 'none' / 'bge' — reranking after retrieval             (1-3)

    prompt_rag/quad/enhanced/no_history/q_clf/q_subj_idf
                       : each prompt filename (relative to assets/prompt/)       (1-9)

    tkg_snapshot       : str | None — prebuilt KG pickle path.
                         If present, overwrite with that session's KG at commit time.
                         If absent, commit-based accumulation.

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
        # no_history: if True, skip all retrieval and use only prompt_no_history.
        self.no_history = bool(cfg.get("no_history", False))
        # Split top-k separately for the RAG path and the TKG path.
        # - num_retrieved_rag (k1): top-k1 of RAG retrieval results
        # - num_retrieved_tkg (k2): top-k2 of TKG retrieval results
        # Behavior:
        #   - retrieval_method != "enhanced": (k1==0 AND k2==0) -> fill up to context_size (existing behavior)
        #                                     (k1>0 AND k2>0) -> cut each path to its top-k
        #                                     mixed (only one is 0) -> ValueError
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
        if _rm == "enhanced":
            # enhanced top-k: quad(tkg) = 100 after cos_sim rerank,
            #   dialogue(rag) = 10 after bm25-cos_sim alternating. Prefer the enhanced-only config keys
            #   (enhanced_num_retrieved_tkg/rag); default 100/10 otherwise.
            #   The num_retrieved=0 budget policy of non-enhanced (tkg_only/rag_only/tkg_rag) is unchanged.
            self.num_retrieved_tkg = int(cfg.get("enhanced_num_retrieved_tkg", 100))
            self.num_retrieved_rag = int(cfg.get("enhanced_num_retrieved_rag", 10))
            if self.num_retrieved_tkg <= 0 or self.num_retrieved_rag <= 0:
                raise ValueError(
                    f"enhanced num_retrieved must be positive, got "
                    f"tkg={self.num_retrieved_tkg}, rag={self.num_retrieved_rag}"
                )
        else:
            # otherwise: both 0 or both positive (mixing forbidden)
            _rag_zero = (self.num_retrieved_rag == 0)
            _tkg_zero = (self.num_retrieved_tkg == 0)
            if _rag_zero != _tkg_zero:
                raise ValueError(
                    f"For retrieval_method='{_rm}', num_retrieved_rag and num_retrieved_tkg "
                    f"must both be 0 (context_size budget) OR both positive, "
                    f"got rag={self.num_retrieved_rag}, tkg={self.num_retrieved_tkg}"
                )
        # Backward compat: keep the old self.num_retrieved (external code compatibility).
        # Meaningful in non-enhanced cases where k1==k2. In enhanced, k1 and k2 may differ so
        # self.num_retrieved is unusable — but all code in this class is split into _rag/_tkg.
        self.num_retrieved = self.num_retrieved_rag if self.num_retrieved_rag == self.num_retrieved_tkg else -1
        self.context_size = int(cfg.get("context_size", 0))  # 0/negative = token limit off (unlimited)
        # If k>0 (top-k specified), automatically lift the context_size (8K upper limit).
        #   When num_retrieved_rag/tkg is positive (both positive per validation), override context_size
        #   to unlimited -> the top-k count becomes the real retrieve limit and the 8K prompt trim
        #   (_render_quads_in_order) is bypassed.
        #   Prompt size is bounded by top-k (top_tkg 250 ~ 12K tok), and the serving max-model-len
        #   (40960) is the real ceiling.
        #   When k=0 (both 0), fill the pool with the context_size (default 8000) budget as before.
        if self.num_retrieved_rag > 0 or self.num_retrieved_tkg > 0:
            self.context_size = 10 ** 9

        # RAG path
        self.rag_method = cfg.get("rag_method", "alternating")
        self.bm25_first = bool(cfg.get("bm25_first", True))
        self.rag_time_filter = bool(cfg.get("rag_time_filter", False))
        # Unified time-filter toggle — controls both the RAG path (narrowing candidates by the
        #   question timestamp) and the TKG path (_filter_quads_by_timestamp filtering the 2-hop
        #   result by the question timestamp).
        #   default = False (time-filtering off by default).
        #   If False, both time filters are bypassed (the question timestamp does not restrict
        #   retrieval). Set True for the original time-filter behavior.
        #   Note: TKG's _retrieve_by_timestamp (the only retrieval for no-subject questions) is
        #   retrieval, not a filter, so it is kept behind a separate toggle (retrieve_by_timestamp);
        #   turning it off leaves subjectless questions with nothing.
        self.time_filter = bool(cfg.get("time_filter", False))
        # retrieve_by_timestamp toggle — timestamp-only retrieval for no-subject questions.
        #   default True = original behavior (retrieve quads for subjectless questions by the question timestamp).
        #   If False, that path is disabled -> subjectless questions retrieve nothing even by timestamp (empty result).
        #   Part of "turning off all time-filtering" (a retrieval path rather than a filter, but
        #   timestamp-based, so turned off together).
        self.retrieve_by_timestamp = bool(cfg.get("retrieve_by_timestamp", True))

        # TKG path
        self.hop = int(cfg.get("hop", 2))
        # quad entity normalization type default = "raw".
        # The entity-consistency loss from max_candidates=200 truncation affects partial_en only,
        # not raw.
        self.triple_normalization = cfg.get("triple_normalization", "raw")
        self.quad_format = cfg.get("quad_format", "quad")  # quad / sentence
        self.num_q_subj_candidates = int(cfg.get("num_q_subj_candidates", 3))  # TKG seed node default = 3 (nq3 wins at k250/40)

        # Reranker (lazy)
        self.reranker_kind = cfg.get("reranker", "none")
        self._reranker = None

        # Prompts
        # question_format -> prompt auto-mapping.
        # If cfg.prompt_X is specified, that takes precedence; otherwise use the question_format default.
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
        # Prompt for the 2nd try of enhanced_fallback = always qa_enhanced.txt
        #   (always qa_enhanced.txt regardless of question_format / prompt_enhanced override).
        #   Injects both RAG retrieved and TKG retrieved (placeholders {triples}, {history}).
        self._p_enhanced_fallback = load_prompt("qa_enhanced.txt")
        self._p_no_hist = load_prompt(cfg.get("prompt_no_history") or _defaults["no_history"])
        self._p_q_clf = load_prompt(cfg.get("prompt_q_clf", "q_clf.txt"))
        self._p_q_subj_idf = load_prompt(cfg.get("prompt_q_subj_idf", "q_subj_idf.txt"))
        # q_subj 2-pass stage2 prompt — merge the stage1 batch candidates, then feed the merged
        #   set back as options and select the final k only from those options (a subset of the current TKG nodes).
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
        #   If absent, empty dict -> live _identify_q_subj (same pattern as q_clf).
        self.precomputed_q_subj: dict[int, list[str]] = cfg.get("precomputed_q_subj") or {}
        # q_subj options batch size (100). Split nodes into batch_size chunks, call, then merge.
        self.q_subj_batch_size: int = int(cfg.get("q_subj_batch_size", 100))
        # q_subj stage1 batch parallelism. default 1 = original sequential (results unchanged).
        #   >1: call the LLM concurrently over batches in an index-order-preserving ThreadPool
        #   (results bit-identical -> fair gemma comparison). For precompute speedup — not applied in
        #   ablation because of the precomputed_q_subj early-return. Total concurrent LLM calls =
        #   (question parallel max-workers) x (this value), so tune to endpoint capacity at launch.
        self.q_subj_batch_workers: int = max(1, int(cfg.get("q_subj_batch_workers", 1)))

        # Set question_embedding_type before the q_embed_path block that references it (otherwise AttributeError).
        # Question embedding type axis (distinct from quad embedding).
        # temporal     = embed the whole question
        # non_temporal = strip_question_timestamp(q) — embed after removing the timestamp + a 1-word time preposition
        # default = "temporal".
        _qet_q = cfg.get("question_embedding_type", "temporal")
        if _qet_q not in ("temporal", "non_temporal"):
            raise ValueError(
                f"question_embedding_type invalid: {_qet_q!r}; expected 'temporal' or 'non_temporal'"
            )
        self.question_embedding_type: str = _qet_q

        # Precomputed question embeddings (auto lookup — shared across all ablation cells).
        # path = {dataset_root}/{data_type}/precomputed/per-trial/t{trial}/question_embeddings_{model_alias}.npz
        # If absent, embed_one fallback.
        self._q_embed_map: dict[str, np.ndarray] = {}
        q_embed_path = cfg.get("precomputed_q_embed_path")  # explicit path takes precedence
        if not q_embed_path:
            ds_root = cfg.get("dataset_root")
            data_type = cfg.get("data_type")
            trial = cfg.get("trial")
            model = cfg.get("embedder_model") or getattr(embedder, "model", None) or "text-embedding-3-small"
            if ds_root and data_type is not None and trial is not None:
                alias = str(model).replace("/", "_").replace(":", "_")
                # question_embedding_type suffix path
                # (build_embeddings.py stores separate _temporal / _non_temporal npz files).
                candidate = (
                    f"{ds_root}/{data_type}/precomputed/per-trial/t{int(trial)}/"
                    f"question_embeddings_{alias}_{self.question_embedding_type}.npz"
                )
                if Path(candidate).exists():
                    q_embed_path = candidate
        if q_embed_path:
            try:
                data = np.load(local_first(q_embed_path), allow_pickle=True)
                qs = list(data["questions"])
                embs = data["embeddings"]
                self._q_embed_map = {str(q): embs[i] for i, q in enumerate(qs) if q is not None}
            except (OSError, ValueError, KeyError, EOFError) as e:
                # On load failure, fall back to embed_one. Keep empty dict.
                logging.warning("[TKGHybridMemory.__init__] q_embed_path load failed -> fallback: %s", e)
                self._q_embed_map = {}

        # Precomputed quadruple embeddings (injected by the entry point via cfg["quad_embeddings"]).
        # format: {quad_string_key: vector}
        #   temporal     = f"{head} {relation} {tail} on {start_date}"
        #   non_temporal = f"{head} {relation} {tail}"
        # If absent, empty dict (the cos_sim re-rank in the retrieve stage becomes a no-op automatically).
        self._quad_emb_map: dict[str, Any] = cfg.get("quad_embeddings") or {}

        # cos_sim re-rank toggle:
        # True  -> re-sort retrieved quads by cosine similarity to the question, descending.
        # False -> skip re-sorting (keep BM25/TKG order). default = False.
        self.cos_sim_rerank: bool = bool(cfg.get("cos_sim_rerank", False))

        # quad embedding type toggle.
        # "temporal"     -> key = f"{h} {r} {t} on {sd}"  (includes start_date, build filename _temporal)
        # "non_temporal" -> key = f"{h} {r} {t}"          (build filename _non_temporal)
        # default = "non_temporal". simulate yaml overrides to "temporal".
        # Can be added as an ablation grid axis (config grid's memory.quad_embedding_type).
        _qet = cfg.get("quad_embedding_type", "non_temporal")
        if _qet not in ("temporal", "non_temporal"):
            raise ValueError(f"quad_embedding_type invalid: {_qet!r}; expected 'temporal' or 'non_temporal'")
        self.quad_embedding_type: str = _qet

        # ====================================================================
        # 4 kinds of EmbeddingCache instances.
        # If cfg keys are absent, None — live embed fallback (keeps original behavior).
        # ====================================================================
        self._init_embedding_caches(cfg, embedder)

    # ====================================================================
    # Embedding cache init — 4 kinds (entire_dialogue/partial_dialogue/question/quad)
    # Required cfg keys: dataset_root, data_type, trial — if absent, cache disabled (live embed fallback).
    # Only the quad cache additionally needs llm_name + ie_type + normalization.
    # ====================================================================
    def _init_embedding_caches(self, cfg: dict[str, Any], embedder: "Embedder") -> None:
        self._entire_dialogue_cache: EmbeddingCache | None = None
        self._partial_dialogue_cache: EmbeddingCache | None = None
        self._question_cache: EmbeddingCache | None = None
        self._partial_quad_cache: EmbeddingCache | None = None
        self._entire_quad_cache: EmbeddingCache | None = None

        ds_root = cfg.get("dataset_root")
        dt = cfg.get("data_type")
        trial = cfg.get("trial")

        # Prevent dt=0 from wrongly passing — isinstance + valid-value check.
        if not isinstance(dt, str) or dt not in {"normal", "newname", "shuffled"}:
            # Required key absent or invalid dt — cache disabled. live embed fallback.
            return
        if not (ds_root and trial is not None):
            return

        ds_root = Path(ds_root)
        dt = str(dt)
        trial = int(trial)

        # Use _get_or_create_cache to build the EmbeddingCache so instances for the same path are
        # reused within the process (better ablation throughput).
        # autoflush_every=50 + fsync_on_flush=False — reduces runtime hot-path latency
        # (the build script pre-fills the cache, so runtime misses are rare).
        _kw = dict(autoflush_every=50, fsync_on_flush=False)

        try:
            # entire dialogue — once per (ds x dt) (trial-independent)
            self._entire_dialogue_cache = _get_or_create_cache(
                ds_root / dt / "entire_session_embedding_cache.pickle",
                embedder.name, embedder.dim, **_kw,
            )
            # partial dialogue — ds x dt x trial
            self._partial_dialogue_cache = _get_or_create_cache(
                ds_root / dt / "precomputed" / "per-trial" / f"t{trial}"
                       / "partial_dialogues_embedding_cache.pickle",
                embedder.name, embedder.dim, **_kw,
            )
            # question — ds x dt x trial x type (temporal/non_temporal)
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

        # quad cache — per (LLM x ie x normalization) (only when LLM info is present)
        llm_name = cfg.get("llm_name")
        ie_type = cfg.get("ie_type", "openie")
        norm = cfg.get("normalization", "raw")
        # cache_budget branch.
        #   data_loader reads quad json under `bNNNNN` in the LLM folder, so the quad's embedding
        #   cache must be read from the same segment for path consistency (if absent, no segment = original behavior).
        from groupchat_qa.pipeline.data_loader import budget_segment as _budget_segment
        _bseg = _budget_segment(cfg.get("cache_budget"))
        if llm_name:
            # Add the embedding_type suffix to the quad cache path
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

        # Lazy-prime precomputed values from the in-memory map into the cache.
        # If _q_embed_map (npz) is populated, register it in the question cache too, so lookups go
        # through a single cache.get path.
        # The quad map can be large, so it is set at first lookup (handled inside _cached_quad_embed below).
        if self._question_cache is not None:
            for _text, _vec in self._q_embed_map.items():
                if self._question_cache.get(_text) is None:
                    try:
                        self._question_cache.set(_text, _vec)
                    except (OSError, ValueError, RuntimeError):
                        pass

    # ------------------------------------------------------------------
    # Common handling for empty text.
    # Return a uniform unit vector instead of a zero-vec to avoid cos_sim ZeroDivisionError.
    # ------------------------------------------------------------------
    def _empty_vec(self) -> np.ndarray:
        dim = self.embedder.dim
        v = np.ones(dim, dtype=np.float32)
        v /= np.sqrt(dim)
        return v

    def _cached_entire_dialogue_embed(self, text: str) -> np.ndarray:
        """entire dialogue embedding used by commit().

        Separates the cache-write path from _rag_order_adhoc's partial pool.
        lookup: entire -> partial chain. On miss, set in the entire cache.
        """
        if not text:
            return self._empty_vec()
        # entire first, then partial on miss
        for c in (self._entire_dialogue_cache, self._partial_dialogue_cache):
            if c is None:
                continue
            v = c.get(text)
            if v is not None:
                return v
        # miss -> live embed + set in entire cache
        vec = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        if self._entire_dialogue_cache is not None:
            try:
                self._entire_dialogue_cache.set(text, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] entire_dialogue cache set failed: {e}")
        return vec

    def _cached_partial_dialogue_embed(self, text: str) -> np.ndarray:
        """partial dialogue embedding used by _rag_order_adhoc's pool dialogue.

        Fully separates the path from commit()'s entire cache write -> prevents trial contamination.
        lookup: partial -> entire chain. On miss, set in the partial cache.
        """
        if not text:
            return self._empty_vec()
        # partial first, then entire on miss
        for c in (self._partial_dialogue_cache, self._entire_dialogue_cache):
            if c is None:
                continue
            v = c.get(text)
            if v is not None:
                return v
        # miss -> live embed + set in partial cache
        vec = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        if self._partial_dialogue_cache is not None:
            try:
                self._partial_dialogue_cache.set(text, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[tkg_hybrid] partial_dialogue cache set failed: {e}")
        return vec

    def _cached_question_embed(self, text: str) -> np.ndarray:
        """question embedding — _q_embed_map (npz) first, cache next, then live + cache on miss.

        When question_embedding_type=non_temporal, transform text via strip_question_timestamp
        before lookup/embed. When temporal, use text as-is.
        The cache key is the transformed text -> uses a different cache file per temporal/non_temporal.

        Note: num_embedding_calls is a combined count including cache hits (not the OpenAI billable count).
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
        """quad embedding — _quad_emb_map (build script output) first, cache next, then live + cache on miss.

        On _quad_emb_map hit, lazy-prime the partial cache too (later lookups converge on cache.get).
        """
        if not key:
            return self._empty_vec()
        v = self._quad_emb_map.get(key)
        if v is not None:
            # lazy prime: on a quad map hit, register in the partial cache too (later converges to the cache-hit path)
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
        # runtime live embed results are stored in the partial cache (the entire cache is managed by the build script).
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
        # 1) KG update
        if self._tkg_snapshot is not None:
            # overwrite with the prebuilt KG at that session's point in time
            snap_tkg = self._tkg_snapshot.get("tkg", {})
            if session.idx in snap_tkg:
                self._kg.tkg = snap_tkg[session.idx]
        elif session.entire_quadruples:
            # TKG/RAG symmetry: commit accumulates the session's *entire* quads into KG (A).
            #   Mirrors RAG's dialogue (entire) add. Sequential evolve: after session s ends, its
            #   entire quads become A for the next session. (At retrieve, the current-session B=partial
            #   is joined by retrieve, not commit.)
            for q in session.entire_quadruples:
                # Pass session.idx — records sess_idx to block future quads at retrieve time.
                self._kg.insert_quad(q.head, q.relation, q.tail, q.start_date, session.idx)

        # 2) add to the dialogue index (for RAG search)
        if session.dialogue:
            # Reuse a precomputed embedding if present (injected in advance by data_loader).
            # Otherwise (e.g. no dialogue or emb npy missing) fall back to a live call.
            if session.dialogue_embedding is not None:
                vec = session.dialogue_embedding
            else:
                t0 = time.perf_counter()
                # commit() uses the dedicated entire-dialogue cache path.
                vec = self._cached_entire_dialogue_embed(session.dialogue)
                metrics.latency_sec += time.perf_counter() - t0
                # Note: num_embedding_calls is a combined count including cache hits (not the OpenAI billable count).
            self._idx.add(session.dialogue, session.date, session.idx, vec)
        return metrics

    # ====================================================================
    # retrieve — dispatcher
    # current_session: the current session's Session object. _retrieve_rag needs it to add
    # partial_dialogue + partial_embedding to the candidates. If None, entire only (legacy).
    # ====================================================================
    def retrieve(self, question: Question, current_session_idx: int,
                 current_session: "Session | None" = None) -> RetrievalResult:
        m = self.retrieval_method
        if m == "tkg_only":
            return self._retrieve_tkg(question, current_session_idx, route="TKG",
                                       current_session=current_session)
        if m == "rag_only":
            return self._retrieve_rag(question, current_session_idx, route="RAG",
                                       current_session=current_session)
        if m in ("tkg_rag", "enhanced_fallback"):
            # The 1st try of enhanced_fallback is identical to tkg_rag
            #   (q_clf routing -> RAG or TKG). Only on fallback (the (E) branch in answer()) does
            #   the 2nd try inject both RAG+TKG results into qa_enhanced.txt (see answer() below).
            q_clf = self._classify_question(question, current_session_idx)
            route = "RAG" if q_clf == "RAG" else "TKG"
            return (
                self._retrieve_rag(question, current_session_idx, route=route,
                                    current_session=current_session)
                if route == "RAG"
                else self._retrieve_tkg(question, current_session_idx, route=route,
                                         current_session=current_session)
            )
        if m == "enhanced":
            return self._retrieve_enhanced(question, current_session_idx,
                                            current_session=current_session)
        if m == "enhanced_fallback_inverse":
            # The exact opposite order of enhanced_fallback.
            #   1st try = always enhanced (both RAG k_rag + TKG k_tkg, prompt=qa_enhanced.txt).
            #   Only on (E) (the answer() branch) does the 2nd try re-query on a single path per the "original q_clf" value.
            #   -> compute q_clf here in advance and store it in extra["q_clf"] (used by the 2nd branch in answer()).
            r_enh = self._retrieve_enhanced(question, current_session_idx,
                                            current_session=current_session)
            q_clf = self._classify_question(question, current_session_idx,
                                            metrics=r_enh.metrics)
            r_enh.extra["q_clf"] = "RAG" if q_clf == "RAG" else "TKG"
            return r_enh
        raise ValueError(f"Unknown retrieval_method: {m!r}")

    # ====================================================================
    # build_qa_prompt — branches by retrieval_method x question_format
    # ====================================================================
    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        # creates properly formatted question: e.g.,) "Joey: So, what does Mrs. bing do?\n\t(A) Butt double\n\t(B) ..."
        q_str = render_question(question, self.question_format)
        route = retrieval.extra.get("route", "RAG")

        # 1) no-history shortcut (route == 'NO_HISTORY' or empty retrieval)
        if route == "NO_HISTORY" or (not retrieval.items and not retrieval.rendered.strip()):
            return self._p_no_hist.replace("{question}", q_str)

        # 2) enhanced always uses qa_enhanced.txt (RAG + TKG together)
        if self.retrieval_method == "enhanced":
            rag_rendered = retrieval.extra.get("rag_rendered", "No relevant history!")
            tkg_rendered = retrieval.extra.get("tkg_rendered", "No relevant triple!")
            return (
                self._p_enhanced.replace("{question}", q_str)
                .replace("{triples}", tkg_rendered)
                .replace("{history}", rag_rendered)
            )

        # 2b) The 1st try of enhanced_fallback_inverse (route == "ENHANCED") = always qa_enhanced.txt
        #   (RAG + TKG together). prompt is self._p_enhanced_fallback (=qa_enhanced.txt, override-independent).
        #   The 2nd try (single path) has route RAG/TKG, so it is handled by branch (3) below.
        if self.retrieval_method == "enhanced_fallback_inverse" and route == "ENHANCED":
            rag_rendered = retrieval.extra.get("rag_rendered", "No relevant history!")
            tkg_rendered = retrieval.extra.get("tkg_rendered", "No relevant triple!")
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
        # If no_history=True, skip retrieval and use only the no-history prompt.
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

        # Stop if fallback is off, the mode doesn't support fallback, or the 1st answer isn't (E)
        if (
            not self.fallback
            or self.retrieval_method not in ("tkg_rag", "enhanced_fallback",
                                             "enhanced_fallback_inverse")
            or process_answer(gen1.text) != "(E)"
        ):
            return gen1.text, r1, metrics

        first_route = r1.extra.get("route", "RAG")

        if self.retrieval_method == "enhanced_fallback_inverse":
            # The exact opposite order of enhanced_fallback.
            #   1st try = enhanced (already run above). (E) -> the 2nd try goes single path per the "original q_clf" value:
            #   q_clf==TKG -> TKG-path (quad only, default tkg prompt),
            #   q_clf==RAG -> RAG-path (dialogue only, default rag prompt). The 2nd output is final.
            q_clf = r1.extra.get("q_clf", "RAG")  # computed and stored in advance by retrieve().
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
            # On (E), the 2nd try re-queries by injecting both the RAG-path (k_rag=40) and
            #   TKG-path (k_tkg=350) results into qa_enhanced.txt. Like enhanced mode, retrieve both
            #   paths simultaneously, but the prompt is always self._p_enhanced_fallback (=qa_enhanced.txt).
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
                    # For retrieval recall analysis (stored per path, same as enhanced).
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
        else:
            # tkg_rag: original behavior — retry once on the opposite path (single path).
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
        # fallback branch — overwrite with the final answer's raw + reasoning + prompt + rendering.
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
                # quad -> sess_idx map for future-leakage prevention. tuple keys are picklable.
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
        # Old-snapshot compatibility — if quad_sess is absent, empty dict (sess_idx None -> keep, legacy behavior).
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
        """q_clf classification. Fixes:
        - Include model_name + prompt fingerprint in the cache key (guards against id(llm) collision + prompt-grid).
        - Accumulate _rec metrics even when structured fails (token usage drop fix).
        - Unify label normalization (precomputed + structured + free-form all take the same path).
        """
        # precomputed first (per-llm cache's q_clf.json) — same path as the v1_2 baseline.
        if idx in self.precomputed_q_clf:
            return _normalize_q_clf_label(self.precomputed_q_clf[idx])
        # Module-level q_clf cache — avoid re-calling for the same (model, prompt, question).
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
        """scope = the previous session's entire dialogues + the current session's partial dialogue.

        - Only dialogue_index (committed entire dialogue) entries with s_idx < idx are candidates.
        - current_session's partial_dialogue + partial_embedding are also appended to candidates.
        - Only when timestamp_filter is ON, filter further within that scope (scope is the same regardless of timestamp).
        - Ad-hoc BM25/dense/alternating ranking (corpus is the pool only).
        """
        metrics = CallMetrics()
        timestamps = {t: {"type": classify_timestamp(t)} for t in extract_timestamps(question.text)}

        # 1) pool = (dialogue, date, sess_idx, embedding)
        pool: list[tuple[str, str, int, Any]] = []
        for dia, date, s_idx, emb in zip(
            self._idx.dialogues, self._idx.dates, self._idx.sess_idxs, self._idx.embeddings
        ):
            if s_idx < idx:  # up to the previous session only (do not use the current session's entire)
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
        # Variable-swap fix: the RAG path cuts by num_retrieved_rag (previously misused _tkg).
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
                #   C = items (what the QA prompt actually contains after context_size). pool elem = (dia, date, s_idx, emb).
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
        """Ranking with the pool only (temporary BM25 build + dense embedding lookup).

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
                    # _rag_order_adhoc's pool uses the partial-dialogue-only path.
                    # Separate from commit()'s entire cache write -> prevents trial contamination.
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
    # internal — get the question embedding (precomputed first, else live embed)
    # ====================================================================
    def _question_embedding(self, question: Question, metrics: "CallMetrics | None" = None):
        """Return the question's embedding vector. Precomputed map first, embed_one fallback if absent.

        Same pattern as the emb_only path of `_rag_order_adhoc` (keeps consistency).
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

        # TKG/RAG symmetry: TKG at retrieve time (C) = A (committed accumulation, sess_idx < idx) +
        #   B (current-session partial quads). Exactly mirrors RAG's "previous entire dialogue +
        #   current-session partial dialogue" pool.
        #   q_subj candidate nodes, 2-hop traversal, and timestamp retrieval all target C.
        #   Future-leakage prevention: A is sess<idx only, B is the current-session partial only ->
        #   zero exposure of future-session quads.
        cur_partial = current_session.partial_quadruples if current_session is not None else None
        kg = self._kg.combined_until(idx, cur_partial)

        # q_subj options = nodes of C (A+B). Use the combined KG's full nodes (already time-accurate) instead of nodes_until.
        nodes = kg.nodes()
        if not nodes:
            return RetrievalResult(
                items=[], rendered="No relevant triple!", extra={"route": route}, metrics=metrics
            )

        # 1) identify q_subj (precomputed_q_subj[idx] first -> else live 100-batch over nodes)
        # _identify_q_subj returns a (kg_nodes, raw_subjects, prompt_text) tuple. Looks up precompute by question_idx=idx.
        #   Ablation uses kg_nodes only -> ignores raw_subjects.
        q_subj_candidates, _, q_subj_idf_prompt = self._identify_q_subj(
            question, nodes, metrics, question_idx=idx,
        )
        q_subj_candidates = [
            c if c.lower() not in {"i", "me"} else question.questioner for c in q_subj_candidates
        ]
        if any(p in f" {question.text.lower()} " for p in (" i ", " my ", " me ")):
            q_subj_candidates.append(question.questioner)
        q_subj_candidates = list({c for c in q_subj_candidates if c})
        # BUG FIX: .capitalize() lowercases everything after the first word of a multi-word entity
        # (e.g. "Paul the wine guy", "Ross Geller"), causing node-match failure. Also, without
        # case-insensitive matching, "andrea" / "ANDREA" / "Andrea" are inconsistent.
        # Fix: case-insensitive lookup (lower -> original mapping).

        # get_clean_entity is a pure helper in utils/common_utils.py (module-level import).
        # The old dict comprehension silently overwrote on collision when two nodes had the same clean
        # form (order-dependent, hurts reproducibility). setdefault preserves first-seen.
        _node_lookup: dict[str, str] = {}
        for n in nodes:
            _node_lookup.setdefault(get_clean_entity(n), n)
        # q_subj->TKG node matching uses exact match (_node_lookup) only. q_subj candidates without
        # an exact match are dropped (not included in q_subj_nodes).
        # The old fuzzy fallback (rapidfuzz score>=60) was removed. The post-exact timestamp retrieval
        # (below `if not q_subj_nodes and ts_retrieval:`, controlled by the retrieve_by_timestamp toggle)
        # is a separate mechanism and is kept (default ON).
        q_subj_nodes = [
            _node_lookup[get_clean_entity(c)]
            for c in q_subj_candidates
            if get_clean_entity(c) in _node_lookup
        ]
        # Force-cut to num_q_subj_candidates (K=number of TKG seed nodes) — keep only the top K after exact-match.
        #   The precompute cache (`q_subj_{norm}_nq{K}.json`) already holds K, but (a) the questioner
        #   append (first-person) can make it K+1, and (b) a bug where the precompute early-return in
        #   _identify_q_subj returned the whole cache made K be ignored. Cutting to K here makes both
        #   precompute and live seed-node counts exactly num_q_subj_candidates (no-op and harmless if the cache is exactly K).
        q_subj_nodes = q_subj_nodes[: self.num_q_subj_candidates]

        # 2) 2-hop traversal (case branches)
        retrieved: list[dict] = []
        # If retrieve_by_timestamp is OFF, disable timestamp-only retrieval (subjectless questions).
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
                # If time_filter is OFF, bypass this whole timestamp-filter block -> keep the 2-hop result as-is.
                # The tkg_timestamp_fallback cfg flag decides behavior.
                # True (default): if filtered is empty, keep unfiltered -> prevents degeneration (future-proof).
                # False: if filtered is empty, empty list -> for legacy/v1_2_2 reproduction.
                # In v1.3.1, _force_session_date guarantees absolute dates, so False is fine.
                _ts_fallback = self.cfg.get("tkg_timestamp_fallback", True)
                filtered = _filter_quads_by_timestamp(retrieved, timestamps)
                if filtered:
                    retrieved = filtered
                elif _ts_fallback:
                    pass  # keep unfiltered retrieved (prevents degeneration)
                else:
                    retrieved = []  # legacy behavior: empty when filtered is empty

        # 2.5) Future-leakage prevention is already guaranteed by combined_until (C=A+B):
        #   A copies sess<idx only, B is the current-session partial only (sess==idx) — C has no future (>idx) quads.
        #   So the old post-hoc `sess_idx < idx` filter is removed (it wrongly dropped B, sess==idx too).
        #   (The old filter was a safety net back when we traversed all of self._kg.)

        # retrieval recall: B = the whole set of retrieved raw evidence (before top-k/reranker).
        #   reranker/cos_sim only reorder and top-k (num_retrieved_tkg) cuts, so snapshot here
        #   (right after 2-hop/timestamp retrieval), i.e. the raw set before any later transformation.
        b_retrieved_snapshot = list(retrieved)

        # 4) (optional) reranker
        if self.reranker is not None and len(retrieved) > 1:
            cands = [_quad_to_string(q) for q in retrieved]
            order = self.reranker.rerank(question.text, cands)
            retrieved = [retrieved[i] for i in order]

        # 4.5) cos_sim re-rank — right before render, sort DESCENDING by cosine similarity between
        #      the question embedding and each quad embedding (highest similarity first -> top-first render).
        # If self.cos_sim_rerank=False (default), skip this whole block.
        # Works even when _quad_emb_map is empty, via partial_quad_cache (file cache) or the live embed
        # fallback -> the cache-only case is not silently skipped.
        if self.cos_sim_rerank and len(retrieved) > 1 and (
            self._quad_emb_map or self._partial_quad_cache or self._entire_quad_cache
        ):
            q_vec = self._question_embedding(question, metrics)
            scored: list[tuple[dict, float]] = []
            for q in retrieved:
                head, rel, tail = q["head"], q["relation"], q["tail"]
                sd = q.get("start_date")
                # Split the key by quad_embedding_type:
                # - "temporal" -> f"{h} {r} {t} on {sd}" (non_temporal fallback if sd absent)
                # - "non_temporal" -> f"{h} {r} {t}" only (ignores sd)
                if self.quad_embedding_type == "temporal":
                    key = f"{head} {rel} {tail} on {sd}" if sd else f"{head} {rel} {tail}"
                else:  # non_temporal
                    key = f"{head} {rel} {tail}"
                emb = self._quad_emb_map.get(key)
                if emb is None and self.quad_embedding_type == "temporal" and sd:
                    # On temporal match failure, non_temporal fallback (in-memory map only).
                    emb = self._quad_emb_map.get(f"{head} {rel} {tail}")
                if emb is None:
                    # in-memory map miss -> cache lookup chain + live embed + cache update.
                    t0 = time.perf_counter()
                    emb = self._cached_quad_embed(key)
                    metrics.latency_sec += time.perf_counter() - t0
                    metrics.num_embedding_calls += 1
                sim = cos_sim(q_vec, emb)
                scored.append((q, sim))
            # similarity descending. Quads without an embedding were live-embedded above, so their
            # sim is always valid — no separate last-handling needed.
            scored.sort(key=lambda x: -x[1])
            retrieved = [q for q, _ in scored]

        # 3') num_retrieved_tkg top-k cut — moved after reranker + cos_sim rerank.
        #   enhanced: 2-hop -> (reranker) -> cos_sim rerank -> top-k(=100) cut -> render.
        #   In the old order, the cut ran before rerank, so the front k of the 2-hop order (not the
        #   cos_sim top) were kept.
        #   If num_retrieved_tkg=0 (tkg_only budget mode), no cut -> fill up to context_size.
        if self.num_retrieved_tkg > 0:
            retrieved = retrieved[: self.num_retrieved_tkg]

        # 5) render + trim to context_size
        # BUG FIX: the old _render_quads_by_date re-groups the cos_sim-sorted `retrieved` by date,
        # destroying the similarity order. The highest-sim quad no longer appears near the front of the
        # prompt, and the most important quad may be dropped when trimmed to context_size.
        # Fix: render without date-grouping, preserving the cos_sim descending order (the current order of retrieved).
        # _render_quads_by_date's definition is kept (may be used on other paths) — unused only on this reranked path.
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
                "retrieval_c": _dedup_quad_items(c_quads),               # C = the set included in the QA prompt after context_size
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
    # internal — enhanced retrieval (RAG top-k + TKG top-k together)
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
                # retrieval recall: enhanced = RAG+TKG together -> store B/C separately per path.
                #   For recall measurement, RAG-path = dialogue unit, TKG-path = quad unit.
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
    # internal — q_subj identification
    # ====================================================================
    def _identify_q_subj(
        self, question: Question, nodes: list[str], metrics: CallMetrics,
        question_idx: "int | None" = None,
        debug_out: "dict | None" = None,
    ) -> tuple[list[str], list[str], str]:
        """Extract q_subj candidates (supports batch + precompute).

        - If precomputed_q_subj[question_idx] exists, skip the LLM (precompute first, same pattern as q_clf).
        - Otherwise, split nodes into q_subj_batch_size (default 100) chunks, call _q_subj_one_batch per
          batch, and merge candidates (dedup, first-seen order). 100-batch.
        Returns: (kg_nodes: list[str], raw_subjects: list[str], last_prompt_text: str)
        - kg_nodes: the final q_subj after mapping the stage2 LLM output to KG nodes and applying the _allowed filter (used by ablation).
        - raw_subjects: the raw question subjects emitted by the stage2 LLM (*before* KG mapping/_allowed filter, for analysis).
        - prompt_text: the full prompt of the last batch (empty string if no LLM call/precompute).
        """
        # precompute first (per-llm q_subj.json) — zero LLM calls.
        if question_idx is not None and question_idx in self.precomputed_q_subj:
            return list(self.precomputed_q_subj[question_idx]), [], ""
        # nodes < k (sparse KG) -> trivial return — skip LLM.
        if not nodes:
            return [], [], ""
        k = min(self.num_q_subj_candidates, len(nodes))
        # Removed the k<=1 (single node) early return — even with 1 node it must go through
        #   q_subj_one_batch (subject identification) to be accurate. (The early return returned
        #   nodes[0] unconditionally regardless of subject = inaccurate.)
        bs = max(1, self.q_subj_batch_size)
        merged: list[str] = []
        seen: set[str] = set()
        last_prompt = ""
        # stage1 batch parallelism. Batches are independent (each looks only at nodes[s:s+bs]) -> concurrent LLM calls.
        #   Results are merged in index order, so bit-identical to sequential (same merged order and last_prompt).
        #   metrics are received per batch as independent CallMetrics and summed in a single-threaded merge
        #   (avoids a shared += race).
        #   q_subj_batch_workers=1 is fully identical to the original sequential path.
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
        # Merge in index order (same first-seen order as sequential) + sum metrics (single-threaded, no race).
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
        # q_subj 2-pass stage2: feed stage1's merged (union of q_subj_idf candidates across batches)
        #   back into the q_subj_idf_select prompt as options, and select the final k q_subj only from
        #   those options (a subset of the current TKG nodes). If merged is empty, skip stage2 (return []).
        #   q_subj_check (verifying TKG node existence) was removed — node matching (exact, no fuzzy) is done by the caller _retrieve_tkg.
        #   The LLM may hallucinate entities outside options, so filter final by the merged set once more.
        if merged:
            final, sel_prompt = _q_subj_one_batch(
                self.llm, self._p_q_subj_select, question.text, question.questioner, k, merged, metrics
            )
            last_prompt = sel_prompt
            _merged_set = set(merged)
            # The final q_subj must always be within (the current TKG node set + questioner).
            #   Even if the stage1/stage2 LLM hallucinates entities outside options, remove them here.
            #   questioner is allowed for first-person questions.
            _allowed = set(nodes)
            if question.questioner:
                _allowed.add(question.questioner)
            filtered = [c for c in final if c in _merged_set and c in _allowed]
            # raw_subjects = the stage2 LLM output final (before KG mapping/_allowed filter).
            raw_subjects = list(final)
            # Prevent a 0-collapse from stage2 LLM whims: if there are stage1 candidates but stage2
            #   drops them all, keep the top k of stage1 (within allowed), avoiding total wipeout of
            #   q_subj-based TKG retrieval. The first-person questioner is additionally guaranteed by
            #   the caller _retrieve_tkg. If truly 'no subject', stage1 already made merged=[].
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
    """Call the q_subj_idf LLM once with nodes_batch as options. Returns (candidates, prompt).

    structured(QSubjList) first -> free-form(ast.literal_eval) fallback on failure — identical to the
    parse semantics of the original _identify_q_subj (protects Qwen2.5 reproducibility). If metrics is
    None, skip accumulation.
    llm is an object with generate_structured/generate (returning GenerationRecord) — tkg_hybrid's self.llm.
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
    """Merge two rankings into one list by alternating (dedup, preserving appearance order)."""
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
    # Render quadruples as a list: [{head, relation, tail, date}, ...]
    # (cos_sim_rerank=false path. quads keep the caller's sort order. No grouping by date.)
    if not quads:
        return "No relevant triple!"
    base_tokens = count_tokens(base_prompt)
    cur = base_tokens + 2  # slack for the enclosing "[" "]" tokens
    items: list[str] = []
    for q in quads:
        date = q.get("start_date") or ""
        if quad_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        st = count_tokens(s) + 2  # slack for the ", " separator token
        if cur + st > context_size:
            break
        items.append(s)
        cur += st
    return "[" + ", ".join(items) + "]"


def _dedup_quad_items(quads: "list[dict]") -> "list[dict]":
    """quad set for retrieval recall B/C — normalize (head,relation,tail,start_date) + dedup (order-independent set)."""
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
    """Render quads that are sorted cos_sim descending in that same order (no date-grouping).

    Returns (for retrieval recall B/C measurement): (rendered_text, included_quads).
      included_quads = the list of quad dicts actually included within context_size (= C, what the QA prompt contains).
      Previously only the rendered text was returned; the included quads are also returned to extract C (the set) for retrieval recall.

    BUG FIX: alternate renderer to fix _render_quads_by_date re-grouping by date and destroying the
    similarity order. quads must already be sorted highest-sim first by the caller. Here, preserving
    that order, add one quad at a time from the top (top-first), filling only up to just before
    base_prompt + accumulated quad tokens exceed context_size.
    -> The highest-similarity quad sits near the front of the prompt and survives truncation.

    Render quadruples as a list: [{head, relation, tail, date}, ...]
    (date is the 4th element of the quad; the whole thing is wrapped in "[" ... "]"). No grouping by date.
    Token-budget accounting starts from base_prompt tokens, same as _render_quads_by_date.
    """
    if not quads:
        return "No relevant triple!", []
    base_tokens = count_tokens(base_prompt)
    cur = base_tokens + 2  # slack for the enclosing "[" "]" tokens
    items: list[str] = []
    included: list[dict] = []  # quads actually included within context_size (= C, for retrieval recall measurement)
    for q in quads:
        date = q.get("start_date") or ""
        if quad_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        st = count_tokens(s) + 2  # slack for the ", " separator token
        # Stop when base_prompt + accumulated so far + this quad exceeds context_size (preserves top-first).
        # context_size<=0 = token limit off -> no cut, fill all (unlimited).
        if context_size > 0 and cur + st > context_size:
            break
        items.append(s)
        included.append(q)
        cur += st
    # If not even the first quad fits (items empty), include at least one quad, trimmed to context_size.
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
