"""GroupChat-Mem — a dual-view memory for long-term multi-party group-chat QA.

GroupChat-Mem maintains two complementary memory views, both induced online from the
dialogue only:

  * Relational memory R — a graph of ``[subject, relation, object]`` records produced by
    Open Information Extraction (OpenIE) over each incoming (partial) session. Subjects and
    objects are nodes, relations are edges, and each record carries its source session's
    timestamp. Records from different sessions are linked through shared nodes.
  * Raw session repository S — each original session stored verbatim, preserving exact
    wording and local discourse context.

Given a query, evidence is retrieved from both views in parallel: the top-n query-relevant
nodes in R seed a two-hop neighborhood whose records are ranked by cosine similarity (top
k_r kept), while k_s sessions are retrieved from S (half by BM25, half by embedding cosine
similarity). The LLM generates an initial response and judges its sufficiency; if
insufficient, GroupChat-Mem traverses R one hop at a time, re-answering after each hop until
the answer is supported or the maximum hop limit is reached ("I don't know.").
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    is_valid_record,
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

# Process-level EmbeddingCache instance pool. Recreating an EmbeddingCache for the same
# path per run / LLM would trigger a pickle load every time; sharing path -> instance
# removes that overhead. Key = Path.resolve() string.
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
# pydantic schema — structured LLM response for query-node identification
# ============================================================================
class QueryNodeList(BaseModel):
    """Response schema for query-node identification: a list of candidate node names."""
    subjects: list[str] = Field(description="query-relevant node name candidates")


# ============================================================================
# Helpers — KG / dialogue index
# ============================================================================
@dataclass
class RelationalGraph:
    """Graph-structured relational memory R. Nested dict head -> relation -> timestamp ->
    [tails], plus a record -> earliest-session map for future-information-leakage protection."""
    graph: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)
    # Leakage guard: maps each record (head, relation, tail, timestamp) to the earliest
    #   session index where it appeared, so future-session records can be excluded at
    #   retrieve time (records reconstructed at each query's timepoint).
    record_sess: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def insert_record(self, head: str, relation: str, tail: str, ts: str,
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
        if not is_valid_record({"head": h, "relation": r, "tail": t, "start_date": ts}):
            return
        node = self.graph.setdefault(h, {})
        rel = node.setdefault(r, {})
        bucket = rel.setdefault(ts, [])
        if t not in bucket:
            bucket.append(t)
        # Record sess_idx — if the same record appears in multiple sessions, keep the earliest (min).
        #   Update even when t is already in the bucket, to reflect an earlier session. Skip if sess_idx is None (legacy).
        if sess_idx is not None:
            key = (h, r, t, ts)
            prev = self.record_sess.get(key)
            self.record_sess[key] = sess_idx if prev is None else min(prev, sess_idx)

    def nodes(self) -> list[str]:
        ns: set[str] = set(self.graph)
        for h in self.graph:
            for r in self.graph[h]:
                for ts in self.graph[h][r]:
                    for t in self.graph[h][r][ts]:
                        ns.add(t)
        return list(ns)

    def nodes_until(self, current_session_idx: "int | None") -> list[str]:
        """Return only head/tail of records that appeared in sessions below current_session_idx (future-leakage guard).

        - current_session_idx is None → the full nodes() (snapshot path — already a point-in-time-correct KG, no future record).
        - Entities of records without a recorded record_sess (sess_idx None) are kept (live commit always records,
          so this only happens on the snapshot path). The comparison `sess is None or sess < idx` matches the
          record-filter semantics of `_filter_records_by_timestamp`, ensuring temporal alignment between query_node options
          and retrieved records.
        - Unlike nodes(), the head `h` is not added unconditionally; it is added only when the (h,r,t,ts) record is
          in the past → blocks exposure of entities that come only from future-only records.
        """
        if current_session_idx is None:
            return self.nodes()
        ns: set[str] = set()
        for h, rel_map in self.graph.items():
            for r, ts_map in rel_map.items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        sess = self.record_sess.get((h, r, t, ts))
                        if sess is None or sess < current_session_idx:
                            ns.add(h)
                            ns.add(t)
        return list(ns)

    def combined_until(
        self,
        current_session_idx: "int | None",
        extra_records: "tuple | list | None" = None,
    ) -> "RelationalGraph":
        """R(C) at retrieve time = A (entire records accumulated up to the previous session) + B (current-session partial records).

        This is exactly symmetric to the RAG path's
          "pool = (entire dialogue with s_idx < idx) + (current-session partial_dialogue)".
        Builds a new ephemeral RelationalGraph:
          - A: copies only records of self.graph with sess_idx < current_session_idx (or sess_idx None = legacy/snapshot).
               Same temporal semantics as nodes_until / _filter_records_by_timestamp.
          - B: inserts extra_records (= current_session.partial_records) with sess_idx = current_session_idx.
               Records of future sessions (>idx) enter through no path → future-leakage guard.
        current_session_idx is None (snapshot path) → A is all of self.graph (already point-in-time-correct) + B.

        The returned RelationalGraph is the target C for query_node candidate nodes, 2-hop one_hop traversal, and
        _retrieve_by_timestamp. self.graph (permanent commit accumulation) is not mutated (read-only copy).
        """
        out = RelationalGraph()
        # A — copy only records with sess_idx < idx (or None)
        for h, rel_map in self.graph.items():
            for r, ts_map in rel_map.items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        sess = self.record_sess.get((h, r, t, ts))
                        if current_session_idx is None or sess is None or sess < current_session_idx:
                            out.insert_record(h, r, t, ts, sess)
        # B — current-session partial records (sess_idx = current_session_idx)
        if extra_records:
            for q in extra_records:
                out.insert_record(q.head, q.relation, q.tail, q.start_date, current_session_idx)
        return out

    def one_hop(self, node: str) -> list[dict]:
        out: list[dict] = []
        if node in self.graph:
            for r, ts_map in self.graph[node].items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        q = {"head": node, "relation": r, "tail": t, "start_date": ts,
                             "sess_idx": self.record_sess.get((node, r, t, ts))}
                        if is_valid_record(q):
                            out.append(q)
        for h in self.graph:
            for r, ts_map in self.graph[h].items():
                for ts, tails in ts_map.items():
                    for t in tails:
                        if t == node and h != node:
                            q = {"head": h, "relation": r, "tail": t, "start_date": ts,
                                 "sess_idx": self.record_sess.get((h, r, t, ts))}
                            if is_valid_record(q):
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
# GroupChatMem — dual-view memory (relational memory R + raw session repository S)
# ============================================================================
class GroupChatMem(Memory):
    """GroupChat-Mem: a dual-view memory that maintains a relational memory R
    (a graph of [subject, relation, object] records induced online from the
    dialogue via Open Information Extraction) alongside a raw session repository S
    (each session stored verbatim). Given a query, evidence is retrieved from both
    views in parallel; if the initial answer is insufficient, a fine-grained
    traversal over R gathers additional evidence one hop at a time, stopping when
    the answer is supported or at the maximum hop limit ("I don't know.").

    Config keys (memory.*):
      num_records        : k_r — number of relational records kept from R (paper default 40).
      num_sessions       : k_s — number of sessions retrieved from S (paper default 250).
      num_query_nodes    : n  — top-n query-relevant nodes identified in R (paper default 3).
      hop                : two-hop neighborhood expansion around each query node (default 2).
      max_hop            : maximum traversal hop limit (default 6).
      session_method     : 'bm25_only' / 'emb_only' / 'alternating' — how S is ranked
                           (paper: half BM25 + half embedding, i.e. 'alternating').
      bm25_first         : prefer BM25 first in 'alternating'.
      record_format      : 'record' / 'sentence' — record rendering style.
      question_format    : 'multi_structured' / 'multi_unstructured' / 'open_ended'.
      cos_sim_rerank     : re-rank records by cosine similarity to the query (default True).
      fallback           : enable the iterative traversal when the initial answer is (E).
      context_size       : optional final-prompt token limit (0 = unlimited).
    """

    NAME = "groupchat_mem"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        self._graph = RelationalGraph()
        self._idx = DialogueIndex()

        # Fixed retrieval policy for GroupChat-Mem: dual-view (records from R + sessions from S).
        self.retrieval_method = "groupchat_mem"
        # Enable the iterative traversal over R when the initial answer is "I don't know."
        self.fallback = bool(cfg.get("fallback", True))

        # Top-k for each memory view (paper: k_r records from R, k_s sessions from S).
        #   num_records  (k_r): number of relational records kept from R (paper default 40).
        #   num_sessions (k_s): number of sessions retrieved from S      (paper default 250).
        self.num_records = int(cfg.get("num_records", 40))
        self.num_sessions = int(cfg.get("num_sessions", 250))
        if self.num_records <= 0 or self.num_sessions <= 0:
            raise ValueError(
                f"num_records and num_sessions must be positive, got "
                f"records={self.num_records}, sessions={self.num_sessions}"
            )
        # Optional final-prompt token limit. 0/negative = off (unlimited); the serving
        #   max-model-len is then the real cap and top-k bounds the retrieval count.
        self.context_size = int(cfg.get("context_size", 0))
        self.context_size = 10 ** 9 if self.context_size <= 0 else self.context_size

        # Session-view (S) ranking. 'alternating' = half BM25 + half embedding cosine
        #   similarity (the paper setting). 'bm25_only' / 'emb_only' use one signal.
        self.session_method = cfg.get("session_method", "alternating")
        self.bm25_first = bool(cfg.get("bm25_first", True))
        # Optional timestamp filters (off by default). When on, candidate sessions/records
        #   are narrowed to those matching a timestamp mentioned in the question.
        self.session_time_filter = bool(cfg.get("session_time_filter", False))
        self.time_filter = bool(cfg.get("time_filter", False))
        # Timestamp-only retrieval for questions with no identifiable query node (default on).
        self.retrieve_by_timestamp = bool(cfg.get("retrieve_by_timestamp", True))

        # Relational-view (R) traversal.
        self.hop = int(cfg.get("hop", 2))  # two-hop neighborhood around each query node
        # Record entity normalization variant (default 'raw' = no normalization).
        self.triple_normalization = cfg.get("triple_normalization", "raw")
        self.record_format = cfg.get("record_format", "record")  # record / sentence
        # n = number of top query-relevant nodes identified in R (paper default 3).
        self.num_query_nodes = int(cfg.get("num_query_nodes", 3))

        # Optional cross-encoder reranker (lazy; default 'none').
        self.reranker_kind = cfg.get("reranker", "none")
        self._reranker = None

        # Prompts. The question_format selects a default template per view; an explicit
        #   cfg.prompt_* overrides it.
        #     records  : {question} + {triples}  (records linearized from R)
        #     sessions : {question} + {history}  (raw sessions from S)
        #     dual     : {question} + {triples} + {history}  (both views together)
        _FMT_PROMPT_MAP = {
            "multi_structured": {
                "records":  "qa_records.multi-unstructured.txt",
                "sessions": "qa_sessions.txt",
                "dual":     "qa_dual.txt",
            },
            "multi_unstructured": {
                "records":  "qa_records.multi-unstructured.txt",
                "sessions": "qa_sessions.multi-unstructured.txt",
                "dual":     "qa_dual.multi-unstructured.txt",
            },
            "open_ended": {
                "records":  "qa_records.open-ended.txt",
                "sessions": "qa_sessions.open-ended.txt",
                "dual":     "qa_dual.open-ended.txt",
            },
        }
        _fmt_key = str(cfg.get("question_format", "multi_structured")).lower().replace("-", "_")
        _defaults = _FMT_PROMPT_MAP.get(_fmt_key, _FMT_PROMPT_MAP["multi_structured"])
        self._p_sessions = load_prompt(cfg.get("prompt_sessions") or _defaults["sessions"])
        self._p_records = load_prompt(cfg.get("prompt_records") or _defaults["records"])
        # Dual-view prompt: used for the initial response and every traversal re-answer.
        self._p_dual = load_prompt(cfg.get("prompt_dual") or _defaults["dual"])
        # Traversal lookup prompt: at each traversal step the LLM decides whether there is a
        #   further query-relevant node to expand (entity / UNKNOWN / "I don't know").
        self._p_traversal_lookup = load_prompt(cfg.get("prompt_traversal_lookup") or "qa_traversal_lookup.txt")
        # Maximum traversal hop limit (paper default 6). Termination is otherwise decided by
        #   the progress guard (UNKNOWN / revisiting a node / no new records).
        self._max_hop = int(cfg.get("max_hop", 6))
        # Over-length guard: cap the input tokens of traversal prompts so that
        #   input + output (thinking budget) stays within the serving max-model-len.
        self._input_token_cap = int(cfg.get("input_token_cap", 40000))

        # question format
        self.question_format = cfg.get("question_format", "multi_structured")

        # misc
        self.delimiter = cfg.get("delimiter", "=" * 40)
        # Precomputed query nodes — {question_idx: [node, ...]} (optional; saves LLM calls).
        #   If absent, query nodes are identified live via _identify_query_nodes.
        self.precomputed_query_nodes: dict[int, list[str]] = cfg.get("precomputed_query_nodes") or {}
        # Query-node identification is batched over the candidate node set.
        self.query_node_batch_size: int = int(cfg.get("query_node_batch_size", 100))
        # Batch parallelism for query-node identification (default 1 = sequential; the merge
        #   is index-ordered so results are identical regardless of worker count).
        self.query_node_batch_workers: int = max(1, int(cfg.get("query_node_batch_workers", 1)))
        # Prompts for the two-stage query-node identification.
        self._p_query_node_id = load_prompt(cfg.get("prompt_query_node_id") or "query_node_id.txt")
        self._p_query_node_select = load_prompt(cfg.get("prompt_query_node_select") or "query_node_id_select.txt")

        # Question embedding type axis (separate from record embedding).
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
                logging.warning("[GroupChatMem.__init__] q_embed_path load failed → fallback: %s", e)
                self._q_embed_map = {}

        # Precomputed record embeddings (injected by the entry point via cfg["record_embeddings"]).
        # Format: {record_string_key: vector}
        #   temporal     = f"{head} {relation} {tail} on {start_date}"
        #   non_temporal = f"{head} {relation} {tail}"
        # Empty dict if absent (the cos_sim re-rank in the retrieve step becomes a no-op automatically).
        self._record_emb_map: dict[str, Any] = cfg.get("record_embeddings") or {}

        # cos_sim re-rank toggle:
        # True  → re-sort retrieved records by cosine similarity to the question, descending.
        # False → skip re-sorting (keep BM25/record order). default = False.
        self.cos_sim_rerank: bool = bool(cfg.get("cos_sim_rerank", False))

        # record embedding type toggle.
        # "temporal"     → key = f"{h} {r} {t} on {sd}"  (includes start_date, build filename _temporal)
        # "non_temporal" → key = f"{h} {r} {t}"          (build filename _non_temporal)
        # default = "non_temporal". simulate yaml overrides to "temporal".
        # Can be added as an ablation grid axis (config's grid memory.record_embedding_type).
        _qet = cfg.get("record_embedding_type", "non_temporal")
        if _qet not in ("temporal", "non_temporal"):
            raise ValueError(f"record_embedding_type invalid: {_qet!r}; expected 'temporal' or 'non_temporal'")
        self.record_embedding_type: str = _qet
        # question_embedding_type is set earlier (must run before the q_embed_path block references it).

        # ====================================================================
        # 4 EmbeddingCache instances.
        # None if the cfg key is absent — live embed fallback (existing behavior).
        # ====================================================================
        self._init_embedding_caches(cfg, embedder)

    # ====================================================================
    # Embedding cache init — 4 kinds (entire_dialogue / partial_dialogue / question / record)
    # Required cfg keys: dataset_root, data_type, trial — cache disabled if absent (live embed fallback).
    # The record cache additionally needs llm_name + ie_type + normalization.
    # ====================================================================
    def _init_embedding_caches(self, cfg: dict[str, Any], embedder: "Embedder") -> None:
        # Use module-level logging / Path (no function-local import).
        self._entire_dialogue_cache: EmbeddingCache | None = None
        self._partial_dialogue_cache: EmbeddingCache | None = None
        self._question_cache: EmbeddingCache | None = None
        self._partial_record_cache: EmbeddingCache | None = None
        self._entire_record_cache: EmbeddingCache | None = None

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
            logging.warning(f"[GroupChatMem] dialogue/question cache init failed: {e}. live embed fallback.")
            self._entire_dialogue_cache = None
            self._partial_dialogue_cache = None
            self._question_cache = None

        # record cache — per LLM × ie × normalization (only when LLM info is present)
        llm_name = cfg.get("llm_name")
        ie_type = cfg.get("ie_type", "openie")
        norm = cfg.get("normalization", "raw")
        # cache_budget segment branch.
        #   data_loader reads the record json under `bNNNNN` in the LLM folder, so the record's embedding cache
        #   must be read under the same segment for path consistency (absent → no segment = existing behavior).
        from groupchat_qa.pipeline.data_loader import budget_segment as _budget_segment
        _bseg = _budget_segment(cfg.get("cache_budget"))
        if llm_name:
            # Add embedding_type suffix to the record cache path
            # (build_embeddings.py stores separate _temporal / _non_temporal pickles).
            suffix = self.record_embedding_type  # "temporal" or "non_temporal"
            try:
                _entire_record_dir = ds_root / dt / "precomputed" / "common" / ie_type / llm_name
                if _bseg:
                    _entire_record_dir = _entire_record_dir / _bseg
                self._entire_record_cache = _get_or_create_cache(
                    _entire_record_dir
                           / f"entire_session_records_embedding_{norm}_{suffix}.pickle",
                    embedder.name, embedder.dim, **_kw,
                )
                _partial_record_dir = (ds_root / dt / "precomputed" / "per-trial" / f"t{trial}"
                                     / "per-llm" / llm_name)
                if _bseg:
                    _partial_record_dir = _partial_record_dir / _bseg
                self._partial_record_cache = _get_or_create_cache(
                    _partial_record_dir / ie_type
                           / f"partial_session_records_embedding_{norm}_{suffix}.pickle",
                    embedder.name, embedder.dim, **_kw,
                )
            except (OSError, ValueError, ImportError, RuntimeError) as e:
                logging.warning(f"[GroupChatMem] record cache init failed: {e}. live embed fallback.")
                self._entire_record_cache = None
                self._partial_record_cache = None

        # Lazy-prime the cache with the in-memory map's precomputed values.
        # If _q_embed_map (npz) is filled, also register in the question cache so lookup goes through a single cache.get path.
        # The record map can be large, so it is set at first lookup (handled inside _cached_record_embed below).
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

        Separates the partial pool of _session_order_adhoc from the cache-write path.
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
                logging.warning(f"[GroupChatMem] entire_dialogue cache set failed: {e}")
        return vec

    def _cached_partial_dialogue_embed(self, text: str) -> np.ndarray:
        """partial dialogue embedding used by the pool dialogue of _session_order_adhoc.

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
                logging.warning(f"[GroupChatMem] partial_dialogue cache set failed: {e}")
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
                logging.warning(f"[GroupChatMem] question cache set failed: {e}")
        return vec

    def _cached_record_embed(self, key: str) -> np.ndarray:
        """record embedding — _record_emb_map (build script output) first, cache as secondary, on miss live + cache.

        On _record_emb_map hit, also lazy-prime the partial cache (subsequent lookups go through a single cache.get path).
        Convert live embed result to np.asarray(float32).
        """
        if not key:
            return self._empty_vec()
        v = self._record_emb_map.get(key)
        if v is not None:
            # lazy prime: on record map hit, also register in the partial cache (subsequent lookups converge to the cache-hit path)
            if self._partial_record_cache is not None and self._partial_record_cache.get(key) is None:
                try:
                    self._partial_record_cache.set(key, np.asarray(v, dtype=np.float32))
                except (OSError, ValueError, RuntimeError):
                    pass
            return v
        for c in (self._entire_record_cache, self._partial_record_cache):
            if c is None:
                continue
            v = c.get(key)
            if v is not None:
                return v
        vec = np.asarray(self.embedder.embed_one(key), dtype=np.float32)
        # Store the runtime live embed result in the partial cache (the build script manages entire).
        if self._partial_record_cache is not None:
            try:
                self._partial_record_cache.set(key, vec)
            except (OSError, ValueError, RuntimeError) as e:
                logging.warning(f"[GroupChatMem] record cache set failed: {e}")
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
        # 1) accumulate the session's records into the relational memory R.
        if session.entire_records:
            # Records extracted from the full session become part of R once the session ends.
            #   The current session's partial records are merged at retrieve time (not here).
            for q in session.entire_records:
                # Pass session.idx so future records can be blocked at retrieve time.
                self._graph.insert_record(q.head, q.relation, q.tail, q.start_date, session.idx)

        # 2) store the raw session verbatim in the session repository S (dialogue index).
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
    # current_session: the Session object of the current session. Needed so _retrieve_sessions can add
    # partial_dialogue + partial_embedding to the candidates. None → entire only (legacy).
    # ====================================================================
    def retrieve(self, question: Question, current_session_idx: int,
                 current_session: "Session | None" = None) -> RetrievalResult:
        """Retrieve evidence from both memory views (relational memory R and the raw
        session repository S) in parallel, without committing to a single source."""
        return self._retrieve_dual(question, current_session_idx,
                                   current_session=current_session)

    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        """Fill the QA prompt with the dual-view evidence (records from R + sessions from S)."""
        q_str = render_question(question, self.question_format)
        records_rendered = retrieval.extra.get("records_rendered") or "No relevant record!"
        sessions_rendered = retrieval.extra.get("sessions_rendered") or "No relevant history!"
        return (
            self._p_dual.replace("{question}", q_str)
            .replace("{triples}", records_rendered)
            .replace("{history}", sessions_rendered)
        )

    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        # 1st retrieve + generate the initial response.
        r1 = self.retrieve(question, current_session_idx, current_session=current_session)
        prompt1 = apply_token_limit(self.build_qa_prompt(question, r1), self.context_size)
        gen1 = self.llm.generate(prompt1)
        metrics = r1.metrics.merge(gen1.to_metrics())
        # Preserve the last QA generation for post-hoc analysis.
        r1.extra["qa_raw_text"] = gen1.text
        r1.extra["qa_reasoning_text"] = getattr(gen1, "reasoning_text", "")
        r1.extra["qa_final_prompt"] = prompt1
        r1.extra["qa_retrieval_rendered"] = r1.rendered

        # If the initial answer is sufficient (not "I don't know"), return it directly.
        if not self.fallback or process_answer(gen1.text) != "(E)":
            return gen1.text, r1, metrics

        # Otherwise, run the fine-grained traversal over the relational memory.
        first_route = r1.extra.get("route", "SESSIONS")
        return self._iterative_traversal(
            question, current_session_idx, r1, metrics, first_route,
            current_session=current_session,
        )

    # ================================================================
    # snapshot / restore
    # ================================================================
    def snapshot(self, path: str) -> None:
        save_pickle(
            {
                "graph": self._graph.graph,
                # record→sess_idx map for future-leakage blocking. tuple keys are picklable.
                "record_sess": self._graph.record_sess,
                "dialogues": self._idx.dialogues,
                "dates": self._idx.dates,
                "sess_idxs": self._idx.sess_idxs,
                "embeddings": [v.tolist() for v in self._idx.embeddings],
            },
            path,
        )

    def restore(self, path: str) -> None:
        d = load_pickle(path)
        self._graph.graph = d["graph"]
        # Old-snapshot compat — empty dict if record_sess is absent (sess_idx None→keep, legacy behavior).
        self._graph.record_sess = d.get("record_sess", {})
        self._idx.dialogues = d["dialogues"]
        self._idx.dates = d["dates"]
        self._idx.sess_idxs = d["sess_idxs"]
        self._idx.embeddings = [np.asarray(v, dtype=np.float32) for v in d["embeddings"]]
        self._idx._bm25 = None

    # ====================================================================
    # internal — q_clf
    # ====================================================================
    def _retrieve_sessions(self, question: Question, idx: int, route: str = "SESSIONS",
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
        if self.session_time_filter and self.time_filter and timestamps:
            pool = [t for t in pool if _date_matches_any(t[1], timestamps)]
        if not pool:
            return RetrievalResult(
                items=[], rendered="No relevant history!", extra={"route": route}, metrics=metrics
            )

        # 3) ranking (ad-hoc corpus)
        order = self._session_order_adhoc(question, pool, metrics)

        # 4) top-k cut (when num_sessions > 0) — RAG path
        # BUG FIX: variable swap — the session path cuts by num_sessions.
        if self.num_sessions > 0:
            order = order[: self.num_sessions]

        # 5) (optional) reranker
        if self.reranker is not None and len(order) > 1:
            cands = [pool[i][0] for i in order]
            reranked = self.reranker.rerank(question.text, cands)
            order = [order[i] for i in reranked]

        # 6) fill up to context_size (full QA prompt = base + question + dialogue)
        items: list[Item] = []
        rendered = ""
        q_str = render_question(question, self.question_format)
        approx_tokens = count_tokens(self._p_sessions) + count_tokens(q_str)
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

    def _session_order_adhoc(self, question: Question,
                          pool: list[tuple[str, str, int, Any]],
                          metrics: CallMetrics) -> list[int]:
        """Ranking using the pool only (temporary BM25 build + dense embedding lookup).

        Prefer the precomputed question embedding. If a pool embedding is None, live embed_one.
        """
        method = self.session_method
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
                    # L1+S4 fix: _session_order_adhoc's pool uses the partial-dialogue-only path.
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

        Same pattern as the emb_only path in `_session_order_adhoc` (keeps them consistent).
        """
        t0 = time.perf_counter()
        q_vec = self._cached_question_embed(question.text)
        if metrics is not None:
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += 1
        return q_vec

    # ====================================================================
    # internal — relational memory R retrieval (two-hop traversal)
    # ====================================================================
    def _retrieve_records(self, question: Question, idx: int, route: str = "RECORDS",
                      current_session: "Session | None" = None) -> RetrievalResult:
        timestamps = {t: {"type": classify_timestamp(t)} for t in extract_timestamps(question.text)}
        metrics = CallMetrics()

        # R/S symmetry: R(C) at retrieve time = A (committed accumulation, sess_idx < idx) + B (current
        #   session's partial record). Exactly symmetric to RAG's "previous entire dialogue + current-session
        #   partial dialogue" pool. query_node candidate nodes, 2-hop traversal, and timestamp retrieval all
        #   operate over C. Future-leakage blocking: A only sess<idx, B only the current session's partial →
        #   0 future-session records exposed.
        cur_partial = current_session.partial_records if current_session is not None else None
        kg = self._graph.combined_until(idx, cur_partial)

        # query_node options = nodes of C (A+B). All nodes of the combined KG instead of nodes_until (already time-accurate).
        nodes = kg.nodes()
        if not nodes:
            return RetrievalResult(
                items=[], rendered="No relevant triple!", extra={"route": route}, metrics=metrics
            )

        # 1) identify query_node (precomputed_query_nodes[idx] first → else 100-batch live over nodes)
        # _identify_query_nodes returns a (kg_nodes, raw_subjects, prompt_text) tuple. Precompute is looked up by question_idx=idx.
        #   ablation uses kg_nodes only → raw_subjects ignored.
        query_node_candidates, _, query_node_id_prompt = self._identify_query_nodes(
            question, nodes, metrics, question_idx=idx,
        )
        query_node_candidates = [
            c if c.lower() not in {"i", "me"} else question.questioner for c in query_node_candidates
        ]
        if any(p in f" {question.text.lower()} " for p in (" i ", " my ", " me ")):
            query_node_candidates.append(question.questioner)
        query_node_candidates = list({c for c in query_node_candidates if c})
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
        # fuzzy fallback removed by design. query_node→R node matching uses exact match (_node_lookup) only.
        # query_node candidates without an exact match are discarded (not included in query_node_matches).
        # The old fuzzy fallback (rapidfuzz score>=60) is removed. The subsequent timestamp retrieval
        # (below, `if not query_node_matches and ts_retrieval:`, controlled by the retrieve_by_timestamp toggle)
        # is a separate mechanism and is kept (default ON).
        query_node_matches = [
            _node_lookup[get_clean_entity(c)]
            for c in query_node_candidates
            if get_clean_entity(c) in _node_lookup
        ]
        # Enforce num_query_nodes (K = number of query nodes) cut — top K after exact-match.
        #   The precompute cache (`query_node_{norm}_nq{K}.json`) already holds K, but (a) questioner append
        #   (first person) could make it K+1, and (b) there was a bug where the precompute-first early-return
        #   returned the entire cache and ignored K. Cutting by K here makes the query-node count exactly
        #   num_query_nodes for both precompute and live (no-op if the cache is exactly K, harmless).
        query_node_matches = query_node_matches[: self.num_query_nodes]

        # 2) 2-hop traversal (case branching)
        retrieved: list[dict] = []
        # When retrieve_by_timestamp is OFF, timestamp-only retrieval (subject-less questions) is disabled.
        ts_retrieval = self.retrieve_by_timestamp and bool(timestamps)
        if not query_node_matches and not ts_retrieval:
            return RetrievalResult(
                items=[], rendered="No relevant triple!",
                extra={
                    "route": route,
                    "records_path_info": {
                        "question_text": question.text,
                        "query_node_candidates": list(query_node_candidates),
                        "graph_nodes": list(nodes),
                        "matched_subjs": [],
                        "query_node_id_prompt": query_node_id_prompt,
                    },
                },
                metrics=metrics,
            )
        if not query_node_matches and ts_retrieval:
            # timestamp retrieval over the combined KG (C=A+B), not self._graph.
            retrieved = _retrieve_by_timestamp(kg, timestamps)
        else:
            seeds = list(query_node_matches)
            seen: set[str] = set()
            for _ in range(self.hop):
                new_seeds: list[str] = []
                for s in seeds:
                    if s in seen:
                        continue
                    seen.add(s)
                    # 2-hop traversal over the combined KG (C=A+B), not self._graph.
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
                # Behavior is decided by the timestamp_fallback cfg flag.
                # True (default): if filtered is empty, keep unfiltered → prevents degeneration (future-proof).
                # False: if filtered is empty, empty list → for legacy/v1_2_2 reproduction.
                # v1.3.1 guarantees absolute dates via _force_session_date, so False is fine too.
                _ts_fallback = self.cfg.get("timestamp_fallback", True)
                filtered = _filter_records_by_timestamp(retrieved, timestamps)
                if filtered:
                    retrieved = filtered
                elif _ts_fallback:
                    pass  # keep unfiltered retrieved (prevents degeneration)
                else:
                    retrieved = []  # legacy behavior: empty when filtered is empty

        # 2.5) Future-leakage blocking is already guaranteed by combined_until(C=A+B):
        #   A copies only sess<idx, B only the current session's partial (sess==idx) — no future (>idx) records in C.
        #   So the old post-filter `sess_idx < idx` wrongly removed B (sess==idx) and is removed.
        #   (The old filter was a safety net from when we traversed all of self._graph.)

        # retrieval recall: B = the whole set of retrieved raw evidence (before top-k/reranker).
        #   reranker/cos_sim only reorder while top-k (num_records) cuts, so we snapshot here
        #   (right after 2-hop/timestamp retrieval), i.e. the raw set before any transformation.
        b_retrieved_snapshot = list(retrieved)

        # 4) (optional) reranker
        if self.reranker is not None and len(retrieved) > 1:
            cands = [_record_to_string(q) for q in retrieved]
            order = self.reranker.rerank(question.text, cands)
            retrieved = [retrieved[i] for i in order]

        # 4.5) cos_sim re-rank — right before render, sort DESCENDING by cosine similarity between the
        #      question embedding and each record embedding (highest similarity first → top-first render).
        # This whole block is skipped when self.cos_sim_rerank=False (default).
        # BUG FIX: works even when _record_emb_map is empty, via partial_record_cache (file cache) or the
        # live embed fallback → no silent skip in the cache-only case.
        if self.cos_sim_rerank and len(retrieved) > 1 and (
            self._record_emb_map or self._partial_record_cache or self._entire_record_cache
        ):
            q_vec = self._question_embedding(question, metrics)
            scored: list[tuple[dict, float]] = []
            for q in retrieved:
                head, rel, tail = q["head"], q["relation"], q["tail"]
                sd = q.get("start_date")
                # Split the key by record_embedding_type:
                # - "temporal" → f"{h} {r} {t} on {sd}" (non_temporal fallback if sd is absent)
                # - "non_temporal" → f"{h} {r} {t}" only (sd ignored)
                if self.record_embedding_type == "temporal":
                    key = f"{head} {rel} {tail} on {sd}" if sd else f"{head} {rel} {tail}"
                else:  # non_temporal
                    key = f"{head} {rel} {tail}"
                emb = self._record_emb_map.get(key)
                if emb is None and self.record_embedding_type == "temporal" and sd:
                    # non_temporal fallback on temporal match miss (in-memory map only).
                    emb = self._record_emb_map.get(f"{head} {rel} {tail}")
                if emb is None:
                    # in-memory map miss → cache lookup chain + live embed + cache update.
                    t0 = time.perf_counter()
                    emb = self._cached_record_embed(key)
                    metrics.latency_sec += time.perf_counter() - t0
                    metrics.num_embedding_calls += 1
                sim = cos_sim(q_vec, emb)
                scored.append((q, sim))
            # Descending by similarity. Records without an embedding were live-embedded above, so they
            # always have a valid sim — no special last-placement needed.
            scored.sort(key=lambda x: -x[1])
            retrieved = [q for q, _ in scored]

        # 3') num_records top-k cut — moved after reranker + cos_sim rerank.
        #   enhanced: 2-hop → (reranker) → cos_sim rerank → top-k(=100) cut → render.
        #   In the old order, the cut preceded the rerank, so the first k in 2-hop order were kept
        #   rather than the cos_sim top ones. With num_records=0 (budget mode) no cut is
        #   done → fill up to context_size.
        if self.num_records > 0:
            retrieved = retrieved[: self.num_records]

        # 5) render + cut to context_size
        # Render preserving the cosine-similarity-descending order (retrieved's current order) without
        # any date-grouping, so the highest-similarity records come first in the prompt and survive the
        # context_size cut.
        # full QA prompt = base + question + records (consistent with the session view)
        q_str = render_question(question, self.question_format)
        base_p = self._p_records + q_str
        rendered, c_records = _render_records_in_order(retrieved, self.record_format, self.context_size, base_p)
        items = [Item(text=_record_to_string(q), meta=q) for q in retrieved]
        return RetrievalResult(
            items=items,
            rendered=rendered,
            extra={
                "route": route,
                "query_node_matches": query_node_matches,
                "retrieval_b": _dedup_record_items(b_retrieved_snapshot),  # B = the whole set of retrieved records (before top-k/reranker)
                "retrieval_c": _dedup_record_items(c_records),               # C = set included in the QA prompt after context_size
                "records_path_info": {
                    "question_text": question.text,
                    "query_node_candidates": list(query_node_candidates),
                    "graph_nodes": list(nodes),
                    "matched_subjs": list(query_node_matches),
                    "query_node_id_prompt": query_node_id_prompt,
                },
            },
            metrics=metrics,
        )

    # ====================================================================
    # internal — enhanced retrieval (sessions top-k + records top-k simultaneously)
    # ====================================================================
    def _retrieve_dual(self, question: Question, idx: int,
                            current_session: "Session | None" = None) -> RetrievalResult:
        sess = self._retrieve_sessions(question, idx, route="SESSIONS_in_dual",
                                  current_session=current_session)
        rec = self._retrieve_records(question, idx, route="RECORDS_in_dual",
                                 current_session=current_session)
        metrics = sess.metrics.merge(rec.metrics)
        return RetrievalResult(
            items=sess.items + rec.items,
            rendered="",  # build_qa_prompt uses sessions_rendered/records_rendered from extra
            extra={
                "route": "DUAL",
                "sessions_rendered": sess.rendered,
                "records_rendered": rec.rendered,
                "session_items": sess.items,
                "record_items": rec.items,
                # retrieval recall: dual = sessions + records simultaneously → store B/C separately per path.
                #   For recall measurement, session-path = dialogue-level, record-path = record-level.
                "retrieval_b_sessions": sess.extra.get("retrieval_b", []),
                "retrieval_c_sessions": sess.extra.get("retrieval_c", []),
                "retrieval_b_records": rec.extra.get("retrieval_b", []),
                "retrieval_c_records": rec.extra.get("retrieval_c", []),
                # The record-path is used in the ENHANCED route too, so propagate records_path_info.
                "records_path_info": rec.extra.get("records_path_info"),
            },
            metrics=metrics,
        )

    # ====================================================================
    # internal — traversal helpers (over-length guard + node resolution)
    # ====================================================================
    def _cap_prompt(self, prompt: str) -> str:
        """Cap the input tokens of a traversal prompt so that input + output stays within
        the serving max-model-len (the front — instruction + question + records — is kept
        first; the dialogue at the back is trimmed first). If context_size > 0, apply the
        smaller of the two limits."""
        cap = self._input_token_cap
        if self.context_size and self.context_size > 0:
            cap = min(cap, self.context_size) if cap > 0 else self.context_size
        return trim_text_to_tokens(prompt, cap) if cap > 0 else prompt

    def _resolve_node(self, node_raw: str, nodes: "list[str]",
                      node_lookup: "dict[str, str]") -> "str | None":
        """Resolve a raw node name to a graph node (exact clean-entity match first, else the
        shortest substring match)."""
        if not node_raw or node_raw.upper() == "UNKNOWN":
            return None
        ce = get_clean_entity(node_raw)
        if ce in node_lookup:
            return node_lookup[ce]
        nr_low = node_raw.lower()
        cands = [n for n in nodes if nr_low and (nr_low in n.lower() or n.lower() in nr_low)]
        return min(cands, key=len) if cands else None

    @staticmethod
    def _classify_next_lookup(text: str) -> "tuple[str, str]":
        """Classify the traversal-lookup output into 3 outcomes. Returns (kind, entity).

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

    def _iterative_traversal(
        self, question: Question, idx: int, r1: RetrievalResult,
        metrics: CallMetrics, first_route: str,
        current_session: "Session | None" = None,
    ) -> "tuple[str, RetrievalResult, CallMetrics]":
        """iterative traversal: the subgraph-exploration re-answer prompt = **initial records +
        records newly discovered by exploration, accumulated** (qa_dual.txt). Dialogue **keeps the initial
        full_sessions as is** in every QA prompt (do NOT clear dialogue during traversal — initial dialogue is always
        included). lookup uses the traversal prompt (qa_traversal_lookup.txt: IDK=can't solve + no node
        to look up / UNKNOWN=there is a node to check but it cannot be pinpointed among the gathered record
        nodes). Each hop:
          ① lookup → IDK=(E) terminate immediately, UNKNOWN=break→initial dual_full, node=resolve.
          resolve failed/revisit/0 new records → break→initial dual_full.
          if resolve+new records exist, ③ try answering over (initial triples + accumulated new records, initial
          dialogue) → terminate unless (E); if (E), next iter. If still (E) after exhausting max_tries
          (max hop limit), finish with initial enhanced_full.
        route="GROUPCHAT_MEM".
        """
        q_str = render_question(question, self.question_format)
        full_sessions = r1.extra.get("sessions_rendered") or "No relevant history!"
        full_records = r1.extra.get("records_rendered") or "No relevant triple!"
        # Initial records (list of record dicts) — re-answer {triples} = these initial records + accumulated new records.
        #   ★ r1.extra["record_items"] is a list of Item objects (_retrieve_records wraps as Item(meta=record)) → extract .meta (record dict).
        #   (Item has no .get — calling q.get(...) directly crashes with "'Item' object has no attribute 'get'".)
        initial_items = [it.meta for it in (r1.extra.get("record_items") or [])
                         if isinstance(getattr(it, "meta", None), dict)]

        cur_partial = current_session.partial_records if current_session is not None else None
        kg = self._graph.combined_until(idx, cur_partial)
        nodes = kg.nodes()
        node_lookup: dict[str, str] = {}
        for n in nodes:
            node_lookup.setdefault(get_clean_entity(n), n)

        visited: list[str] = []
        visited_set: set[str] = set()
        # dedup key seed = initial triple keys → records already present initially don't count as "new records" (avoids duplicates).
        acc_record_keys: set[tuple] = {
            (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
            for q in initial_items
        }
        acc_records: list[dict] = []   # only records newly discovered by exploration (accumulated for {focused} + iter answer {triples}).
        base_p_records = self._p_records + q_str

        iters = 0
        for _hop in range(max(1, self._max_hop)):
            # 1) traversal lookup: any more entity to look up? {focused}=new records, {triples}=initial full_records, {history}=initial full_sessions.
            focused_now, _c = (_render_records_in_order(
                acc_records, self.record_format, self.context_size, base_p_records
            ) if acc_records else ("(none yet)", []))
            ex_prompt = (
                self._p_traversal_lookup
                .replace("{question}", q_str)
                .replace("{resolved}", ", ".join(visited) if visited else "(none)")
                .replace("{focused}", focused_now)
                .replace("{triples}", full_records)
                .replace("{history}", full_sessions)
            )
            ex_prompt = self._cap_prompt(ex_prompt)
            gen_ex = self.llm.generate(ex_prompt)
            metrics = metrics.merge(gen_ex.to_metrics())
            kind, ent = self._classify_next_lookup(gen_ex.text)

            if kind == "IDK":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "GROUPCHAT_MEM", "fallback_from": first_route,
                        "traversal_resolved": False, "traversal_iters": iters,
                        "traversal_chain": list(visited), "traversal_n_records": len(acc_records),
                        "neutral_stop": "IDK",
                        "sessions_rendered": full_sessions, "records_rendered": full_records,
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
                        "route": "GROUPCHAT_MEM", "fallback_from": first_route,
                        "traversal_resolved": bool(visited), "traversal_iters": iters,
                        "traversal_chain": list(visited), "traversal_n_records": len(acc_records),
                        "neutral_stop": "STOP_FIRST_ANSWER",
                        "sessions_rendered": full_sessions, "records_rendered": full_records,
                        "qa_raw_text": first_text,
                        "qa_reasoning_text": r1.extra.get("qa_reasoning_text", ""),
                        "qa_final_prompt": r1.extra.get("qa_final_prompt", ""),
                        "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return first_text, r2, metrics

            node = self._resolve_node(ent, nodes, node_lookup)
            if node is None or node in visited_set:
                break

            # ② among one-hop records, accumulate as new records only those "not in the previous QA prompt (initial triples + existing accumulation)".
            bq = kg.one_hop(node)
            added = 0
            for q in bq:
                k = (q.get("head"), q.get("relation"), q.get("tail"), q.get("start_date"))
                if k not in acc_record_keys:
                    acc_record_keys.add(k); acc_records.append(q); added += 1
            if added == 0:
                break

            visited.append(node); visited_set.add(node); iters += 1

            # ③ try answering over (initial triples + accumulated new records) + initial dialogue.
            combined_items = initial_items + acc_records
            combined_records, _c2 = _render_records_in_order(
                combined_items, self.record_format, self.context_size, base_p_records
            )
            if not combined_items:
                combined_records = "No relevant triple!"
            ans_prompt = (
                self._p_dual.replace("{question}", q_str)
                .replace("{triples}", combined_records)
                .replace("{history}", full_sessions)
            )
            ans_prompt = self._cap_prompt(ans_prompt)
            gen_ans = self.llm.generate(ans_prompt)
            metrics = metrics.merge(gen_ans.to_metrics())
            if process_answer(gen_ans.text) != "(E)":
                r2 = RetrievalResult(
                    items=list(r1.items), rendered="",
                    extra={
                        "route": "GROUPCHAT_MEM", "fallback_from": first_route,
                        "traversal_resolved": True, "traversal_iters": iters,
                        "traversal_chain": list(visited), "traversal_n_records": len(acc_records),
                        "neutral_stop": "ANSWERED",
                        "sessions_rendered": full_sessions, "records_rendered": combined_records,
                        "qa_raw_text": gen_ans.text,
                        "qa_reasoning_text": getattr(gen_ans, "reasoning_text", ""),
                        "qa_final_prompt": ans_prompt, "qa_retrieval_rendered": "",
                    },
                    metrics=CallMetrics(),
                )
                return gen_ans.text, r2, metrics

        else:
            # The for loop exhausted all 6 iters (max hop limit) without a break →
            #   still no answer even with further exploration → terminate with "I don't know (E)" as the final
            #   answer (no extra LLM call).
            #   ⚠ for-else: else runs only when the loop ran to completion without any break.
            #      If it exited via break (resolve failed/revisit/added0), else is skipped and it falls through
            #      to FINAL_ENHANCED below.
            r2 = RetrievalResult(
                items=list(r1.items), rendered="",
                extra={
                    "route": "GROUPCHAT_MEM", "fallback_from": first_route,
                    "traversal_resolved": bool(visited), "traversal_iters": iters,
                    "traversal_chain": list(visited), "traversal_n_records": len(acc_records),
                    "neutral_stop": "MAX_ITER_IDK",
                    "sessions_rendered": full_sessions, "records_rendered": full_records,
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
            self._p_dual.replace("{question}", q_str)
            .replace("{triples}", full_records)
            .replace("{history}", full_sessions)
        )
        final_prompt = self._cap_prompt(final_prompt)
        gen_f = self.llm.generate(final_prompt)
        metrics = metrics.merge(gen_f.to_metrics())
        r2 = RetrievalResult(
            items=list(r1.items), rendered="",
            extra={
                "route": "GROUPCHAT_MEM", "fallback_from": first_route,
                "traversal_resolved": bool(visited), "traversal_iters": iters,
                "traversal_chain": list(visited), "traversal_n_records": len(acc_records),
                "neutral_stop": "FINAL_ENHANCED",
                "sessions_rendered": full_sessions, "records_rendered": full_records,
                "qa_raw_text": gen_f.text,
                "qa_reasoning_text": getattr(gen_f, "reasoning_text", ""),
                "qa_final_prompt": final_prompt, "qa_retrieval_rendered": "",
            },
            metrics=CallMetrics(),
        )
        return gen_f.text, r2, metrics

    def _identify_query_nodes(
        self, question: Question, nodes: list[str], metrics: CallMetrics,
        question_idx: "int | None" = None,
        debug_out: "dict | None" = None,
    ) -> tuple[list[str], list[str], str]:
        """Extract query_node candidates (batch + precompute support).

        - If precomputed_query_nodes[question_idx] exists, skip the LLM (precompute first, same pattern as precomputed routing).
        - Otherwise split nodes into query_node_batch_size (default 100) chunks and call _query_node_one_batch per batch →
          merge candidates (dedup, first-seen order). By design: 100-batch.
        Returns: (kg_nodes: list[str], raw_subjects: list[str], last_prompt_text: str)
        - kg_nodes: the final query_node after mapping the stage2 LLM output to KG nodes and applying the _allowed filter
          (used by ablation).
        - raw_subjects: the raw question subjects emitted by the stage2 LLM (*before* KG mapping/_allowed filter, for
          analysis). By design, both raw + kg_nodes are stored.
        - prompt_text: the full prompt of the last batch (empty string if no LLM call/precompute).
        """
        # precompute first (per-llm query_node.json) — 0 LLM calls.
        if question_idx is not None and question_idx in self.precomputed_query_nodes:
            return list(self.precomputed_query_nodes[question_idx]), [], ""
        # Review #2 MEDIUM-E11: trivial return when nodes < k (sparse KG) — LLM skip.
        if not nodes:
            return [], [], ""
        k = min(self.num_query_nodes, len(nodes))
        # Removed the k<=1 (single node) early return — even with 1 node, it must go through query_node_one_batch
        #   (subject identification) to be accurate. (The early return unconditionally returned nodes[0] regardless
        #   of subject = inaccurate.)
        bs = max(1, self.query_node_batch_size)
        merged: list[str] = []
        seen: set[str] = set()
        last_prompt = ""
        # stage1 batch parallelization. Batches are independent (each sees only nodes[s:s+bs]) → concurrent LLM calls.
        #   Results are merged in index order, so it is bit-identical to sequential (same merged order, same last_prompt).
        #   metrics are received as independent CallMetrics per batch and summed in a single-threaded merge (avoids
        #   shared += race). query_node_batch_workers=1 is fully identical to the original sequential path.
        _starts = list(range(0, len(nodes), bs))
        _bw = max(1, min(self.query_node_batch_workers, len(_starts) or 1))

        def _run_batch(_bi: int) -> "tuple[int, list[str], str, CallMetrics]":
            _s = _starts[_bi]
            _bm = CallMetrics()
            _cands, _prompt = _query_node_one_batch(
                self.llm, self._p_query_node_id, question.text, question.questioner,
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
        # query_node 2-pass stage2: feed stage1's merged (union of each batch's query_node_idf candidates) back into the
        #   query_node_idf_select prompt as options, and select the final k query_node only from the options (= a subset of
        #   current R nodes). If merged is empty, skip stage2 (return []).
        #   query_node_check (R node existence verification) was removed — node matching (exact, no fuzzy) is done by
        #   the caller _retrieve_records. Since the LLM may hallucinate entities outside the options, filter final once
        #   more against the merged set.
        if merged:
            final, sel_prompt = _query_node_one_batch(
                self.llm, self._p_query_node_select, question.text, question.questioner, k, merged, metrics
            )
            last_prompt = sel_prompt
            _merged_set = set(merged)
            # The final query_node must always be within (current R node set + questioner).
            #   Even if the stage1/stage2 LLM hallucinates entities outside the options, they are removed here.
            #   The questioner is allowed for first-person questions.
            _allowed = set(nodes)
            if question.questioner:
                _allowed.add(question.questioner)
            filtered = [c for c in final if c in _merged_set and c in _allowed]
            # raw_subjects = the stage2 LLM output final (before KG mapping/_allowed filter).
            raw_subjects = list(final)
            # Prevent 0-candidate collapse from stage2 LLM caprice: if stage1 has candidates but stage2 discards
            #   them all, keep the top k of stage1 (within allowed) (avoids wiping out query_node-based record retrieval).
            #   The first-person questioner is additionally guaranteed by the caller _retrieve_records. A genuine 'no
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
def _query_node_one_batch(llm, prompt_template: str, question_text: str, questioner: str,
                      k: int, nodes_batch: list[str],
                      metrics: "CallMetrics | None" = None) -> tuple[list[str], str]:
    """One query_node_idf LLM call with nodes_batch as options. Returns (candidates, prompt).

    structured (QueryNodeList) first → free-form (ast.literal_eval) fallback on failure — **identical** to the parse
    semantics of the original _identify_query_nodes (protects Qwen2.5 reproducibility). If metrics is None, skip accumulation.
    llm is an object with generate_structured/generate (returning GenerationRecord) (the memory's self.llm).
    """
    nodes_str = "[" + ", ".join(nodes_batch) + "]"
    prompt = (
        prompt_template.replace("{question}", question_text)
        .replace("{questioner}", questioner)
        .replace("{k}", str(k))
        .replace("{options}", nodes_str)
    )
    try:
        parsed, rec = llm.generate_structured(prompt, QueryNodeList)
        if metrics is not None and rec is not None:
            metrics.latency_sec += rec.latency_sec
            metrics.input_tokens += rec.input_tokens
            metrics.output_tokens += rec.output_tokens
            metrics.num_llm_calls += 1
        if isinstance(parsed, QueryNodeList) and parsed.subjects:
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


def _retrieve_by_timestamp(kg: RelationalGraph, timestamps: dict) -> list[dict]:
    out: list[dict] = []
    for t, meta in timestamps.items():
        tp = meta["type"]
        for h in kg.graph:
            for r in kg.graph[h]:
                for ts in kg.graph[h][r]:
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
                        for tl in kg.graph[h][r][ts]:
                            q = {"head": h, "relation": r, "tail": tl, "start_date": ts,
                                 "sess_idx": kg.record_sess.get((h, r, tl, ts))}
                            if q not in out and is_valid_record(q):
                                out.append(q)
    return out


def _filter_records_by_timestamp(records: list[dict], timestamps: dict) -> list[dict]:
    out: list[dict] = []
    for q in records:
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


def _record_to_string(q: dict) -> str:
    # record list element format: {head, relation, tail, date}
    return f'{{{q["head"]}, {q["relation"].lower()}, {q["tail"]}, {q["start_date"]}}}'


def _dedup_record_items(records: "list[dict]") -> "list[dict]":
    """record set for retrieval recall B/C — normalize (head,relation,tail,start_date) + dedup (order-agnostic set)."""
    seen: set = set()
    out: list = []
    for q in records or []:
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


def _render_records_in_order(records: list[dict], record_format: str, context_size: int,
                           base_prompt: str) -> "tuple[str, list[dict]]":
    """Render records sorted in descending cosine similarity in that exact order (no date-grouping).

    Returns (for retrieval-recall B/C measurement): (rendered_text, included_records).
      included_records = the list of record dicts that actually fit within context_size (= C, the part
      included in the QA prompt).

    Records must already be sorted by the caller so the highest-similarity record comes first. Preserving
    that order, records are added one at a time from the top (top-first), filling only until just before
    base_prompt + accumulated record tokens exceed context_size, so the highest-similarity record sits near
    the front of the prompt and survives even under truncation.

    Each record is rendered as [head, relation, tail, date] (date is the 4th element, the whole wrapped in
    "[" ... "]"). No grouping by date. Token-budget accounting starts from base_prompt tokens.
    """
    if not records:
        return "No relevant triple!", []
    base_tokens = count_tokens(base_prompt)
    cur = base_tokens + 2  # allowance for the enclosing "[" "]" tokens
    items: list[str] = []
    included: list[dict] = []  # records actually fit within context_size (= C, for retrieval recall measurement)
    for q in records:
        date = q.get("start_date") or ""
        if record_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        st = count_tokens(s) + 2  # allowance for the ", " separator tokens
        # stop if base_prompt + accumulated so far + this record exceeds context_size (preserves top-first).
        # context_size<=0 = token limit off → fill all without cutting (unlimited).
        if context_size > 0 and cur + st > context_size:
            break
        items.append(s)
        included.append(q)
        cur += st
    # If not even the first record fits and items is empty, keep at least one record, trimmed to context_size.
    if not items and records:
        q = records[0]
        date = q.get("start_date") or ""
        if record_format == "sentence":
            s = f"{q['head']} {q['relation'].lower()} {q['tail']} {date}"
        else:
            s = f"{{{q['head']}, {q['relation'].lower()}, {q['tail']}, {date}}}"
        items = [trim_text_to_tokens(s, max(1, context_size - 2))]
        included = [records[0]]
    return "[" + ", ".join(items) + "]", included
