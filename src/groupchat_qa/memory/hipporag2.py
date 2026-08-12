"""HippoRAG 2 memory - OpenIE-based relational fact store + Personalized
PageRank retrieval.

Paper: "From RAG to Memory: Non-Parametric Continual Learning for Large
       Language Models" (arXiv 2502.14802, HippoRAG 2)
Official repo: https://github.com/OSU-NLP-Group/HippoRAG

Core idea (faithfully ported from the official repo
`src/hipporag/HippoRAG.py`):

  [offline / commit]
    - Run OpenIE on each session dialogue (= passage) to extract named entities
      (= phrases) and triples. As in the official repo, this uses **2-call
      OpenIE**:
      (1) NER (passage -> named_entities) -> (2) triple extraction
      (passage + named_entities -> triples), mirroring the official ner.py and
      triple_extraction.py templates. openie_single_call=True falls back to
      1-call.
    - Add two kinds of nodes to the (undirected) graph:
        * phrase node  (entity-<text>)   - subject/object of a triple
        * passage node (passage-<idx>)   - session dialogue chunk
      Add three kinds of edges:
        * fact edge    (phrase<->phrase)  - the (s,o) pair of a triple.
          weight = co-occurrence count.
        * passage edge (passage->phrase)  - phrase appearing in the passage.
          weight = 1.0.
        * synonym edge (phrase<->phrase)  - embedding cosine sim >= threshold.
          weight = sim.
    - Store embeddings for both phrases and passages (for dense scoring).

  [online / retrieve]
    1. Compare the query densely against all facts (triples) -> fact score
       (min-max normalized).
    2. Keep only relevant facts among the top candidates via an LLM "filter"
       (recognition memory).
    3. Assign seed weights to the phrases of the remaining facts
       (phrase weight = mean fact score of facts containing the phrase /
       #passages the phrase belongs to).
    4. Add query-passage dense score x passage_node_weight as seed weights for
       passages.
    5. Run Personalized PageRank with this seed (reset/personalization
       distribution).
    6. Rank passages by their PPR score -> return the top-k passages.
    7. If no fact survives the filter, fall back to dense passage retrieval (as
       in the paper).
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx
from pydantic import BaseModel, Field

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder
from groupchat_qa.utils import (
    CallMetrics,
    Item,
    Question,
    RetrievalResult,
    Session,
    cos_sim,
    apply_token_limit,
    count_tokens,
    is_valid_record,
    load_pickle,
    load_prompt,
    remove_brackets,
    remove_quotes,
    render_question,
    save_pickle,
    trim_text_to_tokens,
)

from .base import Memory


# pydantic schema - structured LLM responses
class OpenIEResult(BaseModel):
    """OpenIE response schema - named entity list + triple list.

    Each triple is a 3-element [subject, predicate, object] list.
    """
    named_entities: list[str] = Field(
        default_factory=list, description="named entities in the passage"
    )
    triples: list[list[str]] = Field(
        default_factory=list,
        description="list of [subject, predicate, object] triples",
    )


class NERResult(BaseModel):
    """NER response schema - named entity list (official repo ner.py output
    format).

    Output of the first OpenIE stage (NER). Its entities feed the second stage
    (triple extraction).
    """
    named_entities: list[str] = Field(
        default_factory=list, description="named entities in the passage"
    )


class TripleResult(BaseModel):
    """Triple extraction response schema - triple list (official repo
    triple_extraction.py output).

    Output of the second OpenIE stage - generated from passage + NER entities.
    """
    triples: list[list[str]] = Field(
        default_factory=list,
        description="list of [subject, predicate, object] triples",
    )


class FactFilterResult(BaseModel):
    """Recognition-memory filter response schema - list of relevant facts kept.

    Each fact is a 3-element [subject, predicate, object] list.
    """
    fact: list[list[str]] = Field(
        default_factory=list,
        description=(
            "filtered relevant facts, each [subject, predicate, object]"
        ),
    )


# Graph store - phrase/passage nodes + 3 edge types
@dataclass
class HippoGraphStore:
    """HippoRAG 2 in-memory relational fact store + auxiliary indices.

    Merges the official repo's EmbeddingStore (chunk/entity/fact) + igraph into
    one: a networkx undirected graph plus phrase/passage embedding dicts.
    """
    # phrase node: key="entity-<lower_text>", value=original surface form (for
    # rendering)
    phrase_text: dict[str, str] = field(default_factory=dict)
    phrase_emb: dict[str, np.ndarray] = field(default_factory=dict)
    # passage node: key="passage-<sess_idx>", value=passage metadata
    passage_text: dict[str, str] = field(default_factory=dict)
    passage_date: dict[str, str] = field(default_factory=dict)
    passage_sess_idx: dict[str, int] = field(default_factory=dict)
    passage_emb: dict[str, np.ndarray] = field(default_factory=dict)
    # fact: (s, p, o) tuple - target of dense scoring. index-aligned with
    # fact_emb.
    facts: list[tuple[str, str, str]] = field(default_factory=list)
    fact_emb: list[np.ndarray] = field(default_factory=list)
    # session index each fact was extracted from (index-aligned with facts) -
    # blocks future-session facts at retrieve time.
    fact_sess_idx: list[int] = field(default_factory=list)
    # phrase key -> set of passage keys where the phrase appears (paper's
    # ent_node_to_chunk_ids)
    phrase_to_passages: dict[str, set] = field(default_factory=dict)
    # phrase pairs for which a synonym edge has already been computed (dedup
    # for incremental build)
    _synonym_done: set = field(default_factory=set)

    # networkx graph holds fact/passage/synonym edges.
    graph: Any = field(default_factory=nx.Graph)

    # provenance indices - used to reconstruct PPR from past information only
    # (fully block future leakage)
    # per-session co-occurrence count of fact edges: (sorted phrase pair) ->
    # {sess_idx: count}.
    # at retrieve time, sum only counts with sess_idx < current to restore edge
    # weight point-in-time-accurately.
    fact_edge_sessions: dict[tuple, dict] = field(default_factory=dict)
    # session index where a phrase first appeared (gates the phrase node's
    # existence by time).
    phrase_first_sess: dict[str, int] = field(default_factory=dict)
    # passage key -> set of phrase keys appearing in that passage (for
    # reconstructing passage->phrase edges).
    passage_to_phrases: dict[str, set] = field(default_factory=dict)
    # synonym edge weight: (sorted phrase pair) -> cosine sim (time-invariant,
    # gated by phrase-existence time).
    synonym_weight: dict[tuple, float] = field(default_factory=dict)


@dataclass
class HippoExtractResult:
    """Output of phase 1 (extract) of the 2-phase commit - session-independent
    LLM/embedding result.

    The cumulative store (graph/facts/phrase) is mutated only in phase 2
    (commit_prepared). extract never reads/writes self._store, so it is safe
    for ThreadPool parallelism.
    """
    entities: list[str]
    triples: list[list[str]]
    # dense vector of session.dialogue (precomputed or live).
    passage_emb: np.ndarray
    metrics: CallMetrics


def _phrase_key(text: str) -> str:
    """phrase node key - lower-cased normalized text with prefix."""
    return "entity-" + text.strip().lower()


def _passage_key(sess_idx: int) -> str:
    """passage node key - based on session index."""
    return f"passage-{sess_idx}"


def _format_hippo_chunk(text: str, date: str, sess_idx: int) -> str:
    """Render a retrieved passage into a QA prompt {history} chunk - including
    a timestamp header.

    The hipporag2_qa.txt EXAMPLE gives each History entry a timestamp
    ([Date: ..., Session N]), so the actual rendering includes the timestamp
    too. The format matches the system-wide convention (GroupChat-Mem
    `_format_dialogue_chunk`): "[Date: {date} (Session {sess_idx})]" - unifying
    the history format the model sees. If date is empty, "[Date:  (Session N)]"
    would result, so only Session is shown.
    """
    if date:
        header = f"[Date: {date} (Session {sess_idx})]"
    else:
        header = f"[Session {sess_idx}]"
    return f"\n{header}\n{text}\n"


def _cosine_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Return (N,) cosine similarity between each row of an (N, D) matrix and a
    (D,) vector.

    The official repo unit-normalizes embeddings and computes cosine via dot
    product. The embedder does not guarantee unit norm, so we normalize here.
    Zero-norm rows/vectors are treated as 0.0 (prevents NaN - edge case).
    """
    mat = np.asarray(mat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float64)
    row_norms = np.linalg.norm(mat, axis=1)  # (N,)
    v_norm = float(np.linalg.norm(vec))
    if v_norm == 0.0:
        return np.zeros(mat.shape[0], dtype=np.float64)
    dots = mat @ vec  # (N,)
    denom = row_norms * v_norm
    out = np.zeros_like(dots)
    nz = denom > 0
    out[nz] = dots[nz] / denom[nz]  # zero-norm rows stay 0.
    return out


def _min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Same as the official repo's min_max_normalize - scale to 0~1.

    Non-finite (NaN/inf) values are replaced with 0 before normalization
    (prevents NaN propagation from e.g. zero-norm embeddings - edge case). Not
    in the official repo but a safety net.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    # NaN/inf -> 0 (cos_sim may return NaN for zero-norm vectors).
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo == 0.0:
        # all equal -> avoid division by zero. If positive, return ones
        # (uniform weight):
        # when the filter keeps a single fact or all fact scores are equal,
        # returning
        # zeros would collapse the phrase seed to 0, degenerating PPR to
        # passage-only
        #   (DPR). If 0 (or negative), keep 0.
        return np.ones_like(x) if hi > 0.0 else np.zeros_like(x)
    return (x - lo) / (hi - lo)


# HippoRAG2Memory
class HippoRAG2Memory(Memory):
    """HippoRAG 2 - OpenIE fact store + recognition-memory filter + PPR
    retrieval.

    config dict keys (all optional - defaults are the official repo BaseConfig
    values):
      synonymy_sim_threshold : float - synonym edge sim threshold (default 0.8)
      synonymy_topk          : int   - KNN synonym candidates per phrase
                                       (default 2047, official repo)
      linking_top_k          : int   - #candidate facts passed to filter +
                                       #PPR seed phrases (default 5)
      passage_node_weight    : float - passage seed weight multiplier
                                       (default 0.05)
      damping                : float - PPR damping factor (default 0.5)
      num_retrieved          : int   - #passages returned (default 40; 0 or
                                       None -> 40)
      context_size           : int   - final QA prompt token limit (default 0 =
                                       off/unlimited; >0 = upper bound)
      use_filter             : bool  - use the recognition-memory LLM filter
                                       (default True)
      openie_single_call     : bool  - if True, 1-call OpenIE (prompt_openie).
                                       **default True** (this project's
                                       HippoRAG2 default = 1-call). Only when
                                       explicitly False, use the official
                                       2-call (prompt_ner -> prompt_triple).
      question_format        : str   - multi_structured / multi_unstructured /
                                       open_ended
      prompt_ner/triple      : official 2-call OpenIE prompt filenames (when
                                       used)
      prompt_openie          : 1-call OpenIE prompt (only when
                                       openie_single_call=True)
      prompt_filter/qa       : prompt filenames (relative to assets/prompt/)
      hipporag2_snapshot     : str | None - prebuilt graph pickle path
                                       (restored if present)
    """

    NAME = "hipporag2"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        self._store = HippoGraphStore()

        # graph build parameters (official repo BaseConfig defaults)
        self.synonymy_sim_threshold = float(
            cfg.get("synonymy_sim_threshold", 0.8))
        # official repo BaseConfig synonymy_edge_topk = 2047 (the threshold 0.8
        # break
        # triggers first, so its effective impact is small, but keep the
        # default aligned).
        self.synonymy_topk = int(cfg.get("synonymy_topk", 2047))

        # retrieval parameters
        # If linking_top_k <= 0, fact candidates are truncated to 0 (k=min(...)
        # in retrieve),
        # so every question silently falls back to DPR - measuring DPR rather
        # than the PPR
        # baseline. Force a minimum of 1 to guard against unintended 0/negative
        # config.
        self.linking_top_k = max(1, int(cfg.get("linking_top_k", 5)))
        self.passage_node_weight = float(cfg.get("passage_node_weight", 0.05))
        self.damping = float(cfg.get("damping", 0.5))
        _nr = cfg.get("num_retrieved", 40)  # default 40 (# passages returned)
        self.num_retrieved = 40 if (_nr is None or int(_nr) <= 0) else int(_nr)
        # 0/negative = token limit off (unlimited)
        self.context_size = int(cfg.get("context_size", 0))
        self.use_filter = bool(cfg.get("use_filter", True))
        # HippoRAG2 default = **1-call OpenIE** (hipporag2_openie.txt).
        #   Always 1-call unless openie_single_call: false is explicitly set.
        # (The official repo uses NER->triple 2-call, but this project's
        # default is 1-call.)
        self.openie_single_call = bool(cfg.get("openie_single_call", True))

        # prompts
        # HippoRAG2 variant = **official (2-call) vs 1-call** only, default =
        # **always 1-call**.
        # - 1-call (default): hipporag2_openie.txt extracts entities+triples in
        # one call (openie_single_call=True).
        # - official (only when openie_single_call: false): hipporag2_ner.txt
        # (NER) + hipporag2_triple.txt (triple), 2-call.
        self._p_ner = load_prompt(cfg.get("prompt_ner", "hipporag2_ner.txt"))
        self._p_triple = load_prompt(
            cfg.get("prompt_triple", "hipporag2_triple.txt"))
        # 1-call OpenIE prompt (project default - used when
        # openie_single_call=True).
        self._p_openie = load_prompt(
            cfg.get("prompt_openie", "hipporag2_openie.txt"))
        self._p_filter = load_prompt(
            cfg.get("prompt_filter", "hipporag2_filter.txt"))
        self._p_qa = load_prompt(cfg.get("prompt_qa", "hipporag2_qa.txt"))
        # The QA template token count is constant -> compute once (avoids
        # re-tokenizing per render).
        self._p_qa_tokens = count_tokens(self._p_qa)

        self.question_format = cfg.get("question_format", "multi_structured")

        # prebuilt graph snapshot (optional) - restore immediately if present.
        snap_path = cfg.get("hipporag2_snapshot")
        if snap_path:
            self.restore(snap_path)

    # commit - OpenIE -> graph build
    def commit(self, session: Session) -> CallMetrics:
        """Single-call path (no 2-phase) - run extract + commit_prepared
        sequentially.

        Result/metrics are 100% identical to the 2-phase path (same internal
        logic reused). Works as-is for callers that only use commit(session)
        (e.g. the sequential path).
        """
        ex = self.extract(session)
        return self.commit_prepared(session, ex)

    # phase 1: session-independent LLM/embedding (ThreadPool-safe, store
    # immutable)
    def extract(self, session: "Session") -> "HippoExtractResult | None":
        metrics = CallMetrics()
        if not session.dialogue:
            return None  # empty session -> commit_prepared is a no-op.

        # 1) OpenIE - extract named entities + triples (one LLM call). store
        # unchanged.
        entities, triples, ie_metrics = self._openie(session.dialogue)
        metrics = metrics.merge(ie_metrics)

        # 2) passage embedding - reuse the value injected by data_loader, else
        # live-embed.
        # If the injected value is an empty list / wrong dimension, it would be
        # stored with
        # (0,) shape and raise ValueError in retrieve's cos_sim / np.stack.
        # Trust it only
        #   when 1-D and non-empty; otherwise live-embed.
        p_emb = None
        if session.dialogue_embedding is not None:
            _e = np.asarray(session.dialogue_embedding, dtype=np.float32)
            if _e.ndim == 1 and _e.size > 0:
                p_emb = _e
        if p_emb is None:
            t0 = time.perf_counter()
            vec = self.embedder.embed_one(session.dialogue)
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += 1
            p_emb = np.asarray(vec, dtype=np.float32)
        return HippoExtractResult(
            entities=list(entities), triples=list(triples), passage_emb=p_emb,
            metrics=metrics,
        )

    # phase 2: cumulative build (must be called sequentially in ascending
    # session idx)
    def commit_prepared(
        self, session: "Session", extract_result: "HippoExtractResult | None"
    ) -> CallMetrics:
        if extract_result is None:
            return CallMetrics()  # empty session - nothing to build.
        # Carry over and accumulate extract's metrics (OpenIE LLM + passage
        # embed).
        metrics = extract_result.metrics
        entities = extract_result.entities
        triples = extract_result.triples
        p_emb = extract_result.passage_emb

        # 2) add passage node (+ embedding).
        p_key = _passage_key(session.idx)
        self._store.passage_text[p_key] = session.dialogue
        self._store.passage_date[p_key] = session.date
        self._store.passage_sess_idx[p_key] = session.idx
        self._store.passage_emb[p_key] = p_emb
        self._store.graph.add_node(p_key)

        # 3) triple phrase (=subject/object) nodes + embeddings.
        #    Treat both triple phrases and NER-extracted entities as phrases.
        new_phrases: list[str] = []
        phrases_in_passage: set = set()

        def _register_phrase(raw: str) -> str | None:
            """Normalize + register a phrase. Returns node key (None if
            invalid)."""
            text = remove_quotes(remove_brackets(raw.strip()))
            if not text:
                return None
            key = _phrase_key(text)
            if key not in self._store.phrase_text:
                self._store.phrase_text[key] = text
                new_phrases.append(key)
                self._store.graph.add_node(key)
                # provenance: record the session where the phrase first
                # appeared (for PPR time gating).
                self._store.phrase_first_sess[key] = session.idx
            phrases_in_passage.add(key)
            return key

        # 3-a) fact edge (phrase<->phrase) - the (s, o) pair of a triple.
        for tri in triples:
            if len(tri) != 3:
                continue
            s, p, o = (
                str(tri[0]).strip(),
                str(tri[1]).strip(),
                str(tri[2]).strip(),
            )
            # drop lone pronouns / cyclic triples (same policy as
            # GroupChat-Mem).
            if not is_valid_record({"head": s, "tail": o}):
                continue
            s_key = _register_phrase(s)
            o_key = _register_phrase(o)
            if s_key is None or o_key is None:
                continue
            # drop self-loop facts (s/o are the same phrase after
            # normalization) - avoids PPR
            # distortion from networkx self-loops + double-counting the same
            # phrase in _build_seed.
            if s_key == o_key:
                continue
            # Store the fact's s/o as the *same normalized text as the node*
            # (phrase_text).
            # Previously raw `s.lower()` was stored, so phrases with
            # brackets/quotes had their
            # seed lookup (`_phrase_key(f[0])`) mismatch the registered node
            # key
            # (remove_quotes/remove_brackets applied) -> that phrase's PPR seed
            # was entirely
            # lost (PPR degenerated to DPR). phrase_text[key] is the cleaned
            # original, and
            #   lower() matches `_phrase_key` exactly.
            s_clean = self._store.phrase_text[s_key].lower()
            o_clean = self._store.phrase_text[o_key].lower()
            # store fact (for dense scoring). text unified to lower-case
            # (official repo text_processing).
            self._store.facts.append((s_clean, p.lower(), o_clean))
            # record the fact's source session index (keeps index alignment
            # with facts) - blocks future facts.
            self._store.fact_sess_idx.append(session.idx)
            # fact edge weight = accumulated co-occurrence count (official repo
            # add_fact_edges).
            g = self._store.graph
            if g.has_edge(s_key, o_key):
                g[s_key][o_key]["weight"] += 1.0
            else:
                g.add_edge(s_key, o_key, weight=1.0)
            # provenance: record the fact edge's per-session co-occurrence
            # count (for PPR time restoration).
            pair = (s_key, o_key) if s_key <= o_key else (o_key, s_key)
            sess_counts = self._store.fact_edge_sessions.setdefault(pair, {})
            sess_counts[session.idx] = sess_counts.get(session.idx, 0.0) + 1.0

        # 3-b) register NER-only entities as phrase nodes too (connect to
        # passage even without a triple).
        for ent in entities:
            _register_phrase(str(ent))

        # 4) passage edge (passage->phrase) - weight 1.0.
        for ph_key in phrases_in_passage:
            self._store.graph.add_edge(p_key, ph_key, weight=1.0)
            self._store.phrase_to_passages.setdefault(ph_key, set()).add(p_key)
        # provenance: set of phrases appearing in this passage (for
        # time-reconstructing passage->phrase edges).
        self._store.passage_to_phrases[p_key] = set(phrases_in_passage)

        # 5) compute embeddings for new phrases (needed for synonym edges +
        # fact embeddings).
        if new_phrases:
            texts = [self._store.phrase_text[k] for k in new_phrases]
            t0 = time.perf_counter()
            embs = self.embedder.embed(texts)
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += len(texts)
            # If the embedder returns a different count than the input,
            # phrase<->embedding
            # alignment silently breaks (zip stops at the shorter side). Fail
            # explicitly.
            if len(embs) != len(texts):
                raise RuntimeError(
                    f"embedder returned {len(embs)} embeddings for "
                    f"{len(texts)} phrases"
                )
            for k, e in zip(new_phrases, embs):
                self._store.phrase_emb[k] = np.asarray(e, dtype=np.float32)

        # 6) fact embedding - embed only the facts added in this session.
        # Keeps the index alignment between the facts list and the fact_emb
        # list.
        n_have = len(self._store.fact_emb)
        n_all = len(self._store.facts)
        if n_all > n_have:
            new_facts = self._store.facts[n_have:]
            fact_strings = [f"{s} {p} {o}" for (s, p, o) in new_facts]
            t0 = time.perf_counter()
            f_embs = self.embedder.embed(fact_strings)
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += len(fact_strings)
            # On count mismatch, the facts<->fact_emb index alignment
            # permanently breaks ->
            #   retrieve would use the wrong fact's embedding. Fail explicitly.
            if len(f_embs) != len(fact_strings):
                raise RuntimeError(
                    f"embedder returned {len(f_embs)} embeddings for "
                    f"{len(fact_strings)} facts"
                )
            for e in f_embs:
                self._store.fact_emb.append(np.asarray(e, dtype=np.float32))

        # 7) synonym edge - cosine sim >= threshold between new and existing
        # phrases.
        if new_phrases:
            self._add_synonym_edges(new_phrases)

        return metrics

    # OpenIE call (structured -> free-form fallback)
    def _openie(
        self, passage: str
    ) -> tuple[list[str], list[list[str]], CallMetrics]:
        """passage -> (entities, triples, metrics).

        As in the official repo (openie_openai_gpt.py), **2-call OpenIE**:
          (1) NER (passage -> named_entities)
          (2) triple extraction (passage + named_entities -> triples)
        openie_single_call=True falls back to the 1-call path (_openie_single)
        - an intentional deviation.
        """
        if self.openie_single_call:
            return self._openie_single(passage)
        metrics = CallMetrics()

        # (1) NER - passage -> named_entities.
        entities = self._ner(passage, metrics)
        # (2) triple extraction - passage + named_entities -> triples (entities
        # injected).
        triples = self._triple_extraction(passage, entities, metrics)
        return entities, triples, metrics

    def _ner(self, passage: str, metrics: CallMetrics) -> list[str]:
        """Official NER call (structured first, free-form on failure). metrics
        accumulated in place."""
        prompt = self._p_ner.replace("{passage}", passage)
        try:
            parsed, rec = self.llm.generate_structured(prompt, NERResult)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, NERResult):
                return [str(e) for e in parsed.named_entities]
        except Exception:
            pass
        gen = self.llm.generate(prompt)
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        return _parse_ner_json(gen.text)

    def _triple_extraction(
        self, passage: str, entities: list[str], metrics: CallMetrics
    ) -> list[list[str]]:
        """Official triple extraction call - passage + named_entities input.
        metrics accumulated in place."""
        ent_json = json.dumps([str(e) for e in entities])
        prompt = self._p_triple.replace("{passage}", passage).replace(
            "{named_entities}", ent_json
        )
        try:
            parsed, rec = self.llm.generate_structured(prompt, TripleResult)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, TripleResult):
                # enforce len==3 + str elements (prevents node/embedding
                # pollution from nested list/dict elements).
                return [
                    [str(x) for x in t]
                    for t in parsed.triples
                    if isinstance(t, list) and len(t) == 3
                ]
        except Exception:
            pass
        gen = self.llm.generate(prompt)
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        return _parse_triple_json(gen.text)

    def _openie_single(
        self, passage: str
    ) -> tuple[list[str], list[list[str]], CallMetrics]:
        """1-call OpenIE (entity+triple in one call). Only when
        openie_single_call=True.

        structured first, free-form JSON parsing on failure.
        """
        metrics = CallMetrics()
        prompt = self._p_openie.replace("{passage}", passage)

        # 1) try structured
        try:
            parsed, rec = self.llm.generate_structured(prompt, OpenIEResult)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, OpenIEResult):
                # The structured path also enforces len==3 + str elements, like
                # free-form
                # (_parse_openie_json). Previously only isinstance(list) was
                # checked, so if
                # the model returned nested list/dict elements, broken phrases
                # like
                #   str(['a','b']) permanently polluted nodes/embeddings.
                triples = [
                    [str(x) for x in t]
                    for t in parsed.triples
                    if isinstance(t, list) and len(t) == 3
                ]
                return list(parsed.named_entities), triples, metrics
        except Exception:
            pass

        # 2) free-form fallback - parse the JSON object directly.
        gen = self.llm.generate(prompt)
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        entities, triples = _parse_openie_json(gen.text)
        return entities, triples, metrics

    # synonym edge build
    def _add_synonym_edges(self, new_phrase_keys: list[str]) -> None:
        """KNN between new phrase nodes and all phrase nodes -> synonym edge if
        sim >= threshold.

        Same policy as the official repo add_synonymy_edges:
          - exclude phrases with <= 2 alphanumeric characters (noise).
          - scan in descending sim order, stop below threshold, at most
            synonymy_topk.
          - edge weight = cosine similarity.
        """
        all_keys = list(self._store.phrase_emb.keys())
        if len(all_keys) < 2:
            return
        # (N, D)
        all_mat = np.stack([self._store.phrase_emb[k] for k in all_keys])
        # normalize (cosine = dot after normalization).
        norms = np.linalg.norm(all_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_unit = all_mat / norms

        for q_key in new_phrase_keys:
            q_text = self._store.phrase_text.get(q_key, "")
            # skip if fewer than 3 alphanumeric chars (official repo's
            # re.sub('[^A-Za-z0-9]','') length > 2 condition).
            if len(re.sub(r"[^A-Za-z0-9]", "", q_text)) <= 2:
                continue
            q_vec = self._store.phrase_emb.get(q_key)
            if q_vec is None:
                continue
            qn = q_vec / (np.linalg.norm(q_vec) or 1.0)
            sims = all_unit @ qn  # (N,) cosine sim
            # argsort(-sims, stable): deterministic order on tied sims
            # (argsort(x)[::-1] reverses
            # ties non-deterministically). Ensures reproducibility.
            order = np.argsort(-sims, kind="stable")
            cnt = 0
            for j in order:
                if cnt >= self.synonymy_topk:
                    break
                nn_key = all_keys[j]
                if nn_key == q_key:
                    continue
                sim = float(sims[j])
                if sim < self.synonymy_sim_threshold:
                    # descending order, so everything after is below threshold.
                    break
                pair = tuple(sorted((q_key, nn_key)))
                if pair in self._store._synonym_done:
                    cnt += 1
                    continue
                self._store._synonym_done.add(pair)
                # synonym edge weight = sim. Do not overwrite an existing
                # fact/passage edge.
                if not self._store.graph.has_edge(q_key, nn_key):
                    self._store.graph.add_edge(q_key, nn_key, weight=sim)
                # provenance: store synonym weight (time-invariant, gated by
                # phrase-existence time at PPR reconstruction).
                self._store.synonym_weight[pair] = sim
                cnt += 1

    # retrieve - fact scoring -> filter -> PPR
    def retrieve(
        self, question: Question, current_session_idx: int
    ) -> RetrievalResult:
        metrics = CallMetrics()

        # 0) only passages from sessions before current_session_idx are
        # candidates.
        passage_keys = [
            k
            for k in self._store.passage_text
            if self._store.passage_sess_idx[k] < current_session_idx
        ]
        if not passage_keys:
            return RetrievalResult(
                items=[], rendered="No relevant history!",
                extra={"route": "HIPPORAG2", "ppr_used": False},
                metrics=metrics,
            )

        # query embedding (once - shared by fact/passage scoring).
        t0 = time.perf_counter()
        q_vec = np.asarray(
            self.embedder.embed_one(question.text), dtype=np.float32)
        metrics.latency_sec += time.perf_counter() - t0
        metrics.num_embedding_calls += 1

        # 1) fact dense scoring.
        # - block future-session facts: only facts with fact_sess_idx <
        # current_session_idx are candidates.
        # (a fact extracted from a future dialogue must not enter the PPR seed
        # - prevents leakage.)
        # - top_facts : final selected fact list (filter survivors or all
        # candidates).
        # - top_fact_scores : per-fact dense scores, index-aligned with
        # top_facts
        # (the official repo adds each fact's own score to the seed - not
        # averaged away).
        top_facts: list[tuple[str, str, str]] = []
        top_fact_scores: list[float] = []
        if self._store.fact_emb:
            # keep only indices of past-session facts
            # (facts/fact_emb/fact_sess_idx are all index-aligned).
            valid_idx = [
                i
                for i in range(len(self._store.fact_emb))
                if i < len(self._store.fact_sess_idx)
                and self._store.fact_sess_idx[i] < current_session_idx
            ]
            if valid_idx:
                # (F', D)
                fact_mat = np.stack(
                    [self._store.fact_emb[i] for i in valid_idx])
                # cosine similarity - the official repo unit-normalizes
                # embeddings then dots.
                # the embedder does not guarantee unit norm, so normalize here
                # (avoids long-vector bias).
                raw = _cosine_matvec(fact_mat, q_vec)
                # aligned with valid_idx order.
                scores = _min_max_normalize(raw)
                # top linking_top_k candidate facts (within valid_idx).
                k = min(self.linking_top_k, len(scores))
                # argsort(-scores, stable): deterministic order on tied scores
                # (reproducibility).
                order = np.argsort(-scores, kind="stable")[:k].tolist()
                # original fact indices.
                cand_idx = [valid_idx[j] for j in order]
                # each candidate's dense score.
                cand_scores = [float(scores[j]) for j in order]
                cand_facts = [self._store.facts[i] for i in cand_idx]

                # 2) recognition memory - LLM filter (keep only relevant
                # facts).
                if self.use_filter and cand_facts:
                    kept, fmetrics = self._filter_facts(
                        question.text, cand_facts)
                    metrics = metrics.merge(fmetrics)
                    # As in the official repo (rerank.py DSPyFilter.rerank):
                    # the LLM-returned
                    # fact's surface form may not exactly match a candidate
                    # (re-cased/re-spaced),
                    # so fuzzy-match each returned fact to a candidate
                    # (difflib) and preserve the
                    # matched *candidate's original tuple* and its *precomputed
                    # score* (do not
                    # trust the LLM string). An exact set comparison would
                    # silently drop
                    # re-cased/re-spaced facts and needlessly force a DPR
                    # fallback.
                    cand_strs = [str(c) for c in cand_facts]
                    sel: list[tuple[tuple[str, str, str], float]] = []
                    used_idx: set[int] = set()
                    for f in kept:
                        if len(f) != 3:
                            continue
                        # (a) consider only candidates not yet matched -> two
                        # identical-string
                        # candidates match to different indices (the old
                        # dict{str:idx}
                        # collapsed duplicate strings to the last index,
                        # dropping a fact entirely).
                        # (b) cutoff=0.6 - the old cutoff=0.0 *always* returned
                        # the closest
                        # candidate, so even a hallucinated irrelevant fact was
                        # force-matched,
                        # polluting the seed and making the `if not match`
                        # guard dead code.
                        #   0.6 accepts only sufficiently similar matches.
                        avail = [
                            (i, cand_strs[i])
                            for i in range(len(cand_strs))
                            if i not in used_idx
                        ]
                        if not avail:
                            break
                        avail_strs = [s for _, s in avail]
                        match = difflib.get_close_matches(
                            str(f), avail_strs, n=1, cutoff=0.6)
                        if not match:
                            continue
                        ci = avail[avail_strs.index(match[0])][0]
                        used_idx.add(ci)
                        sel.append((cand_facts[ci], cand_scores[ci]))
                    # if the filter removes everything -> sel is empty -> DPR
                    # fallback below in 3) (as official repo).
                else:
                    sel = list(zip(cand_facts, cand_scores))

                for f, sc in sel:
                    top_facts.append(f)
                    top_fact_scores.append(sc)

        # 3) if no fact survives the filter, fall back to dense passage
        # retrieval (as in the paper).
        if not top_facts:
            return self._dense_passage_fallback(
                question, q_vec, passage_keys, metrics, ppr_used=False
            )

        # 4) build the PPR seed distribution (per-fact scores passed - faithful
        # to official repo).
        # pass current_session_idx - count a phrase's passages over past
        # passages only (prevents future-frequency leakage).
        seed, linking_info = self._build_seed(
            q_vec, passage_keys, top_facts, top_fact_scores,
            current_session_idx,
        )

        # 5) Personalized PageRank (pass current_session_idx - excludes future
        # passage nodes).
        ranked_passages = self._run_ppr(
            seed, passage_keys, current_session_idx)
        if not ranked_passages:
            return self._dense_passage_fallback(
                question, q_vec, passage_keys, metrics, ppr_used=False
            )

        # 6) top-k passages -> render.
        ranked_passages = ranked_passages[: self.num_retrieved]
        items, rendered = self._render_passages(ranked_passages)
        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant history!",
            extra={
                "route": "HIPPORAG2",
                "ppr_used": True,
                "top_facts": top_facts,
                "linking_phrases": linking_info,
            },
            metrics=metrics,
        )

    # filter (recognition memory)
    def _filter_facts(
        self, question: str, cand_facts: list[tuple[str, str, str]]
    ) -> tuple[list[list[str]], CallMetrics]:
        """Use the LLM to keep only candidate facts relevant to the query
        (recognition memory)."""
        metrics = CallMetrics()
        facts_json = json.dumps([list(f) for f in cand_facts])
        prompt = (
            self._p_filter.replace("{question}", question)
            .replace("{facts}", facts_json)
            .replace("{top_k}", str(self.linking_top_k))
        )
        # structured first.
        try:
            parsed, rec = self.llm.generate_structured(
                prompt, FactFilterResult)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, FactFilterResult):
                return [f for f in parsed.fact if len(f) == 3], metrics
        except Exception:
            pass
        # free-form fallback.
        gen = self.llm.generate(prompt)
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        return _parse_filter_json(gen.text), metrics

    # build PPR seed
    def _build_seed(
        self,
        q_vec: np.ndarray,
        passage_keys: list[str],
        top_facts: list[tuple[str, str, str]],
        top_fact_scores: list[float],
        current_session_idx: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Port of the official repo graph_search_with_fact_entities
        seed-weight logic.

        Official repo (HippoRAG.py graph_search_with_fact_entities):
          for each fact, for each phrase in (subject, object):
            weighted_fact_score = (
                fact_score / len(passages containing the phrase))
            phrase_weights[phrase] += weighted_fact_score
            number_of_occurs[phrase] += 1
          phrase_weights /= number_of_occurs   # element-wise average

        - Each fact adds *its own dense score* (not averaged away - key to
          fidelity).
        - passage weight: query-passage dense score (min-max) x
          passage_node_weight.
        Returns: (node_key -> seed_weight dict, phrase_text -> linking_score
        dict)
        """
        # per-phrase accumulation: (score sum, occurrence count). Official
        # repo's phrase_weights / number_of_occurs.
        phrase_score_sum: dict[str, float] = {}
        phrase_count: dict[str, int] = {}
        cur = current_session_idx

        for f, fscore in zip(top_facts, top_fact_scores):
            s_phrase, _, o_phrase = f[0], f[1], f[2]
            for phrase in (s_phrase, o_phrase):
                key = _phrase_key(phrase)
                if key not in self._store.phrase_text:
                    continue
                # official repo: weighted_fact_score = fact_score / (#passages
                # of that phrase)
                # [future-leakage] count passages over *past passages* only.
                # Using the full
                # (future-inclusive) phrase_to_passages would let future
                # passage frequency
                # enter the divisor, leaking future info into the seed weight
                # magnitude
                # (graph topology is blocked, but seed magnitude would still
                # leak).
                past_chunks = sum(
                    1 for pk in self._store.phrase_to_passages.get(key, ())
                    if self._store.passage_sess_idx.get(pk, 1 << 30) < cur
                )
                weighted = fscore / past_chunks if past_chunks > 0 else fscore
                phrase_score_sum[key] = (
                    phrase_score_sum.get(key, 0.0) + weighted)
                phrase_count[key] = phrase_count.get(key, 0) + 1

        # phrase weight = score sum / occurrence count (official repo
        # phrase_weights /= number_of_occurs).
        phrase_weights: dict[str, float] = {}
        linking_info: dict[str, float] = {}
        for key, total in phrase_score_sum.items():
            w = total / max(phrase_count[key], 1)
            phrase_weights[key] = w
            linking_info[self._store.phrase_text[key]] = w

        # link_top_k cut - keep only the top-scoring phrases as seed (official
        # repo get_top_k_weights).
        if self.linking_top_k and len(phrase_weights) > self.linking_top_k:
            top_keys = sorted(
                phrase_weights, key=lambda k: phrase_weights[k], reverse=True
            )[: self.linking_top_k]
            phrase_weights = {k: phrase_weights[k] for k in top_keys}

        # passage seed = dense score x passage_node_weight.
        passage_weights: dict[str, float] = {}
        raw = np.array(
            [cos_sim(q_vec, self._store.passage_emb[k]) for k in passage_keys]
        )
        dpr = _min_max_normalize(raw)
        for k, sc in zip(passage_keys, dpr):
            passage_weights[k] = float(sc) * self.passage_node_weight

        # combine both weights into the seed dict.
        seed: dict[str, float] = {}
        for k, w in phrase_weights.items():
            seed[k] = seed.get(k, 0.0) + max(w, 0.0)
        for k, w in passage_weights.items():
            seed[k] = seed.get(k, 0.0) + max(w, 0.0)
        return seed, linking_info

    # run PPR
    def _run_ppr(
        self, seed: dict[str, float], passage_keys: list[str],
        current_session_idx: int,
    ) -> list[tuple[str, float]]:
        """networkx personalized PageRank -> passage node scores in descending
        order.

        - seed: node_key -> reset probability (networkx normalizes internally
          even if unnormalized).
        - damping(alpha): same meaning as the official repo damping=0.5
          (teleport=1-alpha).
        Returns: [(passage_key, score), ...] descending.

        [fully block future-structure leakage] PPR propagates mass along the
        graph *topology*, so filtering the seed alone is not enough. Therefore,
        for each query, the graph is *fully reconstructed* to the
        current_session_idx point-in-time via provenance, then PPR runs:
          - phrase nodes: only those with phrase_first_sess < current (excludes
            future-only phrases).
          - passage nodes: only those with sess_idx < current (= passage_keys).
          - fact edge weight: sum only fact_edge_sessions counts with sess <
            current (removes future co-occurrence, blocking residual leakage
            that passage-node removal alone missed).
          - passage->phrase edges: only phrases of past passages (weight 1.0).
          - synonym edges: only when both phrases exist in the past (sim is
            time-invariant).
        The merged self._store.graph is kept only for commit
        accumulation/snapshots; PPR runs only on this point-in-time-accurate
        subgraph.
        """
        cur = current_session_idx
        # sets of past phrase / passage nodes.
        past_phrases = {
            k for k, fs in self._store.phrase_first_sess.items() if fs < cur
        }
        past_passages = [
            k for k in self._store.passage_sess_idx
            if self._store.passage_sess_idx[k] < cur
        ]
        if not past_phrases and not past_passages:
            return []

        g = nx.Graph()
        g.add_nodes_from(past_phrases)
        g.add_nodes_from(past_passages)
        # fact edge: restore weight by summing only past-session counts.
        for (a, b), sess_counts in self._store.fact_edge_sessions.items():
            if a in past_phrases and b in past_phrases:
                w = sum(c for s, c in sess_counts.items() if s < cur)
                if w > 0:
                    g.add_edge(a, b, weight=w)
        # passage->phrase edge (phrases of past passages).
        for pk in past_passages:
            for ph in self._store.passage_to_phrases.get(pk, ()):
                if ph in past_phrases:
                    g.add_edge(pk, ph, weight=1.0)
        # synonym edge (both phrases exist in the past). Do not overwrite
        # existing fact/passage edges.
        for (a, b), sim in self._store.synonym_weight.items():
            if a in past_phrases and b in past_phrases:
                if not g.has_edge(a, b):
                    g.add_edge(a, b, weight=sim)

        # if the seed sum is 0, PPR is impossible -> return empty (caller falls
        # back).
        total = sum(v for v in seed.values() if v > 0)
        if total <= 0 or g.number_of_nodes() == 0:
            return []
        # sparse personalization - positive seed items that exist in the
        # (point-in-time) graph.
        personalization = {
            k: v for k, v in seed.items() if v > 0 and g.has_node(k)}
        if not personalization:
            return []
        try:
            pr = nx.pagerank(
                g,
                alpha=self.damping,
                personalization=personalization,
                weight="weight",
                max_iter=200,
                tol=1.0e-6,
            )
        except Exception:
            # Absorb all PPR exceptions (not just
            # PowerIterationFailedConvergence) as a safety net -
            # prevents an uncaught exception from killing all of retrieve and
            # zeroing that question's result.
            pr = _manual_ppr(g, personalization, self.damping)
        # tie-break by passage key (determinism - independent of dict order on
        # tied PPR scores).
        scored = [(k, pr.get(k, 0.0)) for k in passage_keys]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    # dense passage fallback
    def _dense_passage_fallback(
        self,
        question: Question,
        q_vec: np.ndarray,
        passage_keys: list[str],
        metrics: CallMetrics,
        ppr_used: bool,
    ) -> RetrievalResult:
        """Pure dense passage retrieval when there are no facts (the paper's
        fallback path)."""
        scored = [
            (k, cos_sim(q_vec, self._store.passage_emb[k]))
            for k in passage_keys
        ]
        # tie-break by passage key (same policy as _run_ppr - determinism on
        # tied scores).
        scored.sort(key=lambda x: (-x[1], x[0]))
        scored = scored[: self.num_retrieved]
        items, rendered = self._render_passages(scored)
        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant history!",
            extra={"route": "HIPPORAG2_DPR", "ppr_used": ppr_used},
            metrics=metrics,
        )

    # render passages
    def _render_passages(
        self, ranked: list[tuple[str, float]]
    ) -> tuple[list[Item], str]:
        """Ranked passages -> Item list + a rendered string filled up to
        context_size."""
        items: list[Item] = []
        rendered = ""
        approx = self._p_qa_tokens
        for k, score in ranked:
            text = self._store.passage_text[k]
            # if passage_date is None, the string "None" would leak into the
            # prompt, so guard with "".
            # .get(k, "") returns None if the key exists with a None value ->
            # `or ""` fixes it.
            date = self._store.passage_date.get(k) or ""
            sidx = self._store.passage_sess_idx.get(k, -1)
            # include a timestamp header before each hippo passage
            # ([Date: <date> (Session <idx>)]). The hipporag2_qa.txt EXAMPLE
            # gives each
            # History a timestamp, so the actual rendering matches it. Same
            # format as GroupChat-Mem.
            chunk = _format_hippo_chunk(text, date, sidx)
            tk = count_tokens(chunk)
            if self.context_size > 0 and approx + tk > self.context_size:
                # If the first (top-ranked) passage alone exceeds the limit, a
                # plain break would
                # leave rendered="" -> the caller's `or "No relevant history!"`
                # misroutes, so a
                # successful retrieval answers with an empty context (realistic
                # since the
                # entire_session is long). Always include at least one, trimmed
                # to the remaining budget.
                if not items:
                    budget = self.context_size - approx
                    if budget > 0:
                        chunk = trim_text_to_tokens(chunk, budget)
                        rendered += chunk
                        items.append(
                            Item(
                                text=text,
                                meta={"date": date, "session_idx": sidx},
                                score=score,
                            )
                        )
                break
            rendered += chunk
            items.append(
                Item(
                    text=text,
                    meta={"date": date, "session_idx": sidx},
                    score=score,
                )
            )
            approx += tk
        return items, rendered

    # build_qa_prompt
    def build_qa_prompt(
        self, question: Question, retrieval: RetrievalResult
    ) -> str:
        q_str = render_question(question, self.question_format)
        evidence = retrieval.rendered or "No relevant history!"
        prompt = self._fill_qa_template(self._p_qa, q_str, evidence)
        return apply_token_limit(prompt, self.context_size)

    # answer - retrieve -> build prompt -> generate (same contract as
    # GroupChat-Mem)
    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        # current_session is accepted but ignored - the HippoRAG2 baseline uses
        # only committed past sessions.
        result = self.retrieve(question, current_session_idx)
        prompt = self.build_qa_prompt(question, result)
        gen = self.llm.generate(prompt)
        metrics = result.metrics.merge(gen.to_metrics())
        # preserve the final QA generation's raw + reasoning + final prompt +
        # retrieval rendering
        # - used by run_baseline's record (same keys as GroupChat-Mem).
        result.extra["qa_raw_text"] = gen.text
        result.extra["qa_reasoning_text"] = getattr(gen, "reasoning_text", "")
        result.extra["qa_final_prompt"] = prompt
        result.extra["qa_retrieval_rendered"] = result.rendered
        return gen.text, result, metrics

    # snapshot / restore
    def snapshot(self, path: str) -> None:
        s = self._store
        data = {
            "phrase_text": s.phrase_text,
            "phrase_emb": {k: v.tolist() for k, v in s.phrase_emb.items()},
            "passage_text": s.passage_text,
            "passage_date": s.passage_date,
            "passage_sess_idx": s.passage_sess_idx,
            "passage_emb": {k: v.tolist() for k, v in s.passage_emb.items()},
            "facts": s.facts,
            "fact_emb": [v.tolist() for v in s.fact_emb],
            "fact_sess_idx": s.fact_sess_idx,
            "phrase_to_passages": {
                k: list(v) for k, v in s.phrase_to_passages.items()
            },
            "synonym_done": [list(p) for p in s._synonym_done],
            # provenance (for PPR point-in-time reconstruction).
            "fact_edge_sessions": [
                [list(pair), dict(sc)]
                for pair, sc in s.fact_edge_sessions.items()
            ],
            "phrase_first_sess": dict(s.phrase_first_sess),
            "passage_to_phrases": {
                k: list(v) for k, v in s.passage_to_phrases.items()
            },
            "synonym_weight": [
                [list(pair), float(w)]
                for pair, w in s.synonym_weight.items()
            ],
            # serialize the networkx graph as an edge list (u, v, weight).
            "edges": [
                (u, v, d.get("weight", 1.0))
                for u, v, d in s.graph.edges(data=True)
            ],
            "nodes": list(s.graph.nodes()),
        }
        save_pickle(data, path)

    def restore(self, path: str) -> None:
        d = load_pickle(path)
        s = HippoGraphStore()
        s.phrase_text = d.get("phrase_text", {})
        s.phrase_emb = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in d.get("phrase_emb", {}).items()
        }
        s.passage_text = d.get("passage_text", {})
        s.passage_date = d.get("passage_date", {})
        s.passage_sess_idx = d.get("passage_sess_idx", {})
        s.passage_emb = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in d.get("passage_emb", {}).items()
        }
        s.facts = [tuple(f) for f in d.get("facts", [])]
        s.fact_emb = [
            np.asarray(v, dtype=np.float32) for v in d.get("fact_emb", [])
        ]
        s.fact_sess_idx = list(d.get("fact_sess_idx", []))
        s.phrase_to_passages = {
            k: set(v) for k, v in d.get("phrase_to_passages", {}).items()
        }
        s._synonym_done = {tuple(p) for p in d.get("synonym_done", [])}
        # restore provenance (empty dict if absent in older snapshots - then
        # _run_ppr gets an
        # empty graph and safely degrades to DPR fallback; filled on
        # re-snapshot).
        s.fact_edge_sessions = {
            tuple(pair): {int(k): float(v) for k, v in sc.items()}
            for pair, sc in d.get("fact_edge_sessions", [])
        }
        s.phrase_first_sess = {
            k: int(v) for k, v in d.get("phrase_first_sess", {}).items()
        }
        s.passage_to_phrases = {
            k: set(v) for k, v in d.get("passage_to_phrases", {}).items()
        }
        s.synonym_weight = {
            tuple(pair): float(w) for pair, w in d.get("synonym_weight", [])
        }
        g = nx.Graph()
        g.add_nodes_from(d.get("nodes", []))
        for u, v, w in d.get("edges", []):
            g.add_edge(u, v, weight=w)
        s.graph = g

        # Guard against corrupted/partial snapshots.
        # (a) facts/fact_emb/fact_sess_idx must stay index-aligned (retrieve
        # dense scoring
        # depends on it). A length mismatch would silently drop trailing facts,
        # so block explicitly.
        nf = len(s.facts)
        if not (len(s.fact_emb) == nf and len(s.fact_sess_idx) == nf):
            raise RuntimeError(
                f"HippoRAG2 snapshot corrupted: facts({nf})/"
                f"fact_emb({len(s.fact_emb)})/"
                f"fact_sess_idx({len(s.fact_sess_idx)}) length mismatch - "
                f"alignment invariant violated"
            )
        # (b) if facts exist but provenance
        # (phrase_first_sess/fact_edge_sessions) is empty,
        # _run_ppr gets an empty point-in-time graph -> every question silently
        # degrades to
        # DPR (not a leak, but PPR is disabled). This signals an old/partial
        # snapshot; warn.
        if nf > 0 and (not s.phrase_first_sess or not s.fact_edge_sessions):
            print(
                "[HippoRAG2.restore] WARNING: facts exist but provenance"
                "(phrase_first_sess/fact_edge_sessions) is empty, so PPR "
                "degrades to DPR. "
                "Re-snapshotting with the latest code is recommended."
            )
        self._store = s


# Module-level helpers
def _parse_openie_json(text: str) -> tuple[list[str], list[list[str]]]:
    """Robustly parse named_entities + triples from a free-form LLM
    response."""
    entities: list[str] = []
    triples: list[list[str]] = []
    # extract the JSON object candidate (outermost {...}).
    obj = _extract_json_object(text)
    if obj is not None:
        ents = obj.get("named_entities") or obj.get("entities") or []
        if isinstance(ents, list):
            entities = [str(e) for e in ents]
        tris = obj.get("triples") or []
        if isinstance(tris, list):
            for t in tris:
                if isinstance(t, list) and len(t) == 3:
                    triples.append([str(x) for x in t])
    return entities, triples


def _parse_ner_json(text: str) -> list[str]:
    """Parse the named_entities list from a free-form NER response (official
    2-call OpenIE stage 1)."""
    obj = _extract_json_object(text)
    if obj is None:
        return []
    ents = obj.get("named_entities") or obj.get("entities") or []
    return [str(e) for e in ents] if isinstance(ents, list) else []


def _parse_triple_json(text: str) -> list[list[str]]:
    """Parse [s,p,o] 3-element triples from a free-form triple response
    (official 2-call OpenIE stage 2)."""
    obj = _extract_json_object(text)
    out: list[list[str]] = []
    if obj is None:
        return out
    tris = obj.get("triples") or []
    if isinstance(tris, list):
        for t in tris:
            if isinstance(t, list) and len(t) == 3:
                out.append([str(x) for x in t])
    return out


def _parse_filter_json(text: str) -> list[list[str]]:
    """Parse the fact list from a free-form filter response."""
    obj = _extract_json_object(text)
    out: list[list[str]] = []
    if obj is not None:
        facts = obj.get("fact") or obj.get("facts") or []
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, list) and len(f) == 3:
                    out.append([str(x) for x in f])
    return out


def _extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from text (tolerates code
    fences/noise)."""
    if not text:
        return None
    # strip a ```json ... ``` fence.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # greedy match of the outermost brace block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _manual_ppr(
    g: Any,
    personalization: dict[str, float],
    damping: float,
    max_iter: int = 200,
) -> dict[str, float]:
    """Custom power-iteration PPR for when networkx fails to converge (safety
    net).

    p_{t+1} = (1-alpha) * teleport + alpha * W^T p_t
    (alpha = damping, teleport = normalized personalization).
    """
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}
    # normalize the teleport distribution.
    tele = np.zeros(n)
    for k, v in personalization.items():
        if k in idx and v > 0:
            tele[idx[k]] = v
    s = tele.sum()
    if s <= 0:
        return {nd: 0.0 for nd in nodes}
    tele = tele / s
    # weighted transition matrix (column-normalized).
    p = tele.copy()
    # precompute the (weighted) adjacency list.
    out_w = np.zeros(n)
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, d in g.edges(data=True):
        if u == v:
            # exclude self-loops defensively (they double-count out_w/neigh and
            # distort PPR).
            continue
        w = float(d.get("weight", 1.0))
        ui, vi = idx[u], idx[v]
        neigh[ui].append((vi, w))
        neigh[vi].append((ui, w))  # undirected.
        out_w[ui] += w
        out_w[vi] += w
    for _ in range(max_iter):
        new_p = (1.0 - damping) * tele
        for ui in range(n):
            if out_w[ui] == 0 or p[ui] == 0:
                # a dangling node's mass goes to teleport (simplification).
                if out_w[ui] == 0 and p[ui] > 0:
                    new_p += damping * p[ui] * tele
                continue
            share = damping * p[ui] / out_w[ui]
            for vi, w in neigh[ui]:
                new_p[vi] += share * w
        if np.abs(new_p - p).sum() < 1e-6:
            p = new_p
            break
        p = new_p
    return {nodes[i]: float(p[i]) for i in range(n)}
