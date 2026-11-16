"""Mem0 memory - fact extraction + ADD/UPDATE/DELETE/NOOP update phase.

Paper: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
       (arXiv 2504.19413, Chhikara et al., 2025)
Official repo: https://github.com/mem0ai/mem0
               (fact extraction = FACT_RETRIEVAL_PROMPT, update phase =
               DEFAULT_UPDATE_MEMORY_PROMPT)

Core behavior (the core "Mem0" of paper Section 2.1; the graph variant Mem0g is
excluded):

  commit(session) - write phase. Splits the session into chunks of
    turns_per_batch (default 5) utterances and repeats (1)+(2) per chunk (same
    batch unit as a_mem's turns_per_note - for fair comparison across
    baselines). A chunk = up to turns_per_batch consecutive utterances and
    never crosses a session boundary
    (e.g. 13 utterances, batch=5 -> [1-5][6-10][11-13], 3 chunks).
    Note: the official LoCoMo eval uses CHUNK_SIZE=1 (per-message), but this
       implementation uses the same turns_per_batch utterance batch as a_mem so
       the two baselines are compared on the same unit.
    For each chunk:
    (1) Extraction: extract salient facts from the chunk dialogue via LLM
        (fact_extraction prompt).
    (2) Update: use the chunk dialogue as a query to dense-similarity retrieve
        the top_s(=10) existing memories, then decide ADD / UPDATE / DELETE /
        NONE for those candidates plus the chunk's new facts in a single
        update-decision LLM call -> updates the memory store.
    Chunks are processed in order within a session, so a later chunk sees
    earlier chunks' store updates (accumulation).

  retrieve(question, idx) - read phase:
    Embed the question and take the top-k by cosine similarity against memories
    created in sessions *before* idx (future-leakage prevention). rendered =
    concatenated memory texts.

  build_qa_prompt - inject retrieved memory + question into the qa_mem0.txt
    template.
  snapshot / restore - pickle the memory store (text + embedding + metadata).

Differences from the paper/repo (group-chat adaptation):
  - Official mem0 assumes user/assistant 2-party dialogue. Our setting is
    multi-party group chat, so the fact extraction prompt is adapted to extract
    "facts about multiple speakers" (mem0_fact_extraction.txt).
  - Official mem0 receives updates via LLM tool-call (function calling); this
    repo uses pydantic + langchain `generate_structured`
    (with_structured_output) to receive the same JSON contract
    ({"memory": [{id, text, event, old_memory}]}).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

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
    load_pickle,
    load_prompt,
    render_question,
    save_pickle,
)

from groupchat_qa.utils.mem0_scoring import (
    get_bm25_params,
    lemmatize_for_bm25,
    normalize_bm25,
    score_and_rank,
)
from .base import Memory


# pydantic schema - structured LLM response
class FactList(BaseModel):
    """Fact extraction response schema. Same as official mem0's
    {"facts": [...]}."""
    facts: list[str] = Field(
        default_factory=list, description="extracted salient facts"
    )


class MemoryOp(BaseModel):
    """Response schema for a single memory item in the update-decision.

    event in {ADD, UPDATE, DELETE, NONE} - the 4 operations of official mem0's
    DEFAULT_UPDATE_MEMORY_PROMPT. (The paper calls it NOOP and the repo prompt
    writes NONE - same meaning.)
    """
    id: str = Field(
        default="",
        description="memory id (reused on update/delete, new on add)",
    )
    text: str = Field(default="", description="memory content")
    event: Literal["ADD", "UPDATE", "DELETE", "NONE"] = Field(
        default="NONE", description="operation to perform"
    )
    old_memory: str = Field(
        default="", description="previous memory content on UPDATE"
    )

    # Harden the structured path: when a weak model returns event in lowercase
    # ("add"),
    # as an alias ("NOOP"), or misspelled, pydantic Literal validation fails
    # wholesale and
    # the entire plan is dropped as None. Normalize case + map aliases
    # (NOOP/NO_CHANGE/empty) -> "NONE", and map unknown values -> "NONE" -
    # giving the
    # structured path the same leniency as the free-form fallback
    # (_parse_plan_json).
    @field_validator("event", mode="before")
    @classmethod
    def _coerce_event(cls, v: object) -> str:
        s = str(v).strip().upper()
        if s in ("ADD", "UPDATE", "DELETE", "NONE"):
            return s
        # NOOP / NO_CHANGE / NO CHANGE / empty etc. mean "no change" -> NONE.
        return "NONE"


class MemoryUpdatePlan(BaseModel):
    """Full update-decision response schema. {"memory": [MemoryOp, ...]}."""
    memory: list[MemoryOp] = Field(default_factory=list)


# Internal memory store
@dataclass
class MemoryRecord:
    """A single unit of the memory store - text + embedding + metadata."""
    id: str
    text: str
    embedding: np.ndarray
    # metadata: which session this memory was created in (essential for
    # future-leakage
    # prevention) + creation date.
    session_idx: int = 0
    date: str = ""


@dataclass
class Mem0ChunkExtract:
    """Phase-1 output for one chunk (up to turns_per_batch consecutive
    utterances) - chunk-independent LLM/embedding.

    facts          : salient facts extracted from this chunk's dialogue.
    fact_embeddings: embeddings index-aligned with facts.
    query_embedding: embedding of the whole chunk dialogue (candidate-search
                     query for the update phase). Corresponds to the official
                     parsed_messages embedding, but at chunk granularity rather
                     than session.
    """
    facts: list[str]
    fact_embeddings: list[np.ndarray]
    query_embedding: np.ndarray | None


@dataclass
class Mem0ExtractResult:
    """Phase-1 (extract) output of the 2-phase commit - a list of chunks (chunk
    order = dialogue order).

    Splits the session dialogue into chunks of turns_per_batch utterances and,
    for each chunk independently, extracts facts + embeddings (parallel-safe
    since it does not touch self._store). The update-decision (which references
    existing memory and depends on accumulation) runs sequentially in chunk
    order in phase 2 (commit_prepared).
    """
    chunks: list["Mem0ChunkExtract"]
    metrics: CallMetrics


def _group_score(scores: list[float], agg: str) -> float:
    """Aggregate a group's fact scores into a single sorting key (used when
    ``group_order='score'`` to order the session groups).

    agg in {"max","mean","median","min"}:
      - max    : the group's best-fact score (best-evidence group first).
      - mean   : the group's average relevance.
      - median : the group's median (robust to a single high outlier).
      - min    : the group's lowest score (conservative - every fact in the
                 group must be relevant to rank the group highly).
    Empty list -> 0.0. Unknown agg -> max fallback.
    """
    if not scores:
        return 0.0
    if agg == "mean":
        return float(sum(scores) / len(scores))
    if agg == "median":
        s = sorted(scores)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        return float((s[mid - 1] + s[mid]) / 2.0)
    if agg == "min":
        return float(min(scores))
    return float(max(scores))  # "max" (default / fallback)


def _format_grouped(
    recs_scores: list[tuple["MemoryRecord", float]],
    group_order: str,
    group_score_agg: str = "max",
) -> tuple[str, list[Item]]:
    """Render the retrieved (record, score) facts grouped by their source
    session, each group under a "[Date: {date} (Session {idx})]" header.

    Within a group: facts are ordered by score (best evidence first).
    Group placement:
      - "time"  : ascending session_idx (chronological).
      - "score" : descending group score (aggregated by ``group_score_agg``);
                  with agg='max' the best-evidence group is placed first,
                  mitigating the lost-in-the-middle effect.
    Returns (rendered_str, items); items are the score-descending flattened
    facts (same metadata keys as the flat renderer). dataset-agnostic (headers
    use only the date / integer session index).
    """
    groups: dict[tuple[int, str], list[tuple["MemoryRecord", float]]] = {}
    for rec, sc in recs_scores:
        key = (int(rec.session_idx), str(rec.date))
        groups.setdefault(key, []).append((rec, sc))

    for key in groups:  # within-group: score descending
        groups[key].sort(key=lambda rs: -rs[1])

    group_keys = list(groups.keys())
    if group_order == "time":
        group_keys.sort(key=lambda k: (k[0], k[1]))
    else:  # "score" (default) - ties broken by session_idx for determinism
        def _gscore(k: tuple[int, str]) -> float:
            return _group_score([sc for _, sc in groups[k]], group_score_agg)
        group_keys.sort(key=lambda k: (-_gscore(k), k[0]))

    lines: list[str] = []
    items: list[Item] = []
    for key in group_keys:
        idx, date = key
        if date:
            header = f"[Date: {date} (Session {idx})]"
        else:
            header = f"[Session {idx}]"
        lines.append(header)
        for rec, sc in groups[key]:
            lines.append(f"- {rec.text}")
            items.append(
                Item(
                    text=rec.text,
                    meta={"id": rec.id, "session_idx": rec.session_idx,
                          "date": rec.date},
                    score=float(sc),
                )
            )
        lines.append("")  # blank line between groups (readability)
    rendered = "\n".join(lines).strip()
    return rendered, items


# Mem0Memory
class Mem0Memory(Memory):
    """Mem0 baseline - config dict keys:

    top_s            : int - number of existing memories to compare per new
                             fact in the update phase (official: paper s=10).
    top_k            : int - number of memories to retrieve for QA (default
                             600; official github search=20). Max documents to
                             return after re-ranking.
    context_size     : int - final QA prompt token limit (default 0 =
                             off/unlimited; >0 = that value as upper bound).
    fact_render      : 'flat' (one '- fact' per line, score-descending) or
                       'grouped' (facts grouped by source session with date
                       headers). Default 'grouped'.
    group_order      : when fact_render='grouped', how the session groups are
                       ordered: 'score' (by group relevance, default) or
                       'time' (chronological session_idx).
    group_score_agg  : group-relevance aggregation for group_order='score':
                       'max' (default) / 'mean' / 'median' / 'min'.
    dedup            : bool - merge near-duplicate facts before rendering
                       (cosine >= dedup_tau). Default False.
    dedup_tau        : float - dedup cosine threshold. Default 0.95.
    prompt_fact_extraction : fact extraction prompt filename (default
                             mem0_fact_extraction.txt).
    prompt_update_decision : update-decision prompt filename (default
                             mem0_update_decision.txt).
    prompt_qa        : QA prompt filename (default qa_mem0.txt).
    question_format  : 'multi_structured' / 'multi_unstructured' /
                       'open_ended'.
    """

    NAME = "mem0"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        # memory store - id -> MemoryRecord
        self._store: dict[str, MemoryRecord] = {}
        # Counter for issuing new memory ids (monotonically increasing ->
        # avoids collisions).
        self._next_id = 0

        # Hyperparameters - following official defaults:
        # top_s: number of candidate existing memories to search at commit
        # time. Official
        #          main.py `_add_to_vector_store`'s
        # `self.vector_store.search(query=parsed_messages, top_k=10)` = 10
        #          (matches paper section 2.1 s=10).
        # top_k: number of memories to retrieve for QA. Official main.py
        # `search()` default 20.
        self.top_s = int(cfg.get("top_s", 10))
        # mem0 default top_k=600 (best from topk ablation). sweep configs keep
        # their own value.
        self.top_k = int(cfg.get("top_k", 600))
        # Official mem0 hybrid scoring (_search_vector_store) - semantic gate
        # threshold + BM25 lemmatize cache.
        # official _search_vector_store default 0.1
        self.semantic_threshold = float(cfg.get("semantic_threshold", 0.1))
        # memory id -> lemmatized text (BM25 corpus cache, computed once)
        self._lem_cache: dict[str, str] = {}
        # 0/negative = token limit off (unlimited)
        self.context_size = int(cfg.get("context_size", 0))
        self.question_format = cfg.get("question_format", "multi_structured")
        # Retrieval rendering: how the retrieved facts are laid out in the QA
        # prompt. 'grouped' groups the facts by their source session (date
        # headers) - the setting used in the paper; 'flat' is one '- fact' per
        # line in score order.
        self.fact_render = str(cfg.get("fact_render", "grouped")).lower()
        self.group_order = str(cfg.get("group_order", "score")).lower()
        self.group_score_agg = str(cfg.get("group_score_agg", "max")).lower()
        # Optional near-duplicate fact merging before rendering.
        self.dedup = bool(cfg.get("dedup", False))
        self.dedup_tau = float(cfg.get("dedup_tau", 0.95))
        # How many utterances to group into one chunk (= the unit of fact
        # extraction + update),
        # same concept as a_mem's turns_per_note. Default 5 to match a_mem so
        # the two
        # baselines are compared on the same batch unit. The session dialogue
        # is sliced into
        # turns_per_batch utterances and committed chunk by chunk (a later
        # chunk sees the
        #   earlier chunk's store updates).
        _tpb = cfg.get("turns_per_batch", 5)
        # null/unspecified -> 5
        self.turns_per_batch = 5 if _tpb is None else int(_tpb)
        if self.turns_per_batch < 1:
            raise ValueError(
                f"turns_per_batch must be >= 1, got {self.turns_per_batch}"
            )

        # Load prompts.
        self._p_fact = load_prompt(
            cfg.get("prompt_fact_extraction", "mem0_fact_extraction.txt")
        )
        self._p_update = load_prompt(
            cfg.get("prompt_update_decision", "mem0_update_decision.txt")
        )
        self._p_qa = load_prompt(cfg.get("prompt_qa", "qa_mem0.txt"))
        # The QA template token count is constant -> compute once in __init__
        # (avoids
        # re-tokenizing on every retrieve).
        self._p_qa_tokens = count_tokens(self._p_qa)

    # Internal helper - id issuance
    def _new_id(self) -> str:
        """Issue one new memory id - a monotonically increasing integer string
        that does not collide with existing ids."""
        # Guarantee a value always larger than any numeric id already in the
        # store
        # (safe even after restore).
        while str(self._next_id) in self._store:
            self._next_id += 1
        nid = str(self._next_id)
        self._next_id += 1
        return nid

    @staticmethod
    def _mem_hash(text: str) -> str:
        """Duplicate-detection hash - exact md5(text) hash, same as official
        mem0.

        Official main.py `_add_to_vector_store`:
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes: continue
        That is, only *exactly identical text* is skipped as a duplicate (no
        case/whitespace normalization). We use exact md5 rather than normalized
        dedup to faithfully reproduce the official behavior.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # commit - (1) fact extraction + (2) update phase
    def commit(self, session: Session) -> CallMetrics:
        """Single-call path (no 2-phase) - extract + commit_prepared in
        sequence (same result)."""
        ex = self.extract(session)
        return self.commit_prepared(session, ex)

    # phase 1: session-independent LLM/embedding (ThreadPool-parallel safe,
    # store immutable)
    def extract(self, session: "Session") -> "Mem0ExtractResult | None":
        """Split the session dialogue into chunks of turns_per_batch utterances
        and, per chunk, extract facts + embeddings (update-decision/apply are
        excluded - phase 2).

        Never reads or writes self._store - pure computation from the session
        input alone (parallel-safe). A chunk is a group of consecutive
        utterances and never crosses a session boundary (_split_turns splits
        per line = per utterance). Chunks with 0 facts are excluded from the
        result since update is meaningless (empty chunk skipped).
        """
        if not session.dialogue:
            return None  # empty session -> commit_prepared is a no-op.
        metrics = CallMetrics()
        raw_turns = _split_turns(session.dialogue)
        chunks: list[Mem0ChunkExtract] = []
        for i in range(0, len(raw_turns), self.turns_per_batch):
            chunk_turns = raw_turns[i:i + self.turns_per_batch]
            chunk_text = "\n".join(chunk_turns)
            facts = self._extract_facts_from_text(
                chunk_text, session.date, metrics
            )
            if not facts:
                # no facts extracted from this chunk -> no update (no
                # embedding/query needed).
                continue
            fact_embeddings = [self._embed(f, metrics) for f in facts]
            # candidate-search query = embedding of the whole chunk dialogue
            # (corresponds to
            # the official parsed_messages, at chunk granularity). Independent
            # of the store,
            #   so precomputed in the parallel phase 1.
            query_embedding = self._embed(chunk_text, metrics)
            chunks.append(Mem0ChunkExtract(
                facts=facts, fact_embeddings=fact_embeddings,
                query_embedding=query_embedding,
            ))
        # Return an empty result even if chunks is empty (all chunks had 0
        # facts) -
        # commit_prepared is a no-op.
        return Mem0ExtractResult(chunks=chunks, metrics=metrics)

    # phase 2: accumulating build (must be called sequentially in ascending
    # session idx)
    def commit_prepared(
        self, session: "Session", extract_result: "Mem0ExtractResult | None"
    ) -> CallMetrics:
        if extract_result is None:
            return CallMetrics()  # empty session.
        # inherit extract's fact-extraction + embed metrics.
        metrics = extract_result.metrics

        # (2) Update phase: process chunks in dialogue order, sequentially
        # Per chunk: 1 cycle of (candidate search -> update-decision -> apply).
        # - candidate-search query = that chunk's dialogue embedding (official
        # parsed_messages -> chunk).
        # - reads and writes self._store (accumulating), so chunks must be
        # strictly sequential.
        # - a later chunk's update sees the earlier chunk's store updates
        # (ADD/UPDATE/DELETE results).
        for ch in extract_result.chunks:
            if not ch.facts:
                # defensive - extract does not add empty chunks, but skip
                # safely.
                continue
            candidate_ids = self._similar_memory_ids(
                ch.query_embedding, self.top_s
            )
            old_memory = [
                {"id": rid, "text": self._store[rid].text}
                for rid in candidate_ids
            ]

            # update-decision LLM call -> ADD/UPDATE/DELETE/NONE plan
            plan = self._decide_updates(old_memory, ch.facts, metrics)

            # Apply plan - a fact embedding is recomputed if the text changes.
            # Build a fact->emb dict to map a plan's text back to the embedding
            # of some new fact.
            fact_to_emb = {f: e for f, e in zip(ch.facts, ch.fact_embeddings)}
            self._apply_plan(plan, fact_to_emb, session, metrics)
        return metrics

    # Internal - fact extraction
    def _extract_facts_from_text(
        self, dialogue_text: str, date: str, metrics: CallMetrics
    ) -> list[str]:
        """Extract salient facts from the given dialogue (chunk) text via the
        fact_extraction prompt.

        Tries structured (pydantic) first -> falls back to free-form JSON
        parsing on failure. Receives chunk text, not the whole session (extract
        calls this per chunk).
        """
        prompt = self._p_fact.replace("{date}", date or "").replace(
            "{dialogue}", dialogue_text
        )
        # Structured attempt - if parsed is not None (even an empty list),
        # treat as "parse success".
        # An empty fact list ({"facts": []}) is a valid answer meaning "no
        # facts to extract".
        # Treating it as failure and also calling the free-form fallback would
        # double the LLM
        # calls and double-accumulate metrics. If parsed is not None, trust
        # that result as-is.
        try:
            parsed, rec = self.llm.generate_structured(prompt, FactList)
            self._accumulate(metrics, rec)
            if isinstance(parsed, FactList):
                return [str(x).strip() for x in parsed.facts if str(x).strip()]
        except Exception:
            pass
        # Free-form fallback - reached only when structured failed
        # (parsed=None/exception).
        gen = self.llm.generate(prompt)
        self._accumulate_gen(metrics, gen)
        return _parse_facts_json(gen.text)

    # Internal - update-decision
    def _decide_updates(
        self, old_memory: list[dict], new_facts: list[str],
        metrics: CallMetrics,
    ) -> list[MemoryOp]:
        """Decide the ADD/UPDATE/DELETE/NONE plan via the update_decision
        prompt.

        old_memory : [{"id","text"}, ...]  (top-s similar memory union)
        new_facts  : [str, ...]            (new facts from this session)
        returns    : list of MemoryOp.
        """
        old_str = json.dumps(old_memory, ensure_ascii=False)
        new_str = json.dumps(new_facts, ensure_ascii=False)
        prompt = self._p_update.replace(
            "{retrieved_old_memory}", old_str
        ).replace("{new_facts}", new_str)
        # Structured attempt - if parsed is not None (even an empty plan),
        # treat as "parse success".
        # An empty plan ({"memory": []}) can be a valid answer meaning "no
        # change".
        # Treating it as failure and also calling the free-form fallback would
        # double the LLM
        # calls and double-accumulate metrics. If parsed is not None, trust it
        # as-is.
        try:
            parsed, rec = self.llm.generate_structured(
                prompt, MemoryUpdatePlan
            )
            self._accumulate(metrics, rec)
            if isinstance(parsed, MemoryUpdatePlan):
                return list(parsed.memory)
        except Exception:
            pass
        # Free-form fallback - reached only when structured failed
        # (parsed=None/exception).
        gen = self.llm.generate(prompt)
        self._accumulate_gen(metrics, gen)
        return _parse_plan_json(gen.text)

    # Internal - apply plan (ADD / UPDATE / DELETE / NONE)
    def _apply_plan(
        self,
        plan: list[MemoryOp],
        fact_to_emb: dict[str, np.ndarray],
        session: Session,
        metrics: CallMetrics,
    ) -> None:
        """Apply the update-decision plan to the memory store."""
        # Ignore the id given by the LLM and issue a new integer id internally
        # (no uuid_mapping
        # needed - this implementation's ids are already small integers).
        # Intentionally differs
        # from the official repo's id re-indexing.
        #
        # Official mem0 skips exactly-duplicate memories via an md5(text) hash
        # before insert.
        # This implementation dedups the same way with an exact md5 hash, but
        # queries the *live
        # store* on every ADD rather than a snapshot. (If DELETE -> ADD of the
        # same text happens
        # within one plan, a snapshot set would not reflect the DELETE and
        # would wrongly dedup.)
        def _is_dup(t: str) -> bool:
            th = self._mem_hash(t)
            return any(
                self._mem_hash(r.text) == th
                for r in self._store.values()
            )

        def _emb_for(op_text: str, stored_text: str) -> np.ndarray:
            # Compute/look up the embedding based on the *stripped text that
            # will be stored*.
            # fact_to_emb is keyed by the original extracted fact, so op.text
            # as-is hits; if
            # the LLM rephrases or changes whitespace it misses -> re-embed
            # stored_text
            # (stripped) so that "stored text and its embedding" come from the
            # same string.
            vec = fact_to_emb.get(op_text)
            if vec is None:
                vec = self._embed(stored_text, metrics)
            return vec

        # Two-pass application: process all DELETEs first, then UPDATE/ADD.
        # With a single pass, when (ADD "x") precedes (DELETE old_x) within one
        # plan, the ADD
        # sees the still-alive old_x, dedups with _is_dup=True and skips ->
        # then the DELETE
        # removes old_x and the fact is lost entirely. Processing DELETEs first
        # removes this
        #   order-dependent data loss.
        for op in plan:
            if op.event.upper() == "DELETE":
                # Delete the existing memory (only if the id is in the store).
                self._store.pop(op.id, None)

        for op in plan:
            event = op.event.upper()
            text = (op.text or "").strip()
            if event in ("NONE", "DELETE"):
                # NONE = no change. DELETE = already handled in the 1st pass
                # above.
                continue
            if event == "UPDATE":
                # If text is empty, UPDATE would overwrite an existing memory
                # with blanks (noise) -> skip.
                if not text:
                    continue
                # Update the existing memory text -> recompute the embedding
                # (required since the text changed).
                vec = _emb_for(op.text, text)
                if op.id in self._store:
                    rec = self._store[op.id]
                    rec.text = text
                    rec.embedding = vec
                    # Future-leakage prevention: UPDATE merges a
                    # *current-session* new fact into
                    # an existing memory, so its information availability must
                    # be re-stamped to
                    # the current session. Keeping the old session_idx would
                    # let the updated
                    # text (containing future info) be retrieved by
                    # intermediate-session
                    #   questions, causing future leakage.
                    rec.session_idx = session.idx
                    rec.date = session.date
                else:
                    # If the LLM said UPDATE but the id is not in the store
                    # (hallucination/restore
                    # drift), safely treat it as ADD to avoid information loss
                    # - but skip if duplicate.
                    if _is_dup(text):
                        continue
                    nid = self._new_id()
                    self._store[nid] = MemoryRecord(
                        id=nid, text=text, embedding=vec,
                        session_idx=session.idx, date=session.date,
                    )
                continue
            if event == "ADD":
                # If text is empty there is nothing to add -> skip.
                if not text:
                    continue
                # If the same content is already in the store, skip the
                # duplicate insert (matches official mem0 md5 dedup).
                if _is_dup(text):
                    continue
                # Add a new memory. Ignore the LLM-given id and issue a new one
                # internally (avoids collisions).
                vec = _emb_for(op.text, text)
                nid = self._new_id()
                self._store[nid] = MemoryRecord(
                    id=nid, text=text, embedding=vec,
                    session_idx=session.idx, date=session.date,
                )
                continue
            # Unknown event -> ignore (safe).

    # retrieve - dense similarity, future-leakage prevention
    def retrieve(
        self, question: Question, current_session_idx: int
    ) -> RetrievalResult:
        metrics = CallMetrics()
        # Future-leakage prevention: only memories created in sessions before
        # current_session_idx.
        candidates = [
            r for r in self._store.values()
            if r.session_idx < current_session_idx
        ]
        if not candidates:
            return RetrievalResult(
                items=[], rendered="No relevant memory!",
                extra={"route": "MEM0"}, metrics=metrics,
            )

        # Question embedding.
        q_vec = self._embed(question.text, metrics)

        # Official mem0 hybrid scoring (mem0/utils/scoring.py score_and_rank):
        # additive
        # combination of semantic (cosine) + BM25 + entity_boost, then top_k.
        # Matches the
        # official _search_vector_store (threshold 0.1) 1:1 - but we do not use
        # graph memory,
        #   so entity_boosts={} (has_entity=False -> max_possible=2.0).
        semantic_results = [
            {"id": str(r.id), "score": cos_sim(q_vec, r.embedding),
             "payload": r}
            for r in candidates
        ]
        bm25_scores = self._compute_bm25_scores(question.text, candidates)
        # When dedup is on, over-fetch (x3) so that after merging near-dups we
        # can still fill top_k.
        rank_top_k = self.top_k * 3 if self.dedup else self.top_k
        ranked = score_and_rank(
            # entity_boosts={} - graph unused
            semantic_results, bm25_scores, {},
            threshold=self.semantic_threshold, top_k=rank_top_k,
        )

        id_to_rec = {str(r.id): r for r in candidates}

        # Near-duplicate fact merging: walk in ranked order and skip a fact
        # whose embedding cosine >= dedup_tau against an already-kept fact.
        if self.dedup:
            kept: list[tuple[str, float]] = []
            kept_vecs: list[np.ndarray] = []
            for sr in ranked:
                rid = sr["id"]
                vec = id_to_rec[rid].embedding
                if any(cos_sim(vec, kv) >= self.dedup_tau for kv in kept_vecs):
                    continue
                kept.append((rid, float(sr["score"])))
                kept_vecs.append(vec)
                if len(kept) >= self.top_k:
                    break
            ranked = [{"id": rid, "score": sc} for rid, sc in kept]

        recs_scores = [
            (id_to_rec[sr["id"]], float(sr["score"])) for sr in ranked
        ]

        # Render. 'grouped' lays the facts out grouped by source session (the
        # paper setting). The grouped renderer is unbudgeted (context_size=0 is
        # the default), so fall back to the flat budgeted renderer whenever a
        # token budget is set.
        if self.fact_render == "grouped" and self.context_size <= 0:
            rendered, items = _format_grouped(
                recs_scores, self.group_order, self.group_score_agg
            )
        else:
            rendered, items = self._render_flat_with_budget(recs_scores)

        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant memory!",
            extra={"route": "MEM0"},
            metrics=metrics,
        )

    def _render_flat_with_budget(
        self, recs_scores: list[tuple["MemoryRecord", float]]
    ) -> tuple[str, list[Item]]:
        """Flat renderer: one '- fact' per line in score order, truncated at
        the context_size token budget (0 = unlimited)."""
        items: list[Item] = []
        rendered = ""
        approx_tokens = self._p_qa_tokens
        for rec, sc in recs_scores:
            line = f"- {rec.text}\n"
            tk = count_tokens(line)
            if (self.context_size > 0
                    and approx_tokens + tk > self.context_size):
                break
            rendered += line
            items.append(
                Item(
                    text=rec.text,
                    meta={"id": rec.id, "session_idx": rec.session_idx,
                          "date": rec.date},
                    score=float(sc),
                )
            )
            approx_tokens += tk
        return rendered.strip(), items

    def _compute_bm25_scores(
        self, query: str, candidates: list
    ) -> dict[str, float]:
        """BM25 stage of the official _search_vector_store - normalized BM25
        scores of candidate memories.

        The official code uses vector_store.keyword_search's raw_score, but we
        have no such store, so we compute raw directly via BM25Okapi(lemmatized
        corpus) (same BM25 raw concept). lemmatize is lemmatize_for_bm25(spaCy)
        - same as official. Corpus lemmatize is cached once per memory id.
        Returns: {memory_id(str): normalized_bm25} (only raw>0 - same as
        official).
        """
        if not candidates:
            return {}
        from rank_bm25 import BM25Okapi

        # Lemmatize the corpus (candidate memory text) - cached per id (avoids
        # recompute per retrieve).
        corpus_tokens: list[list[str]] = []
        for r in candidates:
            rid = str(r.id)
            lem = self._lem_cache.get(rid)
            if lem is None:
                lem = lemmatize_for_bm25(r.text)
                self._lem_cache[rid] = lem
            corpus_tokens.append(lem.split())

        query_lemmatized = lemmatize_for_bm25(query)
        query_tokens = query_lemmatized.split()
        # Empty query or empty corpus (all tokens stop/punct) -> BM25 has no
        # signal -> {} (semantic-only).
        if not query_tokens or not any(corpus_tokens):
            return {}

        bm25 = BM25Okapi(corpus_tokens)
        raw_scores = bm25.get_scores(query_tokens)
        midpoint, steepness = get_bm25_params(
            query, lemmatized=query_lemmatized
        )

        bm25_scores: dict[str, float] = {}
        for r, raw in zip(candidates, raw_scores):
            # Same as official: normalize and include only raw>0 (0/negative =
            # no keyword signal).
            if raw and raw > 0:
                bm25_scores[str(r.id)] = normalize_bm25(
                    float(raw), midpoint, steepness
                )
        return bm25_scores

    # build_qa_prompt
    def build_qa_prompt(
        self, question: Question, retrieval: RetrievalResult
    ) -> str:
        q_str = render_question(question, self.question_format)
        evidence = retrieval.rendered or "No relevant memory!"
        prompt = self._fill_qa_template(self._p_qa, q_str, evidence)
        # Apply the context_size token limit (<=0 = off/unlimited - same policy
        # as GroupChat-Mem).
        return apply_token_limit(prompt, self.context_size)

    # answer - retrieve -> build prompt -> generate (same contract as
    # GroupChat-Mem)
    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        # current_session is accepted but ignored - the Mem0 baseline uses only
        # committed past-session memory.
        result = self.retrieve(question, current_session_idx)
        prompt = self.build_qa_prompt(question, result)
        gen = self.llm.generate(prompt)
        metrics = result.metrics.merge(gen.to_metrics())
        # Preserve the final QA generation's raw + reasoning + final prompt +
        # retrieval rendering
        # - used by run_baseline's record (same keys as GroupChat-Mem).
        result.extra["qa_raw_text"] = gen.text
        result.extra["qa_reasoning_text"] = getattr(gen, "reasoning_text", "")
        result.extra["qa_final_prompt"] = prompt
        result.extra["qa_retrieval_rendered"] = result.rendered
        return gen.text, result, metrics

    # snapshot / restore
    def snapshot(self, path: str) -> None:
        save_pickle(
            {
                "next_id": self._next_id,
                "records": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "embedding": r.embedding.tolist(),
                        "session_idx": r.session_idx,
                        "date": r.date,
                    }
                    for r in self._store.values()
                ],
            },
            path,
        )

    def restore(self, path: str) -> None:
        d = load_pickle(path)
        self._store = {}
        for rec in d.get("records", []):
            # Defend against corrupt/old snapshots - skip records missing
            # required keys (avoids whole-restore failure).
            has_keys = isinstance(rec, dict) and all(
                k in rec for k in ("id", "text", "embedding")
            )
            if not has_keys:
                continue
            self._store[rec["id"]] = MemoryRecord(
                id=rec["id"],
                text=rec["text"],
                embedding=np.asarray(rec["embedding"], dtype=np.float32),
                session_idx=rec.get("session_idx", 0),
                date=rec.get("date", ""),
            )
        # Recover the id counter - the larger of the snapshot's next_id and
        # "max numeric id in
        # the store + 1", so a new id never collides even if the snapshot
        # next_id is stale/low.
        next_id = int(d.get("next_id", 0))
        max_numeric = -1
        for rid in self._store:
            if rid.isdigit():
                max_numeric = max(max_numeric, int(rid))
        self._next_id = max(next_id, max_numeric + 1)

    # Internal - embedding + similar-memory search + metrics accumulation
    def _embed(self, text: str, metrics: CallMetrics) -> np.ndarray:
        """Embed one text + record latency/count."""
        t0 = time.perf_counter()
        vec = self.embedder.embed_one(text)
        metrics.latency_sec += time.perf_counter() - t0
        metrics.num_embedding_calls += 1
        return np.asarray(vec, dtype=np.float32)

    def _similar_memory_ids(
        self, q_vec: "np.ndarray | None", top_s: int
    ) -> list[str]:
        """Return the top_s existing memory ids most similar to q_vec (over the
        whole store).

        Corresponds to the single batch search (top_k=10) in official main.py
        `_add_to_vector_store` - takes top_s over the whole store from a single
        session-dialogue query embedding.
        """
        if q_vec is None or not self._store:
            return []
        scored = [
            (rid, cos_sim(q_vec, rec.embedding))
            for rid, rec in self._store.items()
        ]
        # Tie-break by id (determinism - independent of dict order on score
        # ties).
        scored.sort(key=lambda x: (-x[1], x[0]))
        if top_s > 0:
            scored = scored[:top_s]
        return [rid for rid, _ in scored]

    @staticmethod
    def _accumulate(metrics: CallMetrics, rec) -> None:
        """Accumulate a generate_structured record into metrics."""
        if rec is None:
            return
        metrics.latency_sec += rec.latency_sec
        metrics.input_tokens += rec.input_tokens
        metrics.output_tokens += rec.output_tokens
        metrics.num_llm_calls += 1

    @staticmethod
    def _accumulate_gen(metrics: CallMetrics, gen) -> None:
        """Accumulate a generate GenerationRecord into metrics."""
        if gen is None:
            return
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1


# Module-level helpers
def _split_turns(dialogue: str) -> list[str]:
    """Split a session dialogue string into utterances (turns) - same rule as
    a_mem._split_turns.

    In this project dialogue is formatted as "Speaker: text" listed line by
    line (\\n) (data_loader output), so per-line split = per-utterance split.
    Blank lines are dropped and each turn is only outer-stripped (inner
    whitespace preserved). If there are no turns (whitespace only), returns an
    empty list -> extract produces an empty result.

    (a_mem.py has an identical function, but it is intentionally duplicated
     here for module independence. The logic is identical, so the two
     baselines' utterance-split unit is consistent.)
    """
    if not dialogue:
        return []
    return [ln.strip() for ln in dialogue.splitlines() if ln.strip()]


# Module-level helpers - free-form JSON parsing (fallback when structured
# fails)
def _extract_json_blob(text: str) -> str:
    """Extract the first {...} JSON block from text (strips markdown fences
    etc.)."""
    s = text.strip()
    # Strip ```json ... ``` fences.
    if "```" in s:
        # Try to extract between the first and last fence.
        parts = s.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("json"):
                p2 = p2[4:].strip()
            if p2.startswith("{"):
                s = p2
                break
    # From the first '{' to the last '}'.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _parse_facts_json(text: str) -> list[str]:
    """Parse {"facts": [...]} from a free-form response. Empty list on
    failure."""
    try:
        obj = json.loads(_extract_json_blob(text))
        facts = obj.get("facts", []) if isinstance(obj, dict) else []
        # If "facts" is not a list (dict/str/None), iterating it wrongly
        # (str->chars, dict->keys)
        # produces garbage facts, so guard with an empty list.
        if not isinstance(facts, list):
            return []
        return [str(x).strip() for x in facts if str(x).strip()]
    except (ValueError, AttributeError, TypeError):
        return []


def _parse_plan_json(text: str) -> list[MemoryOp]:
    """Parse {"memory": [...]} from a free-form response -> list of MemoryOp.
    Empty list on failure."""
    try:
        obj = json.loads(_extract_json_blob(text))
        mem = obj.get("memory", []) if isinstance(obj, dict) else []
    except (ValueError, AttributeError, TypeError):
        return []
    # If "memory" is not a list (dict/str/None), iteration breaks, so guard.
    if not isinstance(mem, list):
        return []
    out: list[MemoryOp] = []
    for m in mem:
        if not isinstance(m, dict):
            continue
        event = str(m.get("event", "NONE")).upper()
        if event not in ("ADD", "UPDATE", "DELETE", "NONE"):
            event = "NONE"
        out.append(
            MemoryOp(
                id=str(m.get("id", "")),
                text=str(m.get("text", "")),
                event=event,  # type: ignore[arg-type]
                old_memory=str(m.get("old_memory", "")),
            )
        )
    return out
