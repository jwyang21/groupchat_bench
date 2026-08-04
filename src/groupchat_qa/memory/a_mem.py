"""A-MEM (Agentic Memory) baseline - Xu et al., NeurIPS 2025 (arXiv:2502.12110).

Reference implementations:
  - System code:     https://github.com/WujiangXu/A-mem-sys
  - Evaluation code: https://github.com/agiresearch/A-mem
Both repos' `agentic_memory/memory_system.py` (AgenticMemorySystem) are essentially identical.

Core of A-MEM (Zettelkasten principle):
  1. Note construction - create a "memory note" for each new piece of information. The LLM
     extracts keywords / context (one-sentence summary) / tags metadata from content
     (analyze_content).
  2. Link generation   - find the k nearest neighbors of the new note in embedding space and
     let the LLM decide which neighbors to link to (the "strengthen" action in the official code).
  3. Memory evolution  - let the LLM decide whether the arrival of the new note should update
     the context / tags of neighbor notes (the "update_neighbor" action in the official code).

Important facts about the official repo:
  - Link generation and memory evolution are decided together in a **single LLM call**
    (`process_memory`, `_evolution_system_prompt`). When should_evolve=True, the actions
    include "strengthen" (= add link + update the new note's tags) and
    "update_neighbor" (= update neighbors' context/tags).
  - Retrieval is ChromaDB (SentenceTransformer embedding) dense similarity top-k.
  - The embedding target is the official A-MEM enhanced document (paper Eq3
    e_i=f_enc[concat(content, keywords, tags, context)]; WujiangXu/A-mem memory_layer.py
    line722 add_documents(["content:"+c+" context:"+ctx+" keywords:"+kw+" tags:"+tags])).
    The old content-only implementation was replaced with enhanced. Future leakage is
    prevented via ranking on fixed creation-time embeddings + session_idx filter.

Project adaptation (Memory ABC mapping):
  - commit(session)  -> split the session into turns (utterances) and add_note per turn
    (note construction + link + evolution). Faithfully reproduces the official LoCoMo eval
    (AgenticMemory/test_advanced.py:301-305) which makes one note per turn via
    `for turn in turns.turns: add_note(...)`.
  - retrieve(q, idx) -> dense top-k over turn-notes from sessions below idx only (prevents future leakage).
  - build_qa_prompt  -> embed the retrieved notes (turns) into the history slot to build the QA prompt.
  - snapshot/restore -> pickle the note store + link graph + embeddings.

Instead of using ChromaDB / SentenceTransformer directly, this reproduces the behavior with the
project's `self.embedder` + an in-memory dense store (project rule: use only self.llm / self.embedder).
"""
from __future__ import annotations

import ast
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    load_pickle,
    load_prompt,
    render_question,
    save_pickle,
    trim_text_to_tokens,
)

from .base import Memory


# pydantic schema - structured LLM responses
class NoteAnalysis(BaseModel):
    """Response schema for note construction (analyze_content).

    Same fields as the official A-MEM analyze_content JSON schema:
    keywords (list), context (one sentence), tags (list).
    """
    keywords: list[str] = Field(default_factory=list, description="salient keywords")
    context: str = Field(default="General", description="one-sentence context summary")
    tags: list[str] = Field(default_factory=list, description="categorical tags")


class EvolutionDecision(BaseModel):
    """Combined link generation + memory evolution decision schema.

    Same as the official A-MEM process_memory JSON schema.
    - strengthen action      -> suggested_connections (link) + tags_to_update (new note tags)
    - update_neighbor action -> new_context_neighborhood + new_tags_neighborhood (neighbor updates)
    """
    should_evolve: bool = Field(default=False)
    actions: list[str] = Field(default_factory=list)
    suggested_connections: list[int] = Field(default_factory=list)  # 0-based neighbor index (int enforced via response_format so even weak models output only the index)
    tags_to_update: list[str] = Field(default_factory=list)
    new_context_neighborhood: list[str] = Field(default_factory=list)
    new_tags_neighborhood: list[list[str]] = Field(default_factory=list)


class LinkDecision(BaseModel):
    """Link-only decision schema (split mode - the standalone link generation stage, paper section 3.3).

    The official repo merges link+evolution; this schema is used when split is enabled via cfg.
    """
    should_link: bool = Field(default=False)
    suggested_connections: list[int] = Field(default_factory=list)  # 0-based neighbor index (int enforced via response_format so even weak models output only the index)
    tags_to_update: list[str] = Field(default_factory=list)


class QueryKeywords(BaseModel):
    """Response schema for generate_query_llm (official A-MEM test_advanced.py L67-87).

    question -> search keyword string. The official code embeds the keyword directly for search
    (retrieve_memory(keywords)). We store this as a precompute cache to eliminate the QA-runtime call.
    """
    keywords: str = Field(default="", description="search keywords for retrieval")


# MemoryNote - the basic unit of A-MEM (same fields as the official MemoryNote)
@dataclass
class MemoryNote:
    """A single unit of the memory system. Holds the same metadata as the official A-MEM MemoryNote."""
    content: str                                  # note body (= session dialogue)
    id: str                                        # unique ID (uuid)
    session_idx: int                               # which session it came from (for future-leakage prevention)
    keywords: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # ids of other linked notes
    context: str = "General"                        # one-sentence summary
    tags: list[str] = field(default_factory=list)
    category: str = "Uncategorized"
    timestamp: str = ""                            # session date (talk start time)
    last_accessed: str = ""
    retrieval_count: int = 0
    evolution_history: list = field(default_factory=list)
    embedding: Any | None = None                   # note's dense vector (enhanced: content+context+keywords+tags concat, fixed at creation time - not re-embedded on evolution)
    # Render-only creation-time originals. Since evolution (update_neighbor) may overwrite
    #   note.context/tags in future sessions, QA rendering uses these immutable originals to prevent
    #   future leakage (ranking is leak-free because the embedding is fixed).
    orig_context: str = "General"
    orig_tags: list[str] = field(default_factory=list)


@dataclass
class AMemTurnExtract:
    """Phase-1 (extract) output for one turn (utterance) - turn-independent LLM/embedding result."""
    content: str           # turn utterance text (= one note's content).
    analysis: "NoteAnalysis"
    embedding: Any         # np.ndarray - turn's dense vector (enhanced: content+context+keywords+tags concat).


@dataclass
class AMemExtractResult:
    """Phase-1 (extract) output of the 2-phase commit - session-independent LLM/embedding result.

    Faithful reproduction of official A-MEM: split the session dialogue into turns (utterances) and
    make **one note per turn** (official LoCoMo eval `for turn in turns.turns: add_note(...)` - one
    add_note per turn, AgenticMemory/test_advanced.py:301-305). So a session's extract result holds a
    list of turns. link/evolution (which have cumulative dependencies) run sequentially in
    phase 2 (commit_prepared) in turn order.
    """
    turns: list["AMemTurnExtract"]
    metrics: CallMetrics


# AMemMemory
class AMemMemory(Memory):
    """A-MEM baseline - note construction + link generation + memory evolution.

    config dict keys:
      k_neighbors        : int - number of nearest neighbors considered during link/evolution (default 5).
                           Same as the official A-MEM process_memory(k=5).
      num_retrieved      : int - top-k notes to retrieve at QA time (default 120).
      context_size       : int - final QA prompt token limit (default 0 = off/unlimited; >0 = that upper bound).
      question_format    : 'multi_structured' / 'multi_unstructured' / 'open_ended'.
      prompt_note        : note construction prompt filename (default a_mem_note_construction.txt).
      prompt_evolution   : combined link+evolution prompt (default a_mem_evolution.txt).
      prompt_link        : link-only prompt (split mode - default a_mem_link_judge.txt).
      prompt_qa          : QA prompt filename. Uses the question_format default if unspecified.
      split_link_evolution: bool - if True, split link and evolution into separate LLM calls
                            (the paper's conceptual separation). Default False = single combined call
                            like the official repo.
      include_linked     : bool - also include the link neighbors of retrieved notes at retrieve time
                           (default False). The official A-MEM QA retrieval (search_agentic) is pure
                           top-k cosine, and link (find_related_memories) is only used at the note
                           construction/evolution stage; expanding to links at QA search deviates from
                           official, so default False.
    """

    NAME = "a_mem"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)

        # note store + link graph (in-memory; ChromaDB replacement)
        self._notes: dict[str, MemoryNote] = {}
        self._order: list[str] = []  # preserve insertion order (snapshot reproducibility)

        # hyperparameters
        self.k_neighbors = int(cfg.get("k_neighbors", 5))
        _nr = cfg.get("num_retrieved", 120)  # default 120
        self.num_retrieved = 120 if _nr is None else int(_nr)
        if self.num_retrieved < 0:
            raise ValueError(f"num_retrieved must be non-negative, got {self.num_retrieved}")
        self.context_size = int(cfg.get("context_size", 0))  # 0/negative = token limit off (unlimited)
        self.split_link_evolution = bool(cfg.get("split_link_evolution", False))
        self.include_linked = bool(cfg.get("include_linked", False))  # official: QA search is pure top-k cosine
        self.question_format = cfg.get("question_format", "multi_structured")
        # How many turns (utterances) to group into one note (user-configurable).
        #   default 1 = original behavior (one utterance = one note, faithful to official A-MEM).
        #   Increasing it reduces note count to ~1/N, cutting commit LLM calls (analyze+link+evolution)
        #   proportionally -> faster commit (at the cost of coarser memory granularity: utterance->chunk).
        _tpn = cfg.get("turns_per_note", 1)
        self.turns_per_note = 1 if _tpn is None else int(_tpn)  # null/unspecified -> 1 (original behavior)
        if self.turns_per_note < 1:
            raise ValueError(f"turns_per_note must be >= 1, got {self.turns_per_note}")

        # Wall-clock hard timeout (seconds) for the extract note-analysis LLM call (analyze_content).
        #   If a thinking-ON model degenerates (infinite/very long response) on a particular session
        #   input, the OpenAI SDK read timeout gets reset by streaming keepalives and cannot cut it,
        #   so Phase 1 hangs indefinitely. This wall-clock timeout gives up on that call and skips the
        #   turn with an empty analysis so the commit can continue. default 180s (normal calls take a
        #   few seconds, so no effect; only hangs are cut). Tunable via config extract_call_timeout.
        self._extract_call_timeout = float(cfg.get("extract_call_timeout", 180.0))

        # prompts
        self._p_note = load_prompt(cfg.get("prompt_note", "a_mem_note_construction.txt"))
        self._p_evolution = load_prompt(cfg.get("prompt_evolution", "a_mem_evolution.txt"))
        self._p_link = load_prompt(cfg.get("prompt_link", "a_mem_link_judge.txt"))

        # QA prompt - default per question_format. A-MEM answers from the retrieved memory notes
        #   embedded into the evidence slot, i.e. the same "answer strictly from the provided
        #   evidence" template used by the Simple RAG baseline (qa_sessions.*).
        _FMT_QA_MAP = {
            "multi_structured": "qa_sessions.txt",
            "multi_unstructured": "qa_sessions.multi-unstructured.txt",
            "open_ended": "qa_sessions.open-ended.txt",
        }
        _fmt_key = str(cfg.get("question_format", "multi_structured")).lower().replace("-", "_")
        _qa_default = _FMT_QA_MAP.get(_fmt_key, _FMT_QA_MAP["multi_structured"])
        self._p_qa = load_prompt(cfg.get("prompt_qa") or _qa_default)
        # QA template token count is constant -> compute once (avoid re-tokenizing per retrieve).
        self._p_qa_tokens = count_tokens(self._p_qa)

        # Load session_{idx-1}.snapshot at retrieve time and search only over notes (enhanced embedding)
        #   up to that point -> use enhanced embeddings without future leakage.
        #   None (unset) keeps the current behavior (self._notes, final commit).
        _snap_dir = cfg.get("session_snapshot_dir")
        self._session_snapshot_dir = Path(_snap_dir) if _snap_dir else None
        # Read-only LRU cache of each session snapshot's (order, notes).
        #   Multi-threaded QA (answer_max_workers>1) safe - _get_session_notes fills under lock and
        #   retrieve uses the returned dict read-only (self._notes not mutated -> no cross-thread race).
        #   The QA loop is monotonically increasing, so evict from smaller idx. 64 = 16 workers x slack.
        self._snap_cache: "OrderedDict[int, tuple[list[str], dict[str, MemoryNote]]]" = OrderedDict()
        self._snap_lock = threading.Lock()
        self._snap_cache_max = int(cfg.get("session_snapshot_cache_max", 64))

        # Official generate_query_llm (test_advanced.py L67-87) - convert the question into search
        #   keywords and embed. Use a precompute cache instead of a per-QA-runtime LLM call: the
        #   keyword depends only on the question text (independent of memory/session/current_session_idx),
        #   so precomputing all of them has no future leakage (same safety as the routing/subject caches).
        #   cache path = session_snapshot_dir/query_keywords.json (per model, session-independent).
        self._use_query_keywords = bool(cfg.get("use_query_keywords", False))
        self._p_query_kw = load_prompt(
            cfg.get("prompt_query_keywords", "a_mem_query_keywords.txt"))
        self._query_keywords: dict[str, str] = {}
        self._query_kw_path = (
            self._session_snapshot_dir / "query_keywords.json"
            if self._session_snapshot_dir is not None else None)

    # commit - add one session as memory notes
    def commit(self, session: Session) -> CallMetrics:
        """Add each turn (utterance) of the session as a memory note (per turn: construction -> link -> evolution).

        Follows the official A-MEM add_note + process_memory flow at turn granularity
        (official LoCoMo eval = one add_note per turn). Single call path (no 2-phase) -
        extract + commit_prepared run sequentially (same result).
        """
        ex = self.extract(session)
        return self.commit_prepared(session, ex)

    # phase 1: session-independent LLM/embedding (ThreadPool-parallel safe, store immutable)
    def extract(self, session: "Session") -> "AMemExtractResult | None":
        """Per turn (utterance): note construction (analyze_content) + content embedding (no link/evolution).

        Faithful reproduction of official A-MEM, which splits the dialogue into turns and
        does one add_note per turn (AgenticMemory/test_advanced.py:301-305:
        `for turn in turns.turns: add_memory("Speaker X says : "+turn.text)`).
        So split the session dialogue into turns and run analyze_content + content embedding per turn.

        Never reads or writes self._notes/_order - pure computation from the session input only
           (ThreadPool-parallel safe). link/evolution have cumulative dependencies and run sequentially in phase 2.
        """
        if not session.dialogue:
            return None  # empty session -> commit_prepared is a no-op.
        metrics = CallMetrics()
        turns: list[AMemTurnExtract] = []
        # Group turns_per_note utterances into one note (configurable).
        #   With default 1, raw_turns[i:i+1] = one utterance = identical to the original turn unit.
        raw_turns = _split_turns(session.dialogue)
        for i in range(0, len(raw_turns), self.turns_per_note):
            chunk = raw_turns[i:i + self.turns_per_note]
            # Join grouped utterances with newlines - preserve the original "Speaker: text\n..." format.
            turn_content = "\n".join(chunk)
            # 1) note construction - extract keywords/context/tags via LLM (per grouped-utterance).
            analysis = self._analyze_content(turn_content, metrics)
            # 2) enhanced embedding as in official A-MEM: content+context+keywords+tags concat.
            #    (memory_layer.py line722 add_note -> add_documents(["content:"+content
            #     +" context:"+context+" keywords: "+kw+" tags: "+tags]); paper Eq3 e_i=f_enc[concat(c,K,G,X)].)
            enhanced_text = (
                "content:" + turn_content
                + " context:" + (analysis.context or "General")
                + " keywords: " + ", ".join(analysis.keywords)
                + " tags: " + ", ".join(analysis.tags)
            )
            embedding = self._embed_turn(enhanced_text, metrics)
            turns.append(AMemTurnExtract(content=turn_content, analysis=analysis, embedding=embedding))
        if not turns:
            return None  # dialogue is whitespace only -> make no note.
        return AMemExtractResult(turns=turns, metrics=metrics)

    # phase 2: cumulative build (must be called in ascending session idx order)
    def commit_prepared(
        self, session: "Session", extract_result: "AMemExtractResult | None"
    ) -> CallMetrics:
        if extract_result is None:
            return CallMetrics()  # empty session - make no note.
        metrics = extract_result.metrics  # inherit the analyze+embed metrics from extract.

        # Build one note per turn cumulatively in turn order (official add_note flow, turn-by-turn).
        for te in extract_result.turns:
            note = MemoryNote(
                content=te.content,
                id=str(uuid.uuid4()),
                session_idx=session.idx,
                keywords=te.analysis.keywords,
                context=te.analysis.context or "General",
                tags=te.analysis.tags,
                timestamp=session.date,
                last_accessed=session.date,
            )
            note.embedding = te.embedding
            # Freeze creation-time context/tags as render-only originals. Even if process_memory / future
            #   evolution overwrites note.context/tags, orig stays immutable -> zero future leakage in QA rendering.
            note.orig_context = note.context
            note.orig_tags = list(note.tags)

            # 3) link generation + memory evolution (judge neighbors before adding the note to the store).
            #    The official process_memory looks only at neighbors in the "existing store" (past turn
            #    notes already registered), so the new note is not yet added. Cumulative dependency
            #    (references prior notes) - run sequentially in turn order in phase 2 (prevents future leakage).
            self._process_memory(note, metrics)

            # 4) register in the store (so the next turn note can see this one as a neighbor).
            self._notes[note.id] = note
            self._order.append(note.id)
        return metrics

    # per-session snapshot load - enhanced embedding without future leakage
    def _get_session_notes(self, snap_idx: int) -> "tuple[list[str], dict[str, MemoryNote]]":
        """[DEAD] retrieve no longer uses per-session snapshots
        (self._notes + orig rendering + session_idx filter achieve zero leakage). This method and
        `_snap_cache` have no callers (kept as a marker rather than deleted). _read_snapshot is still
        used by restore.

        Returns the (order, notes) of session_{snap_idx}.snapshot read-only.

        Notes committed+evolved up to that point (sessions 0~snap_idx), using enhanced embeddings -
        no future evolution reflected, so no future leakage. **Multi-threaded QA (answer_max_workers>1)
        safe**:
          - Does not touch self._notes/_order and returns local (order, notes) only -> no cross-thread swap race.
          - Only cache filling is serialized under lock. retrieve uses the returned dict read-only.
          - LRU (_snap_cache_max) bounds memory (QA loop is monotone, so FIFO-evict from smaller idx).
        snap_idx<0 (first session) or missing file (commit incomplete) or corruption -> ([], {}) = no history.
        """
        if snap_idx < 0:
            return [], {}
        with self._snap_lock:
            cached = self._snap_cache.get(snap_idx)
            if cached is None:
                snap_path = self._session_snapshot_dir / f"session_{snap_idx:04d}.snapshot.pkl"
                if snap_path.exists():
                    try:
                        cached = self._read_snapshot(str(snap_path))
                    except Exception:
                        cached = ([], {})  # corrupted snapshot -> graceful no-history
                else:
                    cached = ([], {})
                self._snap_cache[snap_idx] = cached
                while len(self._snap_cache) > self._snap_cache_max:
                    self._snap_cache.popitem(last=False)  # FIFO evict (oldest = smallest idx)
            return cached

    # retrieve - dense top-k over notes from sessions below idx
    def _generate_query_keywords(self, question_text: str) -> str:
        """Official generate_query_llm (test_advanced.py L67-87) - question -> search keywords.

        Prefer structured output; fall back to the raw question on failure (search works with raw too).
        """
        prompt = self._p_query_kw.replace("{question}", question_text)
        try:
            parsed, rec = self.llm.generate_structured(prompt, QueryKeywords)
            if isinstance(parsed, QueryKeywords) and parsed.keywords.strip():
                return parsed.keywords.strip()
        except Exception:
            pass
        return question_text  # fallback: raw question

    def precompute_query_keywords(self, questions: list) -> None:
        """Once before QA - generate/load and cache the search keywords for all questions.

        No-op if use_query_keywords=False (keeps raw-question embedding). If the cache file exists, load
        it; otherwise generate in parallel via that model's generate_query_llm and save. Depends only on
        question text, so there is no future leakage (safe to precompute all of them).
        """
        if not self._use_query_keywords:
            return
        # cache load - use as-is if present (load-only at experiment time).
        if self._query_kw_path is not None and self._query_kw_path.exists():
            try:
                loaded = json.loads(self._query_kw_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._query_keywords = {str(k): str(v) for k, v in loaded.items()}
                    return
            except Exception:
                self._query_keywords = {}
        # collect unique question texts (avoid duplicate calls).
        uniq: list[str] = []
        seen: set[str] = set()
        for q in questions:
            if q is None:
                continue
            t = getattr(q, "text", None)
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        if not uniq:
            return
        # parallel generation (load-aware client is thread-safe). Assemble the result dict on the main thread only.
        result: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(self._generate_query_keywords, t): t for t in uniq}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    result[t] = fut.result()
                except Exception:
                    result[t] = t  # fallback: raw question
        self._query_keywords = result
        # save (best-effort) - subsequent runs load only.
        if self._query_kw_path is not None:
            try:
                self._query_kw_path.parent.mkdir(parents=True, exist_ok=True)
                self._query_kw_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def retrieve(self, question: Question, current_session_idx: int) -> RetrievalResult:
        """Top-k search by cosine similarity between the question embedding and note embeddings.

        Future-leakage prevention: only notes with session_idx < current_session_idx are candidates.
        """
        metrics = CallMetrics()

        # retrieve candidates = final store (self._notes) + session_idx<current filter.
        #   Ranking uses note.embedding (fixed at creation = same as official) -> zero future leakage;
        #   rendering uses orig_context/orig_tags (immutable) -> zero future leakage. This removed the old
        #   dependency on per-session snapshots (_get_session_notes), eliminating the sparse-snapshot
        #   empty-late-session bug. retrieve accesses self._notes read-only and evolution (writes) finish
        #   in the commit stage, so multi-threaded QA is safe.
        #   [render-field leakage-free rationale] context/tags render from orig (immutable). keywords/links
        #   are safe because evolution (_apply_neighbor_updates=context/tags only, strengthen=the new
        #   note's own tags/links only) never touches past notes, so they stay fixed at creation time.
        #   The links expansion is also filtered by session_idx<current.
        order, notes = self._order, self._notes

        # candidate = notes from past sessions only
        cand_ids = [
            nid for nid in order
            if notes[nid].session_idx < current_session_idx
        ]
        if not cand_ids:
            return RetrievalResult(
                items=[], rendered="No relevant history!",
                extra={"route": "A_MEM"}, metrics=metrics,
            )

        # question embedding (live - questions are not pre-embedded).
        #   If use_query_keywords, embed the precomputed search keyword (official generate_query_llm).
        #   On cache miss (no keyword), fall back to the raw question.
        q_text = question.text
        if self._use_query_keywords:
            _kw = self._query_keywords.get(question.text)
            if _kw:
                q_text = _kw
        t0 = time.perf_counter()
        try:
            q_vec = self.embedder.embed_one(q_text)
        except Exception:
            # A transient embedding-API failure (rate-limit/timeout etc.) must not kill all of answer()
            #   -> graceful no-history fallback. QA proceeds without history (= same as empty candidates).
            metrics.latency_sec += time.perf_counter() - t0
            return RetrievalResult(
                items=[], rendered="No relevant history!",
                extra={"route": "A_MEM", "embed_error": True}, metrics=metrics,
            )
        metrics.latency_sec += time.perf_counter() - t0
        metrics.num_embedding_calls += 1

        # sort by cosine similarity.
        q_dim = _vec_dim(q_vec)
        scored: list[tuple[str, float]] = []
        for nid in cand_ids:
            emb = notes[nid].embedding
            if emb is None:
                continue
            # skip notes with a dimension mismatch (e.g., embedder swap) - avoids cos_sim np.dot crash.
            if q_dim is not None and _vec_dim(emb) != q_dim:
                continue
            scored.append((nid, cos_sim(q_vec, emb)))
        # tie-break by note id (determinism/reproducibility - no dependence on insertion order or float ULP on score ties).
        scored.sort(key=lambda x: (-x[1], x[0]))

        top = scored[: self.num_retrieved] if self.num_retrieved > 0 else scored

        # include link neighbors (official search_agentic / find_related_memories_raw behavior).
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for nid, _ in top:
            if nid not in seen:
                ordered_ids.append(nid)
                seen.add(nid)
            if self.include_linked:
                for link_id in notes[nid].links:
                    # only links to past-session notes (future-leakage prevention).
                    nb = notes.get(link_id)
                    if nb is not None and nb.session_idx < current_session_idx and link_id not in seen:
                        ordered_ids.append(link_id)
                        seen.add(link_id)

        # Cutting to num_retrieved after merging linked neighbors into the dense top-k would nearly
        #   truncate the linked ones (dense already fills k), making include_linked meaningless.
        #   -> if include_linked, skip the cut (final length is decided by context_size at render);
        #     otherwise cut to num_retrieved as before.
        if self.num_retrieved > 0 and not self.include_linked:
            ordered_ids = ordered_ids[: self.num_retrieved]

        # render - fill note content up to the context_size token limit.
        items: list[Item] = []
        rendered = ""
        approx_tokens = self._p_qa_tokens
        score_map = {nid: sc for nid, sc in top}
        # memory index = the global commit rank of the note (batch) (order position, cumulative across sessions).
        nid_to_memidx = {nid: i for i, nid in enumerate(order)}
        for nid in ordered_ids:
            note = notes[nid]
            chunk = _format_note_chunk(note, nid_to_memidx.get(nid, -1))
            tk = count_tokens(chunk)
            if self.context_size > 0 and approx_tokens + tk > self.context_size:
                break
            rendered += chunk
            # retrieve must be read-only (one snapshot -> repeated across many current_session_idx
            #   questions). Mutating note.retrieval_count per retrieve would (a) pollute shared state and
            #   (b) accumulate into the snapshot, making re-runs non-deterministic. retrieval_count is not
            #   used in ranking, so the mutation is removed. (Official A-MEM's access tracking has no
            #   effect on this baseline's search score.)
            items.append(Item(
                text=note.content,
                meta={
                    "date": note.timestamp,
                    "session_idx": note.session_idx,
                    "context": note.orig_context,
                    "keywords": note.keywords,
                    "tags": note.orig_tags,
                    "id": note.id,
                },
                score=score_map.get(nid),
            ))
            approx_tokens += tk

        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant history!",
            extra={"route": "A_MEM", "note_ids": ordered_ids},
            metrics=metrics,
        )

    # build_qa_prompt
    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        # render the question string per question_format (multi-structured etc.).
        q_str = render_question(question, self.question_format)
        evidence = retrieval.rendered or "No relevant history!"
        prompt = self._fill_qa_template(self._p_qa, q_str, evidence)
        return apply_token_limit(prompt, self.context_size)

    # answer - retrieve -> build prompt -> generate (same contract as GroupChat-Mem)
    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        # current_session is accepted but ignored - the A-MEM baseline uses only committed past-session notes.
        result = self.retrieve(question, current_session_idx)
        prompt = self.build_qa_prompt(question, result)
        gen = self.llm.generate(prompt)
        metrics = result.metrics.merge(gen.to_metrics())
        # preserve the final QA generation's raw + reasoning + final prompt + retrieval rendering
        # - used by run_baseline's record (same keys as GroupChat-Mem).
        result.extra["qa_raw_text"] = gen.text
        result.extra["qa_reasoning_text"] = getattr(gen, "reasoning_text", "")
        result.extra["qa_final_prompt"] = prompt
        result.extra["qa_retrieval_rendered"] = result.rendered
        return gen.text, result, metrics

    # snapshot / restore
    def snapshot(self, path: str) -> None:
        payload = {
            "order": list(self._order),
            "notes": {
                nid: {
                    "content": n.content,
                    "id": n.id,
                    "session_idx": n.session_idx,
                    "keywords": n.keywords,
                    "links": n.links,
                    "context": n.context,
                    "tags": n.tags,
                    "orig_context": n.orig_context,
                    "orig_tags": n.orig_tags,
                    "category": n.category,
                    "timestamp": n.timestamp,
                    "last_accessed": n.last_accessed,
                    "retrieval_count": n.retrieval_count,
                    "evolution_history": n.evolution_history,
                    "embedding": None if n.embedding is None else np.asarray(n.embedding).tolist(),
                }
                for nid, n in self._notes.items()
            },
        }
        save_pickle(payload, path)

    def _read_snapshot(self, path: str) -> "tuple[list[str], dict[str, MemoryNote]]":
        """Read the snapshot pickle and return (order, notes) (does not touch self).
        Shared by restore (which swaps self) and _get_session_notes (read-only cache)."""
        d = load_pickle(path)
        order = list(d.get("order", []))
        notes: dict[str, MemoryNote] = {}
        for nid, nd in d.get("notes", {}).items():
            emb = nd.get("embedding")
            notes[nid] = MemoryNote(
                content=nd["content"],
                id=nd["id"],
                session_idx=nd["session_idx"],
                keywords=nd.get("keywords", []),
                links=nd.get("links", []),
                context=nd.get("context", "General"),
                tags=nd.get("tags", []),
                # orig is fail-closed: it always exists in current snapshots. Older snapshots lack orig
                #   -> falling back to evolved context/tags would leak the future, so use neutral ("General"/[]).
                #   None-safe (coerce to str) + list copy (avoid aliasing - orig_tags must not share the tags list).
                orig_context=(str(nd["orig_context"]) if nd.get("orig_context") is not None else "General"),
                orig_tags=(list(nd["orig_tags"]) if nd.get("orig_tags") is not None else []),
                category=nd.get("category", "Uncategorized"),
                timestamp=nd.get("timestamp", ""),
                last_accessed=nd.get("last_accessed", ""),
                retrieval_count=nd.get("retrieval_count", 0),
                evolution_history=nd.get("evolution_history", []),
                embedding=None if emb is None else np.asarray(emb, dtype=np.float32),
            )
        return order, notes

    def restore(self, path: str) -> None:
        # atomic restore: swap self only if _read_snapshot succeeds (no exception) -> prevents partial
        #   corruption from a damaged snapshot (on resume, if restore fails the caller commits from
        #   scratch, so keeping an empty state is safe).
        order, notes = self._read_snapshot(path)
        self._order = order
        self._notes = notes

    # internal - note construction (analyze_content)
    def _run_with_timeout(self, fn, timeout_sec: float):
        """Run fn() in a daemon thread and raise TimeoutError if it does not finish within timeout_sec.

        Purpose: cut hung LLM calls (no response / degenerate very-long response) by *wall clock* -
        the OpenAI SDK timeout is a read (idle) timeout that gets reset by streaming keepalives, so it
        cannot cut an infinite hang. A **daemon** thread means: (a) the main flow proceeds immediately
        after timeout, and (b) a hung thread does not block join at interpreter exit (a non-daemon
        ThreadPoolExecutor makes process exit wait forever via the atexit `_python_exit` join). One hung
        thread leaks, but hangs are rare (specific sessions) so it is acceptable. Normal calls finish
        before timeout -> returned as-is.
        """
        box: dict = {}
        done = threading.Event()

        def _target():
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - propagate any exception to the caller
                box["exc"] = e
            finally:
                done.set()

        threading.Thread(target=_target, daemon=True).start()
        if not done.wait(timeout_sec):
            raise TimeoutError(f"a_mem extract LLM call exceeded {timeout_sec}s (hung)")
        if "exc" in box:
            raise box["exc"]
        return box.get("result")

    def _analyze_content(self, content: str, metrics: CallMetrics) -> NoteAnalysis:
        """Extract keywords/context/tags from content via LLM.

        Same as the official analyze_content. Prefer structured output; fall back to free-form JSON on
        failure. Both LLM calls are wrapped in a wall-clock hard timeout (_extract_call_timeout) - on a
        hang, skip that turn with an empty analysis (keywords/tags=[], context="General") so commit continues.
        """
        prompt = self._p_note.replace("{content}", content)

        # prefer pydantic + langchain structured output.
        #   Wrapped in a wall-clock timeout - on a hang (infinite/degenerate response), exit via
        #   TimeoutError and try the fallback.
        try:
            parsed, rec = self._run_with_timeout(
                lambda: self.llm.generate_structured(prompt, NoteAnalysis),
                self._extract_call_timeout,
            )
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, NoteAnalysis):
                return parsed
        except Exception:  # includes TimeoutError (wall clock) -> fall back.
            pass

        # free-form fallback - parse raw JSON like the official code. Also wrapped in a wall-clock timeout.
        try:
            gen = self._run_with_timeout(lambda: self.llm.generate(prompt), self._extract_call_timeout)
        except Exception:
            # wall-clock timeout (hang) or generation failure -> skip this turn with an empty analysis so
            #   commit continues. (Prevents one session's hang from stalling all of Phase 1 indefinitely.
            #   The note is still searchable via the content embedding - only keywords/tags are empty.)
            return NoteAnalysis(keywords=[], context="General", tags=[])
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        data = _safe_json(gen.text)
        return NoteAnalysis(
            keywords=list(data.get("keywords", []) or []),
            context=str(data.get("context", "General") or "General"),
            tags=list(data.get("tags", []) or []),
        )

    # internal - note embedding (official A-MEM enhanced: content+context+keywords+tags concat)
    def _embed_turn(self, text: str, metrics: CallMetrics):
        """Note embedding (official A-MEM enhanced document: text = content+context+keywords+tags concat).

        Official A-MEM (memory_layer.py line722 add_note -> add_documents(["content:"+content
        +" context:"+context+" keywords: "+kw+" tags: "+tags])) embeds the enhanced document
        (paper Eq3 e_i=f_enc[concat(c_i,K_i,G_i,X_i)], "for retrieval and linking"). The caller (extract)
        assembles and passes enhanced_text. The note unit is a turn, so embed live per turn.
        """
        t0 = time.perf_counter()
        vec = self.embedder.embed_one(text)
        metrics.latency_sec += time.perf_counter() - t0
        metrics.num_embedding_calls += 1
        return np.asarray(vec, dtype=np.float32)

    # internal - link generation + memory evolution
    def _process_memory(self, note: MemoryNote, metrics: CallMetrics) -> None:
        """Find nearest neighbors for the new note and run link/evolution.

        Same flow as the official process_memory:
          - skip if the store is empty (first note).
          - find the k neighbors by the new note's embedding.
          - split_link_evolution=False (default): single combined link+evolution call like official.
          - True: separate calls for link (paper section 3.3) and evolution (paper section 3.4).
        """
        if not self._notes:
            return  # first note - no neighbors.

        neighbor_ids = self._nearest_neighbor_ids(note, self.k_neighbors)
        if not neighbor_ids:
            return
        neighbors_text = self._format_neighbors(neighbor_ids)

        if self.split_link_evolution:
            self._link_only(note, neighbor_ids, neighbors_text, metrics)
            self._evolve_neighbors(note, neighbor_ids, neighbors_text, metrics)
        else:
            self._link_and_evolve(note, neighbor_ids, neighbors_text, metrics)

    def _link_and_evolve(
        self, note: MemoryNote, neighbor_ids: list[str],
        neighbors_text: str, metrics: CallMetrics,
    ) -> None:
        """Official _evolution_system_prompt - combined link (strengthen) + evolution (update_neighbor)."""
        prompt = (
            self._p_evolution
            .replace("{context}", note.context)
            .replace("{content}", note.content)
            .replace("{keywords}", str(note.keywords))
            .replace("{nearest_neighbors_memories}", neighbors_text)
            .replace("{neighbor_number}", str(len(neighbor_ids)))
        )
        decision = self._call_evolution(prompt, metrics)
        if decision is None or not decision.should_evolve:
            return

        for action in decision.actions:
            if action == "strengthen":
                # add link + update the new note's tags.
                for cid in self._resolve_connections(decision.suggested_connections, neighbor_ids):
                    if cid not in note.links:
                        note.links.append(cid)
                if decision.tags_to_update:
                    note.tags = decision.tags_to_update
            elif action == "update_neighbor":
                # update neighbor notes' context / tags (positional mapping).
                self._apply_neighbor_updates(
                    neighbor_ids,
                    decision.new_context_neighborhood,
                    decision.new_tags_neighborhood,
                )

    def _link_only(
        self, note: MemoryNote, neighbor_ids: list[str],
        neighbors_text: str, metrics: CallMetrics,
    ) -> None:
        """split mode - link generation only (paper section 3.3)."""
        prompt = (
            self._p_link
            .replace("{context}", note.context)
            .replace("{content}", note.content)
            .replace("{keywords}", str(note.keywords))
            .replace("{nearest_neighbors_memories}", neighbors_text)
        )
        try:
            parsed, rec = self.llm.generate_structured(prompt, LinkDecision)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            decision = parsed if isinstance(parsed, LinkDecision) else None
        except Exception:
            decision = None
        if decision is None:
            gen = self.llm.generate(prompt)
            metrics.latency_sec += gen.latency_sec
            metrics.input_tokens += gen.input_tokens
            metrics.output_tokens += gen.output_tokens
            metrics.num_llm_calls += 1
            data = _safe_json(gen.text)
            decision = LinkDecision(
                should_link=bool(data.get("should_link", False)),
                suggested_connections=_coerce_int_list(data.get("suggested_connections")),
                tags_to_update=list(data.get("tags_to_update", []) or []),
            )
        if not decision.should_link:
            return
        for cid in self._resolve_connections(decision.suggested_connections, neighbor_ids):
            if cid not in note.links:
                note.links.append(cid)
        if decision.tags_to_update:
            note.tags = decision.tags_to_update

    def _evolve_neighbors(
        self, note: MemoryNote, neighbor_ids: list[str],
        neighbors_text: str, metrics: CallMetrics,
    ) -> None:
        """split mode - memory evolution only (paper section 3.4).

        Reuses the combined prompt but applies only the update_neighbor result (link is handled by _link_only).
        """
        prompt = (
            self._p_evolution
            .replace("{context}", note.context)
            .replace("{content}", note.content)
            .replace("{keywords}", str(note.keywords))
            .replace("{nearest_neighbors_memories}", neighbors_text)
            .replace("{neighbor_number}", str(len(neighbor_ids)))
        )
        decision = self._call_evolution(prompt, metrics)
        if decision is None or not decision.should_evolve:
            return
        if "update_neighbor" in decision.actions:
            self._apply_neighbor_updates(
                neighbor_ids,
                decision.new_context_neighborhood,
                decision.new_tags_neighborhood,
            )

    def _call_evolution(self, prompt: str, metrics: CallMetrics) -> "EvolutionDecision | None":
        """Call the evolution prompt - prefer structured; fall back to free-form JSON on failure."""
        try:
            parsed, rec = self.llm.generate_structured(prompt, EvolutionDecision)
            if rec is not None:
                metrics.latency_sec += rec.latency_sec
                metrics.input_tokens += rec.input_tokens
                metrics.output_tokens += rec.output_tokens
                metrics.num_llm_calls += 1
            if isinstance(parsed, EvolutionDecision):
                return parsed
        except Exception:
            pass
        gen = self.llm.generate(prompt)
        metrics.latency_sec += gen.latency_sec
        metrics.input_tokens += gen.input_tokens
        metrics.output_tokens += gen.output_tokens
        metrics.num_llm_calls += 1
        data = _safe_json(gen.text)
        if not data:
            return None
        return EvolutionDecision(
            should_evolve=bool(data.get("should_evolve", False)),
            actions=list(data.get("actions", []) or []),
            suggested_connections=_coerce_int_list(data.get("suggested_connections")),
            tags_to_update=list(data.get("tags_to_update", []) or []),
            new_context_neighborhood=list(data.get("new_context_neighborhood", []) or []),
            # A weak model may return a flat list-of-strings instead of per-neighbor tag lists ->
            # keep only list elements (str elements would pollute tags per-character, so drop them).
            new_tags_neighborhood=[
                x for x in (data.get("new_tags_neighborhood", []) or []) if isinstance(x, list)
            ],
        )

    def _apply_neighbor_updates(
        self, neighbor_ids: list[str],
        new_contexts: list[str], new_tags: list[list[str]],
    ) -> None:
        """Update neighbor notes' context / tags (official update_neighbor behavior).

        The prompt requires the lengths to match the input neighbor count, but map safely via min().
        Embeddings are not updated - the official repo does not re-embed immediately after evolution and
        only rebuilds during consolidate once evo_threshold is reached. [assumption] This implementation
        uses the content embedding for search, so context/tags updates do not directly affect the search score.
        """
        # [assumption] The positional mapping neighbor_ids[i] <-> new_*[i] is kept per the paper's intent.
        # (The official repo has a positional-index bug, but this implementation intentionally does not follow it.)
        n = len(neighbor_ids)
        for i in range(n):
            nid = neighbor_ids[i]
            note = self._notes.get(nid)
            if note is None:
                continue
            # A weak model may return a flat string instead of a tag list -> apply only when it is a list
            # (a str would split tags per-character and pollute them, so ignore it).
            if i < len(new_tags) and isinstance(new_tags[i], list) and new_tags[i]:
                note.tags = list(new_tags[i])
            if i < len(new_contexts) and new_contexts[i]:
                note.context = str(new_contexts[i])

    # internal - neighbor search / formatting
    def _nearest_neighbor_ids(self, note: MemoryNote, k: int) -> list[str]:
        """Return top-k ids by cosine similarity between the new note's embedding and existing notes."""
        if note.embedding is None:
            return []
        base_dim = _vec_dim(note.embedding)
        scored: list[tuple[str, float]] = []
        for nid in self._order:
            other = self._notes[nid]
            if other.embedding is None:
                continue
            # skip notes with a dimension mismatch - avoids cos_sim np.dot crash.
            if base_dim is not None and _vec_dim(other.embedding) != base_dim:
                continue
            scored.append((nid, cos_sim(note.embedding, other.embedding)))
        # tie-break by id (determinism - reproducible neighbor selection).
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [nid for nid, _ in scored[:k]]

    def _resolve_connections(self, suggested: list, neighbor_ids: list[str]) -> list[str]:
        """Convert the LLM's suggested_connections into actual note ids.

        Official A-MEM exposes neighbors to the prompt as 0-based 'memory index' and stores
        suggested_connections directly into links (an index goes into a link, which is invalid at
        retrieve time - an official bug). This implementation maps the index to the actual neighbor note
        id so links work. It also fixes the case where weak models cannot copy 36-char uuids and links
        collapse entirely (indices 0/1/2 are easy to copy). (Backward compat) directly-given uuids are
        also matched as a fallback.
        """
        resolved: list[str] = []
        for raw in suggested:
            idx = None
            try:
                idx = int(str(raw).strip())
            except (ValueError, TypeError):
                idx = None
            if idx is not None and 0 <= idx < len(neighbor_ids):
                cid = neighbor_ids[idx]
            elif str(raw) in self._notes:
                cid = str(raw)
            else:
                continue
            if cid not in resolved:
                resolved.append(cid)
        return resolved

    def _format_neighbors(self, neighbor_ids: list[str]) -> str:
        """Serialize neighbor notes. Start with a 0-based 'memory index' like official A-MEM
        (exposing uuids makes weak models unable to copy the 36-char id and links collapse -> use index)."""
        out = ""
        for i, nid in enumerate(neighbor_ids):
            n = self._notes[nid]
            out += (
                f"memory index:{i}\ttalk start time:{n.timestamp}"
                f"\tmemory content: {n.content}\tmemory context: {n.context}"
                f"\tmemory keywords: {n.keywords}\tmemory tags: {n.tags}\n"
            )
        return out


# Module-level helpers
def _coerce_int_list(raw: Any) -> list[int]:
    """Safely convert free-form fallback JSON's suggested_connections into list[int].

    The structured (response_format) path enforces int index, but when that fails and falls to
    free-form, the LLM may mix strings like 'memory index:0' with integers. Extract only the numbers so
    the list[int] validation of LinkDecision/EvolutionDecision does not kill commit with a ValidationError.
    """
    out: list[int] = []
    for x in (raw or []):
        try:
            out.append(int(str(x).strip()))
        except (ValueError, TypeError):
            m = re.search(r"-?\d+", str(x))
            if m:
                out.append(int(m.group()))
    return out


def _vec_dim(vec: Any) -> int | None:
    """Return the vector's last-dim size (for the cos_sim dim-mismatch guard). None on failure."""
    try:
        return int(np.asarray(vec).shape[-1])
    except Exception:
        return None


def _split_turns(dialogue: str) -> list[str]:
    """Split the session dialogue string into turns (utterances).

    Faithful reproduction of official A-MEM - the official LoCoMo eval does one add_note per turn
    (AgenticMemory/test_advanced.py: `for turn in turns.turns:
    add_memory("Speaker "+turn.speaker+" says : "+turn.text)`). This project's session dialogue is
    "Speaker: text" laid out one per line (`\n`) (data_loader output, e.g. "Patricia: ...\nJohn: ..."),
    so splitting by line = splitting by turn.

    Blank lines are dropped and each turn is stripped only at its edges (inner whitespace preserved).
    If there are no turns (whitespace only) an empty list is returned -> the caller makes no note.
    """
    if not dialogue:
        return []
    return [ln.strip() for ln in dialogue.splitlines() if ln.strip()]


def _format_note_chunk(note: MemoryNote, mem_idx: int) -> str:
    """Render one note (= a turns_per_note utterance batch) for the memory slot of the QA prompt.

    Official A-MEM (memory_layer_robust.py L422) format = same as the memory example in the A-MEM QA prompt:
      memory index / talk start time / memory content / memory context / memory keywords / memory tags.
    - memory index = the **global rank** at which this note (batch) was committed (_order/snapshot order
      position, cumulative across sessions - if the first session has 3 batches they are 0,1,2; the
      second session's first batch=3). The caller (retrieve) computes it from the order position.
    - talk start time = note.timestamp (= the date of the session the batch was drawn from).
    Future-leakage prevention: context/tags render from note.orig_context/orig_tags (immutable at
    creation, hiding future evolution). keywords are fixed at creation (evolution does not touch them), so used as-is.
    """
    return (
        f"\nmemory index: {mem_idx}\n"
        f"talk start time: {note.timestamp}\n"
        f"memory content:\n{note.content}\n"
        f"memory context: {note.orig_context}\n"
        f"memory keywords: {', '.join(note.keywords)}\n"
        f"memory tags: {', '.join(note.orig_tags)}\n"
    )


def _safe_json(text: str) -> dict:
    """Safely parse a JSON object from a free-form LLM response.

    Even with ```json ... ``` code fences or surrounding junk text, try the first '{' ~ last '}' span,
    and retry tolerantly on Python literals (True/False/None) via `ast.literal_eval`.

    Previously a regex `_pythonize_json` replaced True/False/None->true/false/null, but it also
    mis-replaced words inside string *values* (e.g. context: "that is None of my business"), corrupting
    data. `ast.literal_eval` parses Python literals precisely without touching string interiors, so it
    is corruption-free. raw_decode also avoids mis-extraction caused by trailing extra '}'/junk text.
    """
    if not text:
        return {}
    s = text.strip()
    # strip code fences (including language labels like ```json / ```python).
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        if s.endswith("```"):
            s = s[:-3]
    # first '{' ~ last '}' span (drop surrounding prose).
    start = s.find("{")
    end = s.rfind("}")
    candidate = s[start : end + 1] if (start != -1 and end != -1 and end > start) else s

    # 1) standard JSON.
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) raw_decode from the first '{' - extract only the first complete object even with trailing junk/extra '}'.
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3) Python literals (True/False/None) - parse without corrupting string values.
    try:
        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}
