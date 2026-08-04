"""Simple RAG and Direct-QA reference baselines.

Two lightweight reference settings reported alongside GroupChat-Mem:

  * SimpleRAG — retrieve sessions from the raw session repository S and answer over them.
    ``retriever`` selects the ranking signal:
      - ``bm25``      : sparse lexical retrieval (BM25)
      - ``embedding`` : dense semantic retrieval (embedding cosine similarity)
    These correspond to the paper's "Simple RAG (BM25)" and
    "Simple RAG (OpenAI Embedding)" baselines.
  * DirectQA — no history at all: the model answers from parametric knowledge only
    (the paper's "Direct QA (No History)" reference), quantifying residual leakage.

Both store each session verbatim and share the Memory interface (commit / retrieve /
build_qa_prompt / answer / snapshot / restore).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder
from groupchat_qa.utils import (
    CallMetrics,
    Item,
    Question,
    RetrievalResult,
    Session,
    apply_token_limit,
    cos_sim,
    count_tokens,
    load_pickle,
    load_prompt,
    render_question,
    save_pickle,
)

from .base import Memory
from .groupchat_mem import DialogueIndex, _format_dialogue_chunk


class SimpleRAGMemory(Memory):
    """Simple retrieval-augmented baseline over the raw session repository S.

    Config keys (memory.*):
      retriever      : 'bm25' or 'embedding' (default 'embedding').
      num_sessions   : number of sessions retrieved (default 40).
      context_size   : optional final-prompt token limit (0 = unlimited).
      no_history     : if True, behave as Direct QA (answer with no retrieved history).
      question_format: 'multi_structured' / 'multi_unstructured' / 'open_ended'.
    """

    NAME = "simple_rag"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        self._idx = DialogueIndex()
        self.retriever = str(cfg.get("retriever", "embedding")).lower()
        if self.retriever not in ("bm25", "embedding"):
            raise ValueError(f"retriever must be 'bm25' or 'embedding', got {self.retriever!r}")
        self.num_sessions = int(cfg.get("num_sessions", 40))
        self.context_size = int(cfg.get("context_size", 0))
        self.context_size = 10 ** 9 if self.context_size <= 0 else self.context_size
        self.no_history = bool(cfg.get("no_history", False))
        self.question_format = cfg.get("question_format", "multi_structured")

        _FMT = {
            "multi_structured": ("qa_sessions.txt", "qa_no-history.txt"),
            "multi_unstructured": ("qa_sessions.multi-unstructured.txt", "qa_no-history.multi-unstructured.txt"),
            "open_ended": ("qa_sessions.open-ended.txt", "qa_no-history.open-ended.txt"),
        }
        _fmt_key = str(self.question_format).lower().replace("-", "_")
        _sess_prompt, _nohist_prompt = _FMT.get(_fmt_key, _FMT["multi_structured"])
        self._p_sessions = load_prompt(cfg.get("prompt_sessions") or _sess_prompt)
        self._p_no_hist = load_prompt(cfg.get("prompt_no_history") or _nohist_prompt)

    # ------------------------------------------------------------------
    # Write side — store each session verbatim in S.
    # ------------------------------------------------------------------
    def commit(self, session: Session) -> CallMetrics:
        metrics = CallMetrics()
        if not session.dialogue:
            return metrics
        if session.dialogue_embedding is not None:
            vec = session.dialogue_embedding
        else:
            t0 = time.perf_counter()
            vec = np.asarray(self.embedder.embed_one(session.dialogue), dtype=np.float32)
            metrics.latency_sec += time.perf_counter() - t0
            metrics.num_embedding_calls += 1
        self._idx.add(session.dialogue, session.date, session.idx, vec)
        return metrics

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    def retrieve(self, question: Question, current_session_idx: int,
                 current_session: "Session | None" = None) -> RetrievalResult:
        metrics = CallMetrics()
        if self.no_history:
            return RetrievalResult(items=[], rendered="", extra={"route": "NO_HISTORY"}, metrics=metrics)

        # Candidate pool: previous sessions + the current session's partial dialogue.
        pool: list[tuple[str, str, int, Any]] = []
        for dia, date, s_idx, emb in zip(
            self._idx.dialogues, self._idx.dates, self._idx.sess_idxs, self._idx.embeddings
        ):
            if s_idx < current_session_idx:
                pool.append((dia, date, s_idx, emb))
        if current_session is not None and current_session.partial_dialogue:
            pool.append((
                current_session.partial_dialogue,
                current_session.date,
                current_session.idx,
                current_session.partial_embedding,
            ))
        if not pool:
            return RetrievalResult(
                items=[], rendered="No relevant history!", extra={"route": "SESSIONS"}, metrics=metrics
            )

        order = self._rank(question, pool, metrics)
        if self.num_sessions > 0:
            order = order[: self.num_sessions]

        items: list[Item] = []
        rendered = ""
        q_str = render_question(question, self.question_format)
        approx = count_tokens(self._p_sessions) + count_tokens(q_str)
        for i in order:
            dia, date, s_idx, _ = pool[i]
            chunk = _format_dialogue_chunk(dia, date, s_idx)
            tk = count_tokens(chunk)
            if self.context_size > 0 and approx + tk > self.context_size:
                break
            rendered += chunk
            items.append(Item(text=dia, meta={"date": date, "session_idx": s_idx}))
            approx += tk
        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant history!",
            extra={"route": "SESSIONS"},
            metrics=metrics,
        )

    def _rank(self, question: Question, pool: list[tuple[str, str, int, Any]],
              metrics: CallMetrics) -> list[int]:
        ids = list(range(len(pool)))
        if self.retriever == "bm25":
            from nltk.tokenize import word_tokenize
            from rank_bm25 import BM25Okapi
            tokenized = [word_tokenize(p[0].lower()) for p in pool]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(word_tokenize(question.text.lower()))
            return sorted(ids, key=lambda i: -scores[i])
        # embedding
        t0 = time.perf_counter()
        q_vec = np.asarray(self.embedder.embed_one(question.text), dtype=np.float32)
        metrics.latency_sec += time.perf_counter() - t0
        metrics.num_embedding_calls += 1
        embs: list[Any] = []
        for p in pool:
            e = p[3]
            if e is None:
                t0 = time.perf_counter()
                e = np.asarray(self.embedder.embed_one(p[0]), dtype=np.float32)
                metrics.latency_sec += time.perf_counter() - t0
                metrics.num_embedding_calls += 1
            embs.append(e)
        scored = [(i, cos_sim(q_vec, embs[i])) for i in ids]
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored]

    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        q_str = render_question(question, self.question_format)
        if retrieval.extra.get("route") == "NO_HISTORY":
            return self._p_no_hist.replace("{question}", q_str)
        return self._p_sessions.replace("{question}", q_str).replace(
            "{history}", retrieval.rendered or "No relevant history!"
        )

    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        r = self.retrieve(question, current_session_idx, current_session=current_session)
        prompt = apply_token_limit(self.build_qa_prompt(question, r), self.context_size)
        gen = self.llm.generate(prompt)
        metrics = r.metrics.merge(gen.to_metrics())
        r.extra["qa_raw_text"] = gen.text
        r.extra["qa_reasoning_text"] = getattr(gen, "reasoning_text", "")
        r.extra["qa_final_prompt"] = prompt
        r.extra["qa_retrieval_rendered"] = r.rendered
        return gen.text, r, metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def snapshot(self, path: str) -> None:
        save_pickle(
            {
                "dialogues": self._idx.dialogues,
                "dates": self._idx.dates,
                "sess_idxs": self._idx.sess_idxs,
                "embeddings": [v.tolist() for v in self._idx.embeddings],
            },
            path,
        )

    def restore(self, path: str) -> None:
        d = load_pickle(path)
        self._idx.dialogues = d["dialogues"]
        self._idx.dates = d["dates"]
        self._idx.sess_idxs = d["sess_idxs"]
        self._idx.embeddings = [np.asarray(v, dtype=np.float32) for v in d["embeddings"]]
        self._idx._bm25 = None
