"""Memory ABC - common interface for all memory methods.

Design: only 4 methods are enforced so baselines can be compared fairly:

  1. `commit(session)`              - called when a session ends (write-side).
  2. `retrieve(question, idx)`      - called when a question arrives (read-side).
  3. `snapshot(path)` / `restore(path)` - pickle save/load (absorbs build cost across repeated experiments).

Commit 2-phase parallelization (commit cost is the baseline bottleneck - e.g.
HippoRAG2's sequential per-session OpenIE takes ~28h):
  - `extract(session) -> Any`  - session-*independent* LLM extraction (read-only,
    never mutates self's accumulated store). The caller runs extract for all sessions
    in a ThreadPool (spread across the endpoint pool), then passes results to
    commit_prepared in session order.
  - `commit_prepared(session, extract_result) -> CallMetrics` - takes the extract
    result and does the accumulated build (KG/note/memory). Any accumulation
    dependency (link/evolution/update-decision) must be done here sequentially so
    there is no temporal leakage.
  The Memory ABC default is `extract`=None and `commit_prepared`=plain
  `commit(session)` (ignoring extract_result), so memories that do not support
  2-phase (e.g. GroupChat-Mem) still work safely as sequential commits in the caller's
  2-phase loop.

baseline-specific behavior:
  - A-MEM: commit does LLM-driven note metadata + link evolution
  - HippoRAG2: commit does OpenIE, retrieve does PPR
  - Mem0: commit does UPDATE/ADD/DELETE decisions, retrieve is plain dense sim
  - GroupChat-Mem: commit accumulates records, retrieve does dual-view retrieval + traversal
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder
from groupchat_qa.utils.types import (
    CallMetrics,
    Question,
    RetrievalResult,
    Session,
)


class Memory(ABC):
    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        self.llm = llm
        self.embedder = embedder
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------
    @abstractmethod
    def commit(self, session: Session) -> CallMetrics:
        """Write session info into memory. Returns metrics (if any LLM call was made)."""
        ...

    # ------------------------------------------------------------------
    # Commit 2-phase parallelization hook (default = no-op split)
    # ------------------------------------------------------------------
    def extract(self, session: Session) -> Any:
        """phase 1 - session-*independent* LLM extraction (read-only, accumulated store unchanged).

        default: return None (no extraction step). If None, the caller knows
        commit_prepared falls back to plain commit(session). Baselines
        (a_mem/hipporag2/mem0) override this to return (LLM extraction result +
        precomputed embeddings).

        This method is called in parallel from a ThreadPool, so it must never read
        or write self's accumulated state (_store/_notes/_order/graph, etc.) - it
        must compute purely from the session dialogue/embedding input to stay
        thread-safe (no races). self.llm/self.embedder calls are thread-safe
        (RoundRobinLLM lock + EmbeddingCache lock).
        """
        return None

    def commit_prepared(self, session: Session, extract_result: Any) -> CallMetrics:
        """phase 2 - accumulated build from the extract result. Must be called *sequentially in ascending session idx*.

        default: ignore extract_result and run commit(session) directly (compatible
        with memories that do not support 2-phase). Baselines override this to reuse
        the LLM extraction from extract_result (no re-call) for the accumulated build.
        Any accumulation dependency (link/evolution/update-decision) must be done
        here sequentially so there is no future leakage.
        """
        return self.commit(session)

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    @abstractmethod
    def retrieve(self, question: Question, current_session_idx: int) -> RetrievalResult:
        """Return the context payload for the question."""
        ...

    @abstractmethod
    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        """Return the final QA prompt with the retrieved result embedded.

        Each baseline uses its own prompt template. When `retrieval.rendered` is a
        single text, embed it directly; when session/record slots are separated (e.g.
        enhanced), use the baseline-specific payload in `retrieval.extra`.
        """
        ...

    def answer(
        self,
        question: Question,
        current_session_idx: int,
        current_session: Session | None = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        """**default**: single retrieve + generate.

        Memories with a special policy (e.g. (E)-fallback in GroupChatMem) override this.

        (1) The caller always passes `current_session=`, so we accept it in the
        signature (avoids TypeError even if unused). (2) Fill the diagnostic extra
        keys the caller reads (qa_raw_text / qa_reasoning_text / qa_final_prompt /
        qa_retrieval_rendered / route) even on the default path to avoid empty logs.
        All baselines currently override answer, but this default must match the
        contract so future baselines can safely reuse it.

        Returns: (model_answer_text, retrieval_result, accumulated_metrics)
        """
        result = self.retrieve(question, current_session_idx)
        prompt = self.build_qa_prompt(question, result)
        gen = self.llm.generate(prompt)
        metrics = result.metrics.merge(gen.to_metrics())
        result.extra.setdefault("qa_raw_text", gen.text)
        result.extra.setdefault("qa_reasoning_text", getattr(gen, "reasoning_text", ""))
        result.extra.setdefault("qa_final_prompt", prompt)
        result.extra.setdefault("qa_retrieval_rendered", result.rendered)
        result.extra.setdefault("route", getattr(self, "NAME", self.__class__.__name__))
        return gen.text, result, metrics

    # ------------------------------------------------------------------
    # Fill QA prompt (shared) - unify evidence placeholders
    # ------------------------------------------------------------------
    def _fill_qa_template(self, qa_template: str, q_str: str, evidence: str) -> str:
        """Fill the QA prompt's {question} + evidence placeholders.

        The evidence placeholder name differs by baseline/QA mode
        ({history}/{memories}/{passages}/{notes}/{context}). All are replaced with
        the same evidence, so changing only the `prompt_qa` file lets you toggle
        (1) method-mcq / (2) unified-rag / (3) official-generative modes without code
        changes (placeholders absent from the template are no-ops).
        """
        prompt = qa_template.replace("{question}", q_str)
        # Fix: {memory} (singular, used by some A-MEM QA templates) was missing, leaving
        #   {memory} unreplaced in the a_mem QA prompt. Other method templates have no
        #   {memory}, so this is a no-op there. "{memory}" is not a substring of
        #   "{memories}" (memory vs memori), so order is safe.
        for ph in ("{history}", "{memory}", "{memories}", "{passages}", "{notes}", "{context}"):
            prompt = prompt.replace(ph, evidence)
        return prompt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @abstractmethod
    def snapshot(self, path: str) -> None:
        """Save the current memory state (e.g. via pickle)."""
        ...

    @abstractmethod
    def restore(self, path: str) -> None:
        """Load the saved state."""
        ...
