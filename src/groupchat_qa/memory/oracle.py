"""Oracle memory — retrieval upper-bound (perfect evidence + perfect routing).

Background: for time-indexed group-chat QA, measure "how well the model does if
retrieval were perfect" (the retrieval ceiling). Bypasses the memory backend's live
retrieval and injects an **already-built oracle bundle** (output of
`preprocess/v1_3_1/build_oracle.py`) as the retrieved context.

Isolation principle: a separate module that never touches the memory methods. Registered in the factory under
`name == "oracle"`. QA prompt assembly reuses the **same** templates and render
functions as GroupChat-Mem to match prompt format (same format + oracle evidence = a fair ceiling).

Oracle bundle structure (build_oracle.py):
  - dialogue bundle (LLM-independent): {root}/{dt}/oracle/trial{N}.oracle.dialogue.json
        list[entry], entry = {session_idx, question, questioner, q_kind, q_type, timestamp,
                              oracle_dialogues_by_{cosine,bm25,alternating}}
  - record bundle (per LLM & norm): {root}/{dt}/oracle/{llm}/trial{N}.oracle_{raw|en_node|en_triple}.json
        list[entry], entry = {... , oracle_records_by_cosine: list[{head,relation,tail,start_date}]}
  Both bundles are in the same order (one entry per question). entry.session_idx = the session the question was asked in = current_session_idx.

Routing: select modality by GT q_kind — perfect routing + perfect retrieval = the true ceiling.
  - q_kind == 'records' → records route, oracle records (cosine) into the {triples} slot.
  - else (fan, etc.) → RAG route, oracle dialogues (alternating) into the {history} slot.
  If fallback=True and the first answer is (E), try once more with the opposite modality (matches GroupChat-Mem's (E)-fallback).

Ordering: build_oracle already sorts (dialogue=alternating, records=cosine) → used **as-is without re-sorting**.
      "use all as evidence" (context_size off = unlimited). raw normalization.

Future leakage: the oracle bundle is collected at build time by date-matching against
  the question's GT timestamp (past evidence sessions, 'future' → []), so it does not
  preserve per-item source session_idx (no future filter at retrieve time). By the
  ceiling's definition it holds only GT (past) evidence, relying on the build-time guarantee.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from groupchat_qa.client import LLM
from groupchat_qa.embedder import Embedder
from groupchat_qa.utils import (
    CallMetrics,
    Item,
    Question,
    RetrievalResult,
    Session,
    apply_token_limit,
    load_prompt,
    process_answer,
    render_question,
)

from .base import Memory
# Reuse GroupChat-Mem's record renderer to match prompt format (for a fair comparison).
from .groupchat_mem import _render_records_in_order

LOGGER = logging.getLogger(__name__)


def _entry_key(session_idx: Any, question: Any) -> tuple:
    """lookup key — (session, question text).

    questioner is excluded from the key: build_oracle's provenance questioner uses the
    **normal name** (e.g. 'Phoebe'), while a newname/shuffled run's Question.questioner
    uses the **perturbed name** (e.g. 'Jennifer'), so they mismatch (question text and
    evidence still match under perturbation). (session_idx, question text) is unique
    across all bundles (0 duplicates), so it matches safely without questioner."""
    return (int(session_idx), str(question).strip())


class OracleMemory(Memory):
    NAME = "oracle"

    def __init__(self, llm: LLM, embedder: Embedder, cfg: dict[str, Any]):
        super().__init__(llm, embedder, cfg)
        self.question_format = cfg.get("question_format", "multi_structured")
        self.record_format = cfg.get("record_format", "record")
        # context_size: 0/negative = token limit off (unlimited). serving max-model-len is the real ceiling.
        self.context_size = int(cfg.get("context_size", 0) or 0)
        self.fallback = bool(cfg.get("fallback", True))
        # dialogue_only=True → no records route, always sessions route.
        #   Oracle = uses only the set of entire-session dialogues (alternating order)
        #   from each question's oracle-timestamp sessions (no records). Usually
        #   single-shot with fallback=false. In this mode, load succeeds even without a
        #   record bundle (record_by_key stays empty).
        self.dialogue_only = bool(cfg.get("oracle_dialogue_only", False))
        # dual=True → inject **both** the entire dialogue and (GT) records from each
        #   question's oracle-timestamp sessions via qa_dual.txt simultaneously
        #   (all, k unlimited; a separate path isolated from route-select / dialogue_only).
        #   The GT records are usually loaded from the llm_name='gt' bundle.
        self.dual = bool(cfg.get("oracle_dual", False))
        # records_only=True → always records route + qa_records regardless of q_kind.
        #   Inject only the GT record union (oracle_records_by_cosine) from each
        #   question's oracle-timestamp sessions (no dialogue; the mirror of
        #   dialogue_only). k_records=0 means all. Usually single-shot with fallback=false.
        self.records_only = bool(cfg.get("oracle_records_only", False))
        self.delimiter = cfg.get("delimiter", "=" * 40)
        # top-k truncation — a fair ceiling using the same budget as the memory methods.
        #   Uses only the top-k of the already-sorted evidence (records=cosine, dials=alternating).
        #   num_records (records, default 250), num_sessions (dialogue, default 40).
        #   0/negative = truncation off (all) — but the main policy defaults to records=250, sessions=40.
        self.k_records = int(cfg.get("num_records", 250) or 0)
        self.k_sessions = int(cfg.get("num_sessions", 40) or 0)
        # dialogue ordering choice — build_oracle policy defaults to alternating.
        self.dialogue_rank = str(cfg.get("oracle_dialogue_rank", "alternating")).lower()

        # QA prompt templates — same files as GroupChat-Mem.
        self._p_sessions = load_prompt(cfg.get("prompt_sessions") or "qa_sessions.txt")
        self._p_records = load_prompt(cfg.get("prompt_records") or "qa_records.txt")
        self._p_no_hist = load_prompt(cfg.get("prompt_no_history") or "qa_no-history.txt")
        # For dual mode — same template as GroupChat-Mem's dual prompt ({question}/{triples}/{history}).
        self._p_dual = load_prompt(cfg.get("prompt_dual") or "qa_dual.txt")

        # Load oracle bundles + build lookup.
        self._lookup: dict[tuple, dict] = {}
        self._load_bundles(cfg)

    # ------------------------------------------------------------------
    # Load bundles
    # ------------------------------------------------------------------
    def _load_bundles(self, cfg: dict[str, Any]) -> None:
        ds_root = cfg.get("dataset_root")
        dt = cfg.get("data_type")
        trial = cfg.get("trial")
        llm_name = cfg.get("llm_name")
        # Record-bundle LLM override. run_baseline injects mem_cfg["llm_name"] as the
        #   QA LLM (qwen, etc.), so to use the GT records (oracle/gt/) set
        #   oracle_record_llm='gt' in cfg to point only the record-bundle path elsewhere
        #   (dialogue is LLM-independent, one file, so unchanged). If unset, use
        #   llm_name (fully preserving the original oracle behavior).
        record_llm = cfg.get("oracle_record_llm") or llm_name
        norm = cfg.get("normalization", "raw")
        if not (ds_root and dt and trial is not None and llm_name):
            raise ValueError(
                "[oracle] cfg requires dataset_root/data_type/trial/llm_name — "
                f"got: dataset_root={ds_root!r} data_type={dt!r} trial={trial!r} llm_name={llm_name!r}"
            )
        trial = int(trial)
        oracle_dir = Path(ds_root) / dt / "oracle"
        dlg_path = oracle_dir / f"trial{trial}.oracle.dialogue.json"
        record_path = oracle_dir / record_llm / f"trial{trial}.oracle_{norm}.json"
        if not dlg_path.exists():
            raise FileNotFoundError(f"[oracle] dialogue bundle not found: {dlg_path} (run build_oracle.py first)")
        if not self.dialogue_only and not record_path.exists():
            raise FileNotFoundError(f"[oracle] record bundle not found: {record_path} (run build_oracle.py first)")

        with open(dlg_path, encoding="utf-8") as f:
            dlg_list = json.load(f)

        # Index the record bundle by key (same order, but match by key to be safe). If dialogue_only, records are unneeded → empty dict.
        record_by_key: dict[tuple, list] = {}
        if not self.dialogue_only:
            with open(record_path, encoding="utf-8") as f:
                record_list = json.load(f)
            for q in record_list:
                record_by_key[_entry_key(q["session_idx"], q["question"])] = list(
                    q.get("oracle_records_by_cosine", []) or []
                )

        for e in dlg_list:
            key = _entry_key(e["session_idx"], e["question"])
            dials = self._pick_dialogues(e)
            self._lookup[key] = {
                "q_kind": str(e.get("q_kind", "")).lower(),
                "dials": dials,
                "records": record_by_key.get(key, []),
            }
        LOGGER.info(
            "[oracle] bundles loaded: dialogue=%s record=%s → lookup %d entries (llm=%s norm=%s t%d)",
            dlg_path.name, record_path.name, len(self._lookup), llm_name, norm, trial,
        )

    def _pick_dialogues(self, entry: dict) -> list:
        """Select the dialogue list build_oracle pre-sorted (no re-sorting)."""
        # oracle experiment default = time-ordered accumulation (no ranking).
        if self.dialogue_rank in ("time_ordered", "time", "session_order", "accumulated"):
            v = entry.get("oracle_dialogues_time_ordered")
        elif self.dialogue_rank == "cosine":
            v = entry.get("oracle_dialogues_by_cosine")
        elif self.dialogue_rank == "bm25":
            v = entry.get("oracle_dialogues_by_bm25")
        elif self.dialogue_rank == "alternating":
            v = entry.get("oracle_dialogues_by_alternating")
        else:
            v = entry.get("oracle_dialogues_time_ordered")
        if not v:  # if the selected list is empty, fall back time_ordered → alternating → cosine
            v = (entry.get("oracle_dialogues_time_ordered")
                 or entry.get("oracle_dialogues_by_alternating")
                 or entry.get("oracle_dialogues_by_cosine") or [])
        return list(v)

    # ------------------------------------------------------------------
    # Write side — oracle is precomputed, commit is a no-op.
    # ------------------------------------------------------------------
    def commit(self, session: Session) -> CallMetrics:
        return CallMetrics()

    # ------------------------------------------------------------------
    # Rendering — same format as GroupChat-Mem
    # ------------------------------------------------------------------
    def _render_records(self, records: list) -> str:
        rendered, _ = _render_records_in_order(
            records, self.record_format, self.context_size, self._p_records
        )
        return rendered

    def _render_sessions(self, dials: list) -> str:
        if not dials:
            return "No relevant history!"
        sep = f"\n{self.delimiter}\n"
        return sep.join(str(d) for d in dials)

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    def _lookup_entry(self, question: Question, current_session_idx: int) -> dict | None:
        key = _entry_key(current_session_idx, question.text)
        e = self._lookup.get(key)
        if e is None:
            LOGGER.warning(
                "[oracle] lookup miss: idx=%s q=%r — treating as empty evidence.",
                current_session_idx, (question.text or "")[:60],
            )
        return e

    def _retrieve_route(self, question: Question, current_session_idx: int,
                        route: str) -> RetrievalResult:
        """Render oracle evidence for route ('RECORDS'|'SESSIONS')."""
        e = self._lookup_entry(question, current_session_idx)
        records = e["records"] if e else []
        dials = e["dials"] if e else []
        # top-k truncation (already sorted: records=cosine, dials=alternating, top first). k<=0 = all.
        if self.k_records > 0:
            records = records[: self.k_records]
        if self.k_sessions > 0:
            dials = dials[: self.k_sessions]
        if route == "RECORDS":
            rendered = self._render_records(records)
            items = [Item(text=str(q)) for q in records]
        else:
            rendered = self._render_sessions(dials)
            items = [Item(text=str(d)) for d in dials]
        return RetrievalResult(
            items=items,
            rendered=rendered,
            metrics=CallMetrics(),
            extra={"route": route, "oracle_records": records, "oracle_dials": dials,
                   "q_kind": (e["q_kind"] if e else "")},
        )

    def _retrieve_dual(self, question: Question,
                           current_session_idx: int) -> RetrievalResult:
        """Dual oracle: gather **both** dialogue + records and render with qa_dual (all, k unlimited).

        Same spirit as GroupChat-Mem's dual view (provide records + dialogue together),
        but oracle is a retrieval ceiling that injects the (GT) evidence of the oracle-
        timestamp sessions directly instead of retrieving. k_records/k_sessions = 0 (default)
        means all; >0 uses only the top-k (already sorted).
        """
        e = self._lookup_entry(question, current_session_idx)
        records = e["records"] if e else []
        dials = e["dials"] if e else []
        if self.k_records > 0:
            records = records[: self.k_records]
        if self.k_sessions > 0:
            dials = dials[: self.k_sessions]
        records_rendered = self._render_records(records)
        sessions_rendered = self._render_sessions(dials)
        items = [Item(text=str(q)) for q in records] + [Item(text=str(d)) for d in dials]
        return RetrievalResult(
            items=items,
            # rendered is for compatibility (logging) — build_qa_prompt uses records/sessions_rendered in extra.
            rendered=records_rendered,
            metrics=CallMetrics(),
            extra={"route": "DUAL", "records_rendered": records_rendered,
                   "sessions_rendered": sessions_rendered, "oracle_records": records,
                   "oracle_dials": dials, "q_kind": (e["q_kind"] if e else "")},
        )

    def retrieve(self, question: Question, current_session_idx: int,
                 current_session: "Session | None" = None) -> RetrievalResult:
        # dual: dialogue + records both (bypasses route selection).
        if self.dual:
            return self._retrieve_dual(question, current_session_idx)
        e = self._lookup_entry(question, current_session_idx)
        q_kind = (e["q_kind"] if e else "")
        # records_only → always records; dialogue_only → always sessions; else by q_kind.
        if self.records_only:
            route = "RECORDS"
        elif self.dialogue_only:
            route = "SESSIONS"
        else:
            route = "RECORDS" if q_kind == "records" else "SESSIONS"
        return self._retrieve_route(question, current_session_idx, route)

    def build_qa_prompt(self, question: Question, retrieval: RetrievalResult) -> str:
        q_str = render_question(question, self.question_format)
        route = retrieval.extra.get("route", "SESSIONS")
        if route == "DUAL":
            # qa_dual.txt: inject {question} + {triples} (records) + {history} (dialogue) together.
            return (
                self._p_dual.replace("{question}", q_str)
                .replace("{triples}", retrieval.extra.get("records_rendered") or "No relevant triple!")
                .replace("{history}", retrieval.extra.get("sessions_rendered") or "No relevant history!")
            )
        if route == "RECORDS":
            return self._p_records.replace("{question}", q_str).replace(
                "{triples}", retrieval.rendered or "No relevant triple!"
            )
        return self._p_sessions.replace("{question}", q_str).replace(
            "{history}", retrieval.rendered or "No relevant history!"
        )

    # ------------------------------------------------------------------
    # answer — supports (E)-fallback (opposite modality), matching GroupChat-Mem
    # ------------------------------------------------------------------
    def answer(
        self, question: Question, current_session_idx: int,
        current_session: "Session | None" = None,
    ) -> tuple[str, RetrievalResult, CallMetrics]:
        r1 = self.retrieve(question, current_session_idx, current_session=current_session)
        prompt1 = apply_token_limit(self.build_qa_prompt(question, r1), self.context_size)
        gen1 = self.llm.generate(prompt1)
        metrics = r1.metrics.merge(gen1.to_metrics())
        r1.extra["qa_raw_text"] = gen1.text
        r1.extra["qa_reasoning_text"] = getattr(gen1, "reasoning_text", "")
        r1.extra["qa_final_prompt"] = prompt1
        r1.extra["qa_retrieval_rendered"] = r1.rendered

        # If dual (dialogue+records already provided together), fallback off, or not
        #   (E), finalize the first answer — enhanced has no opposite modality, so (E)-fallback is meaningless.
        if self.dual or self.records_only or not self.fallback or process_answer(gen1.text) != "(E)":
            return gen1.text, r1, metrics

        # (E) → try once more with the opposite modality.
        first_route = r1.extra.get("route", "SESSIONS")
        fb_route = "RECORDS" if first_route == "SESSIONS" else "SESSIONS"
        r2 = self._retrieve_route(question, current_session_idx, fb_route)
        # If the opposite modality is empty (no evidence), fallback is meaningless → keep the first answer.
        if not r2.items:
            return gen1.text, r1, metrics
        prompt2 = apply_token_limit(self.build_qa_prompt(question, r2), self.context_size)
        gen2 = self.llm.generate(prompt2)
        metrics = metrics.merge(gen2.to_metrics())
        r2.extra["qa_raw_text"] = gen2.text
        r2.extra["qa_reasoning_text"] = getattr(gen2, "reasoning_text", "")
        r2.extra["qa_final_prompt"] = prompt2
        r2.extra["qa_retrieval_rendered"] = r2.rendered
        r2.extra["fallback_from"] = first_route
        return gen2.text, r2, metrics

    # ------------------------------------------------------------------
    # Persistence — oracle has zero build cost (JSON load) → no-op.
    # ------------------------------------------------------------------
    def snapshot(self, path: str) -> None:
        # oracle has no accumulated state, so write only an empty marker (for snapshot/restore wiring compatibility).
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"oracle": True}, f)
        except Exception:  # best-effort
            pass

    def restore(self, path: str) -> None:
        # No accumulated state to restore — no-op (bundles are loaded in __init__).
        return None
