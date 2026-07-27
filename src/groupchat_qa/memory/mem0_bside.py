"""Mem0 B-side QA-only variant — a lens experiment on retrieve/render/QA-prompt.

A **subclass** so the original `mem0.py` (Mem0Memory) is never modified.
The commit/extract/update phases are 100% identical to the original (inherited) —
guaranteeing snapshot reuse (QA-only). Only retrieve (retrieval, dedup, top_k),
render (grouping), and build_qa_prompt (QA prompt file) are varied.

Toggle flags (cfg["memory"]):
  bside_render        : "flat" (original flat `- fact`) | "grouped" (session/date-header groups). default flat.
  bside_group_order   : group placement order when grouped. "time" (ascending session idx) |
                        "score" (descending group max-score). default score (mitigates lost-in-middle).
  bside_dedup         : whether to merge near-dup facts (bool). default false.
  bside_dedup_tau     : dedup cosine threshold (float). default 0.95. (cos >= tau → same cluster → keep only one)
  top_k               : number of memories to retrieve (same key as original). The top_k sweep changes only this.
  prompt_qa           : QA prompt file (same key as original).

dataset-agnostic: render headers use only date/Session integer metadata; QA-prompt few-shot uses only synthetic entities.
"""
from __future__ import annotations

import numpy as np

from groupchat_qa.utils import (
    CallMetrics,
    Item,
    Question,
    RetrievalResult,
    cos_sim,
    count_tokens,
)
from groupchat_qa.utils.mem0_scoring import score_and_rank
from .mem0 import Mem0Memory, MemoryRecord


def _group_score(scores: list[float], agg: str) -> float:
    """Group-score aggregation — compute the sort key for group placement when group_order="score".

    agg ∈ {"max","mean","median","min"}:
      - max    : group's best-fact score (best-evidence group first — original behavior).
      - mean   : group's overall average relevance.
      - median : group median (less swayed by an outlier single high score).
      - min    : group's lowest score (conservative — the whole group must be relevant to rank high).
    Empty list → 0.0 (guard). Unknown agg falls back to max.
    """
    if not scores:
        return 0.0
    if agg == "mean":
        return float(sum(scores) / len(scores))
    if agg == "median":
        s = sorted(scores)
        n = len(s)
        mid = n // 2
        return float(s[mid]) if n % 2 == 1 else float((s[mid - 1] + s[mid]) / 2.0)
    if agg == "min":
        return float(min(scores))
    # "max" (default / fallback)
    return float(max(scores))


def _format_grouped(
    recs_scores: list[tuple[MemoryRecord, float]],
    group_order: str,
    group_score_agg: str = "max",
) -> tuple[str, list[Item]]:
    """Group the retrieved (record, score) pairs by (session_idx, date) and render with headers.

    Within a group: descending score (best evidence first).
    Group placement:
      - "time"  : ascending session_idx (chronological — good for temporal reasoning but best can be pushed back).
      - "score" : descending group score (aggregated by group_score_agg: max/mean/median/min).
                  With group_score_agg=max, the best-evidence group comes first (mitigates lost-in-middle; original behavior).
    Header format follows the same convention as hipporag2._format_hippo_chunk:
    "[Date: {date} (Session {idx})]" (or "[Session {idx}]" if date is empty). dataset-agnostic.
    Returns: (rendered_str, items) — items flattened in descending score (same meta keys as the original).
    """
    # Group by (session_idx, date). The group key is an (idx, date) tuple.
    groups: dict[tuple[int, str], list[tuple[MemoryRecord, float]]] = {}
    for rec, sc in recs_scores:
        key = (int(rec.session_idx), str(rec.date))
        groups.setdefault(key, []).append((rec, sc))

    # Sort within each group by descending score.
    for key in groups:
        groups[key].sort(key=lambda rs: -rs[1])

    # Decide group placement order.
    group_keys = list(groups.keys())
    if group_order == "time":
        # Ascending session idx (chronological). Ties broken by date string.
        group_keys.sort(key=lambda k: (k[0], k[1]))
    else:  # "score" (default)
        # Descending group score (aggregated: max/mean/median/min). Ties broken by ascending session_idx (determinism).
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
                    meta={"id": rec.id, "session_idx": rec.session_idx, "date": rec.date},
                    score=float(sc),
                )
            )
        lines.append("")  # blank line between groups (readability).
    rendered = "\n".join(lines).strip()
    return rendered, items


class Mem0BsideMemory(Mem0Memory):
    """Mem0 retrieve/render/QA-prompt variant (QA-only). commit/update inherited from the original (unchanged)."""

    NAME = "mem0_bside"

    def __init__(self, llm, embedder, cfg):
        super().__init__(llm, embedder, cfg)
        # B-side toggles — if unset, behave identically to the original (flat render, no dedup).
        self.bside_render = str(cfg.get("bside_render", "flat")).lower()
        self.bside_group_order = str(cfg.get("bside_group_order", "score")).lower()
        # Group-score aggregation (for group placement when group_order="score"): max (original)|mean|median|min.
        self.bside_group_score_agg = str(cfg.get("bside_group_score_agg", "max")).lower()
        self.bside_dedup = bool(cfg.get("bside_dedup", False))
        self.bside_dedup_tau = float(cfg.get("bside_dedup_tau", 0.95))

    # ====================================================================
    # retrieve — original logic + dedup + grouped render. top_k sweep changes only top_k.
    # ====================================================================
    def retrieve(self, question: Question, current_session_idx: int) -> RetrievalResult:
        metrics = CallMetrics()
        # Prevent future leakage — same as the original.
        candidates = [
            r for r in self._store.values() if r.session_idx < current_session_idx
        ]
        if not candidates:
            return RetrievalResult(
                items=[], rendered="No relevant memory!",
                extra={"route": "MEM0_BSIDE"}, metrics=metrics,
            )

        q_vec = self._embed(question.text, metrics)

        # Official hybrid scoring (semantic gate + BM25) — same as the original.
        semantic_results = [
            {"id": str(r.id), "score": cos_sim(q_vec, r.embedding), "payload": r}
            for r in candidates
        ]
        bm25_scores = self._compute_bm25_scores(question.text, candidates)
        # If dedup is on, pull more than top_k and fill top_k after dedup.
        #   Since dedup shrinks the set, use a larger rank pool to have enough candidates.
        rank_top_k = self.top_k * 3 if self.bside_dedup else self.top_k
        ranked = score_and_rank(
            semantic_results, bm25_scores, {},
            threshold=self.semantic_threshold, top_k=rank_top_k,
        )

        id_to_rec = {str(r.id): r for r in candidates}

        # near-dup fact merge/dedup — walk ranked in order and skip any with cos>=tau to an already-kept fact.
        if self.bside_dedup:
            kept: list[tuple[str, float]] = []  # (id, combined_score)
            kept_vecs: list[np.ndarray] = []
            for sr in ranked:
                rid = sr["id"]
                rec = id_to_rec[rid]
                vec = rec.embedding
                is_dup = False
                for kv in kept_vecs:
                    if cos_sim(vec, kv) >= self.bside_dedup_tau:
                        is_dup = True
                        break
                if is_dup:
                    continue
                kept.append((rid, float(sr["score"])))
                kept_vecs.append(vec)
                if len(kept) >= self.top_k:
                    break
            ranked = [{"id": rid, "score": sc} for rid, sc in kept]

        # ---- render ----
        recs_scores = [(id_to_rec[sr["id"]], float(sr["score"])) for sr in ranked]

        if self.bside_render == "grouped":
            # Session/date grouped render. The context_size budget is not applied in grouped
            #   (the default experiment uses ctx0=unlimited). If budget is on (>0), fall back to flat.
            if self.context_size > 0:
                rendered, items = self._render_flat_with_budget(recs_scores)
            else:
                rendered, items = _format_grouped(
                    recs_scores, self.bside_group_order, self.bside_group_score_agg
                )
        else:
            # flat (same as the original) — budget applied.
            rendered, items = self._render_flat_with_budget(recs_scores)

        return RetrievalResult(
            items=items,
            rendered=rendered or "No relevant memory!",
            extra={"route": "MEM0_BSIDE"},
            metrics=metrics,
        )

    def _render_flat_with_budget(
        self, recs_scores: list[tuple[MemoryRecord, float]]
    ) -> tuple[str, list[Item]]:
        """Original mem0 flat render + context_size budget (same as the original retrieve)."""
        items: list[Item] = []
        rendered = ""
        approx_tokens = self._p_qa_tokens
        for rec, sc in recs_scores:
            line = f"- {rec.text}\n"
            tk = count_tokens(line)
            if self.context_size > 0 and approx_tokens + tk > self.context_size:
                break
            rendered += line
            items.append(
                Item(text=rec.text, meta={"id": rec.id, "session_idx": rec.session_idx,
                                          "date": rec.date}, score=float(sc))
            )
            approx_tokens += tk
        return rendered.strip(), items
