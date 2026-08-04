"""Faithful reimplementation of mem0's official hybrid scoring (mem0/utils/scoring.py + lemmatization.py).

Source: https://github.com/mem0ai/mem0/blob/b55c51e0049f6fb8dbf4f39b3405e80de905ab22/mem0/utils/scoring.py
      (lemmatize: .../mem0/utils/lemmatization.py, caller: .../mem0/memory/main.py:_search_vector_store)

Official algorithm (score_and_rank):
  combined = min((semantic + bm25 + entity_boost) / max_possible, 1.0)
  - candidates with semantic < threshold (=0.1) are gated out before the hybrid combination.
  - max_possible = 1.0 (+1.0 if bm25, +0.5 if entity). entity weight = 0.5.
  - sort by combined descending -> top_k.
  get_bm25_params: (midpoint, steepness) by query length. normalize_bm25: logistic sigmoid.

Our adaptation (1:1 with the official _search_vector_store):
  - semantic = cosine similarity (official vector_store.search score).
  - bm25 raw = BM25Okapi.get_scores (replaces the official vector_store.keyword_search raw_score) -> normalize_bm25.
  - entity_boosts = {} - our mem0 does not use graph memory, so there are no query_entities/graph
    (the official code also sets entity_boosts={} when there are no query_entities; has_entity=False -> max_possible=2.0).
  - threshold = 0.1 (official _search_vector_store default).
"""
from __future__ import annotations

import math
from typing import Any

# Constants copied from the official scoring.py.
ENTITY_BOOST_WEIGHT = 0.5

# One-time spaCy lemmatizer load cache (mirrors the official lemmatization.py get_nlp_lemma).
_NLP: Any = None


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize for BM25 matching (as in the official mem0/utils/lemmatization.py).

    Extract lemmas with spaCy (lowercase, excluding punct/stop). For -ing forms, if the
    surface form differs from the lemma, also add the surface form (noun/verb ambiguity,
    e.g. meeting/meet). If spaCy is unavailable, fall back to the original text (as in the official code).
    """
    global _NLP
    if _NLP is None:
        try:
            import spacy

            # tagger/attribute_ruler suffice for lemmas - disable parser/ner for efficiency.
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            _NLP = False  # load-failure marker -> always fall back to the original text.
    if not _NLP:
        return text

    doc = _NLP(text.lower())
    tokens: list[str] = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        # If the surface form ends in -ing and differs from the lemma, also add the surface form
        # (handles noun/verb ambiguity - as in the official code).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)
    return " ".join(tokens)


def get_bm25_params(query: str, *, lemmatized: str | None = None) -> tuple[float, float]:
    """BM25 sigmoid parameters by query length (as in the official scoring.py).

    Longer queries produce larger raw BM25 scores, so midpoint/steepness are adjusted.
    Returns: (midpoint, steepness).
    """
    if lemmatized is None:
        lemmatized = lemmatize_for_bm25(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize raw BM25 to [0,1] via a logistic sigmoid (as in the official scoring.py)."""
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def score_and_rank(
    semantic_results: list[dict[str, Any]],
    bm25_scores: dict[str, float],
    entity_boosts: dict[str, float],
    threshold: float,
    top_k: int,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """As in the official mem0/utils/scoring.py score_and_rank.

    Per candidate: semantic_score = result["score"]. Gate out (before combining) if semantic < threshold.
    combined = min((semantic + bm25 + entity_boost) / max_possible, 1.0).
    max_possible = 1.0 (+1.0 if bm25, +0.5 if entity). Sort by combined descending, take top_k.
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: list[dict[str, Any]] = []

    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        semantic_score = result.get("score") or 0.0
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)

        scored_result = {
            "id": mem_id_str,
            "score": combined,
            "payload": result.get("payload"),
        }
        if explain:
            scored_result["score_details"] = {
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "entity_boost": entity_boost,
                "raw_score": raw_combined,
                "max_possible_score": max_possible,
                "final_score": combined,
                "threshold": threshold,
            }
        scored.append(scored_result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
