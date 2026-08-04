"""Match a generative answer to a multiple-choice option (SQuAD normalize + rapidfuzz token_set_ratio).

Used for scoring the official-generative QA mode. The model's free-form answer (gen)
is normalized, then compared against each normalized option:
  1) exact match (equal after normalization) -> that option
  2) otherwise, the option with the highest token_set_ratio (argmax)
  3) if fuzzy_floor > 0 and the best score < floor, fall back to (E) (the last option = None of the above)

Per-option scores are returned as well, so one can review after the experiment
which answer matched which option and why.

Normalization follows the SQuAD official `normalize_answer` family (general - no dataset-specific token hardcoding).
"""
from __future__ import annotations

import re
import string

from rapidfuzz import fuzz

# Components of the SQuAD official normalize_answer
_PUNC = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str | None) -> str:
    """SQuAD official normalize_answer: lowercase -> remove punctuation -> remove articles (a/an/the) -> normalize whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNC)  # remove punctuation
    s = _ARTICLES.sub(" ", s)                         # remove articles
    return " ".join(s.split())                        # normalize whitespace (split -> join)


def match_option(gen: str | None, options: list[str], fuzzy_floor: float = 0.0) -> dict:
    """Map the generative answer `gen` to one of `options`.

    Args:
        gen        : the model's free-form answer.
        options    : list of multiple-choice option texts (by convention, the last = (E) "I don't know. (None of the above)").
        fuzzy_floor: if the best token_set_ratio (0-100) is below this value, fall back to (E) (the last option).
                     0 (default) disables fallback -> always argmax ((E) competes on equal footing).

    Returns a dict:
        matched_idx  : selected option index (-1 if options is empty)
        matched_text : selected option text
        match_type   : 'exact' | 'fuzzy' | 'floor_fallback' | 'no_options'
        gen_norm     : normalized gen
        scores       : [{option, option_norm, score}] - token_set_ratio for every option (for post-hoc review)
    """
    g = normalize_answer(gen)
    if not options:
        return {"matched_idx": -1, "matched_text": "", "match_type": "no_options",
                "gen_norm": g, "scores": []}

    norm_opts = [normalize_answer(o) for o in options]
    # Always compute scores for all options (kept for review, even on an exact match).
    scores = [float(fuzz.token_set_ratio(g, o)) for o in norm_opts]
    score_payload = [{"option": options[k], "option_norm": norm_opts[k], "score": scores[k]}
                     for k in range(len(options))]

    # 1) exact match (equal after normalization). No exact match if gen is empty.
    if g:
        for i, o in enumerate(norm_opts):
            if g == o:
                return {"matched_idx": i, "matched_text": options[i], "match_type": "exact",
                        "gen_norm": g, "scores": score_payload}

    # 2) fuzzy token_set_ratio best match (ties -> earliest index).
    best_idx = max(range(len(options)), key=lambda i: scores[i])
    match_type = "fuzzy"

    # 3) floor fallback -> (E) (the last option).
    if fuzzy_floor > 0 and scores[best_idx] < fuzzy_floor:
        best_idx = len(options) - 1
        match_type = "floor_fallback"

    return {"matched_idx": best_idx, "matched_text": options[best_idx], "match_type": match_type,
            "gen_norm": g, "scores": score_payload}
