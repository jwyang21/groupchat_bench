"""SQuAD-style generative answer scoring for the multiple-choice QA formats.

Used for the generative QA scoring mode (the multi_unstructured /
open_ended question formats), where the model sees no (A)-(E) labels and
answers in free text. The free-form answer is matched against each option
with the SQuAD official token-overlap F1 (Rajpurkar et al. 2016, EMNLP,
D16-1264), and the argmax-F1 option is selected.

Functions:
  normalize_answer     : SQuAD official normalization (lowercase -> remove
                         punctuation -> remove articles a/an/the -> collapse
                         whitespace).
  f1_score             : SQuAD official token-level F1 (precision/recall
                         harmonic mean).
  exact_match_score    : SQuAD official exact match (equal after
                         normalization).
  match_option_squad   : map a free-form prediction onto one option index
                         via argmax F1.

No dataset-specific token hardcoding - the same scoring applies to every
dataset.
"""
from __future__ import annotations

import re
import string
from collections import Counter

# Components of the SQuAD official normalize_answer.
_PUNC = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str | None) -> str:
    """SQuAD official normalize_answer.

    Order (identical to the SQuAD official evaluate-v1.1):
      white_space_fix(remove_articles(remove_punc(lower(s))))
    """
    if not s:
        return ""
    s = s.lower()                                          # lowercase
    s = "".join(ch for ch in s if ch not in _PUNC)        # remove punctuation
    # remove articles (a/an/the)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())                            # collapse whitespace


def f1_score(prediction: str, ground_truth: str) -> float:
    """SQuAD official token-level F1.

    Token-overlap precision/recall harmonic mean after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    # If either side has no tokens, there is no overlap (num_same=0), so F1 is
    # 0.
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """SQuAD official exact match (equal after normalization)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def match_option_squad(pred: str | None, options: list[str]) -> dict:
    """Map a free-form prediction onto one of `options` via SQuAD F1.

    Compute f1_score + exact_match against every option (including the (E)
    option), then:
      1) select the argmax-F1 option.
      2) resolve F1 ties by EM=True first, then the lower index.
      3) if every option has F1<=0 (no token overlap with any option),
         return matched_idx=-1 (no match). The prediction matched no option,
         so scoring it as any option would be a gift/penalty; instead it
         counts as wrong. (E) is not auto-assigned.

    Returns a dict:
        matched_idx  : index of the selected option (-1 if no match or
                       options is empty).
        matched_text : text of the selected option ("" if matched_idx == -1).
        f1_scores    : per-option F1 scores (rounded to 6 decimals, for audit).
        em_flags     : per-option exact-match flags.
    """
    if not options:
        return {
            "matched_idx": -1,
            "matched_text": "",
            "f1_scores": [],
            "em_flags": [],
        }
    f1s = [f1_score(pred or "", o) for o in options]
    ems = [exact_match_score(pred or "", o) for o in options]
    # No option shares any token with the prediction -> no match (wrong; do not
    # auto-assign (E)).
    if max(f1s) <= 1e-12:
        return {"matched_idx": -1, "matched_text": "",
                "f1_scores": [round(x, 6) for x in f1s], "em_flags": ems}
    # argmax F1, tie-break: EM True first -> lower index.
    best_idx = 0
    for i in range(1, len(options)):
        if f1s[i] > f1s[best_idx] + 1e-12:
            best_idx = i
        elif abs(f1s[i] - f1s[best_idx]) <= 1e-12:
            # Tie -> prefer EM=True (replace only when the current best is not
            # EM and i is EM).
            if ems[i] and not ems[best_idx]:
                best_idx = i
    return {
        "matched_idx": best_idx,
        "matched_text": options[best_idx],
        "f1_scores": [round(x, 6) for x in f1s],
        "em_flags": ems,
    }
