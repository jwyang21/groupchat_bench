"""Small helpers imported across multiple files."""
from __future__ import annotations

import re

import numpy as np


# Cosine similarity
def cos_sim(vec1, vec2) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for a zero vector."""
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(v1, v2) / denom)
    # NaN/inf guard: a corrupted embedding (NaN/inf components) can have a
    # non-zero
    # denom, so it passes the guard above and leaks a NaN score. NaN
    # comparisons
    #   are all False during sorting, which breaks top-k non-deterministically.
    # Coerce non-finite scores to 0.0 (finite scores unchanged -> no
    # reproducibility impact).
    return sim if np.isfinite(sim) else 0.0


# String cleanup
def remove_quotes(text: str) -> str:
    """Remove quotes/backticks."""
    return text.replace("'", "").replace('"', "").replace("`", "")


def remove_brackets(text: str) -> str:
    """Remove brackets."""
    return "".join(c for c in text if c not in "(){}[]")


def get_clean_entity(entity: str) -> str:
    """Normalize an entity - lowercase + remove whitespace/punctuation.

    Enables case-insensitive matching and unifies
    spacing/hyphen/underscore/quote variants of a multi-word entity (e.g.
    "Paul the Wine Guy") into the same form. Used by the relational-memory
    retrieval for query-node matching.
    """
    return (
        entity.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace('"', "")
        .replace("'", "")
    )


# Multiple-choice response parsing
_OPTION_TOKENS = [
    "(A)", "(B)", "(C)", "(D)", "(E)",
    "(a)", "(b)", "(c)", "(d)", "(e)",
    "{A}", "{B}", "{C}", "{D}", "{E}",
    "{a}", "{b}", "{c}", "{d}", "{e}",
]


def process_answer(model_answer: str, options: list[str] | None = None) -> str:
    """Extract and normalize only the (A)-(E) label from the model's
    free-form response.

    Args:
        model_answer: raw model output.
        options: optional list of option texts. When provided, and no label
                 is found, attempt one more recovery via option-text
                 matching (e.g. the model output the option body / "I don't
                 know" instead of a label).
                 If omitted (for caller compatibility), option matching is
                 skipped.

    Policy (with long-form/free-text fallback):
      1. If a "final answer is" / "answer is" / "answer:" / "the answer"
         keyword exists, look only after it (strong signal).
      2. Collect all (A)-(E) labels (including lowercase / brace variants)
         found in that segment.
      3. If at most one label was collected, return it as is.
      4. If two or more labels were collected, return only the LAST (X)
         match in the original segment.
      5. "Option A" / "answer is A" single-letter keyword fallback.
      6. If the whole response is a bare single letter (e.g. "E", "E.",
         "B)"), return that label.
      7. (when options given) If the response exactly matches some option
         text, return that label; for "I don't know" / "none of the above"
         cases, return the label of the option holding that phrase (usually
         (E)).
      8. If nothing is found, return an empty string.
    """
    # None guard (makes the .lower()/.strip() calls below safe).
    model_answer = model_answer or ""
    text = model_answer
    # 1. Strong keyword split - segment after the last match.
    for kw in ("final answer is", "the answer is", "answer is", "answer:"):
        if kw in text.lower():
            text = text.lower().split(kw)[-1].strip()
            break
    # Preserve the segment (before replacing option tokens) for the
    # multi-option fallback.
    segment = text
    # 2. Collect option tokens.
    out = ""
    for tok in _OPTION_TOKENS:
        if tok in text:
            out += tok.upper()
            text = text.replace(tok, "")
    # 3. Single option -> return as is.
    if len(out) <= 3:
        if out:
            return out
    else:
        # 4. Multi-option - take only the last (X) in the segment (not raw
        # model_answer, to avoid reasoning-on bias).
        last = None
        for m in re.finditer(r"[\({][A-Ea-e][\)}]", segment):
            last = m.group(0)
        if last:
            return f"({last[1].upper()})"

    # 5. Fallback - "Option A" / "answer is A" / single letter.
    low = model_answer.lower()
    for letter in ("a", "b", "c", "d", "e"):
        for pat in (
            f"option {letter}",
            f"answer is {letter}",
            f"answer: {letter}",
        ):
            if pat in low:
                return f"({letter.upper()})"

    # 6. Whole response is a bare single letter (e.g. "E", "E.", "B)", "(C") -
    # case missed by token matching.
    m_bare = re.fullmatch(r"\(?([A-Ea-e])[\)\.\:]?", model_answer.strip())
    if m_bare:
        return f"({m_bare.group(1).upper()})"

    # 7. (when options given) Option-text matching - recover when the model
    # output the option body / "I don't know" instead of a label.
    if options:
        def _norm(s: str) -> str:
            return str(s).strip().lower().rstrip(". ")
        raw_norm = _norm(model_answer)
        # 7a. Response exactly matches some option text -> that label (no
        # partial match, to avoid false positives).
        if raw_norm:
            for i, opt in enumerate(options):
                if raw_norm == _norm(opt):
                    return f"({chr(65 + i)})"
        # 7b. "I don't know" / "none of the above" -> label of the option
        # holding that phrase (usually (E)).
        # But if an explicit letter signal ((X) / "it's C" / "answer is C" /
        # "option C") is also present,
        # it is a mixed answer ("I don't know ... but it's C"), so do not
        # overwrite with (E).
        has_explicit_letter = bool(
            re.search(r"[\({][A-Ea-e][\)}]", model_answer)
            or re.search(
                r"\b(?:it'?s|answer is|option)\s+[A-Ea-e]\b",
                model_answer,
                re.IGNORECASE,
            )
        )
        _neg = any(
            neg in raw_norm for neg in ("don't", "do not", "dont")
        )
        is_idk = (not has_explicit_letter) and (
            "none of the above" in raw_norm
            or ("know" in raw_norm and _neg))
        if is_idk:
            for i, opt in enumerate(options):
                ol = str(opt).lower()
                _idk_opt = any(
                    p in ol
                    for p in ("none of the above", "don't know", "do not know")
                )
                if _idk_opt:
                    return f"({chr(65 + i)})"

    # Reaching here means single-label disambiguation failed. `out` is either
    # "" or multiple
    # labels ("(A)(B)" etc.); returning multiple labels would never match
    # true_answer ("(X)")
    # and would pollute the degenerate-analysis buckets. Return a single label
    # (len<=3) only, else "".
    return out if len(out) <= 3 else ""


# Record validity
_PRONOUNS = {
    "he", "his", "him", "she", "her", "i",
    "my", "me", "you", "your", "man", "woman",
}


def is_valid_record(record: dict | object) -> bool:
    """Validity policy for a record.

    1) Drop cyclic records where head == tail.
    2) Drop records whose head or tail is a bare pronoun.
    """
    if isinstance(record, dict):
        head = record.get("head", "")
        tail = record.get("tail", "")
    else:
        head = getattr(record, "head", "")
        tail = getattr(record, "tail", "")
    # If head/tail is an explicit None (e.g. LLM JSON null), .lower() would
    # raise
    #   AttributeError and abort the whole commit. Coerce None/non-str to "".
    h, t = str(head or "").lower().strip(), str(tail or "").lower().strip()
    if h == t:
        return False
    if h in _PRONOUNS or t in _PRONOUNS:
        return False
    return True


# Prompt instruction split
def extract_prompt_instruction(prompt: str, delimiter: str) -> str:
    """Return the part before the delimiter (instruction only)."""
    return prompt.split(delimiter)[0].strip()
