"""Question format converter — render the same question/options in three ways.

  - multi_structured : the default multiple-choice form (options A-E listed explicitly)
  - multi_unstructured: "{question} {opt_a}? {opt_b}? ... or {opt_e}?"
  - open_ended       : "{question}" (options removed)

This converts the same data (multi-structured form from the DialSim trial pickle) by
changing *only the format*, on the fly, without generating a separate pickle.
"""
from __future__ import annotations

from groupchat_qa.utils.types import Question

OPTION_LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")


def render_question(q: Question, format: str = "multi_structured") -> str:
    """Question -> string, per format.

    Args:
        q     : Question object
        format: 'multi_structured' / 'multi_unstructured' / 'open_ended'

    Returns: the final question string including the questioner prefix.
          e.g. "Joey: So, what does Mrs. bing do?\n\t(A) Butt double\n\t(B) ..."
    """
    fmt = format.lower().replace("-", "_")
    if fmt == "multi_structured":
        return _multi_structured(q)
    if fmt == "multi_unstructured":
        return _multi_unstructured(q)
    if fmt == "open_ended":
        return _open_ended(q)
    raise ValueError(
        f"Unknown question format: {format!r}. "
        f"Expected: multi_structured / multi_unstructured / open_ended."
    )


def _multi_structured(q: Question) -> str:
    """`questioner: text\n\t(A) ...\n\t(B) ...` (the default form)."""
    out = f"{q.questioner}: {q.text}\n"
    for i, opt in enumerate(q.options):
        # If there are more than 5 options, OPTION_LABELS[i] would IndexError, so generate
        #   labels dynamically ((F),(G)..) as a guard (DialSim has 5, but this avoids a crash
        #   on malformed/extended data).
        label = OPTION_LABELS[i] if i < len(OPTION_LABELS) else f"({chr(65 + i)})"
        out += f"\t{label} {opt}\n"
    return out.strip()


def _multi_unstructured(q: Question) -> str:
    """`questioner: text {opt_a}? {opt_b}? ... or {opt_e}?` (options phrased as natural-language questions)."""
    if len(q.options) < 2:
        return _open_ended(q)
    opts = list(q.options)
    head = " ".join(f"{o}?" for o in opts[:-1])
    tail = f"or {opts[-1]}?"
    return f"{q.questioner}: {q.text} {head} {tail}".strip()


def _open_ended(q: Question) -> str:
    """`questioner: text` (options removed entirely)."""
    return f"{q.questioner}: {q.text}".strip()
