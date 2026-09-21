"""Token counting / text trimming.

Uses `tiktoken cl100k_base` as a *common counter*, ignoring per-provider
tokenizer differences (for approximate context-length control only). Exact
cost accounting should use the usage metadata from the LLM response.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _gpt_tokenizer():
    """Return the cl100k_base tokenizer, or None if tiktoken is not installed.

    tiktoken is an *optional* dependency (see requirements.txt): when it is
    absent we fall back to a ~chars/4 heuristic in count_tokens /
    trim_text_to_tokens. This keeps the package importable and runnable
    without tiktoken (used only for approximate context-length control,
    never for exact cost accounting).
    """
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count.

    Uses cl100k_base; ~chars/4 heuristic if tiktoken absent.
    """
    enc = _gpt_tokenizer()
    if enc is None:
        # Heuristic fallback: cl100k tokens are roughly len(text)/4 chars. Good
        # enough for
        # the approximate context-length control this counter is used for.
        return (len(text) + 3) // 4 if text else 0
    return len(enc.encode(text))


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """If text exceeds max_tokens, keep the leading max_tokens tokens."""
    # If max_tokens <= 0 (e.g. a budget calculation went negative), return an
    # empty string.
    # Previously len(ids) <= -1 was False, so enc.decode(ids[:-1]) returned
    # nearly the whole
    # text minus one token, bypassing trimming and triggering a vLLM
    # over-length 400.
    if max_tokens <= 0:
        return ""
    enc = _gpt_tokenizer()
    if enc is None:
        # Heuristic (tiktoken absent): ~4 chars per token; trim by character
        # budget.
        if count_tokens(text) <= max_tokens:
            return text
        return text[: max_tokens * 4]
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def apply_token_limit(text: str, limit: int) -> str:
    """Apply a token limit (e.g. to the final QA prompt), unifying on/off
    and upper bound.

    limit <= 0 -> token limit OFF: return the text unchanged (no limit;
        serving max-model-len is the real cap).
    limit > 0  -> token limit ON:  trim to `limit` tokens via
        trim_text_to_tokens(text, limit).

    The final QA prompt length limit is toggled and bounded by config
    (context_size). context_size <= 0 means off. Since trim_text_to_tokens
    returns an empty string when max_tokens <= 0 (over-length guard), it must
    not be called directly; the unlimited case is branched through this
    helper.
    """
    if limit <= 0:
        return text
    return trim_text_to_tokens(text, limit)
