"""Token counting / text trimming.

Uses `tiktoken cl100k_base` as a *common counter*, ignoring per-provider tokenizer
differences (for approximate context-length control only). Exact cost accounting should
use the usage metadata from the LLM response.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _gpt_tokenizer():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count (cl100k_base)."""
    return len(_gpt_tokenizer().encode(text))


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """If text exceeds max_tokens, return it trimmed from the front."""
    # If max_tokens <= 0 (e.g. a budget calculation went negative), return an empty string.
    #   Previously len(ids) <= -1 was False, so enc.decode(ids[:-1]) returned nearly the whole
    #   text minus one token, bypassing trimming and triggering a vLLM over-length 400.
    if max_tokens <= 0:
        return ""
    enc = _gpt_tokenizer()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


def apply_token_limit(text: str, limit: int) -> str:
    """Apply a token limit (e.g. to the final QA prompt), unifying on/off and upper bound.

    limit <= 0 -> token limit OFF: return the text unchanged (no limit; serving max-model-len is the real cap).
    limit > 0  -> token limit ON:  trim to `limit` tokens via trim_text_to_tokens(text, limit).

    The final QA prompt length limit is toggled and bounded by config (context_size).
    context_size <= 0 means off. Since trim_text_to_tokens returns an empty string when
    max_tokens <= 0 (over-length guard), it must not be called directly; the unlimited case
    is branched through this helper.
    """
    if limit <= 0:
        return text
    return trim_text_to_tokens(text, limit)
