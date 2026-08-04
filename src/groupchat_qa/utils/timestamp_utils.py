"""Extract / classify / parse time expressions.

The fanquiz question timestamp supports a **date range** format:
  - same-month  : "September 22-23, 1994"
  - cross-month : "September 23-October 15, 2007"
  - (cross-year "Y/M/D-Y/M/D" ranges do not occur in the data and are unsupported)
For a range, `extract_timestamps` **expands every included day into its own "Month D, Year"
string** and returns them. Each day is then classified by classify_timestamp as
'month_day_year' and filtered by the existing exact-match logic in `_date_matches_any`
(experiment code unchanged).
"""
import calendar
import re
from datetime import date, datetime, timedelta

_MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_MONTHS_ALT = (r"January|February|March|April|May|June|July|"
               r"August|September|October|November|December")
_MONTH_TO_NUM = {m.lower(): i for i, m in enumerate(
    ["", "January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]) if m}

_PATTERN_MONTH_DAY_YEAR = re.compile(rf"\b{_MONTH} \d{{1,2}}, \d{{4}}\b")
_PATTERN_MONTH_YEAR = re.compile(rf"\b{_MONTH} \d{{4}}\b")
_PATTERN_YEAR = re.compile(r"(?<!\S)\d{4}(?!\s*bucks\b)")

_PATTERN_MDY_FULL = re.compile(rf"^{_MONTH} \d{{1,2}}, \d{{4}}$")
_PATTERN_MY_FULL = re.compile(rf"^{_MONTH} \d{{4}}$")

# Date range: "Month D - [Month] D, YYYY" (same-month + cross-month), with a single trailing
# year. The connector is only hyphen/en-dash/em-dash - word connectors like
# "to"/"and" are excluded because they risk false matches in question prose (and the data uses hyphens only).
_RANGE_PAT = re.compile(
    rf"\b(?P<m1>{_MONTHS_ALT})\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?"
    r"\s*[---]\s*"
    rf"(?:(?P<m2>{_MONTHS_ALT})\s+)?"
    r"(?P<d2>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s*(?P<y>\d{4}))",
    re.IGNORECASE,
)


def _enumerate_range_days(y: int, m1: int, d1: int, m2: int, d2: int) -> list[str]:
    """Expand the range [m1/d1 .. m2/d2] (year y) into a list of "Month D, Year" day strings.

    For cross-month, iterate day by day with datetime. If m2 < m1, the year wraps around (y+1).
    Returns an empty list if the date is invalid (e.g. leap-year edge cases).
    """
    y2 = y + 1 if m2 < m1 else y
    try:
        _, max_d1 = calendar.monthrange(y, m1)
        _, max_d2 = calendar.monthrange(y2, m2)
        start = date(y, m1, min(d1, max_d1))
        stop = date(y2, m2, min(d2, max_d2))
    except ValueError:
        return []
    if stop < start:
        return []
    out: list[str] = []
    cur = start
    while cur <= stop:
        out.append(f"{calendar.month_name[cur.month]} {cur.day}, {cur.year}")
        cur += timedelta(days=1)
    return out


def extract_timestamps(q: str) -> list[str]:
    """Extract all timestamps from a question string.

    1) Match date ranges (fanquiz) first -> expand every included day into "Month D, Year".
    2) Then match single patterns (month_day_year / month_year / year_only) - skip any that overlap a range span.
    Overlap avoidance: process from the most specific pattern first, skipping matches that overlap an already-taken span.
    """
    used_spans: list[tuple[int, int]] = []

    def _overlap(span: tuple[int, int]) -> bool:
        for u in used_spans:
            if not (span[1] <= u[0] or span[0] >= u[1]):
                return True
        return False

    out: list[str] = []
    # 1) Ranges first (most specific) - expand into individual days.
    for m in _RANGE_PAT.finditer(q):
        if _overlap(m.span()):
            continue
        y = int(m.group("y"))
        m1 = _MONTH_TO_NUM[m.group("m1").lower()]
        d1 = int(m.group("d1"))
        m2 = _MONTH_TO_NUM[(m.group("m2") or m.group("m1")).lower()]
        d2 = int(m.group("d2"))
        days = _enumerate_range_days(y, m1, d1, m2, d2)
        if days:
            out.extend(days)
            used_spans.append(m.span())
    # 2) Single date patterns (avoiding range spans).
    for pat in (_PATTERN_MONTH_DAY_YEAR, _PATTERN_MONTH_YEAR, _PATTERN_YEAR):
        for m in pat.finditer(q):
            if not _overlap(m.span()):
                out.append(m.group())
                used_spans.append(m.span())
    return out


def classify_timestamp(t: str) -> str:
    """Classify a timestamp's type - 'month_day_year' / 'month_year' / 'year_only'.

    A day expanded from a range is in "Month D, Year" form, so it classifies as 'month_day_year'.
    """
    if _PATTERN_MDY_FULL.match(t):
        return "month_day_year"
    if _PATTERN_MY_FULL.match(t):
        return "month_year"
    return "year_only"


def parse_timestamp(timestamp: str) -> datetime:
    """Parse a timestamp string into a datetime, trying the formats in order."""
    for fmt in ("%B %d, %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid timestamp format: {timestamp}")


# For the "question embedding type" axis - generate the non-temporal question.
#   - temporal     = embed the whole question
#   - non_temporal = remove each timestamp plus the one time-preposition word before it
#                    (on/in/by, ...) and strip. If the question has no timestamp, the result
#                    equals the original question (non_temporal == temporal).
# Time prepositions commonly preceding a timestamp; matched on word boundaries.
# `at` is time/place ambiguous but included since time expressions like "at 1994" are common.
_TIME_PREP_WORDS = (
    "on", "in", "by", "at", "of", "during", "from", "after", "before",
    "since", "until", "till", "around", "about",
)
_TIME_PREP_TRAILING = re.compile(
    rf"\b(?:{'|'.join(_TIME_PREP_WORDS)})\s*$",
    re.IGNORECASE,
)


def _collect_timestamp_spans(q: str) -> list[tuple[int, int]]:
    """Collect all timestamp spans (start, end) from a question string.

    Applies the same patterns as `extract_timestamps` (range + 3 singles), except a range
    yields its whole match span once (not per expanded day).
    """
    spans: list[tuple[int, int]] = []
    used: list[tuple[int, int]] = []

    def _overlap(span: tuple[int, int]) -> bool:
        for u in used:
            if not (span[1] <= u[0] or span[0] >= u[1]):
                return True
        return False

    for m in _RANGE_PAT.finditer(q):
        if _overlap(m.span()):
            continue
        spans.append(m.span())
        used.append(m.span())
    for pat in (_PATTERN_MONTH_DAY_YEAR, _PATTERN_MONTH_YEAR, _PATTERN_YEAR):
        for m in pat.finditer(q):
            if not _overlap(m.span()):
                spans.append(m.span())
                used.append(m.span())
    return spans


def strip_question_timestamp(q: str) -> str:
    """Remove every timestamp plus the one word before it from question q, then clean whitespace.

    If there is no timestamp, return the original (strip + collapsing repeated whitespace is always applied).
    """
    if not q:
        return q
    spans = _collect_timestamp_spans(q)
    if not spans:
        return re.sub(r"\s+", " ", q).strip()
    # Remove from back to front to keep earlier indices valid.
    spans_sorted = sorted(spans, key=lambda s: s[0], reverse=True)
    out = q
    for start, end in spans_sorted:
        before = out[:start]
        after = out[end:]
        # Always remove the single word immediately before the timestamp, regardless of part of speech.
        #   "the word immediately before" = the last whitespace-separated token (\S+) of `before`;
        #   if there is none (empty/whitespace), remove nothing.
        stripped_before = before.rstrip()
        m = re.search(r"\S+\s*$", stripped_before)
        if m:
            new_before = stripped_before[: m.start()]
        else:
            new_before = before
        out = new_before + after
    # Collapse repeated whitespace / punctuation adjacency.
    out = re.sub(r"\s+", " ", out)
    # Clean whitespace before punctuation (e.g. " , " -> " ", keep " ?").
    out = re.sub(r"\s+([?!.,;:])", r"\1", out)
    # Remove leading/trailing punctuation/whitespace garbage.
    # "In 1994, what..." -> removing "In 1994" leaves ", what..." -> strip the ", " prefix.
    out = re.sub(r"^[\s,;:.]+", "", out)
    # Dedup adjacent commas ("June 1994, on July 1995, what" -> "June 1994, July 1995, what").
    out = re.sub(r",\s*,+", ",", out)
    # Normalize spacing after commas.
    out = re.sub(r"\s*,\s*", ", ", out)
    return out.strip()
