"""Step 1 - raw GroupChat-Bench pickles -> text inputs (questions, dialogues, dates).

For every show and trial this samples one question per session from the raw question pool and
cuts the session dialogue at a random utterance (the moment the question is asked).

Sampling rules (all seeded by the trial index, so the output is deterministic):
  * question type ratio TKG-1hop : TKG-2hop : fan-quiz = 1 : 1 : 2 (largest-remainder rounding,
    each session only receives a type it actually has questions for);
  * within a type, the correct-answer letter is balanced (A..E used equally), then the least
    used question template is preferred, then a seeded random tie-break;
  * the questioner is a random character other than the assistant (Ross / Sheldon / Michael).

Inputs  : {raw_dir}/{show}_groupchat_bench.pickle
Outputs : {data_root}/{show}/date.json                                   list[str], one per session
          {data_root}/{show}/{dt}/entire_session.json                    list[str], full dialogues
          {data_root}/{show}/{dt}/precomputed/per-trial/t{N}/qa.json     question/option/true_answer/
                                                                         true_answer_choice/questioner/tkg_hop
          .../t{N}/partial_dialogues.json                                dialogue up to the question
          {data_root}/{show}/normal/precomputed/per-trial/t{N}/qa_provenance.json
                                                                         where each question came from
                                                                         (used by step 5, oracle)
`dt` is each of normal / newname / shuffled; newname and shuffled are the normal text with the
characters renamed (see common.rename). The answer letter is never renamed.

Usage:
  python preprocess/step1_build_text.py --raw-dir /path/to/groupchat_bench --data-root data
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from common import (CHARACTER_NAMES, CHATBOT_BY_SHOW, DATA_TYPES, OPTION_CHOICES, SHOWS,
                    TRIALS, load_pickle, parse_list, rename, save_json, show_root, trial_dir)

RATIO = (1, 1, 2)  # tkg1 : tkg2 : fan-quiz

HARD_Q_TYPES_1HOP = ("past", "cur", "fu")
HARD_Q_TYPES_2HOP = ("past_past", "past_cur", "past_fu", "cur_past", "cur_cur", "cur_fu",
                     "fu_past", "fu_cur", "fu_fu")
FAN_TYPES = ("ans_w_time", "ans_wo_time", "before_event_unans", "dont_know_unans",
             "dont_know_unans_time")


# --------------------------------------------------------------------------- name capitalization
def _build_name_pattern(names: list[str]) -> re.Pattern:
    variants: list[str] = []
    for n in names:
        for v in (n.lower(), n.upper(), n):
            if v not in variants:
                variants.append(v)
    return re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(v) for v in variants)
                      + r")(?![A-Za-z])", flags=re.IGNORECASE)


_NAME_PATTERNS = {s: _build_name_pattern(n) for s, n in CHARACTER_NAMES.items()}


def capitalize_names(text: str, show: str) -> str:
    """`normal` setting: write every main character name as 'Name' (raw text mixes cases)."""
    return _NAME_PATTERNS[show].sub(lambda m: m.group(0).capitalize(), text)


# --------------------------------------------------------------------------- dialogue splitting
def split_utterances(script: str) -> list[str]:
    """Split a session script into utterances ('Speaker: text').

    Header lines ('Teleplay: ', 'Story: ') are skipped; a line without ':' continues the
    previous utterance.
    """
    out: list[str] = []
    for line in script.split("\n"):
        if not line.strip():
            continue
        if "Teleplay: " in line or "Story: " in line:
            continue
        if ":" in line:
            out.append(line.strip())
        elif out:
            out[-1] = out[-1] + "\n" + line.strip()
    return out


# --------------------------------------------------------------------------- question pools
def _has_valid_questioner(q: dict, chatbot: str) -> bool:
    for k in (q or {}).get("questions", {}):
        if k.lower() != "default" and chatbot.lower() not in k.lower():
            return True
    return False


def _valid_items(bucket: Any, chatbot: str) -> list[tuple[Any, dict]]:
    """(raw_key, question) pairs that have at least one valid questioner.

    raw_key is the list index (list buckets) or the dict key (dict buckets); step 5 uses it to
    look up the question's oracle timestamps.
    """
    if isinstance(bucket, dict):
        pairs = list(bucket.items())
    elif isinstance(bucket, list):
        pairs = list(enumerate(bucket))
    else:
        return []
    return [(k, q) for k, q in pairs if isinstance(q, dict) and _has_valid_questioner(q, chatbot)]


def _eligible_types(hard_q: dict, easy_q: dict, chatbot: str) -> set[str]:
    types: set[str] = set()
    if any(_valid_items(hard_q.get(k), chatbot) for k in HARD_Q_TYPES_1HOP):
        types.add("tkg1")
    if any(_valid_items(hard_q.get(k), chatbot) for k in HARD_Q_TYPES_2HOP):
        types.add("tkg2")
    if any(_valid_items(easy_q.get(k), chatbot) for k in FAN_TYPES):
        types.add("fan")
    return types


def _target_counts(total: int) -> dict[str, int]:
    """Split `total` sessions into tkg1/tkg2/fan by RATIO (largest-remainder rounding)."""
    keys = ("tkg1", "tkg2", "fan")
    raw = [total * r / sum(RATIO) for r in RATIO]
    floors = [int(x) for x in raw]
    out = dict(zip(keys, floors))
    fracs = sorted(enumerate(x - f for x, f in zip(raw, floors)), key=lambda t: -t[1])
    for idx, _ in fracs[: total - sum(floors)]:
        out[keys[idx]] += 1
    return out


def _assign_types(eligibilities: list[set[str]], targets: dict[str, int],
                  rng: random.Random) -> list[str]:
    """Give each session one type it is eligible for, getting as close to `targets` as possible."""
    order = list(range(len(eligibilities)))
    rng.shuffle(order)
    remaining = dict(targets)
    assigned = [""] * len(eligibilities)
    for sidx in order:
        elig = eligibilities[sidx]
        if not elig:
            continue
        best = max(remaining.get(t, 0) for t in elig)
        chosen = rng.choice(sorted(t for t in elig if remaining.get(t, 0) == best))
        if remaining.get(chosen, 0) <= 0:
            chosen = rng.choice(sorted(elig))
        assigned[sidx] = chosen
        remaining[chosen] = max(0, remaining.get(chosen, 0) - 1)
    return assigned


# --------------------------------------------------------------------------- template signature
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|"
           "december")
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b(?:" + _MONTHS + r")\s+\d{1,2}(?:st|nd|rd|th)?"
    r"\s*[-–—]\s*(?:(?:" + _MONTHS + r")\s+)?\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s*\d{4})?\b"
    r"|\b(?:" + _MONTHS + r")(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s*\d{4})?\b"
    r"|\b\d{4}\b"
    r"|\b(?:" + _WEEKDAYS + r")\b",
    re.IGNORECASE,
)
_ADJ_TIME_RE = re.compile(r"<time>(?:[\s,\-]+<time>)+")
_TIME_PREP_RE = re.compile(r"\b(?:on|in|during|around|at)\s+<time>", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+(?:st|nd|rd|th)?\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _template_signature(q: dict) -> str:
    """Question text with dates/numbers masked: questions with the same signature are near-duplicates."""
    questions = (q or {}).get("questions", {}) or {}
    text = questions.get("default")
    if not text:
        text = next((questions[k] for k in sorted(questions) if questions[k]), "")
    if not text:
        return ""
    s = _DATE_RE.sub("<time>", text)
    s = _ADJ_TIME_RE.sub("<time>", s)
    s = _TIME_PREP_RE.sub("<time>", s)
    s = _NUM_RE.sub("<num>", s)
    return _WS_RE.sub(" ", s).strip().lower()


def _answer_letter(q: dict) -> str:
    options = q.get("options", [])
    target = (q.get("answer") or "").lower()
    if not target or len(options) < 5:
        return ""
    for letter, opt in zip(OPTION_CHOICES, options):
        if opt.lower() == target:
            return letter
    return ""


def _pick_question(candidates: list[tuple[dict, str, Any]], sig_usage: dict[str, int],
                   letter_usage: dict[str, int], rng: random.Random):
    """Least-used answer letter first, then least-used template, then a seeded tie-break."""
    keyed = [(_template_signature(q), _answer_letter(q), q, t, rk) for q, t, rk in candidates]
    valid = [k for k in keyed if k[1]]
    if valid:
        low = min(letter_usage.get(k[1], 0) for k in valid)
        pool = [k for k in valid if letter_usage.get(k[1], 0) == low]
    else:
        pool = keyed
    low_sig = min(sig_usage.get(k[0], 0) for k in pool)
    sig, letter, q, t, rk = rng.choice([k for k in pool if sig_usage.get(k[0], 0) == low_sig])
    sig_usage[sig] = sig_usage.get(sig, 0) + 1
    if letter:
        letter_usage[letter] = letter_usage.get(letter, 0) + 1
    return deepcopy(q), t, rk


def _pick_questioner(q: dict, chatbot: str, rng: random.Random) -> str:
    keys = (q or {}).get("questions", {})
    cands = [k for k in keys if k.lower() != "default" and chatbot.lower() not in k.lower()]
    return rng.choice(cands) if cands else ""


# --------------------------------------------------------------------------- one trial
def build_trial(raw: dict, show: str, trial: int):
    """Return (qa, partial_dialogues, provenance, dates, entire_sessions) for one trial (normal)."""
    chatbot = CHATBOT_BY_SHOW[show]
    rng = random.Random(trial)
    sessions = [(ep, sc) for ep in raw for sc in raw[ep]]

    eligibilities = [_eligible_types(raw[e][s].get("hard_q", {}) or {},
                                     raw[e][s].get("easy_q", {}) or {}, chatbot)
                     for e, s in sessions]
    types = _assign_types(eligibilities, _target_counts(len(sessions)), rng)

    qa: dict[str, list] = {k: [] for k in ("question", "option", "true_answer",
                                            "true_answer_choice", "questioner", "tkg_hop")}
    partials: list[str] = []
    provenance: list[dict] = []
    dates: list[str] = []
    entire: list[str] = []
    sig_usage: dict[str, int] = {}
    letter_usage = {c: 0 for c in OPTION_CHOICES}

    for sidx, (ep, sc) in enumerate(sessions):
        sess = raw[ep][sc]
        utts = split_utterances(sess["script"])
        assert utts, f"session {sidx} ({ep}/{sc}) has an empty transcript"
        hq, eq = sess.get("hard_q", {}) or {}, sess.get("easy_q", {}) or {}

        qtype = types[sidx]
        if qtype == "tkg1":
            cands = [(q, t, rk) for t in HARD_Q_TYPES_1HOP for rk, q in _valid_items(hq.get(t), chatbot)]
            category, oracle_kind, hop = "hard_q", "tkg", 1
        elif qtype == "tkg2":
            cands = [(q, t, rk) for t in HARD_Q_TYPES_2HOP for rk, q in _valid_items(hq.get(t), chatbot)]
            category, oracle_kind, hop = "hard_q", "tkg", 2
        elif qtype == "fan":
            cands = [(q, t, rk) for t in FAN_TYPES for rk, q in _valid_items(eq.get(t), chatbot)]
            category, oracle_kind, hop = "easy_q", "fan", -1
        else:
            raise AssertionError(f"session {sidx} ({ep}/{sc}) has no eligible question")

        q, subbucket, raw_key = _pick_question(cands, sig_usage, letter_usage, rng)
        questioner = _pick_questioner(q, chatbot, rng)
        anchor = rng.randrange(len(utts))  # the question is asked right after this utterance
        text = q.get("questions", {}).get(questioner, "")
        options = list(q.get("options", []))
        letter = _answer_letter(q)
        assert questioner and text and len(options) == 5 and letter, \
            f"session {sidx} ({ep}/{sc}): malformed question"

        qa["question"].append(capitalize_names(text, show))
        qa["option"].append([capitalize_names(o, show) for o in options])
        qa["true_answer"].append(capitalize_names(options[OPTION_CHOICES.index(letter)], show))
        qa["true_answer_choice"].append(letter)
        qa["questioner"].append(capitalize_names(questioner, show))
        qa["tkg_hop"].append(hop)
        partials.append("".join(capitalize_names(u, show) + "\n" for u in utts[: anchor + 1]).strip())
        provenance.append({"session_idx": sidx, "episode": ep, "session_num": sc,
                           "question_type": qtype, "tkg_hop": hop, "category": category,
                           "subbucket": subbucket, "raw_key": raw_key,
                           "oracle_kind": oracle_kind, "questioner": questioner})
        dates.append(sess["date"])
        entire.append("".join(capitalize_names(u, show) + "\n" for u in utts).strip())
    return qa, partials, provenance, dates, entire


def rename_qa(qa: dict, show: str, dt: str) -> dict:
    out = dict(qa)
    for key in ("question", "true_answer", "questioner"):
        out[key] = [rename(x, show, dt) for x in qa[key]]
    out["option"] = [[rename(o, show, dt) for o in opts] for opts in qa["option"]]
    return out  # true_answer_choice / tkg_hop are unchanged


def build_show(raw_dir: Path, data_root: Path, show: str, trials: list[int],
               data_types: list[str]) -> None:
    raw = load_pickle(raw_dir / f"{show}_groupchat_bench.pickle")
    root = show_root(data_root, show)
    for trial in trials:
        qa, partials, prov, dates, entire = build_trial(raw, show, trial)
        save_json(dates, root / "date.json")  # identical for every trial
        save_json(prov, trial_dir(data_root, show, "normal", trial) / "qa_provenance.json")
        for dt in data_types:
            tdir = trial_dir(data_root, show, dt, trial)
            dt_partials = [rename(p, show, dt) for p in partials]
            save_json(rename_qa(qa, show, dt), tdir / "qa.json")
            save_json(dt_partials, tdir / "partial_dialogues.json")
            save_json([rename(e, show, dt) for e in entire], root / dt / "entire_session.json")
        c = Counter(p["question_type"] for p in prov)
        n = len(prov)
        print(f"[{show} t{trial}] sessions={n} tkg1={c['tkg1']} tkg2={c['tkg2']} fan={c['fan']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="folder holding {show}_groupchat_bench.pickle")
    ap.add_argument("--data-root", required=True, type=Path, help="output data root")
    ap.add_argument("--shows", default=",".join(SHOWS))
    ap.add_argument("--trials", default=",".join(map(str, TRIALS)))
    ap.add_argument("--data-types", default=",".join(DATA_TYPES))
    args = ap.parse_args()
    for show in parse_list(args.shows):
        build_show(args.raw_dir, args.data_root, show, parse_list(args.trials, int),
                   parse_list(args.data_types))


if __name__ == "__main__":
    main()
