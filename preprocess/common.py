"""Shared helpers for the GroupChat-Bench preprocessing pipeline.

Contents:
  * character-name tables and the three name settings (normal / newname / shuffled),
  * small JSON / pickle I/O helpers (atomic JSON writes),
  * the canonical on-disk layout used by the experiment loader.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

SHOWS = ("friends", "bigbang", "theoffice")
DATA_TYPES = ("normal", "newname", "shuffled")
TRIALS = (0, 1, 2)
OPTION_CHOICES = ["(A)", "(B)", "(C)", "(D)", "(E)"]

# Every per-LLM artifact lives under this budget segment (the loader reads `b06000`).
BUDGET_SEGMENT = "b06000"

# Main characters per show (used to capitalize names in the `normal` setting).
CHARACTER_NAMES: dict[str, list[str]] = {
    "friends":   ["Monica", "Ross", "Joey", "Chandler", "Rachel", "Phoebe"],
    "bigbang":   ["Howard", "Leonard", "Raj", "Penny", "Bernadette", "Amy", "Sheldon"],
    "theoffice": ["Dwight", "Jim", "Pam", "Ryan", "Michael"],
}

# The character the assistant plays in each show (never chosen as a questioner).
CHATBOT_BY_SHOW = {"friends": "Ross", "bigbang": "Sheldon", "theoffice": "Michael"}

# `newname`: every main character gets an unrelated common name.
NEW_NAME_MAP: dict[str, dict[str, str]] = {
    "friends":   {"Monica": "Patricia", "Chandler": "James", "Joey": "John",
                  "Phoebe": "Jennifer", "Rachel": "Linda", "Ross": "Nate"},
    "bigbang":   {"Howard": "Robert", "Leonard": "James", "Raj": "Michael",
                  "Penny": "Jennifer", "Bernadette": "Linda", "Amy": "Patricia",
                  "Sheldon": "John"},
    "theoffice": {"Dwight": "John", "Jim": "Robert", "Pam": "Jennifer",
                  "Ryan": "William", "Michael": "James"},
}

# `shuffled`: main characters swap names with each other.
SHUFFLE_MAP: dict[str, dict[str, str]] = {
    "friends":   {"Monica": "Joey", "Chandler": "Rachel", "Joey": "Monica",
                  "Phoebe": "Ross", "Rachel": "Chandler", "Ross": "Phoebe"},
    "bigbang":   {"Howard": "Amy", "Leonard": "Howard", "Raj": "Penny",
                  "Penny": "Raj", "Bernadette": "Sheldon", "Amy": "Leonard",
                  "Sheldon": "Bernadette"},
    "theoffice": {"Dwight": "Ryan", "Jim": "Michael", "Pam": "Dwight",
                  "Ryan": "Jim", "Michael": "Pam"},
}

NAME_MAPS = {"newname": NEW_NAME_MAP, "shuffled": SHUFFLE_MAP}


def rename(text: str, show: str, data_type: str) -> str:
    """Apply a name setting to (capitalized) text.

    Two-step substring replacement (name -> NAME placeholder -> mapped name), so swaps such as
    Rachel<->Chandler do not chain. Matching is case-sensitive on the capitalized name; this is
    exactly the rule used to build the released data, so it must not be changed.
    `normal` returns the text unchanged.
    """
    if data_type == "normal":
        return text
    mapping = NAME_MAPS[data_type][show]
    for name in mapping:
        text = text.replace(name, name.upper())
    for name in mapping:
        text = text.replace(name.upper(), mapping[name])
    return text


# --------------------------------------------------------------------------- layout
def show_root(data_root: Path, show: str) -> Path:
    return Path(data_root) / show


def trial_dir(data_root: Path, show: str, data_type: str, trial: int) -> Path:
    return show_root(data_root, show) / data_type / "precomputed" / "per-trial" / f"t{trial}"


def per_llm_dir(data_root: Path, show: str, data_type: str, trial: int, llm: str) -> Path:
    return trial_dir(data_root, show, data_type, trial) / "per-llm" / llm / BUDGET_SEGMENT


def entire_records_path(data_root: Path, show: str, data_type: str, llm: str) -> Path:
    return (show_root(data_root, show) / data_type / "precomputed" / "common" / "openie"
            / llm / BUDGET_SEGMENT / "entire_session_records_raw.json")


def partial_records_path(data_root: Path, show: str, data_type: str, trial: int,
                         llm: str) -> Path:
    return (per_llm_dir(data_root, show, data_type, trial, llm) / "openie"
            / "partial_session_records_raw.json")


def query_nodes_path(data_root: Path, show: str, data_type: str, trial: int, llm: str,
                     k: int = 3) -> Path:
    return per_llm_dir(data_root, show, data_type, trial, llm) / f"query_node_raw_nq{k}.json"


# --------------------------------------------------------------------------- I/O
def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    """Atomic JSON write (temp file + rename), so an interrupted run never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def load_pickle(path: Path) -> Any:
    """Load a released GroupChat-Bench pickle. Only load files from the official download:
    unpickling runs code, so never point this at untrusted files."""
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_list(arg: str, cast=str) -> list:
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]
