"""Helper to auto-discover and load the project root's .env.

Principles:
  - Never hardcode absolute paths. Traverse upward from the calling file to find the
    first directory containing a .env (the groupchat_qa project root).
  - Never put secrets like OPENAI_API_KEY in yaml; inject them from .env instead.

Usage (right after imports in the run entry points):
    from groupchat_qa.utils import load_project_env
    load_project_env()  # sets OPENAI_API_KEY etc. as environment variables
"""
from __future__ import annotations

import os
from pathlib import Path


def find_project_env(start: Path | str | None = None) -> Path | None:
    """Traverse upward from the current file (or `start`) and return the .env path of the first directory containing `.env`.

    Args:
        start: directory to start searching from. If None, uses this module's location (`env_utils.py`).
               Pass `__file__` to search relative to the caller's location.

    Returns: the Path to `.env`, or None if not found.
    """
    if start is None:
        start_path = Path(__file__).resolve().parent
    else:
        start_path = Path(start).resolve()
        if start_path.is_file():
            start_path = start_path.parent
    for p in [start_path] + list(start_path.parents):
        env = p / ".env"
        if env.is_file():
            return env
    return None


def load_project_env(start: Path | str | None = None, override: bool = False) -> Path | None:
    """Auto-discover `.env`, then call `load_dotenv`. Returns None if not found.

    Args:
        start    : where to start searching. If None, uses this module's location.
        override : if True, overwrite already-set env vars with values from .env (default False).

    Returns: the loaded .env path, or None. Logging is the caller's responsibility.
    """
    env_path = find_project_env(start)
    if env_path is None:
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv not installed — parse manually instead.
        _manual_load_env(env_path, override=override)
        return env_path
    load_dotenv(env_path, override=override)
    return env_path


def _manual_load_env(env_path: Path, override: bool = False) -> None:
    """Used when python-dotenv is not installed. Simple KEY=VALUE parsing."""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if override or k not in os.environ:
                os.environ[k] = v
