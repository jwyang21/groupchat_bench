"""Unified file I/O helpers (pickle / yaml / prompt loading)."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------
# Local-first read-path resolution (optional shared-storage load reduction).
#   If files on shared storage are pre-copied to a faster local disk, this hook can
#   redirect reads to the local copy. Set GROUPCHAT_QA_SHARED_ROOT and
#   GROUPCHAT_QA_LOCAL_ROOT to enable; otherwise reads always use the original path.
#   Read-only: writes always go to the original for consistency.
_SHARED_ROOT = os.getenv("GROUPCHAT_QA_SHARED_ROOT", "")
_LOCAL_ROOT = os.getenv("GROUPCHAT_QA_LOCAL_ROOT", "")
def local_first(fname: "str | Path") -> Path:
    # Always return a Path (consistent type): local Path if a local copy exists, else the original.
    #   Returning str on a local hit but a Path on fallback would split types and break callers that
    #   call .read_text()/.exists()/.stat() on the result. np.load/open/load_pickle accept both.
    if _SHARED_ROOT and _LOCAL_ROOT:
        p = str(fname)
        if p.startswith(_SHARED_ROOT):
            lp = _LOCAL_ROOT + p[len(_SHARED_ROOT):]
            if os.path.exists(lp):
                return Path(lp)
    return Path(fname)


# ----------------------------------------------------------------------------
# Pickle
# ----------------------------------------------------------------------------
def load_pickle(fname: str | Path) -> Any:
    # Load from the local copy if present (avoiding a shared-storage read). Only embedding
    # files are copied locally, so everything else falls back to the original.
    with open(local_first(fname), "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, fname: str | Path) -> None:
    """Atomic write — tmp file + os.replace.

    Even if interrupted by SIGKILL / OOM / ENOSPC, the destination file is not truncated
    (only the tmp file is partial; on success, os.replace does an atomic rename).
    """
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    tmp = fname.with_suffix(fname.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fname)
    except Exception:
        # Clean up the tmp file.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------
# YAML config
# ----------------------------------------------------------------------------
def load_yaml(fname: str | Path) -> dict:
    """Load a yaml file into a dict — simple fallback for environments without sconf."""
    import yaml

    with open(fname, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    *paths: str | Path,
    default: str | Path | None = None,
    argv: list[str] | None = None,
):
    """sconf-based config loader (used in place of hydra).

    Args:
        *paths : yaml files in merge order (later ones take precedence).
        default: bundled default (e.g. `assets/configs/base.yaml`).
        argv   : `sys.argv[1:]` form. If None, no CLI override is applied.

    Returns: an sconf.Config object supporting both dict-style and attr-style access, e.g.:
        cfg = load_config(args.config, default=BASE_CONFIG)
        cfg.argv_update(left_argv)
        print(cfg['memory']['name'])  # or cfg.memory.name
    """
    from sconf import Config

    cfg = Config(*paths, default=default) if default is not None else Config(*paths)
    if argv:
        cfg.argv_update(argv)
    return cfg


# ----------------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------------
def load_prompt(prompt_name: str, prompt_dir: str | Path | None = None) -> str:
    """Read one prompt file from the prompt directory.

    Args:
        prompt_name: with or without extension
                     (e.g. "qa_records.multi-unstructured.txt" or "qa_dual").
        prompt_dir : if None, uses `groupchat_qa.assets.PROMPT_DIR` (the repo's
                     top-level `prompts/` directory).
    """
    if prompt_dir is None:
        from groupchat_qa.assets import PROMPT_DIR
        prompt_dir = PROMPT_DIR
    prompt_dir = Path(prompt_dir)
    fname = prompt_name if prompt_name.endswith(".txt") else f"{prompt_name}.txt"
    path = prompt_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
