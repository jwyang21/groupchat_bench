"""Runtime immutable assets: prompt templates and the bundled default config.

Prompt templates live in the repository's top-level ``prompts/`` directory (see the
README structure diagram); ``PROMPT_DIR`` resolves there so that
``load_prompt("qa_records.txt")`` works out of the box under the documented
``PYTHONPATH=src python ...`` invocation from the repo root. If a top-level
``prompts/`` directory is not found (e.g. when the package is installed as a wheel
without it), it falls back to a ``prompt/`` directory bundled next to this file.

``CONFIG_DIR`` points at the package-bundled ``configs/`` holding ``base.yaml`` — the
internal default that each user-facing config in the repo's top-level ``configs/``
directory is merged on top of.
"""
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
# assets -> groupchat_qa -> src -> <repo root>
REPO_ROOT = ASSETS_DIR.parents[2]

_TOP_LEVEL_PROMPTS = REPO_ROOT / "prompts"
_BUNDLED_PROMPTS = ASSETS_DIR / "prompt"

PROMPT_DIR = _TOP_LEVEL_PROMPTS if _TOP_LEVEL_PROMPTS.is_dir() else _BUNDLED_PROMPTS
CONFIG_DIR = ASSETS_DIR / "configs"
