"""run_groupchat_mem.py — grid / resume entry point for the proposed groupchat-mem method.

The config's top-level key is the method name, selected via `--method <name>` (use
`groupchat_mem` for the clean configs). Each method section holds `llms` (list) +
`grid` (dict of axis -> values). Every (LLM x grid combo) combination is executed.

Core policy:
- `data.per_llm_precomputed: true` -> use per-LLM cache (per-llm/{llm} route/record cache + {ie_type}/...).
  On cache miss, raise FileNotFoundError immediately (assistant == QA model policy).
- `data.session_filter`: all / even / odd (0-based — even = idx 0,2,4,...).
- `data.data_type`: normal / newname / shuffled. `data.ie_type`: openie / closedie / gt.
  Set via yaml edit or sconf argv override (`--data.data_type newname`).

Grid axis naming:
- "seed"            -> run section
- "data.{key}"      -> data section override (e.g., data.ie_type)
- "memory.{key}"    -> memory section override
- otherwise (e.g., "hop", "context_size") -> memory section (backward compat)

Parallelism: cfg.run.parallel_max_workers (default = len(llms_list)). Each LLM uses
its own vLLM port, so there is no GPU contention.

Checkpoints:
- Every 100 sessions, overwrite-save `.ckpt.grid_NNN.{llm}.ckpt.pkl` per grid.
- After a grid's final result (`{llm}.grid_NNN.pkl`) is saved, delete that ckpt.
- After all grids finish, clean up residual ckpts in bulk.

result_dir:
- Used as-is if `run.result_dir` is set.
- Otherwise composed dynamically as
  `./result/ablation/{method}/{dataset}.{data_type}.t{trial}.{ie_type}`,
  so re-running with a different dataset/data_type never overwrites.

Example:
  python src/run/run_groupchat_mem.py --config configs/groupchat_mem/<...>.yaml --method groupchat_mem --llms <model>
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util
# Load the baseline entry module (run_baseline.py) by file path and reuse its shared helpers
# (run_one_llm, _make_llm_from_cfg, _check_vllm_endpoint, aggregation, atomic JSON save). The
# memory factory routes the groupchat_mem name to its class and delegates other names unchanged.
_baseline_path = Path(__file__).with_name("run_baseline.py")
_spec = importlib.util.spec_from_file_location("_run_baseline", _baseline_path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

from sconf import Config

from groupchat_qa.embedder import CachedEmbedder, make_embedder
from groupchat_qa.pipeline.data_loader import load_converted_dataset, load_query_nodes_for
from groupchat_qa.utils import save_pickle, seed_everything, load_project_env

# Auto-discover .env (override=True: the .env's real key takes priority even if a launcher
# exports OPENAI_API_KEY=EMPTY).
load_project_env(start=__file__, override=True)


# ============================================================================
# Helpers
# ============================================================================
def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of grid axes. Empty grid -> [{}] (= 1 combo, no override)."""
    if not grid: return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*[grid[k] for k in keys])]


def _apply_overrides(base_cfg: dict, overrides: dict, default_target: str = "memory") -> dict:
    """Route each axis to its target section dynamically.

    - "seed"          -> run.seed
    - "data.{key}"    -> data.{key}
    - "memory.{key}"  -> memory.{key}
    - "run.{key}"     -> run.{key}
    - otherwise (no dot) -> default_target.{key}  (backward compat — default "memory")
    """
    # Use copy.deepcopy: a json round-trip would coerce int keys -> str, tuple -> list, and
    # silently lose non-JSON types (Path, datetime, set). Safe even if a nested dict value is present.
    out = copy.deepcopy(base_cfg)
    # Guard against a typo creating an unknown section (silent fail) — whitelist known sections.
    _KNOWN_SECTIONS = {"run", "data", "memory", "embedder", "llms"}
    for k, v in overrides.items():
        if k == "seed":
            out.setdefault("run", {})["seed"] = v
        elif "." in k:
            section, key = k.split(".", 1)
            if section not in _KNOWN_SECTIONS and section not in out:
                raise KeyError(f"Override section {section!r} unknown (key={k!r}). Allowed: {_KNOWN_SECTIONS}")
            out.setdefault(section, {})[key] = v
        else:
            out.setdefault(default_target, {})[k] = v
    return out


def _find_best(rows: list[dict]) -> dict | None:
    """Best accuracy across LLM x combo. Excludes error / n_sessions=0 rows.
    On ties, prefer the cell with larger n_scored (or n_sessions) — avoids oversell where a
    small sample's high accuracy wins."""
    if not rows: return None
    valid = [r for r in rows if not r.get("error") and r.get("n_sessions", 0) > 0]
    if not valid: return None
    best = max(valid, key=lambda r: (r["accuracy"], r.get("n_sessions", 0)))
    return {"llm": best["llm"], "combo": best["combo"], "accuracy": best["accuracy"]}


def _apply_answer_budget(llm_cfg: dict, answer_budget: int | None) -> dict:
    """Return a copy of llm_cfg with the answer budget (= reasoning budget for the final QA
    inference) injected.

    The answer budget is the reasoning limit at answer-generation time, independent of the
    cache budget (which cache to load).

    - answer_budget=None -> return llm_cfg as-is (no copy) = existing behavior. The answer-time
      reasoning budget stays whatever llm_cfg.extra_body.thinking_token_budget was (or unset).
    - answer_budget=positive int -> shallow-copy, overwrite extra_body.thinking_token_budget with
      that value, and if max_tokens is below answer_budget + 6384 (content headroom), raise it so
      the budget is not truncated by max_tokens.

    Note: thinking_token_budget is meaningless for enable_thinking=false (non-thinking) models —
    the server ignores it, but to avoid confusion we only overwrite the budget when
    enable_thinking=true.
    """
    if answer_budget is None or answer_budget == "":
        return llm_cfg
    try:
        ab = int(answer_budget)
    except (TypeError, ValueError) as e:
        raise ValueError(f"qa.answer_budget must be None or a positive integer, got {answer_budget!r}") from e
    if ab <= 0:
        raise ValueError(f"qa.answer_budget must be a positive integer, got {ab}")

    new_cfg = dict(llm_cfg)  # Shallow copy — rebuild only extra_body deeply (avoid mutating original).
    extra_body = dict(new_cfg.get("extra_body") or {})
    ctk = dict(extra_body.get("chat_template_kwargs") or {})
    # Skip budget injection for models where enable_thinking is explicitly False (meaningless + confusing).
    if ctk.get("enable_thinking") is False:
        logging.warning(
            f"[ablation] qa.answer_budget={ab} set, but LLM "
            f"{new_cfg.get('name', new_cfg.get('model_name', '?'))} has enable_thinking=false — "
            f"skipping budget injection."
        )
        return llm_cfg
    # reasoning_effort models (e.g., gpt-oss/harmony) have no thinking_token_budget knob.
    #   Injecting a budget makes the server return 400 (BadRequest: thinking_token_budget is set
    #   but reasoning_config is not configured) -> answer fails. reasoning_effort controls the
    #   reasoning amount instead, so skip budget injection.
    if "reasoning_effort" in ctk:
        logging.warning(
            f"[ablation] qa.answer_budget={ab} set, but LLM "
            f"{new_cfg.get('name', new_cfg.get('model_name', '?'))} is a reasoning_effort model "
            f"(e.g., gpt-oss, no thinking_token_budget knob) — skipping budget injection."
        )
        return llm_cfg
    extra_body["thinking_token_budget"] = ab
    new_cfg["extra_body"] = extra_body
    # If max_tokens is below budget + headroom, raise it (prevents budget truncation).
    # A None/unset max_tokens is treated as 0 -> raised to needed (guards int(None) TypeError).
    _needed = ab + 6384
    _mt = new_cfg.get("max_tokens")
    if _mt is None or int(_mt) < _needed:
        new_cfg["max_tokens"] = _needed
    return new_cfg


def _write_grid_map(out_dir: Path, combos: list[dict]) -> None:
    """Save the grid_idx->combo mapping to out_dir/grid_map.json (once at run start).
    Lets you read straight from the result directory what each grid_NNN configures
    (retrieval_method / top_k / etc.). Best-effort — failure does not stop the experiment
    (grid_combo is also stored in each grid pkl summary). Ablation-specific (simulate has no
    grid concept, so no sync needed)."""
    try:
        gm = {str(i): dict(c) for i, c in enumerate(combos)}
        # Atomic write (tmp+replace) — prevents partial reads even when several LLM procs share
        #   the same out_dir (as in top-k) and write concurrently (same combos -> same content).
        _tmp = out_dir / "grid_map.json.tmp"
        _tmp.write_text(json.dumps(gm, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(_tmp, out_dir / "grid_map.json")
        # Only grid_map.json is kept (no README). The grid_idx->combo mapping is read from
        #   grid_map.json plus each {llm}.grid_NNN.json's summary.grid_combo.
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the experiment
        logging.warning(f"[ablation] failed to write grid_map (ignored): {e}")


def _compose_default_out_dir(method_cfg: dict, method_name: str) -> Path:
    """Compose a dynamic dataset/data_type/trial/ie_type path when `run.result_dir` is unset."""
    data_cfg = method_cfg.get("data", {})

    # Path(<str>).name -> the last component of the path string (the final directory or file
    #   name, i.e., the single piece after the last '/').
    # dataset: the TV show title.
    dataset = Path(str(data_cfg.get("dataset_root", "unknown"))).name

    # {tv_show}/{data_type}/{trial_idx}/{ie_type}
    parts = [
        dataset,
        str(data_cfg.get("data_type", "unknown")),
        f"t{int(data_cfg.get('trial', 0))}",
        str(data_cfg.get("ie_type", "openie")),
    ]

    # return result directory
    return Path(f"./result/ablation/{method_name}") / ".".join(parts)


def _in_scope_fn(session_filter: str):
    """Predicate over session_idx matching session_filter (all/even/odd). 0-based (even=0,2,4,...)."""
    _sf = str(session_filter or "all").lower()
    if _sf == "even":
        return lambda si: si % 2 == 0
    if _sf == "odd":
        return lambda si: si % 2 == 1
    return lambda si: True  # all


def _dataset_n_sessions(data_cfg: dict) -> int | None:
    """Total session count of the dataset (len of entire_session.json). Computed dynamically —
    never hardcode a number like 394. Combined with session_filter this yields the "expected
    scored session count" independent of dataset (friends 788 -> even 394, bigbang 805 -> even 403,
    theoffice 2347 -> even 1174; all -> the full count). Returns None on load failure."""
    try:
        root = data_cfg.get("dataset_root")
        dt = data_cfg.get("data_type", "")
        if not root:
            return None
        es = Path(root) / dt / "entire_session.json"
        obj = json.loads(es.read_text())
        return len(obj) if isinstance(obj, list) else len(obj.get("sessions", obj))
    except Exception:
        return None


def _expected_scope_sessions(data_cfg: dict, session_filter: str) -> set[int] | None:
    """Set of expected (to-be-scored) session idxs = range(n_total) filtered by session_filter.
    Dynamic. Returns None on failure."""
    n = _dataset_n_sessions(data_cfg)
    if n is None:
        return None
    ins = _in_scope_fn(session_filter)
    return {i for i in range(n) if ins(i)}


# ============================================================================
# Main runner
# ============================================================================
def run_ablation(method_cfg: dict[str, Any], method_name: str, llm_names: list[str] | None = None,
                 grid_start: int = 0, grid_end: int | None = None,
                 rerun_missing: bool = False, rerun_scope: str = "config") -> dict[str, Any]:
    """
    Run ablation for each config combination

    Args:
        method_cfg: config values set in the yaml for the current method (overridden by CLI).
        method_name: name of the current method (here the proposed method name: groupchat_mem).
        llm_names: list of LLM names to use in the ablation.

    Returns:
        Summarized result
    """
    # Options to sweep, one axis per config key (e.g. hyperparameters such as num_records / num_sessions).
    grid = method_cfg.get("grid", {})
    # Non-rectangular grid support: if grid_combos is present, use that explicit combo list as-is
    #   (bypassing the cartesian product). Use it to specify exactly the combinations to run.
    #   Each combo is a dict whose keys are _apply_overrides dotted paths.
    explicit_combos = method_cfg.get("grid_combos")
    if explicit_combos is not None:
        if not isinstance(explicit_combos, list) or not all(isinstance(c, dict) for c in explicit_combos):
            raise ValueError(
                f"method {method_name!r}: grid_combos must be a list of dicts (got {type(explicit_combos).__name__})"
            )
        if not explicit_combos:
            raise ValueError(f"method {method_name!r}: grid_combos is empty")
        combos = [dict(c) for c in explicit_combos]
    else:
        # cartesian product of all options to be tested (e.g., ['openie', True], ['openie', False], ['closedie', True], ..., ['gt', False])
        combos = _expand_grid(grid)

    # If the yaml has "llms:" -> a list of dicts (each dict is the config for one LLM).
    # If "llms:" is absent or empty -> wrap the single dict method_cfg["llm"] in a list.
    llms_list = method_cfg.get("llms", [])
    if not llms_list:
        raise ValueError(f'"llms" must be defined under method {method_name!r} in the config yaml')

    # User passes --llms llm1,llm2 -> llm_names = [llm1, llm2].
    # User omits --llms on the CLI -> llm_names = [all LLMs named in the yaml].
    # If the user specified --llms on the CLI:
    if llm_names:

        # Set of unique LLM names the user requested.
        wanted = set(llm_names)

        # Get the names of each LLM defined in the yaml via c.get("name", c["model_name"]):
        #   use the "name" value if present, else fall back to "model_name". c["model_name"] is a
        #   lazy fallback to avoid a KeyError when a per-LLM config dict lacks a "name" key.
        # filtered: config dicts for the yaml-defined LLMs whose names are among those requested.
        filtered = [c for c in llms_list if c.get("name", c["model_name"]) in wanted]

        # missing: requested names with no matching config in the yaml -> error out if any exist.
        missing = wanted - {c.get("name", c["model_name"]) for c in filtered}
        if missing:
            raise ValueError(f"names given in --llms not found in method.llms: {sorted(missing)}")

        # Keep only the config dicts whose names match the user-requested LLMs.
        llms_list = filtered

    # session filter: use only a subset of sessions rather than all sessions in the data.
    sess_filter = method_cfg.get("data", {}).get("session_filter", "")

    # dict of data-related config values.
    data_cfg = method_cfg.get("data", {})

    # value of 'per_llm_precomputed' in the data config, or False if the key is absent.
    per_llm_mode = bool(data_cfg.get("per_llm_precomputed", False))

    # base: config template shared by all experiments.
    # A dict(method_cfg) shallow copy would leave nested dicts like base['data']/['memory'] as the
    # same objects as in method_cfg. _apply_overrides protects via deepcopy, but if code that
    # mutates the nested dicts before that is added, it silently corrupts. Isolate with deepcopy up front.
    base = copy.deepcopy(method_cfg)

    # grid, grid_combos, llms are per-experiment metadata, so remove them from the template.
    # grid: which options to test per hyperparameter -> not a value for a single experiment.
    base.pop("grid", None)
    base.pop("grid_combos", None)  # An explicit combo list is also metadata -> drop from per-run cfg.

    # llms: which LLMs to run -> not per-experiment (one experiment runs a single LLM).
    base.pop("llms", None)

    # get result directory
    explicit = method_cfg.get("run", {}).get("result_dir")

    # NoneType(None), bool(False), number 0(0, 0.0, 0j), empty string(''), empty container([], (), {}, set()) -> falsy in "if"
    out_dir = Path(explicit) if explicit else _compose_default_out_dir(method_cfg, method_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[ablation] result_dir = {out_dir}")

    # Auto-generate grid_map.json — read straight from the result directory what each grid_NNN
    #   configures (grid_idx->combo). Written once at run start from the full grid definition (combos).
    _write_grid_map(out_dir, combos)

    # vLLM endpoint health check.
    # Sequential calls incur N LLM x timeout of cold start; a parallel ping shortens it.
    logging.info(f"[ablation] vLLM endpoint ping (num LLMs={len(llms_list)})")
    with ThreadPoolExecutor(max_workers=max(1, min(len(llms_list), 16))) as _hc_pool:
        _hc_futures = [_hc_pool.submit(_baseline._check_vllm_endpoint, c) for c in llms_list]
        for _f in as_completed(_hc_futures):
            _f.result()   # propagate exceptions

    # ---- create the embedder only once ----
    emb_cfg = method_cfg["embedder"]
    embedder = make_embedder(backend=emb_cfg["backend"], model=emb_cfg["model"])

    # File-backed embedding cache — if `embedder.cache: true`, wrap the base embedder in
    #   CachedEmbedder (lookup -> hit / miss embeds live and stores). Each text is embedded once
    #   and reused. Since the ablation grid may vary data_type, the cache lives at the dataset_root
    #   level (text-keyed, so mixing data_types is safe as keys differ + (model,dim) validated).
    #   Wrapped once and shared across all threads.
    if emb_cfg.get("cache"):
        _cpath = emb_cfg.get("cache_path")
        if not _cpath:
            _safe = embedder.name.replace(":", "_").replace("/", "_")
            _droot = method_cfg.get("data", {}).get("dataset_root", ".")
            _cpath = str(Path(_droot) / f"baseline_emb_cache.{_safe}.pickle")
        embedder = CachedEmbedder(embedder, _cpath)
        logging.info(f"[ablation][embedder] file-backed cache enabled -> {_cpath}")

    # ---- bundle cache (per (data_type, trial, ie_type, normalization, per_llm_name) tuple) ----
    # A single shared_bundle load would silently ignore the grid's data.* axis sweep. Including
    # every data.* axis in the bundle cache key reloads the correct bundle per grid override
    # (no reload on cache hit).
    _bundle_cache: dict[tuple, dict] = {}
    _bundle_cache_lock = threading.Lock()

    def _cache_budget_of(method_cfg_run: dict) -> int | str | None:
        """Extract cache_budget from cfg_run.

        Priority: memory.cache_budget (explicit) -> None (= legacy single layout, backward-compat).
        Budget-partitioned cache — no segment when unset.
        """
        return (method_cfg_run.get("memory") or {}).get("cache_budget")

    def _bundle_key(d_cfg: dict, per_llm_name: str | None, cache_budget) -> tuple:
        return (
            str(d_cfg.get("dataset_root")),
            str(d_cfg.get("data_type", "normal")),
            int(d_cfg.get("trial", 0)),
            str(d_cfg.get("ie_type", "openie")),
            str(d_cfg.get("normalization", "raw")),
            per_llm_name,
            # Include cache_budget in the key — reload the correct bundle if the grid varies budget
            #   (prevents silent stale reuse).
            str(cache_budget),
        )

    def _get_bundle(d_cfg: dict, per_llm_name: str | None, cache_budget=None) -> dict:
        key = _bundle_key(d_cfg, per_llm_name, cache_budget)
        with _bundle_cache_lock:
            if key not in _bundle_cache:
                logging.info(f"[ablation] bundle load key={key}")
                _bundle_cache[key] = load_converted_dataset(
                    dataset_root=d_cfg["dataset_root"],
                    data_type=d_cfg.get("data_type", "normal"),
                    trial_idx=int(d_cfg.get("trial", 0)),
                    ie_type=d_cfg.get("ie_type", "openie"),
                    normalization=d_cfg.get("normalization", "raw"),
                    per_llm_name=per_llm_name,
                    cache_budget=cache_budget,
                )
            return _bundle_cache[key]

    rows: list[dict] = []
    total = len(llms_list) * len(combos)
    rows_lock = threading.Lock()
    counter = {"cur": 0}

    def _run_one_llm_all_grids(llm_cfg: dict) -> None:
        """Grid sweep for a single LLM (inner). Thread-safe.

        When per_llm_mode=true, load the per-LLM bundle once. The LLM instance is created once per LLM.
        Wrapped in try/except so an exception across all grids of one LLM does not affect other LLM
        threads. A failed row is recorded in rows with an error field.
        """
        llm_name = llm_cfg.get("name", llm_cfg["model_name"])
        try:
            return _run_one_llm_all_grids_inner(llm_cfg, llm_name)
        except Exception as e:
            logging.error(f"[ablation] [{llm_name}] thread failed: {type(e).__name__}: {e}", exc_info=True)
            with rows_lock:
                rows.append({"method": method_name, "llm": llm_name, "grid_idx": -1, "combo": {},
                             "grid_combo": {},  # whole-thread failure -> no specific combo, use {}.
                             "accuracy": 0.0, "n_sessions": 0, "total_llm_latency_sec": 0,
                             "total_input_tokens": 0, "total_output_tokens": 0, "wall_sec": 0,
                             "error": f"{type(e).__name__}: {e}"})

    # answer budget — reasoning budget for the final QA inference (independent of cache budget).
    #   If qa.answer_budget is unset, keep llm_cfg's thinking_token_budget (existing behavior).
    _answer_budget = (method_cfg.get("qa") or {}).get("answer_budget")

    def _run_one_llm_all_grids_inner(llm_cfg: dict, llm_name: str) -> None:
        # Create the LLM instance once (the LLM itself is grid-independent).
        # seed: inject run.seed as the request seed (reproducibility for temperature>0). Use base's
        # run.seed (client created once outside the grid loop — run.seed is not a grid axis here).
        # Inject the answer budget (original if unset) — changes only the answer-gen reasoning limit.
        _llm_cfg_eff = _apply_answer_budget(llm_cfg, _answer_budget)
        llm = _baseline._make_llm_from_cfg(_llm_cfg_eff, seed=int(base.get("run", {}).get("seed", 42)))

        for grid_idx, combo in enumerate(combos):
            # Grid range split — process half the grids per proc for slow models to raise endpoint
            #   utilization. grid_idx outside the range belongs to another proc, so skip.
            #   grid_idx / filenames follow enumerate to stay consistent.
            if grid_idx < grid_start or (grid_end is not None and grid_idx > grid_end):
                continue
            with rows_lock:
                counter["cur"] += 1
                cur = counter["cur"]
            # Wrap each grid body in try/except — one cell failing does not stop the rest.
            try:
                # Skip-existing: skip if a result file already exists (preserve prior cell results).
                # Results are saved as json only, so skip-existing prefers json but also honors a pkl
                #   made by an older proc (preserve prior cell results, avoid recomputation).
                _result_file = out_dir / f"{llm_name}.grid_{grid_idx:03d}.json"
                # rerun_missing mode: reprocess only the missing sessions (is_correct=None from
                #   timeout etc.) in the existing json. If the json is absent, this grid is not a
                #   rerun target (an incomplete grid is handled by a normal launch). Skip if none missing.
                _rerun_targets: set[int] | None = None
                if rerun_missing:
                    if not _result_file.exists():
                        continue
                    try:
                        _ex_check = json.loads(_result_file.read_text())
                    except Exception as _ee:
                        logging.warning(f"[ablation rerun] grid={grid_idx} failed to load json, skip: {_ee}")
                        continue
                    # session_filter parity-aware filter: an even-filter config scores only even
                    #   sessions, so odd placeholders (is_correct=None) must not go into _miss (during
                    #   rerun, do_qa = session_idx in rerun_sessions ignores the even filter, which
                    #   would score odd sessions too and pollute n_scored). all-filter targets both
                    #   even and odd.
                    # rerun_scope: "config" = keep the config's session_filter (default),
                    #   "even"/"odd"/"entire(=all)" = explicit override. Rerun even experiments as even,
                    #   all experiments as entire.
                    _cfg_sf = str(method_cfg.get("data", {}).get("session_filter", "all")).lower()
                    _rs = str(rerun_scope or "config").lower()
                    _sf = _cfg_sf if _rs == "config" else ("all" if _rs == "entire" else _rs)
                    _in_scope = _in_scope_fn(_sf)
                    # Missing sessions = expected (scope) sessions − already-scored sessions.
                    #   Expected sessions = range(dataset session count) matching session_filter
                    #   (dynamic; never hardcode numbers like 394 — reused for other datasets:
                    #   bigbang 403 / theoffice 1174 / all = full count). This makes both
                    #   (a) sessions with is_correct=None in records and (b) sessions absent from
                    #   records (crashed before reaching them) rerun targets (previously only (a)).
                    _scored_scope = {
                        r.get("session_idx") for r in _ex_check.get("records", [])
                        if r.get("is_correct") is not None
                        and isinstance(r.get("session_idx"), int)
                        and _in_scope(r.get("session_idx"))
                    }
                    _expected_scope = _expected_scope_sessions(method_cfg.get("data", {}), _sf)
                    if _expected_scope is not None:
                        _miss = _expected_scope - _scored_scope
                    else:
                        # Fallback when the dataset session count cannot be computed: previous method
                        #   (only is_correct=None in records).
                        _miss = {
                            r.get("session_idx") for r in _ex_check.get("records", [])
                            if r.get("is_correct") is None
                            and isinstance(r.get("session_idx"), int)
                            and _in_scope(r.get("session_idx"))
                        }
                    if not _miss:
                        continue  # no missing sessions (n_scored == expected) -> skip
                    _rerun_targets = _miss
                    logging.info(f"[ablation rerun] {method_name} llm={llm_name} grid={grid_idx} reprocessing missing sessions: {sorted(_miss)}")
                else:
                    # Refine skip-existing by "completion" (supports resume after interruption).
                    #   - Old pkl result exists -> treat as complete, skip.
                    #   - json exists -> decide via summary.resume_complete:
                    #       True  = file marked complete by new code -> skip.
                    #       None  = file without the flag (old code wrote json only once at completion
                    #               -> old completed result) -> skip (no regression).
                    #       False = incomplete (interrupted/timeout) partial result from new code ->
                    #               do not skip, resume (run_one_llm below).
                    #   - Corrupt json -> skip conservatively (protect existing result).
                    _pkl_file = out_dir / f"{llm_name}.grid_{grid_idx:03d}.pkl"
                    _skip = False
                    if _result_file.exists():
                        try:
                            _ex_done = json.loads(_result_file.read_text())
                            _rc = (_ex_done.get("summary") or {}).get("resume_complete")
                            if (_rc is None) or (_rc is True):
                                _skip = True   # old or new completed result -> skip
                            else:
                                # rc=False but actually complete (0 missing sessions): every plain
                                #   relaunch would enter resume -> whack-a-mole (0 reprocessing yet
                                #   launched repeatedly, rc never updated so stays False -> a completed
                                #   cell keeps re-running). Using the same logic as rerun-missing,
                                #   compute missing sessions (expected scope − scored scope); if 0,
                                #   confirm completion (atomically mark rc=True) and skip. If any
                                #   missing, resume as before (run_one_llm).
                                _cfg_sf2 = str(method_cfg.get("data", {}).get("session_filter", "all")).lower()
                                _in_scope2 = _in_scope_fn(_cfg_sf2)
                                _scored2 = {
                                    r.get("session_idx") for r in _ex_done.get("records", [])
                                    if r.get("is_correct") is not None
                                    and isinstance(r.get("session_idx"), int)
                                    and _in_scope2(r.get("session_idx"))
                                }
                                _exp2 = _expected_scope_sessions(method_cfg.get("data", {}), _cfg_sf2)
                                if _exp2 is not None and not (_exp2 - _scored2):
                                    (_ex_done.setdefault("summary", {}))["resume_complete"] = True
                                    _tmp = _result_file.with_suffix(".json.rcfix.tmp")
                                    _tmp.write_text(json.dumps(_ex_done))
                                    _tmp.replace(_result_file)   # atomic
                                    logging.info(f"[ablation] {method_name} grid={grid_idx} rc=False but 0 missing sessions -> confirm complete (mark rc=True), skip")
                                    _skip = True
                                # else: missing sessions exist (or expected scope unavailable) -> resume below
                        except Exception as _le:
                            logging.warning(f"[ablation] grid={grid_idx} failed to load json ({_le}) -> conservative skip")
                            _skip = True
                    elif _pkl_file.exists():
                        _skip = True
                    if _skip:
                        logging.info(f"[ablation] {method_name} {cur}/{total} llm={llm_name} grid={grid_idx} — already complete (file exists), skip")
                        continue
                    # Incomplete (resume_complete=False, missing sessions) partial result -> run_one_llm resumes.

                # base is the "default config for this experiment" each time; combo values (cartesian
                # product of grids) are overlaid on top to form the final cfg_run. So base must hold
                # only the default config shared by all experiments.
                cfg_run = _apply_overrides(base, combo)
                seed_everything(int(cfg_run.get("run", {}).get("seed", 42)))

                # Reload the bundle keyed on cfg_run["data"] (no reload on cache hit).
                cfg_data = cfg_run.get("data", data_cfg)
                # cache_budget — which reasoning budget the cache was extracted with (independent of
                #   the answer budget = llm_cfg.extra_body.thinking_token_budget).
                _cache_budget = _cache_budget_of(cfg_run)
                # no_history combo -> the per-llm record/route cache is unnecessary (answer uses only
                #   the no-history prompt). Load the base bundle (per_llm_name=None) so models with
                #   incomplete records also run. commit (KG build) is auto-skipped via run_one_llm's
                #   _skip_commit (kept in sync).
                _no_hist = bool((cfg_run.get("memory") or {}).get("no_history"))
                _bundle_per_llm = None if _no_hist else (llm_name if per_llm_mode else None)
                bundle = _get_bundle(cfg_data, _bundle_per_llm, _cache_budget)
                sessions = bundle["sessions"]
                questions = bundle["questions"]

                # Read sess_filter from cfg_data inside the loop (it may be a grid axis).
                run_cfg = dict(cfg_run.get("run", {}))
                sf_this = cfg_data.get("session_filter", sess_filter)
                if sf_this:
                    run_cfg["session_filter"] = sf_this
                # Separate per-grid prompt logs — add a grid_NNN subdir to log_prompts_dir so multiple
                #   combos do not mix into the same {llm}.prompts.jsonl (each grid's prompts get a
                #   distinct absolute path).
                if run_cfg.get("log_prompts") and run_cfg.get("log_prompts_dir"):
                    run_cfg["log_prompts_dir"] = str(Path(run_cfg["log_prompts_dir"]) / f"grid_{grid_idx:03d}")

                # Rebuild mem_cfg (hop/context_size etc. vary per grid).
                mem_cfg = dict(cfg_run["memory"])
                # For auto-composing the memory EmbeddingCache path.
                # Needs dataset_root / data_type / trial / llm_name / ie_type / normalization.
                mem_cfg["dataset_root"] = cfg_data.get("dataset_root")
                mem_cfg["data_type"] = cfg_data.get("data_type", "normal")
                mem_cfg["trial"] = int(cfg_data.get("trial", 0))
                mem_cfg["llm_name"] = llm_name
                mem_cfg["ie_type"] = cfg_data.get("ie_type", "openie")
                mem_cfg["normalization"] = cfg_data.get("normalization", "raw")
                mem_cfg["embedder_model"] = method_cfg["embedder"]["model"]
                # Pass so the record embedding cache is also read from the same budget segment
                #   (mem_cfg["cache_budget"] is already present from cfg_run["memory"] — copied via
                #    dict(cfg_run["memory"]) so re-injection is not required, but unify with _cache_budget).
                mem_cfg["cache_budget"] = _cache_budget
                # Inject precomputed query nodes (produced by the query-node precompute step, per_llm mode).
                #   key = session_idx (= current_session_idx). If present, skip the query-node LLM call on
                #   the record route.
                # precomputed query nodes — load the per-grid nq{K} cache (K = num_query_nodes).
                #   bundle["query_nodes"] is loaded once per (llm, trial) and is K-independent (default file),
                #   so when the grid varies num_query_nodes over 1/3/5/7/9, each must read its own
                #   query_node_{norm}_nq{K}.json for the query-node count to actually change.
                #   Fallback: nq{K} absent -> query_node_{norm}.json -> bundle["query_nodes"] -> {} (live extraction).
                _K = mem_cfg.get("num_query_nodes")
                _K = int(_K) if _K is not None else None
                _query_nodes = load_query_nodes_for(
                    dataset_root=cfg_data.get("dataset_root"),
                    data_type=cfg_data.get("data_type", "normal"),
                    trial_idx=int(cfg_data.get("trial", 0)),
                    llm_name=llm_name,
                    normalization=cfg_data.get("normalization", "raw"),
                    cache_budget=_cache_budget,
                    num_query_nodes=_K,
                ) or (bundle.get("query_nodes") or {})
                if _query_nodes:
                    mem_cfg["precomputed_query_nodes"] = {int(k): v for k, v in _query_nodes.items()}
                # Inject record embeddings (for the memory's cos_sim re-rank).
                # Prefer partial (experiments use the partial memory), fall back to entire.
                # A prior `_record_emb = None` (temporarily disabled due to an underscore-key mismatch)
                # silently no-op'd the cos_sim_rerank=True grid axis -> ON/OFF gave identical results.
                # EmbeddingCache keys are unified as `f"{h} {r} {t} on {sd}"` (temporal) +
                # `f"{h} {r} {t}"` (non_temporal), and the memory also uses the same key -> re-enabled.
                # On in-memory map miss, the file cache + live embed fallback protects.
                _record_emb = bundle.get("partial_record_emb") or bundle.get("entire_record_emb")
                if _record_emb:
                    mem_cfg["record_embeddings"] = _record_emb

                logging.info(f"[ablation] {method_name} {cur}/{total} llm={llm_name} grid={grid_idx} combo={combo}")
                # Add PID to the ckpt name — avoids ckpt collisions across concurrent sweep runs.
                ckpt_path = out_dir / f".ckpt.grid_{grid_idx:03d}.{llm_name}.pid{os.getpid()}.ckpt.pkl"

                t0 = time.perf_counter()
                # A normal launch (not rerun_missing) enables per-session incremental save + resume:
                #   run_one_llm owns _result_file (json): load -> skip done sessions -> QA only the
                #   incomplete -> atomic incremental save. rerun_missing mode keeps the caller-patch
                #   logic below (result_json_path not passed).
                # result_save_every default 1->10: instead of rewriting the whole (huge) result json
                #   (theoffice ~450MB) every session, do it every 10 (10x fewer writes). On crash only
                #   <=10 sessions are reprocessed on resume.
                _abl_save_every = int(run_cfg.get("result_save_every", 10))
                run_out = _baseline.run_one_llm(
                    llm, llm_name, embedder, sessions, questions, mem_cfg, run_cfg,
                    checkpoint_path=ckpt_path, checkpoint_every=100,
                    rerun_sessions=_rerun_targets,   # rerun_missing mode answers only missing sessions, else None
                    result_json_path=(None if rerun_missing else _result_file),
                    save_every_sessions=(0 if rerun_missing else _abl_save_every),
                    resume=(not rerun_missing),
                )
                # Known issue, fix deferred: rerun_missing saves only at the end (save_every=0), so a
                #   glacial theoffice job freezes n_scored until completion, and a kill loses reprocessed
                #   work. Enabling incremental save (result_json_path+save_every+resume=True) is the fix,
                #   but apply it only after unit-testing run_one_llm's resume+rerun_sessions behavior
                #   (core save path — no unverified changes). TODO.
                wall = time.perf_counter() - t0

                # rerun_missing mode: without changing the existing result json's structure/format,
                #   replace only the missing session element values + recompute the records-derived
                #   summary values. (Never alter the file structure/format — only fix error session
                #   element values.)
                if rerun_missing:
                    try:
                        _ex = json.loads(_result_file.read_text())
                    except Exception as _re:
                        logging.warning(f"[ablation rerun] grid={grid_idx} failed to reload existing json, keeping existing: {_re}")
                        continue
                    # Extract only the reprocessed sessions from run_out (do_qa=True -> model_answer not None)
                    _new_by_idx = {
                        r.get("session_idx"): r for r in run_out.get("records", [])
                        if r.get("session_idx") in (_rerun_targets or set()) and r.get("model_answer") is not None
                    }
                    if not _new_by_idx:
                        logging.warning(f"[ablation rerun] grid={grid_idx} no reprocessed result (still timing out?), keeping existing")
                        continue
                    # Replace only those session_idx elements in existing records (leave others untouched)
                    _existing_idx = {
                        _r.get("session_idx") for _r in _ex.get("records", [])
                        if isinstance(_r.get("session_idx"), int)
                    }
                    for _i, _r in enumerate(_ex.get("records", [])):
                        _si = _r.get("session_idx")
                        if _si in _new_by_idx:
                            _ex["records"][_i] = _new_by_idx[_si]
                    # Append reprocessed sessions that were absent from records (crashed before reaching).
                    #   (In-place replacement only updates present sessions -> absent ones must be added
                    #   here for n_scored to reach expected.)
                    for _si, _nr in _new_by_idx.items():
                        if _si not in _existing_idx:
                            _ex.setdefault("records", []).append(_nr)
                    # Recompute summary (same key structure, only records-derived values; preserve
                    #   non-records keys like grid_combo/grid_idx/n_sessions/total_wall_sec)
                    _recs = _ex.get("records", [])
                    _ns = sum(1 for r in _recs if r.get("is_correct") is not None)
                    _nc = sum(1 for r in _recs if r.get("is_correct"))
                    _ex["summary"]["n_scored"] = _ns
                    _ex["summary"]["n_correct"] = _nc
                    _ex["summary"]["accuracy"] = _nc / max(1, _ns)
                    _ex["summary"]["accuracy_by_hop"] = _baseline._accuracy_by_hop(_recs)
                    _ex["summary"]["total_input_tokens"] = sum(int(r.get("input_tokens", 0) or 0) for r in _recs)
                    _ex["summary"]["total_output_tokens"] = sum(int(r.get("output_tokens", 0) or 0) for r in _recs)
                    _ex["summary"]["total_llm_latency_sec"] = sum(float(r.get("latency_sec", 0) or 0) for r in _recs)
                    # Save records sorted by session_idx (0->N).
                    #   Even if the original json's array order was scrambled by resume-append, save a
                    #   sorted file. In-place replacement (above) preserves position, so scrambled input
                    #   yields scrambled output -> sort here. Content (each record) unchanged, only the
                    #   array order is sorted (session_idx exists on every record).
                    if isinstance(_ex.get("records"), list):
                        _ex["records"].sort(key=lambda _r: _r.get("session_idx", 0))
                    # Save (same format: ensure_ascii=False, indent=2, default=str). Atomic.
                    _baseline._atomic_write_json(_result_file, _ex)
                    logging.info(f"[ablation rerun] grid={grid_idx} patched sessions {sorted(_new_by_idx)} "
                                 f"-> n_scored {_ns}, acc {_ex['summary']['accuracy']:.4f}")
                    continue

                # Record "which config combination" this grid was in the result summary.
                #   grid_combo = the combo dict actually used (axis overrides; the `combo` from either
                #   explicit grid_combos or _expand_grid). Empty dict ({}, no override) stays {}.
                #   grid_idx = NNN (0-based). dict(combo) is a defensive copy (avoids later mutation /
                #   shared-object pollution). Leave existing summary keys (accuracy etc.), add only these two.
                run_out["summary"]["grid_combo"] = dict(combo)
                run_out["summary"]["grid_idx"] = grid_idx

                row = {
                    "method": method_name, "llm": llm_name,
                    "grid_idx": grid_idx, "combo": combo,
                    # grid_combo: same key name as the summary so aggregation (json/tsv) also records this grid's config.
                    "grid_combo": dict(combo),
                    "accuracy": run_out["summary"]["accuracy"],
                    "n_sessions": run_out["summary"]["n_sessions"],
                    "total_llm_latency_sec": run_out["summary"]["total_llm_latency_sec"],
                    "total_input_tokens": run_out["summary"]["total_input_tokens"],
                    "total_output_tokens": run_out["summary"]["total_output_tokens"],
                    "wall_sec": wall,
                }
                with rows_lock:
                    rows.append(row)
                # Results are saved as json only; grid_combo/summary are embedded in the json.
                # On the resume path run_one_llm already atomic-saved _result_file, but here we
                #   atomic re-save the final version with grid_combo/grid_idx added to summary
                #   (prevents truncation).
                try:
                    # Save records sorted by session_idx (0->N).
                    #   run_one_llm's resume incremental save appends reprocessed sessions to the end
                    #   of the array (scrambling order); restore order at final save (content unchanged).
                    if isinstance(run_out.get("records"), list):
                        run_out["records"].sort(key=lambda _r: _r.get("session_idx", 0))
                    _baseline._atomic_write_json(
                        out_dir / f"{llm_name}.grid_{grid_idx:03d}.json", run_out)
                except Exception as _je:
                    logging.warning(f"[ablation] failed to save grid json (ignored): {type(_je).__name__}: {_je}")
            except Exception as e:
                logging.error(f"[ablation] [{llm_name}] grid={grid_idx} failed: {type(e).__name__}: {e}", exc_info=True)
                with rows_lock:
                    rows.append({
                        "method": method_name, "llm": llm_name, "grid_idx": grid_idx, "combo": combo,
                        "grid_combo": dict(combo),  # record which config the failed cell was.
                        "accuracy": 0.0, "n_sessions": 0, "total_llm_latency_sec": 0,
                        "total_input_tokens": 0, "total_output_tokens": 0, "wall_sec": 0,
                        "error": f"{type(e).__name__}: {e}",
                    })
                # The except branch also cleans up the ckpt — otherwise a failed cell's stale ckpt
                # would linger on disk forever (older code skipped cleanup via continue).
                try:
                    if 'ckpt_path' in locals():
                        ckpt_path.unlink(missing_ok=True)
                except Exception as _ce:
                    logging.warning(f"[ablation] ckpt cleanup (except branch) failed: {_ce}")
                continue
            # checkpoint cleanup (delete the ckpt right after the final save)
            try:
                ckpt_path.unlink(missing_ok=True)
            except Exception as e:
                logging.warning(f"[ablation] ckpt cleanup failed ({ckpt_path}): {e}")

    # ---- parallel sweep + partial-failure cleanup ----
    max_workers = int(method_cfg.get("run", {}).get("parallel_max_workers", len(llms_list)))
    max_workers = max(1, min(max_workers, len(llms_list)))
    if max_workers == 1 or len(llms_list) == 1:
        for llm_cfg in llms_list:
            _run_one_llm_all_grids(llm_cfg)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one_llm_all_grids, cfg) for cfg in llms_list]
            try:
                for fut in as_completed(futures):
                    fut.result()
            except Exception:
                for f in futures:
                    f.cancel()
                logging.error(
                    "[ablation] LLM thread failed — unstarted futures canceled. Running threads "
                    "continue in the background (Python ThreadPoolExecutor limitation). That grid's "
                    "ckpt is preserved."
                )
                raise

    # If CachedEmbedder, flush the cache to disk after all LLMs/grids finish — persist embeddings
    #   still in memory (autoflush-interval leftovers) so the next ablation run reuses them. (If a
    #   thread failure raised, autoflush already persisted most; leftovers are re-embedded next run.)
    if isinstance(embedder, CachedEmbedder):
        try:
            embedder.flush()
        except Exception as _fe:
            logging.warning(f"[ablation][embedder] cache flush failed: {_fe}")

    summary = {
        "method": method_name,
        "n_llms": len(llms_list),
        "n_grid_points": len(combos),
        "n_total_runs": total,
        "rows": rows,
        "best": _find_best(rows),
    }
    # ablation_summary.{pkl,json,tsv} are generated only on explicit request -> not by default.
    #   Each grid result is in {llm}.grid_NNN.json; the summary is printed to stdout below.
    # Safety net — residual ckpt cleanup. PID filter protects concurrent sweeps' ckpts.
    pid_pat = f".ckpt.grid_*.pid{os.getpid()}.ckpt.pkl"
    for ckpt in out_dir.glob(pid_pat):
        try:
            ckpt.unlink()
        except Exception as e:
            logging.warning(f"[ablation] residual ckpt cleanup failed ({ckpt}): {e}")
    return summary


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="HP search entry: grid sweep for one method (multi-LLM parallel).",
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--method", required=True, help="the config's top-level key (e.g., groupchat_mem).")
    parser.add_argument("--llms", type=str, default="",
                        help="comma-separated LLM names. If omitted, sweep all of method.llms.")
    parser.add_argument("--grid-start", type=int, default=0,
                        help="grid_idx start (inclusive). For splitting a slow model across 2 procs.")
    parser.add_argument("--grid-end", type=int, default=None,
                        help="grid_idx end (inclusive). Runs to the end if omitted.")
    parser.add_argument("--rerun-missing", action="store_true",
                        help="For grids with n_scored<expected (missing due to timeout etc.) in an "
                             "existing result json, reprocess only the missing sessions -> replace only "
                             "those record elements + recompute summary (structure/format unchanged).")
    parser.add_argument("--rerun-scope", type=str, default="config",
                        choices=["config", "even", "odd", "entire"],
                        help="Session scope to reprocess for rerun-missing. config = keep the config's "
                             "session_filter (default; even experiment -> even, all experiment -> entire). "
                             "even = even only, entire = all sessions.")

    # argparse parses only the args it knows and collects unknown ones into the second list.
    # (parse_args would error immediately on unknown args.)
    # left_argv: CLI args other than config, method, llms.
    args, left_argv = parser.parse_known_args()

    # read yaml file: sconf reads the yaml into a nested object.
    cfg = Config(args.config)

    # If the CLI has args other than config/method/llms, override the yaml config values.
    if left_argv:
        # Take the remaining CLI args (flat string list) and overwrite yaml config values, descending
        # the yaml tree along the dot-path.
        # e.g., ["--groupchat_mem.data.trial", "1", "--groupchat_mem.run.seed", "7"]
        cfg.argv_update(left_argv)

    # Convert the sconf object to a python dict.
    full_cfg = cfg.asdict()

    # If the method value given as an arg is not a top-level key in the yaml -> its config is absent,
    # so error out.
    if args.method not in full_cfg:
        raise KeyError(f"Method {args.method!r} not in config. Available: {list(full_cfg.keys())}")

    # INFO / WARNING / ERROR / CRITICAL -> log only INFO and above (ignore DEBUG).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # To run multiple LLM types at once, pass --llms llm1,llm2,llm3 -> split into llm_names.
    llm_names = [s.strip() for s in args.llms.split(",") if s.strip()] or None

    # run ablation
    summary = run_ablation(full_cfg[args.method], method_name=args.method, llm_names=llm_names,
                           grid_start=args.grid_start, grid_end=args.grid_end,
                           rerun_missing=args.rerun_missing, rerun_scope=args.rerun_scope)

    # print out summarized result
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
