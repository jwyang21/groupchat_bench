"""run_baseline.py - baseline / oracle / simple-RAG / groupchat-mem entry point (commit + QA).

Uses:
  - Baselines + oracle + simple_rag + groupchat_mem: `configs/baselines/<...>.yaml`.
    Both the retrieval records and the retrieval route are precomputed by the QA LLM
    (assistant == QA model policy).

Common assumptions:
  - Input: converted dataset tree (`data/.../<dataset>/`).
  - memory: chosen by config via the make_memory factory (live retrieval).
  - data_type (normal/newname/shuffled) + ie_type (openie/closedie/gt) are chosen by config
    (sconf argv override supported).
  - route source (`per_llm_precomputed: true`): per-llm/{llm_name} cache.
  - vLLM endpoint: cfg.llms[i].host / cfg.llms[i].port (one entry per model).
    base_url auto-composed as `http://{host}:{port}/v1` (host default = localhost). base_url may also
    be set directly.

Multi-LLM sweep: run each model in cfg.llms in parallel. On each thread's completion, save
`<out>.{llm}.pkl` (preserve intermediate results). Before starting, ping every vLLM endpoint's
`/health` to fail fast.

Example:
  python src/run/run_baseline.py --config configs/baselines/<...>.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sconf import Config

from groupchat_qa.assets import CONFIG_DIR
from groupchat_qa.client import make_llm
from groupchat_qa.embedder import CachedEmbedder, make_embedder
from groupchat_qa.memory import make_memory
from groupchat_qa.pipeline.data_loader import load_converted_dataset, load_query_nodes_for, load_v1_2_1_trial
from groupchat_qa.utils import process_answer, save_pickle, load_pickle, seed_everything, load_project_env
from groupchat_qa.utils.squad_score import match_option_squad

# Auto-discover .env - traverse upward from this .py file and load the first .env (caller-relative).
load_project_env(start=__file__, override=True)


def _atomic_write_json(path: "Path | str", obj: Any) -> None:
    """Save the result JSON atomically (tmp + os.replace).

    Shared helper for per-session incremental save + resume (simulate/ablation). To avoid partial-write
    corruption even when called every N sessions, write a temp file in the same directory then replace
    via os.replace (an atomic rename within the same filesystem). The tmp filename includes the PID so
    concurrent writers never collide on tmp - assuming the result file itself has a single writer
    (simulate=per-llm, ablation=per-grid_idx). Callers wrap this best-effort so a save failure does not
    kill the experiment (default=str guards serialization)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.pid{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _per_llm_out_path(base: Path, llm_name: str) -> Path:
    """Per-llm result path formed by appending `.{llm}.json` to the base path.
    Note: Path.stem mistakes the last dot token (`.t1`/`.t2` etc.) for an extension and strips it, so
    `--out ...t1` would wrongly save to `...newname.{llm}.json` (aggregate), making t0/t1/t2 clobber
    each other and freezing per-trial files.
    -> Strip only the `.json` extension and preserve any trial token like `.t1`."""
    nm = base.name
    if nm.endswith(".json"):
        nm = nm[:-5]
    return base.with_name(f"{nm}.{llm_name}.json")


def _apply_answer_budget(llm_cfg: dict, answer_budget: int | None) -> dict:
    """Return a copy of llm_cfg with the answer budget (= reasoning budget for the final QA
    inference) injected.

    Ported verbatim from the grid entry's `_apply_answer_budget` (same behavior). The answer
    budget is the reasoning limit at answer-generation time, independent of the cache budget
    (which cache to load).

    - answer_budget=None -> return llm_cfg as-is (no copy) = existing behavior. The answer-time
      reasoning budget stays whatever llm_cfg.extra_body.thinking_token_budget was (or unset).
    - answer_budget=positive int -> shallow-copy, overwrite extra_body.thinking_token_budget with
      that value, and if max_tokens is below answer_budget + 6384 (content headroom), raise it so
      the budget is not truncated by max_tokens.

    Note: thinking_token_budget is meaningless for enable_thinking=false (non-thinking) models -
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

    new_cfg = dict(llm_cfg)  # Shallow copy - rebuild only extra_body deeply (avoid mutating original).
    extra_body = dict(new_cfg.get("extra_body") or {})
    ctk = dict(extra_body.get("chat_template_kwargs") or {})
    # Skip budget injection for models where enable_thinking is explicitly False (meaningless + confusing).
    if ctk.get("enable_thinking") is False:
        logging.warning(
            f"[simulate] qa.answer_budget={ab} set, but LLM "
            f"{new_cfg.get('name', new_cfg.get('model_name', '?'))} has enable_thinking=false - "
            f"skipping budget injection."
        )
        return llm_cfg
    # reasoning_effort models (e.g., gpt-oss/harmony) have no thinking_token_budget knob.
    #   Injecting a budget makes the server return 400 (BadRequest, reasoning_config not configured)
    #   -> answer fails. reasoning_effort controls the reasoning amount instead, so skip injection.
    if "reasoning_effort" in ctk:
        logging.warning(
            f"[simulate] qa.answer_budget={ab} set, but LLM "
            f"{new_cfg.get('name', new_cfg.get('model_name', '?'))} is a reasoning_effort model "
            f"(e.g., gpt-oss, no thinking_token_budget knob) - skipping budget injection."
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


def _normalize_llms(cfg_d: dict) -> list[dict]:
    """Normalize cfg's llms (list) or a single llm (dict) into a list of dicts."""
    if "llms" in cfg_d and cfg_d["llms"]:
        return [dict(c) for c in cfg_d["llms"]]
    if "llm" in cfg_d:
        return [dict(cfg_d["llm"])]
    raise KeyError("cfg must specify 'llms' (list) or 'llm' (dict).")


def _resolve_base_url(llm_cfg: dict) -> str:
    """Prefer an explicit base_url; otherwise auto-compose from host+port."""
    base_url = llm_cfg.get("base_url")
    if base_url:
        return base_url
    port = llm_cfg.get("port")
    if port is None:
        raise KeyError(f"LLM {llm_cfg.get('name', '?')}: must specify either base_url or port")
    host = llm_cfg.get("host", "localhost")
    return f"http://{host}:{port}/v1"


def _make_llm_from_cfg(llm_cfg: dict, seed: int | None = None):
    # Pass every key except the known ones straight through to ChatOpenAI
    # (e.g., extra_body={"chat_template_kwargs":{"enable_thinking": true}} for the Qwen3 thinking toggle)
    # seed: request seed for reproducibility (run.seed). The caller passes cfg.run.seed - even with
    #   temperature>0, the same seed+HW+vLLM version reproduces sampling (vLLM Reproducibility docs).
    #   This is the server-side sampling RNG, separate from the client-side seed_everything
    #   (python/numpy/torch RNG). If None, it is not passed (existing behavior).
    #   Note: the LLM client is created once outside the grid loop, so varying run.seed as a grid axis
    #   does not change the request seed (current configs fix run.seed, so this is moot).
    # Include "seed" in the known set - avoids a duplicate/keyword collision with run.seed if a
    #   per-llm seed key exists.
    _known = {"provider", "name", "model_name", "base_url", "host", "port",
              "api_key", "temperature", "top_p", "max_tokens", "system_prompt", "seed",
              "max_model_len",  # max_model_len: for context_size computation (not passed to client - excluded)
              "extra_base_urls"}  # for endpoint-pool distribution (excluded - not passed to make_llm)
    extra = {k: v for k, v in llm_cfg.items() if k not in _known}

    def _mk(burl: str):
        return make_llm(
            provider=llm_cfg["provider"],
            model_name=llm_cfg["model_name"],
            base_url=burl,
            api_key=llm_cfg.get("api_key", "EMPTY"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            top_p=float(llm_cfg.get("top_p", 0.95)),
            max_tokens=int(llm_cfg.get("max_tokens", 1000)),
            system_prompt=llm_cfg.get("system_prompt", "You are a helpful assistant."),
            seed=seed,
            **extra,
        )

    # endpoint pool - if extra_base_urls (list) is present, wrap primary + extras in RoundRobinLLM to
    #   distribute per-cell parallel question calls across several vLLM endpoints (the same qwen-on on
    #   multiple servers). Otherwise use a single client (existing behavior).
    _extra_urls = llm_cfg.get("extra_base_urls") or []
    primary = _resolve_base_url(llm_cfg)
    if _extra_urls:
        from groupchat_qa.client.round_robin import RoundRobinLLM
        urls = [primary] + [u for u in _extra_urls if u and u != primary]
        return RoundRobinLLM([_mk(u) for u in urls])
    return _mk(primary)


def _check_vllm_endpoint(llm_cfg: dict, timeout: float = 30.0) -> None:
    """Verify vLLM endpoint reachability + served-model-name match.

    - timeout 5s -> 30s (avoids false negatives from a busy vLLM's slow /health response).
    - Besides /health 200, query /v1/models - confirm the yaml model_name matches the actual served
      alias (prevents a silent mismatch when the yaml was not updated after a vLLM restart).

    Fail-fast: validate on the main thread before launching all LLM threads, which would otherwise
    each fail with a connection error.
    """
    if llm_cfg.get("provider") != "vllm":
        return
    base_url = _resolve_base_url(llm_cfg)
    name = llm_cfg.get("name", llm_cfg.get("model_name", "?"))
    expected_model = llm_cfg.get("model_name", name)
    health_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    # str.rstrip("/v1") is a char-set strip, so it would also strip a trailing 1 of the port (e.g.,
    # 8001) and query the wrong port -> absorbed into a silent warning. Use rsplit("/v1", 1)[0] for a
    # substring strip.
    models_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/v1/models"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM {name}: /health status={resp.status}")
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        raise RuntimeError(
            f"LLM {name}: vLLM endpoint unreachable ({health_url})\n"
            f"  -> {e}\n"
            f"  -> check that the server is running."
        ) from e
    # Validate the served alias via /v1/models (matches the yaml model_name).
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        served = {m.get("id") for m in data.get("data", [])}
        if expected_model not in served:
            raise RuntimeError(
                f"LLM {name}: yaml model_name={expected_model!r} not among vLLM served aliases {served}.\n"
                f"  -> check the yaml model_name or vLLM's --served-model-name (prevents silent-fail mismatch)."
            )
    except (urllib.error.URLError, OSError) as e:
        # A /v1/models query failure passed health, so warn only - do not force a failure.
        logging.warning(f"LLM {name}: /v1/models query failed ({e}) - skipping model_name validation.")


def _accuracy_by_hop(records: list[dict]) -> dict[str, float]:
    """Accuracy aggregated by hop_count.

    hop_count convention (the `hop_count` field in qa.json):
       -1 : fanquiz  (fan-knowledge / non-relational question)
        1 : hop1     (1-hop reasoning over the relational memory)
        2 : hop2     (2-hop reasoning over the relational memory)
    Data must follow this convention for accurate bucket classification.
    """
    buckets: dict[int, list[bool]] = {}
    for r in records:
        if r.get("is_correct") is None:
            continue
        hop = r.get("hop_count")
        if hop is None:
            continue
        buckets.setdefault(hop, []).append(bool(r["is_correct"]))
    label_map = {-1: "fanquiz", 1: "hop1", 2: "hop2"}
    return {label_map.get(h, f"hop_{h}"): sum(v) / max(1, len(v)) for h, v in buckets.items()}


def _commit_all_sessions(
    memory, sessions, start: int, end: int, llm_name: str,
    commit_extract_workers: int,
    extract_cache_dir: "Path | None" = None,
    session_ckpt_dir: "Path | None" = None,
) -> None:
    """Commit all sessions in 2 phases (extract in parallel -> build sequentially).

    Phase 1 (extract): run each session's *session-independent* LLM extraction (OpenIE / note
      analysis / fact extraction) in parallel via a ThreadPool, distributed over the endpoint pool
      (RoundRobinLLM). memory.extract does not touch the accumulated store, so it is thread-safe
      (self.llm/embedder are lock-protected).
    Phase 2 (build): feed the extraction results to commit_prepared in *ascending session idx,
      sequentially*, accumulating fact store/note/memory. Even baselines with accumulation dependencies
      (link/evolution/update-decision) preserve ordering, so there is no temporal leakage.

    If commit_extract_workers <= 1, phase 1 is also sequential (identical commit order to before).
    If memory does not support 2 phases (extract -> None, commit_prepared -> commit(session)
      fallback), it still works safely as a plain commit.
    """
    n = end - start
    if n <= 0:
        return
    workers = max(1, int(commit_extract_workers))

    # Disk-cache extraction results - analyze(LLM)+embed (session-independent) is expensive, so avoid
    #   recomputation on re-run. The caller composes a unique cache path per (dataset, data_type,
    #   turns_per_note, llm), with session_idx as the filename -> never mixes with other tpn/model/data
    #   outputs. Cache hit -> load, miss -> memory.extract then save. extract leaves the store
    #   unchanged (parallel-safe).
    def _extract_one(i: int):
        sess = sessions[i]
        if extract_cache_dir is not None:
            cp = extract_cache_dir / f"session_{sess.idx:04d}.pkl"
            if cp.exists():
                try:
                    return load_pickle(str(cp))
                except Exception as e:  # corrupt cache -> re-extract (fail-safe).
                    logging.warning(f"[{llm_name}] extract cache load failed ({cp}): {e} -> re-extract")
        ex = memory.extract(sess)
        if extract_cache_dir is not None:
            try:
                extract_cache_dir.mkdir(parents=True, exist_ok=True)
                save_pickle(ex, str(extract_cache_dir / f"session_{sess.idx:04d}.pkl"))
            except Exception as e:  # save failure is best-effort (experiment continues).
                logging.warning(f"[{llm_name}] extract cache save failed (idx={sess.idx}): {e}")
        return ex

    # Resume: continue commit from the latest session checkpoint
    #   When re-running a run whose commit_prepared (phase 2) was interrupted, restore the last saved
    #   session checkpoint (full snapshot) and continue committing from the next session. The snapshot
    #   is the entire accumulated store up to that point, so restoring only the single latest suffices
    #   (even with missing intermediate saves, the latest snapshot covers 0..idx). extract (phase 1) is
    #   reused from the disk cache, but on resume the extracts before resume_from are unused in phase 2,
    #   so extract only from resume_from, saving cache-load I/O.
    #   Since commit is sequential (ascending idx), the largest loop index among existing checkpoints
    #   is the latest.
    resume_from = start
    if session_ckpt_dir is not None and session_ckpt_dir.is_dir():
        last_done_i: int | None = None
        for i in range(start, end):
            if (session_ckpt_dir / f"session_{sessions[i].idx:04d}.snapshot.pkl").exists():
                last_done_i = i
        if last_done_i is not None:
            ckpt = session_ckpt_dir / f"session_{sessions[last_done_i].idx:04d}.snapshot.pkl"
            try:
                memory.restore(str(ckpt))
                resume_from = last_done_i + 1
                logging.info(
                    f"[{llm_name}] commit resume: session {sessions[last_done_i].idx} "
                    f"checkpoint restored -> {resume_from - start}/{n} sessions done, resuming from the rest"
                )
            except Exception as e:  # corrupt checkpoint -> from scratch (fail-safe). Caller creates a new memory.
                logging.warning(f"[{llm_name}] checkpoint restore failed ({ckpt}): {e} -> committing from scratch")
                resume_from = start
    if resume_from >= end:
        logging.info(f"[{llm_name}] all sessions already committed -> skip commit (resume)")
        return
    n_remaining = end - resume_from

    # Phase 1: extract (session-independent LLM extraction + disk caching)
    # Store results in session-order slots (idx-start) - phase 2 consumes them in ascending order
    # regardless of completion order. On resume, extract only from resume_from (earlier session
    # extracts are unused in phase 2).
    extracts: list[Any] = [None] * n
    if workers > 1:
        logging.info(
            f"[{llm_name}] commit extract parallel start (workers={workers}, sessions={n_remaining})"
        )
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_extract_one, i): i for i in range(resume_from, end)}
            for fut in as_completed(futs):
                i = futs[fut]
                extracts[i - start] = fut.result()  # an extract exception raises here -> cell fails (intended).
                done += 1
                if done % 50 == 0:
                    logging.info(f"[{llm_name}] extracted {done}/{n_remaining}")
    else:
        for i in range(resume_from, end):
            extracts[i - start] = _extract_one(i)

    # Phase 2: build (ascending session idx, sequential) + per-session checkpoint
    #   Save a full snapshot at that point right after each session's commit_prepared.
    #   Even before commit finishes for all sessions, you can restore an arbitrary checkpoint to check
    #   intermediate QA performance over "sessions processed so far" (future leakage is blocked by
    #   retrieve's session_idx < current filter, so an intermediate snapshot is still point-accurate).
    if session_ckpt_dir is not None:
        try:
            session_ckpt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.warning(f"[{llm_name}] failed to create session checkpoint dir: {e}")
    for i in range(resume_from, end):
        memory.commit_prepared(sessions[i], extracts[i - start])
        # Save the per-session snapshot every 10 sessions (+ the last one) instead of every session.
        #   Cuts the huge snapshot pkl writes 10x. Crash resume: restore recovers from the last existing
        #   ckpt (a multiple of 10) then re-commits <=10 sessions (lossless). The restore logic above
        #   finds the "last existing ckpt", so gaps are fine.
        if session_ckpt_dir is not None and ((i + 1) % 10 == 0 or i == end - 1):
            try:
                memory.snapshot(str(session_ckpt_dir / f"session_{sessions[i].idx:04d}.snapshot.pkl"))
            except Exception as e:  # checkpoint save failure is best-effort (build continues).
                logging.warning(
                    f"[{llm_name}] session checkpoint save failed (idx={sessions[i].idx}): {e}"
                )
        if (i + 1) % 50 == 0:
            logging.info(f"[{llm_name}] commit-build {i + 1}/{end}")


def run_one_llm(llm, llm_name: str, embedder, sessions, questions, mem_cfg: dict, run_cfg: dict,
                checkpoint_path: Path | None = None, checkpoint_every: int = 100,
                answer_llm=None, commit_only: bool = False,
                rerun_sessions: set[int] | None = None,
                result_json_path: "Path | None" = None,
                save_every_sessions: int = 0,
                resume: bool = False) -> dict[str, Any]:
    """Run the session loop for a single LLM. A new memory instance is created per LLM.

    session_filter (run_cfg):
      'all'  : all sessions (default)
      'even' : 0-based even index (i=0,2,4,...) - ablation standard
      'odd'  : 0-based odd index  (i=1,3,5,...)

    checkpoint_path: if given, overwrite-save a partial result every checkpoint_every sessions.
                     None disables it. The final save is handled by the caller (run()'s save_per_llm
                     or the ablation grid save).

    answer_llm: an LLM for final QA answering, independent of commit/build.
      None (default) -> answer also uses `llm` (100% identical to existing behavior).
      If set -> commit/build uses `llm` (original budget, e.g., 6000), and answer (QA generation)
      uses `answer_llm` (e.g., budget 10000). The memory code is unmodified - the split is done by
      swapping memory.llm to the right LLM right before commit / right before answer (both
      memory.answer/commit read self.llm). For baselines (a_mem/hipporag2/mem0), commit() is a live
      LLM build and answer()'s final generation is self.llm.generate - so the swap separates them exactly.

    commit_only: if True, commit() all sessions, save only the snapshot, and skip QA (answer
      generation). Snapshot-build-only mode. Returns with empty records (summary.commit_only=True).
      If snapshot_dir is unset, commit still runs but there is nothing to save, so it only warns -
      normally used together with snapshot_dir. Default False (existing behavior: commit+QA).

    result_json_path / save_every_sessions / resume - per-session incremental save + resume after
      interruption (for oracle=run_baseline and the grid entry=run_groupchat_mem):
      - result_json_path=None (default) -> incremental/resume disabled, 100% identical behavior
        (summary is accumulator-based, records = the result list). commit_only=True forces it off
        (commit-only has no records).
      - result_json_path set -> this function owns that JSON file:
          (a) resume=True and the file exists -> load; skip QA for already-scored sessions (record
              exists + is_correct not None), and QA only the incomplete eligible (session_filter-passing)
              sessions. (If rerun_sessions is specified, target that set exactly - explicit wins.)
          (b) atomic-save the merged (existing done + new) result every save_every_sessions and at exit.
          (c) if every eligible session is already done, skip commit/QA and return the existing result.
        The returned records/summary are also merged. summary.resume_complete=True means no eligible
        sessions remain (= complete). thinking-ON nondeterminism note: never regenerate/overwrite the
        answer/record of an already-scored session.
    """
    # Log all LLM prompts + responses (for baseline experiments) - if run.log_prompts: true, wrap
    #   the base llm in LoggingLLM. *Every* self.llm.generate(_structured) call in the baseline
    #   (note/evolution/openie/filter/fact/update/QA, etc.) is recorded as JSONL. If unset, behavior
    #   is unchanged. Path: run.log_prompts_dir (default log/baseline_prompts/{memory_name}) /
    #   {llm}.prompts.jsonl.
    if run_cfg.get("log_prompts"):
        from groupchat_qa.client import LoggingLLM
        # Add data_type/trial to the default path - otherwise the same memory x LLM run with different
        #   trial / data_type accumulates into the same file, blocking split during analysis.
        # None-safe - `.get(k, default)` does not apply the default when the key exists but the value
        # is None. `or 'unknown_dt'` treats falsy as default too. Same for trial.
        _dt = mem_cfg.get('data_type') or 'unknown_dt'
        _tr = mem_cfg.get('trial') if mem_cfg.get('trial') is not None else 0
        _default_dir = (
            f"log/baseline_prompts/{mem_cfg.get('name', 'mem')}/{_dt}/t{_tr}"
        )
        _ldir = run_cfg.get("log_prompts_dir") or _default_dir
        # Record trial/data_type/ie_type as extra_metadata - enables per-grid-cell analysis in ablation
        #   and trial separation in simulate.
        _extra_meta = {
            "llm_name": llm_name,
            "data_type": mem_cfg.get("data_type"),
            "trial": mem_cfg.get("trial"),
            "ie_type": mem_cfg.get("ie_type"),
            "normalization": mem_cfg.get("normalization"),
        }
        llm = LoggingLLM(llm, Path(_ldir) / f"{llm_name}.prompts.jsonl", extra_metadata=_extra_meta)
        logging.info(f"[{llm_name}] LLM prompt logging enabled -> {Path(_ldir) / f'{llm_name}.prompts.jsonl'}")
        # Log answer_llm to the same file too (commit/answer accumulate into one prompts.jsonl - call_idx
        #   monotonic). Keeps logging consistent even if answer_llm is a separate instance.
        if answer_llm is not None:
            answer_llm = LoggingLLM(
                answer_llm, Path(_ldir) / f"{llm_name}.prompts.jsonl", extra_metadata=_extra_meta
            )

    # commit/answer budget split - if answer_llm is set, swap memory.llm per stage.
    #   Default self.llm = `llm` for commit/build. Set answer_llm right before answer, and set llm
    #   back right before commit. If answer_llm=None, _answer_llm = llm (= single LLM, existing
    #   behavior, swap is a no-op).
    _commit_llm = llm
    _answer_llm = answer_llm if answer_llm is not None else llm
    # _ckpt_name: the name used for commit artifact (snapshot/ckpt/extract) paths.
    #   Note: a_mem session_snapshot_dir (below), _session_ckpt_dir, snapshot_path, and the extract
    #   path all use this for consistency.
    _ckpt_name = mem_cfg["name"]
    # a_mem loads session_{idx-1}.snapshot at retrieve time (uses the enhanced embedding + no future
    #   leakage). Inject that path (same rule as _session_ckpt_dir below) into cfg so the a_mem
    #   constructor reads it. Only when both session_checkpoint + snapshot_dir are set.
    if (mem_cfg["name"] == "a_mem" and mem_cfg.get("session_checkpoint")
            and mem_cfg.get("snapshot_dir")):
        # mem0 uses the turns_per_batch (=5) key, so reflect that in the path tpn (a_mem=turns_per_note).
        _tpn_inj = mem_cfg.get("turns_per_note") or mem_cfg.get("turns_per_batch") or 1
        mem_cfg["session_snapshot_dir"] = str(
            Path(mem_cfg["snapshot_dir"]) / f"{_ckpt_name}_ckpt"
            / f"tpn{_tpn_inj}" / llm_name
        )
    memory = make_memory(name=mem_cfg["name"], llm=_commit_llm, embedder=embedder, cfg=mem_cfg)

    # snapshot/restore wiring - optionally absorb build cost.
    #   If mem_cfg["snapshot_dir"] is set, use a per-LLM snapshot file (`{name}.{llm}.snapshot.pkl`):
    #     - file exists -> restore, skipping the all-session commit (retrieve filters future sessions
    #       by current_session_idx, so preloading the full memory is still point-accurate = same result
    #       as a live build).
    #     - not exists -> commit as usual, then save a snapshot at the end of the loop -> reused next run.
    #   Unset (default) keeps existing behavior (live commit every run). Common to a_mem/mem0/hipporag2/groupchat_mem.
    #   Note: snapshot_dir must be unique per (dataset, trial, data_type, ie_type, config) - do not
    #      reuse another experiment's output, so the caller must separate paths per experiment.
    snapshot_path: Path | None = None
    restored = False
    _snap_dir = mem_cfg.get("snapshot_dir")
    if _snap_dir:
        snapshot_path = Path(_snap_dir) / f"{_ckpt_name}.{llm_name}.snapshot.pkl"
        if snapshot_path.exists():
            try:
                memory.restore(str(snapshot_path))
                restored = True
                logging.info(f"[{llm_name}] memory restored from {snapshot_path} -> commit skipped")
            except Exception as e:
                logging.warning(f"[{llm_name}] snapshot restore failed ({e}) -> proceeding with live commit")
                restored = False

    # If no_history, skip commit (fact store build) entirely - answer does not use the fact store/retrieval
    #   (it uses only the no-history prompt), so the build result is discarded (wasteful). Treated the
    #   same as restored (snapshot reuse). Baselines (a_mem/mem0/hippo) have no no_history attribute ->
    #   getattr default False -> no effect (existing behavior preserved).
    _skip_commit = restored or bool(getattr(memory, "no_history", False))

    # Compose extract-caching + per-session checkpoint paths for 2-phase memory (e.g., a_mem).
    #   The path includes (dataset_root -> TV script, data_type, turns_per_note, llm) with session_idx
    #   as the filename -> never mixes with other model/data/tpn outputs. Unset (default) -> None =
    #   existing behavior (memory only, no caching/checkpoint).
    # mem0 uses the turns_per_batch (=5) key -> reflect it in the path tpn (mem0_ckpt/tpn5). a_mem=turns_per_note.
    _tpn = mem_cfg.get("turns_per_note") or mem_cfg.get("turns_per_batch") or 1
    _extract_cache_dir: Path | None = None
    _session_ckpt_dir: Path | None = None
    # extract_cache is always ON. Whenever dataset_root exists, per-session-cache the Phase 1 (LLM
    #   extraction) results. Since hipporag2 no longer makes _ckpt (see below), the only cost-saver on
    #   crash/relaunch resume is the extract cache (avoids re-extraction). O(N), small. Forced ON
    #   regardless of the config's extract_cache value.
    if mem_cfg.get("dataset_root"):
        _dt = mem_cfg.get("data_type") or "normal"
        _extract_cache_dir = (
            Path(mem_cfg["dataset_root"]) / _dt
            / f"{_ckpt_name}_extract" / f"tpn{_tpn}" / llm_name
        )
    # hipporag2 does not create _ckpt (per-session accumulated snapshots).
    #   Reason: hipporag2 retrieve reconstructs an arbitrary time point from the final snapshot's
    #   provenance (fact_sess_idx/passage_sess_idx/phrase_first_sess/fact_edge_sessions), so _ckpt is
    #   unnecessary for QA. _ckpt is only for commit resume, but as an accumulating graph it causes
    #   O(N^2) disk blowup (100~200GB per dataset). a_mem (retrieve loads a session snapshot) / mem0
    #   (resume) keep it.
    if mem_cfg.get("session_checkpoint") and _snap_dir and mem_cfg.get("name") != "hipporag2":
        _session_ckpt_dir = (
            Path(_snap_dir) / f"{_ckpt_name}_ckpt" / f"tpn{_tpn}" / llm_name
        )

    # a_mem's official generate_query_llm - precompute (cache) question keywords before QA.
    #   No-op if use_query_keywords=False (default). Other baselines (mem0/hippo/groupchat_mem) lack this method
    #   (hasattr branch). Keywords depend on question text, so no future leakage.
    if hasattr(memory, "precompute_query_keywords"):
        try:
            memory.precompute_query_keywords([q for q in questions if q is not None])
        except Exception as _e:
            logging.warning(
                f"[{llm_name}] precompute_query_keywords failed ({_e}) -> raw question embed")

    start = int(run_cfg.get("start_session_idx", 0))
    end = min(int(run_cfg.get("end_session_idx", len(sessions))), len(sessions))
    sess_filter = run_cfg.get("session_filter", "all")

    # session_filter validation - prevents a bad value from silently becoming a no-op.
    if sess_filter not in {"all", "even", "odd", ""}:
        raise ValueError(f"session_filter must be all/even/odd, got {sess_filter!r}")

    # Index-preserving slots (fill results in session order regardless of parallel answer completion order).
    results: list[dict | None] = [None] * (end - start)
    total_in = total_out = 0
    total_latency = 0.0
    t0 = time.perf_counter()

    # Per-cell question parallelism - if answer_max_workers > 1:
    #   (1) commit all sessions in order first (or snapshot-restore), then
    #   (2) call answer (QA) in parallel via a ThreadPool (vLLM batching, using max-num-seqs).
    #   retrieve filters by current_session_idx (session_idx < current), so post-full-commit answers
    #   give the same result as a live build / snapshot restore (zero future leakage).
    #   answer is read-only (retrieve = cos_sim + filter + LLM call), hence thread-safe; only commit
    #   (write) is sequential. default=1 keeps the interleaved (answer -> commit) sequential behavior
    #   (baseline-safe).
    answer_max_workers = max(1, int(run_cfg.get("answer_max_workers", 1)))
    # commit extract parallelism (phase 1) - distribute commit's session-independent LLM extraction
    #   (OpenIE/note/fact) over the endpoint pool via a ThreadPool. If unset, safe default:
    #   answer_max_workers if set (same endpoint-pool distribution scale), else 1 (existing sequential
    #   commit). build (phase 2) is always sequential, so it is unaffected.
    _cew = run_cfg.get("commit_extract_workers")
    if _cew is None or str(_cew) == "":
        commit_extract_workers = answer_max_workers
    else:
        commit_extract_workers = max(1, int(_cew))
    # QA scoring mode is tied to question_format so it can never be mismatched:
    #   - multi_unstructured / open_ended: the model sees no (A)-(E) labels, so a free-form answer
    #     is the only sensible output. These formats therefore use generative scoring - the model's
    #     free-form answer is mapped onto one of the options via match_option_squad (SQuAD official
    #     normalize -> token-overlap F1, argmax option), and the resulting option letter is compared
    #     to the true_answer letter. Scoring is driven purely by question_format (no qa_eval_mode knob).
    #   - multi_structured: the model is shown explicit (A)-(E) labels, so scoring uses
    #     process_answer letter extraction (extract the (A)-(E) label, compare to true_answer letter).
    _q_fmt = str(mem_cfg.get("question_format", "multi_structured")).lower().replace("-", "_")
    if _q_fmt in ("multi_unstructured", "open_ended"):
        qa_eval_mode = "generative"
    else:
        qa_eval_mode = "mcq"

    # Prepare per-session incremental save + resume after interruption
    #   If result_json_path is set, this function owns that JSON file (load -> resume -> incremental
    #   save -> final merged save). Disabled (forced None) if commit_only (there are no records).
    _result_json_path = None if commit_only else result_json_path
    _save_every = max(0, int(save_every_sessions))
    existing_by_idx: dict[Any, dict] = {}   # session_idx -> existing record (preserved on resume)
    existing_order: list[Any] = []          # session_idx order of existing records (preserve save order)
    done_idxs: set[Any] = set()             # already-scored (is_correct not None) session_idx
    if _result_json_path is not None and resume and Path(_result_json_path).exists():
        try:
            _ex = json.loads(Path(_result_json_path).read_text(encoding="utf-8"))
            for _r in _ex.get("records", []):
                _si = _r.get("session_idx")
                if _si is None:
                    continue
                if _si not in existing_by_idx:
                    existing_order.append(_si)
                existing_by_idx[_si] = _r
                if _r.get("is_correct") is not None:
                    done_idxs.add(_si)
            logging.info(
                f"[{llm_name}] resume: loaded existing result {len(existing_by_idx)} records "
                f"({len(done_idxs)} scored) <- {_result_json_path}"
            )
        except Exception as _e:  # corrupt result -> from scratch (fail-safe).
            logging.warning(f"[{llm_name}] resume: failed to load existing result ({_result_json_path}): {_e} -> from scratch")
            existing_by_idx, existing_order, done_idxs = {}, [], set()

    def _is_eligible(i: int) -> bool:
        """Whether session loop index i is a QA scoring target (session_filter + question exists)."""
        if questions[i] is None:
            return False
        if sess_filter == "even" and i % 2 != 0:
            return False
        if sess_filter == "odd" and i % 2 != 1:
            return False
        return True

    # Set of eligible session_idx (for completion decision / resume target computation). Not needed
    # when result_json_path is unused.
    _elig_idxs: set[Any] = set()
    if _result_json_path is not None:
        for i in range(start, end):
            if _is_eligible(i):
                _elig_idxs.add(sessions[i].idx)
        # resume: if rerun_sessions is unspecified and prior done sessions exist, target only the
        # incomplete eligible for QA.
        if resume and rerun_sessions is None and done_idxs:
            rerun_sessions = _elig_idxs - done_idxs
            logging.info(
                f"[{llm_name}] resume: {len(done_idxs)} of {len(_elig_idxs)} eligible done "
                f"-> reprocessing only {len(rerun_sessions)} sessions"
            )

    def _summary_from_records(recs: list[dict], wall: float, complete: bool = False) -> dict:
        """Recompute summary from merged records (token/latency also summed from records - equivalent to the accumulator)."""
        _ns = sum(1 for r in recs if r.get("is_correct") is not None)
        _nc = sum(1 for r in recs if r.get("is_correct"))
        return {
            "llm": llm_name,
            "n_sessions": len(recs), "n_scored": _ns, "n_correct": _nc,
            "accuracy": _nc / max(1, _ns),
            "accuracy_by_hop": _accuracy_by_hop(recs),
            "total_llm_latency_sec": sum(float(r.get("latency_sec") or 0.0) for r in recs),
            "total_wall_sec": wall,
            "total_input_tokens": sum(int(r.get("input_tokens") or 0) for r in recs),
            "total_output_tokens": sum(int(r.get("output_tokens") or 0) for r in recs),
            "resume_complete": bool(complete),
        }

    def _merge_records() -> list[dict]:
        """Merge existing + new records by session_idx.

        - If a new record is an actual QA attempt (has model_answer_raw or answer_error), it overrides
          the existing one.
        - If new is bare (do_qa=False, no QA attempt), keep the existing record (never overwrite an
          already-scored answer). If the session was absent before, add even a bare record (full-session
          visibility).
        - Order: preserve existing order + append sessions that appear only in new (session-loop order).
        """
        new_recs = [r for r in results if r is not None]
        if not existing_order and not existing_by_idx:
            return new_recs   # fresh run - nothing existing.
        by_idx = dict(existing_by_idx)
        order = list(existing_order)
        seen = set(order)
        for r in new_recs:
            _si = r.get("session_idx")
            _attempted = ("model_answer_raw" in r) or ("answer_error" in r)
            if _attempted:
                by_idx[_si] = r
            elif _si not in by_idx:
                by_idx[_si] = r
            # else: bare + already exists -> keep (do not overwrite).
            if _si not in seen:
                order.append(_si)
                seen.add(_si)
        return [by_idx[_si] for _si in order]

    def _flush_result_json(complete: bool = False) -> None:
        """Atomic-save the merged result so far (best-effort - failure does not kill the experiment)."""
        if _result_json_path is None:
            return
        try:
            _recs = _merge_records()
            _summ = _summary_from_records(_recs, time.perf_counter() - t0, complete=complete)
            _atomic_write_json(_result_json_path, {"summary": _summ, "records": _recs})
        except Exception as _e:
            logging.warning(f"[{llm_name}] incremental result save failed ({_result_json_path}): {_e}")

    # If every eligible session is already done on resume -> commit/QA unnecessary, return existing
    # result (+ update the flag).
    if (_result_json_path is not None and resume and rerun_sessions is not None
            and len(rerun_sessions) == 0 and done_idxs):
        _merged0 = [existing_by_idx[_si] for _si in existing_order]
        _summ0 = _summary_from_records(_merged0, time.perf_counter() - t0, complete=True)
        logging.info(f"[{llm_name}] resume: all eligible sessions done -> skip commit/QA, return existing result")
        try:
            _atomic_write_json(_result_json_path, {"summary": _summ0, "records": _merged0})
        except Exception as _e:
            logging.warning(f"[{llm_name}] resume complete-result re-save failed: {_e}")
        return {"summary": _summ0, "records": _merged0}

    def _process_qa(i: int):
        """Generate the QA answer for session i (read-only, parallel-safe). The caller performs commit
        separately. Returns: (i, rec, metrics or None)."""
        # session_filter skips only the QA call; commit runs for all sessions.
        s, q = sessions[i], questions[i]
        do_qa = not (sess_filter == "even" and i % 2 != 0) and not (sess_filter == "odd" and i % 2 != 1)
        # If rerun_sessions is set, reprocess only those session_idx (skip do_qa for the rest).
        #   Used to patch only the missing (timeout) n_scored-violating sessions in the existing result.
        #   Target session_idx directly (do NOT assume it equals list index i) - ignore even_filter
        #   (rerun_sessions is already the exact set of scoring targets). commit runs for all sessions
        #   regardless of do_qa (handled separately below).
        if rerun_sessions is not None:
            do_qa = (s.idx in rerun_sessions)
        rec: dict[str, Any] = {"session_idx": s.idx, "date": s.date, "llm": llm_name}
        m = None
        if q is not None and do_qa:
            # Isolate exceptions per session (BadRequest/timeout/length etc.) so one session's answer
            #   failure does not kill the whole cell. In parallel (ThreadPool), fut.result() raising
            #   would lose completed futures; guard against that. Sequential is isolated the same way
            #   (record answer_error only for the failed session, continue with the rest).
            try:
                # The answer (QA generation) stage uses _answer_llm (e.g., budget 10000).
                #   If _answer_llm == _commit_llm, this is a no-op (existing behavior). The swap is
                #   idempotent - parallel answer threads set it to the same value, so no harmful race.
                memory.llm = _answer_llm
                text, retrieval, metrics = memory.answer(q, current_session_idx=s.idx, current_session=s)
                if qa_eval_mode == "generative":
                    # Map the free-form answer onto an option with SQuAD official F1, recording the
                    #   per-option F1 / exact-match scores for post-hoc review. The chosen option's
                    #   letter is then compared to the true_answer letter (same binary is_correct as
                    #   the (A)-(E) path below). matched_idx=-1 means no option shared any token with
                    #   the answer -> counts as wrong (empty label).
                    _mm = match_option_squad(text, list(q.options))
                    processed = f"({chr(65 + _mm['matched_idx'])})" if _mm["matched_idx"] >= 0 else ""
                    rec["squad_f1_scores"] = _mm["f1_scores"]  # per-option SQuAD F1
                    rec["squad_em_flags"] = _mm["em_flags"]    # per-option exact-match flags
                    rec["squad_matched_idx"] = _mm["matched_idx"]
                else:
                    processed = process_answer(text, list(q.options))
                is_correct = processed == q.true_answer if q.true_answer else None
                rec.update({
                    "question": q.text, "questioner": q.questioner, "options": list(q.options),
                    "true_answer": q.true_answer, "model_answer_raw": text, "model_answer": processed,
                    # Preserve raw output - the reasoning-parser-separated thinking trace.
                    "model_answer_reasoning": retrieval.extra.get("qa_reasoning_text", ""),
                    # For post-hoc analysis: the final QA prompt + retrieval's rendered triple/history.
                    "final_qa_prompt": retrieval.extra.get("qa_final_prompt", ""),
                    "retrieval_rendered": retrieval.extra.get("qa_retrieval_rendered", ""),
                    "is_correct": is_correct, "hop_count": q.hop_count, "route_label": q.route_label,
                    "route": retrieval.extra.get("route"), "fallback_from": retrieval.extra.get("fallback_from"),
                    "records_path_info": retrieval.extra.get("records_path_info"),
                    # retrieval recall: B (full retrieved set) / C (set included in the QA prompt) - per session.
                    #   non-enhanced: retrieval_b/c (per route: session route=dialogue{session_idx} /
                    #     record route=record{h,r,t,sd}).
                    #   enhanced: retrieval_b/c_{sessions,records} (session-path=dialogue, record-path separated).
                    "retrieval_b": retrieval.extra.get("retrieval_b", []),
                    "retrieval_c": retrieval.extra.get("retrieval_c", []),
                    "retrieval_b_sessions": retrieval.extra.get("retrieval_b_sessions", []),
                    "retrieval_c_sessions": retrieval.extra.get("retrieval_c_sessions", []),
                    "retrieval_b_records": retrieval.extra.get("retrieval_b_records", []),
                    "retrieval_c_records": retrieval.extra.get("retrieval_c_records", []),
                    "latency_sec": metrics.latency_sec, "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens, "num_llm_calls": metrics.num_llm_calls,
                })
                # legacy-style per-session log
                logging.info(
                    "\n{Start answering question}\n"
                    "========================================\n"
                    f"[Session {s.idx} ({s.date})]\n\n"
                    f"- questioner: {q.questioner}\n\n"
                    f"- partial_dialogue:\n{s.partial_dialogue}\n\n"
                    f"- route: {retrieval.extra.get('route','?')}    "
                    f"(fallback_from={retrieval.extra.get('fallback_from','None')})\n\n"
                    f"question: {q.text}\n\n"
                    f"options: {list(q.options)}\n\n"
                    f"final_qa_prompt:\n{retrieval.extra.get('qa_final_prompt','')}\n\n"
                    f"retrieved_history (rendered):\n{retrieval.extra.get('qa_retrieval_rendered','')}\n\n"
                    f"model_answer_raw:\n{text}\n\n"
                    f"model_answer_reasoning (len={len(retrieval.extra.get('qa_reasoning_text',''))}):\n"
                    f"{retrieval.extra.get('qa_reasoning_text','')}\n\n"
                    f"model_answer_parsed: {processed}\n\n"
                    f"true_answer: {q.true_answer}\n\n"
                    f"is_correct: {is_correct}\n\n"
                    f"hop_count: {q.hop_count}\n\n"
                    f"elapsed_time: {metrics.latency_sec:.5f}\n\n"
                    f"input_tokens: {metrics.input_tokens}\n\n"
                    f"output_tokens: {metrics.output_tokens}\n\n"
                    f"num_llm_calls: {metrics.num_llm_calls}\n\n"
                    "========================================\n"
                )
                m = metrics
            except Exception as e:
                logging.warning(f"[{llm_name}] session {s.idx} answer failed (isolated, continuing): {e}")
                rec["answer_error"] = str(e)
        return i, rec, m

    def _save_checkpoint():
        if checkpoint_path is None:
            return
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _recs = [r for r in results if r is not None]
            _partial = {
                "summary": {"llm": llm_name, "n_processed": len(_recs),
                            "n_scored": sum(1 for r in _recs if r.get("is_correct") is not None),
                            "n_correct": sum(1 for r in _recs if r.get("is_correct"))},
                "records": _recs,
            }
            save_pickle(_partial, checkpoint_path)
        except Exception as e:
            logging.warning(f"[{llm_name}] checkpoint save failed: {e}")

    # The commit stage always uses _commit_llm (original budget). If answer_llm=None, _commit_llm==llm (no-op).
    memory.llm = _commit_llm

    if commit_only:
        # commit-only (snapshot build) mode - commit all sessions, save only the snapshot, skip QA
        #   If restored (snapshot already exists), skip commit too (already built). If snapshot_dir is
        #   unset, commit runs but there is nothing to save -> warn (the built result is discarded).
        if not restored:
            # commit 2-phase parallel (extract in parallel -> build sequential).
            _commit_all_sessions(memory, sessions, start, end, llm_name, commit_extract_workers,
                                 extract_cache_dir=_extract_cache_dir, session_ckpt_dir=_session_ckpt_dir)
            if snapshot_path is not None:
                try:
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    memory.snapshot(str(snapshot_path))
                    logging.info(f"[{llm_name}] (commit-only) memory snapshot saved -> {snapshot_path}")
                except Exception as e:
                    logging.warning(f"[{llm_name}] (commit-only) snapshot save failed: {e}")
            else:
                logging.warning(
                    f"[{llm_name}] commit_only=True but snapshot_dir is unset - "
                    f"the built result is not saved (QA skipped too). Setting snapshot_dir is recommended."
                )
        else:
            logging.info(f"[{llm_name}] (commit-only) snapshot already exists -> skip commit/save")
        # The QA (answer) stage is not performed -> return the summary with empty results.
        wall = time.perf_counter() - t0
        summary = {
            "llm": llm_name, "commit_only": True,
            "n_sessions": 0, "n_scored": 0, "n_correct": 0, "accuracy": 0.0,
            "accuracy_by_hop": {},
            "total_llm_latency_sec": total_latency, "total_wall_sec": wall,
            "total_input_tokens": total_in, "total_output_tokens": total_out,
        }
        return {"summary": summary, "records": []}

    if answer_max_workers > 1:
        # Phase 1: commit all (session order) - skip if restored/no_history
        if not _skip_commit:
            # commit 2-phase parallel (extract in parallel -> build sequential).
            _commit_all_sessions(memory, sessions, start, end, llm_name, commit_extract_workers,
                                 extract_cache_dir=_extract_cache_dir, session_ckpt_dir=_session_ckpt_dir)
            # Save a snapshot after build completes (if snapshot_dir is set).
            if snapshot_path is not None:
                try:
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    memory.snapshot(str(snapshot_path))
                    logging.info(f"[{llm_name}] memory snapshot saved -> {snapshot_path}")
                except Exception as e:
                    logging.warning(f"[{llm_name}] snapshot save failed: {e}")
        # Phase 2: answer in parallel (vLLM batching)
        logging.info(f"[{llm_name}] answer parallel start (max_workers={answer_max_workers}, sessions={end - start})")
        done_cnt = 0
        with ThreadPoolExecutor(max_workers=answer_max_workers) as pool:
            futs = [pool.submit(_process_qa, i) for i in range(start, end)]
            for fut in as_completed(futs):
                i, rec, m = fut.result()
                results[i - start] = rec
                if m is not None:
                    total_latency += m.latency_sec
                    total_in += m.input_tokens
                    total_out += m.output_tokens
                done_cnt += 1
                if done_cnt % 50 == 0:
                    logging.info(f"[{llm_name}] answered {done_cnt}/{end - start}")
                if checkpoint_every >= 1 and done_cnt % checkpoint_every == 0:
                    _save_checkpoint()
                # Per-session incremental JSON save (interruption handling). The main thread
                #   (as_completed) runs alone, so there is no read race on results. No-op if unset
                #   (_save_every=0).
                if _save_every and done_cnt % _save_every == 0:
                    _flush_result_json(complete=False)
    else:
        # existing sequential (answer -> commit interleaved) - default, baseline-safe
        for i in range(start, end):
            _, rec, m = _process_qa(i)
            results[i - start] = rec
            if m is not None:
                total_latency += m.latency_sec
                total_in += m.input_tokens
                total_out += m.output_tokens
            # Do not re-commit if restored via snapshot (prevents double commit).
            # _process_qa set memory.llm=_answer_llm right before answer, so restore _commit_llm
            #   right before commit (no-op if answer_llm=None).
            if not _skip_commit:
                memory.llm = _commit_llm
                memory.commit(sessions[i])
            if (i + 1) % 50 == 0:
                logging.info(f"[{llm_name}] processed {i + 1}/{end}")
            if checkpoint_path is not None and checkpoint_every >= 1 and (i + 1) % checkpoint_every == 0:
                _save_checkpoint()
            # Per-session incremental JSON save (interruption handling). The sequential loop runs alone, so no race.
            if _save_every and (i + 1 - start) % _save_every == 0:
                _flush_result_json(complete=False)
        # Save a snapshot after build completes (only when snapshot_dir is set and not restore/no_history).
        if snapshot_path is not None and not _skip_commit:
            try:
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                memory.snapshot(str(snapshot_path))
                logging.info(f"[{llm_name}] memory snapshot saved -> {snapshot_path}")
            except Exception as e:
                logging.warning(f"[{llm_name}] snapshot save failed: {e}")

    # Drop None slots (do_qa=False still yields a rec -> no None; defensive filter).
    results = [r for r in results if r is not None]
    wall = time.perf_counter() - t0

    # If result_json_path is used -> final save + return based on merged (existing done + new).
    #   Completion (resume_complete) = every eligible session is scored (is_correct not None).
    if _result_json_path is not None:
        merged = _merge_records()
        _done_now = {r.get("session_idx") for r in merged if r.get("is_correct") is not None}
        complete = _elig_idxs.issubset(_done_now) if _elig_idxs else True
        summary = _summary_from_records(merged, wall, complete=complete)
        _flush_result_json(complete=complete)   # final atomic save (reflects the complete flag).
        return {"summary": summary, "records": merged}

    # Existing path (result_json_path unused) - accumulator-based summary, 100% identical behavior.
    n_scored = sum(1 for r in results if r.get("is_correct") is not None)
    n_correct = sum(1 for r in results if r.get("is_correct"))
    summary = {
        "llm": llm_name,
        "n_sessions": len(results), "n_scored": n_scored, "n_correct": n_correct,
        "accuracy": n_correct / max(1, n_scored),
        "accuracy_by_hop": _accuracy_by_hop(results),
        "total_llm_latency_sec": total_latency, "total_wall_sec": wall,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
    }
    return {"summary": summary, "records": results}


def run(
    cfg: Config | dict[str, Any],
    llm_names: list[str] | None = None,
    save_per_llm: Path | None = None,
    checkpoint_per_llm: Path | None = None,
    checkpoint_every: int = 100,
    commit_only: bool | None = None,
    rerun_missing_sessions: dict[str, set[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Returns: {llm_name: {"summary": ..., "records": ...}, ...}.

    Args:
        cfg: sconf Config or dict.
        llm_names: list of LLM names to run. None means all of cfg.llms. If given, filter to those names.
        save_per_llm: if given, disk-save `<save_per_llm>.{llm_name}.{suffix}` on each thread's
                      completion. Preserves partial results during a long-running experiment. If None,
                      only main() saves in bulk.
        checkpoint_per_llm: if given, overwrite-save `<checkpoint_per_llm>.{llm_name}.ckpt.pkl` per LLM
                            every `checkpoint_every` sessions. Ablation gives a separate path per grid
                            and deletes that ckpt after the final result save.
        checkpoint_every: ckpt save interval (in sessions, default 100).
        commit_only: if True, do only commit+snapshot and skip QA (snapshot build only).
                     None (default) uses cfg.run.commit_only (False if unset = existing behavior).
                     If the CLI (`--commit-only`) is specified, this arg overrides the cfg value.

    answer budget split: if cfg.qa.answer_budget is positive, commit/build uses cfg's original budget
      (extra_body.thinking_token_budget) and final QA answering uses answer_budget, run separately
      (_apply_answer_budget + memory.llm swap). If unset, a single LLM (existing behavior).
    """
    cfg_d = cfg.asdict() if isinstance(cfg, Config) else cfg
    # run: guard against AttributeError when the key exists but the value is None (yaml has `run:`
    #   with no children) - None.get would fail.
    seed_everything(int((cfg_d.get("run") or {}).get("seed", 42)))

    # commit_only: arg takes priority; if unset (None), use cfg.run.commit_only (default False).
    if commit_only is None:
        commit_only = bool((cfg_d.get("run") or {}).get("commit_only", False))
    # answer budget - reasoning limit at final-QA-answering time (independent of cache budget).
    #   Unset -> None (single LLM, existing behavior). The grid entry reads method_cfg.qa.answer_budget;
    #   this entry reads top-level cfg.qa.answer_budget.
    _answer_budget = (cfg_d.get("qa") or {}).get("answer_budget")

    llms_list = _normalize_llms(cfg_d)
    if llm_names:
        wanted = set(llm_names)
        filtered = [c for c in llms_list if c.get("name", c["model_name"]) in wanted]
        missing = wanted - {c.get("name", c["model_name"]) for c in filtered}
        if missing:
            raise ValueError(f"names given in --llms not found in cfg.llms: {sorted(missing)}")
        llms_list = filtered

    # Pre-ping every vLLM endpoint - raise immediately if any is dead.
    for llm_cfg in llms_list:
        _check_vllm_endpoint(llm_cfg)

    emb_cfg = dict(cfg_d["embedder"])
    embedder = make_embedder(backend=emb_cfg["backend"], model=emb_cfg["model"])

    data_cfg = dict(cfg_d["data"])
    # assistant == QA policy - load the bundle per LLM.
    per_llm_mode = bool(data_cfg.get("per_llm_precomputed", False))

    # File-backed embedding cache - if `embedder.cache: true`, wrap the base embedder in
    #   CachedEmbedder: look up the cache per embedding target -> use on hit, on miss embed live then
    #   add + save (autoflush) -> reused by later sessions/runs. Each text is embedded once regardless
    #   of LLM/session. If path is unset, {dataset_root}/{data_type}/baseline_emb_cache.{model}.pickle.
    #   Wrapped *once* here so all LLM threads share the same instance (EmbeddingCache path-lock safe).
    if emb_cfg.get("cache"):
        _cpath = emb_cfg.get("cache_path")
        if not _cpath:
            _safe = embedder.name.replace(":", "_").replace("/", "_")
            # Guard against KeyError when dataset_root is absent (v1_2_1 format has only pickle_path)
            #   - fall back to the parent directory of pickle_path.
            _root = data_cfg.get("dataset_root") or str(
                Path(data_cfg.get("pickle_path", ".")).parent
            )
            _cpath = str(
                Path(_root) / data_cfg.get("data_type", "normal")
                / f"baseline_emb_cache.{_safe}.pickle"
            )
        embedder = CachedEmbedder(embedder, _cpath)
        logging.info(f"[embedder] file-backed cache enabled -> {_cpath}")

    shared_bundle: dict | None = None
    _data_format = data_cfg.get("format", "v1_2_2")  # "v1_2_1" | "v1_2_2" (default)
    if not per_llm_mode:
        if _data_format == "v1_2_1":
            # legacy raw pickle loader - reproduce/data/{data_type}/{ie_type}/trial{N}.pickle
            shared_bundle = load_v1_2_1_trial(pickle_path=data_cfg["pickle_path"])
        else:
            shared_bundle = load_converted_dataset(
                dataset_root=data_cfg["dataset_root"],
                data_type=data_cfg.get("data_type", "normal"),
                trial_idx=int(data_cfg.get("trial", 0)),
                ie_type=data_cfg.get("ie_type", "openie"),
                normalization=data_cfg.get("normalization", "raw"),
            )

    # deepcopy - a nested dict (cos_sim settings, etc.) shared across threads would race.
    # Aligned with the deepcopy pattern in the grid entry (asymmetry fix).
    mem_cfg_base = copy.deepcopy(cfg_d["memory"])
    # run: guard against a None value for the key.
    run_cfg = dict(cfg_d.get("run") or {})

    # Prefer the yaml's parallel_max_workers, default 1 (not len(llms_list) - avoids endpoint overload).
    max_workers = int(run_cfg.get("parallel_max_workers", 1))
    max_workers = max(1, min(max_workers, len(llms_list)))

    def _one(llm_cfg: dict) -> tuple[str, dict[str, Any]]:
        # A friendly error instead of KeyError when both name/model_name are absent.
        if "name" not in llm_cfg and "model_name" not in llm_cfg:
            raise KeyError(f"LLM cfg has neither name nor model_name: {llm_cfg!r}")
        llm_name = llm_cfg.get("name", llm_cfg["model_name"])
        logging.info(f"[run_baseline] sweep LLM: {llm_name} ({_resolve_base_url(llm_cfg)})")
        # per-session incremental save + resume variables - computed outside try (before bundle load,
        #   so unaffected by exceptions and safely referenced in the save block below). Resume is
        #   active only on a normal launch (not explicit --rerun-missing, not commit_only); run_one_llm
        #   owns the _p JSON (load -> resume -> incremental -> final).
        _rr = (rerun_missing_sessions or {}).get(llm_name)
        _p = (_per_llm_out_path(save_per_llm, llm_name)
              if save_per_llm is not None else None)
        _use_resume = (_p is not None) and (not commit_only) and (_rr is None)
        _save_every = int(run_cfg.get("result_save_every", 1)) if _use_resume else 0
        # no_history = parametric answer with no retrieval/history (uses only question + no-history
        #   prompt; no fact store/retrieval at all). So the per-llm record/route cache is unnecessary ->
        #   load only the base dataset (sessions+questions) so models with incomplete records (e.g.,
        #   gemma t1/t2) also run immediately. commit (fact store build) is skipped in run_one_llm (_skip_commit below).
        _no_history = bool(mem_cfg_base.get("no_history"))
        try:
            if per_llm_mode and not _no_history:
                if _data_format == "v1_2_1":
                    raise ValueError(
                        "data.format=v1_2_1 cannot be used with per_llm_precomputed=true. "
                        "A v1.2.1 raw pickle has no per-LLM cache."
                    )
                bundle = load_converted_dataset(
                    dataset_root=data_cfg["dataset_root"],
                    data_type=data_cfg.get("data_type", "normal"),
                    trial_idx=int(data_cfg.get("trial", 0)),
                    ie_type=data_cfg.get("ie_type", "openie"),
                    normalization=data_cfg.get("normalization", "raw"),
                    per_llm_name=llm_name,
                    # Pass cache_budget - in sync with the grid entry. If omitted, the per-llm
                    #   record/route cache would load from top-level (non-budget) and ignore the b06000
                    #   cache. None keeps existing behavior.
                    cache_budget=mem_cfg_base.get("cache_budget"),
                )
            elif per_llm_mode and _no_history:
                # no_history -> do not load the per-llm record/route cache (load base). Since answer uses
                #   only the no-history prompt, the result is identical even without records (result-preserving).
                bundle = load_converted_dataset(
                    dataset_root=data_cfg["dataset_root"],
                    data_type=data_cfg.get("data_type", "normal"),
                    trial_idx=int(data_cfg.get("trial", 0)),
                    ie_type=data_cfg.get("ie_type", "openie"),
                    normalization=data_cfg.get("normalization", "raw"),
                )
            else:
                bundle = shared_bundle
            sessions = bundle["sessions"]
            questions = bundle["questions"]
            # deepcopy - a nested dict mutation (e.g., cos_sim_rerank settings) shared across threads
            # would race. Same policy as the grid entry.
            mem_cfg = copy.deepcopy(mem_cfg_base)
            # Auto-inject the keys the memory's _init_embedding_caches expects (previously not injected,
            # so all caches were disabled and the entry embedded live every time).
            # Same pattern as the grid entry.
            _trial_idx = int(data_cfg.get("trial", 0))
            mem_cfg["dataset_root"] = data_cfg.get("dataset_root")
            mem_cfg["data_type"] = data_cfg.get("data_type", "normal")
            mem_cfg["trial"] = _trial_idx
            mem_cfg["llm_name"] = llm_name
            # Compute context_size dynamically from each LLM's serving max-model-len:
            #   context_size = max_model_len - max_tokens - 1000 (reserves max_tokens output space + a
            #   1000 sentinel -> guarantees prompt + output < serving limit, avoiding vLLM 400). If each
            #   LLM's config sets max_model_len, use this instead of memory.context_size (serving limits
            #   differ per model/variant).
            _mml = llm_cfg.get("max_model_len")
            if _mml:
                _cs = int(_mml) - int(llm_cfg.get("max_tokens", 1024)) - 1000
                mem_cfg["context_size"] = max(512, _cs)  # lower-bound guard (avoids negative/too-small)
            mem_cfg["ie_type"] = data_cfg.get("ie_type", "openie")
            mem_cfg["normalization"] = data_cfg.get("normalization", "raw")
            # None fallback - with yaml `embedder.model: null`, .get("model", default) returns None
            # (the value is None, not a missing key), yielding cache path `.../None/...`. Use `or default`
            # so None also becomes the default (prevents silent miss).
            _embedder_model = (cfg_d.get("embedder") or {}).get("model") or "text-embedding-3-small"
            mem_cfg["embedder_model"] = _embedder_model
            # Inject precomputed query nodes (per_llm mode, same pattern as the grid entry).
            #   key = session_idx (= current_session_idx). If present, skip the query-node LLM call on the
            #   record route.
            # precomputed query nodes - load the per-K nq{K} cache (K = num_query_nodes), same convention
            #   as the grid entry. Fallback: nq{K} absent -> query_node_{norm}.json ->
            #   bundle["query_nodes"] -> {} (live extraction).
            _K = mem_cfg.get("num_query_nodes")
            _K = int(_K) if _K is not None else None
            _query_nodes = load_query_nodes_for(
                dataset_root=data_cfg.get("dataset_root"),
                data_type=data_cfg.get("data_type", "normal"),
                trial_idx=int(data_cfg.get("trial", 0)),
                llm_name=llm_name,
                normalization=data_cfg.get("normalization", "raw"),
                cache_budget=mem_cfg.get("cache_budget"),
                num_query_nodes=_K,
            ) or (bundle.get("query_nodes") or {})
            if _query_nodes:
                mem_cfg["precomputed_query_nodes"] = {int(k): v for k, v in _query_nodes.items()}
            # Inject record embeddings (for the memory's cos_sim re-rank).
            # Prefer partial (experiments use the partial memory), fall back to entire.
            # The memory's record-embedding map is read-only (`.get(key)` only, no mutate). So sharing the
            # same dict from shared_bundle as mem_cfg["record_embeddings"] across all threads (by reference)
            # does not race. If a mutation is added later, deepcopy or an RLock would be needed.
            _record_emb = bundle.get("partial_record_emb") or bundle.get("entire_record_emb")
            if _record_emb:
                mem_cfg["record_embeddings"] = _record_emb
            # shared_bundle trial mismatch check.
            # In non-per-LLM mode, shared_bundle is loaded at module level for one trial.
            # If mem_cfg["trial"] overrides to a different trial, record_emb would be the wrong trial's ->
            # a silent, wrong cos_sim.
            # ERROR level + raise to prevent a silent wrong-metric comparison (policy: a small difference
            # invalidates the metric).
            if (not per_llm_mode) and _record_emb:
                _bundle_trial = bundle.get("trial_idx")
                if _bundle_trial is not None and _bundle_trial != _trial_idx:
                    _msg = (
                        f"[run_baseline] shared_bundle trial mismatch - bundle.trial={_bundle_trial} vs mem_cfg.trial={_trial_idx}. "
                        f"In non-per-LLM mode the module-level shared_bundle is loaded for a single trial only. "
                        f"cos_sim_rerank's record_emb may be the wrong trial. "
                        f"If you need per-trial-accurate record_emb, use per_llm_precomputed=true or launch a single-trial run."
                    )
                    logging.error(_msg)
                    raise ValueError(_msg)
            _seed = int((cfg_d.get("run") or {}).get("seed", 42))
            # LLM for commit/build (original budget - cfg's thinking_token_budget as-is).
            llm = _make_llm_from_cfg(llm_cfg, seed=_seed)
            # answer budget split - if answer_budget is set, create a separate budget-overridden LLM
            #   for final QA answering. If unset (None), answer_llm=None (run_one_llm uses the commit
            #   llm for answer too = existing behavior).
            answer_llm = None
            if _answer_budget is not None and _answer_budget != "":
                _ans_cfg = _apply_answer_budget(llm_cfg, _answer_budget)
                # If _apply_answer_budget returned the original as-is (e.g., enable_thinking=false), no separate LLM needed.
                if _ans_cfg is not llm_cfg:
                    answer_llm = _make_llm_from_cfg(_ans_cfg, seed=_seed)
            ckpt_path = None
            if checkpoint_per_llm is not None:
                # Add PID - avoids ckpt collisions when the same LLM runs in parallel.
                import os as _os
                ckpt_path = checkpoint_per_llm.with_name(
                    checkpoint_per_llm.name + f".{llm_name}.pid{_os.getpid()}.ckpt.pkl"
                )
            out = run_one_llm(llm, llm_name, embedder, sessions, questions, mem_cfg, run_cfg,
                              checkpoint_path=ckpt_path, checkpoint_every=checkpoint_every,
                              answer_llm=answer_llm, commit_only=commit_only,
                              rerun_sessions=_rr,
                              result_json_path=(_p if _use_resume else None),
                              save_every_sessions=_save_every, resume=_use_resume)
        except Exception as e:
            # Catch so one LLM thread's failure does not affect others - return an empty result.
            # A partial failure is an inconvenience; proceed automatically.
            logging.error(f"[run_baseline] [{llm_name}] thread failed: {type(e).__name__}: {e}", exc_info=True)
            out = {"summary": {"llm": llm_name, "n_sessions": 0, "n_scored": 0, "n_correct": 0,
                               "accuracy": 0.0, "accuracy_by_hop": {},
                               "total_llm_latency_sec": 0.0, "total_wall_sec": 0.0,
                               "total_input_tokens": 0, "total_output_tokens": 0,
                               "error": f"{type(e).__name__}: {e}"},
                   "records": []}
        # Partial save immediately inside the thread - protects long-running experiments.
        if _p is not None:
            try:
                if _rr and _p.exists():
                    # Explicit --rerun-missing mode: without changing the existing json's structure/format,
                    #   replace only the rerun (missing) sessions' record elements with out's reprocessed
                    #   results + recompute summary (preserving metadata). Sessions that failed again
                    #   (model_answer None) are not replaced (existing None kept).
                    _ex = json.loads(_p.read_text(encoding="utf-8"))
                    _new = {r.get("session_idx"): r for r in out["records"]
                            if r.get("session_idx") in _rr and r.get("model_answer") is not None}
                    for _i, _r in enumerate(_ex["records"]):
                        _si = _r.get("session_idx")
                        if _si in _new:
                            _ex["records"][_i] = _new[_si]
                    _recs = _ex["records"]
                    _sm = _ex["summary"]   # preserve keys/metadata (n_sessions, total_wall_sec, llm), recompute only values
                    _sm["n_scored"] = sum(1 for r in _recs if r.get("is_correct") is not None)
                    _sm["n_correct"] = sum(1 for r in _recs if r.get("is_correct"))
                    _sm["accuracy"] = (_sm["n_correct"] / _sm["n_scored"]) if _sm["n_scored"] else 0.0
                    _sm["accuracy_by_hop"] = _accuracy_by_hop(_recs)
                    _sm["total_input_tokens"] = sum(int(r.get("input_tokens") or 0) for r in _recs)
                    _sm["total_output_tokens"] = sum(int(r.get("output_tokens") or 0) for r in _recs)
                    _sm["total_llm_latency_sec"] = sum(float(r.get("latency_sec") or 0.0) for r in _recs)
                    _atomic_write_json(_p, _ex)
                    logging.info(f"[run_baseline rerun] patched {len(_new)}/{len(_rr)} sessions -> {_p}")
                elif _use_resume:
                    # resume path: run_one_llm already owns and atomic-saved _p (incremental + final).
                    #   A duplicate (non-atomic) caller write risks truncation/overwrite, so skip it -
                    #   but on thread error, preserve the disk partial (out may be an empty-records error).
                    logging.info(f"[run_baseline] resume-owned result saved (run_one_llm): {llm_name} -> {_p}")
                elif not out.get("summary", {}).get("error"):
                    # non-resume path (e.g., commit_only) - save out (atomic). On thread error, skip to preserve disk.
                    _atomic_write_json(_p, out)
                    logging.info(f"[run_baseline] partial save: {llm_name} -> {_p}")
                else:
                    logging.warning(
                        f"[run_baseline] {llm_name} thread error - preserving existing result (skip overwrite): {_p}")
            except Exception as e:
                logging.warning(f"[run_baseline] partial save failed ({llm_name}): {e}")
        return llm_name, out

    outputs: dict[str, dict[str, Any]] = {}
    try:
        if max_workers == 1 or len(llms_list) == 1:
            for llm_cfg in llms_list:
                name, out = _one(llm_cfg)
                outputs[name] = out
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_one, cfg): cfg for cfg in llms_list}
                try:
                    for fut in as_completed(futures):
                        name, out = fut.result()
                        outputs[name] = out
                except Exception:
                    # One thread raised -> cancel unstarted futures. Already-running threads continue
                    # in the background (Python ThreadPoolExecutor limitation - cannot force-terminate).
                    # Partial results are preserved via save_per_llm.
                    for f in futures:
                        f.cancel()
                    logging.error(
                        "[run_baseline] LLM thread failed - unstarted futures canceled. "
                        "Background running threads continue to completion (Python limitation). "
                        "Partial results are preserved at the save_per_llm path."
                    )
                    raise
    finally:
        # If CachedEmbedder, flush the cache to disk at exit (normal or exception) - persist embeddings
        #   still in memory (autoflush-interval leftovers) so the next run reuses them.
        if isinstance(embedder, CachedEmbedder):
            try:
                embedder.flush()
            except Exception as _e:
                logging.warning(f"[embedder] cache flush failed: {_e}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline entry (multi-LLM sweep, parallel).")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out", type=Path, default=Path("./result/simulate_result.json"),
                        help="result base path. Each LLM's result is saved separately as "
                             ".{llm_name}.json (partial save on thread completion, then an idempotent "
                             "re-save by main() when all finish). Results are json only (no pkl).")
    parser.add_argument("--llms", type=str, default="",
                        help="comma-separated LLM names. If omitted, sweep all of cfg.llms.")
    parser.add_argument("--commit-only", action="store_true",
                        help="Do only commit+snapshot and skip QA (answer) - snapshot build only. "
                             "If unset, uses cfg.run.commit_only (default False).")
    parser.add_argument("--rerun-missing", action="store_true",
                        help="Reprocess only the missing sessions (is_correct=None, timeout, etc.) in an "
                             "existing --out result -> replace only those record elements + recompute "
                             "summary (structure/format unchanged). Commit is skipped via snapshot restore.")
    args, left_argv = parser.parse_known_args()

    base = CONFIG_DIR / "base.yaml"
    cfg = Config(args.config, default=str(base))
    if left_argv:
        cfg.argv_update(left_argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    llm_names = [s.strip() for s in args.llms.split(",") if s.strip()] or None
    # If --commit-only is given, override to True; if not, None (= use cfg.run.commit_only).
    _commit_only = True if args.commit_only else None

    # rerun_missing: identify missing (is_correct=None) sessions in the existing result file and
    #   reprocess only those. Commit is skipped via snapshot restore; only missing sessions are QA'd ->
    #   _one()'s save branch patches the existing json.
    _rerun_map: dict[str, set[int]] | None = None
    if args.rerun_missing:
        _cfg_d = cfg.asdict() if isinstance(cfg, Config) else cfg
        _names = llm_names or [c.get("name", c["model_name"]) for c in _normalize_llms(_cfg_d)]
        _rerun_map = {}
        for _ln in _names:
            _p = _per_llm_out_path(args.out, _ln)
            if not _p.exists():
                logging.warning(f"[rerun] no existing result -> skip: {_p}"); continue
            _exj = json.loads(_p.read_text(encoding="utf-8"))
            _miss = {r.get("session_idx") for r in _exj.get("records", [])
                     if r.get("is_correct") is None and r.get("session_idx") is not None}
            if _miss:
                _rerun_map[_ln] = _miss
                logging.info(f"[rerun] {_ln}: {len(_miss)} missing sessions to reprocess")
        if not _rerun_map:
            logging.info("[rerun] 0 missing sessions - nothing to reprocess. Exiting."); return

    # save_per_llm=args.out -> each thread disk-saves its own result on completion.
    outputs = run(cfg, llm_names=llm_names, save_per_llm=args.out, commit_only=_commit_only,
                  rerun_missing_sessions=_rerun_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.rerun_missing:
        # rerun mode: _one()'s save branch already patched the existing json. Overwriting here with out
        #   (only missing sessions filled, the rest bare recs) would lose the patch -> skip the final
        #   re-save, print only the summary.
        for llm_name, out in outputs.items():
            print(f"=== {llm_name} (rerun) -> {_per_llm_out_path(args.out, llm_name)} ===")
            print(json.dumps(out.get("summary", {}), indent=2, ensure_ascii=False))
        return
    for llm_name, out in outputs.items():
        # Results are json only (no pkl). If _one already saved, this is an idempotent re-save.
        # On thread error (out.summary.error), _one preserved the disk partial/complete result, so do
        #   not overwrite it here with an empty-records error (prevents data loss). A normal out is atomic re-saved.
        p = _per_llm_out_path(args.out, llm_name)
        if not out.get("summary", {}).get("error"):
            _atomic_write_json(p, out)
        else:
            logging.warning(f"[run_baseline] {llm_name} thread error - preserving existing result (skip final re-save): {p}")
        print(f"=== {llm_name} -> {p} ===")
        print(json.dumps(out["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
