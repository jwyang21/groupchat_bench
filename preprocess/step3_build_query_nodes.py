"""Step 3 - identify the query nodes of every question (GroupChat-Mem relational view).

For question i (asked in session i), GroupChat-Mem looks at the relational memory available at
that moment,

    R_i = entire-session records of sessions 0 .. i-1   +   partial records of session i

and asks the LLM which of R_i's nodes the question is about (two-stage prompt:
`prompts/query_node_id.txt` over node batches of 100, then `prompts/query_node_id_select.txt`
over the merged candidates). The top-K nodes (K = 3) seed the 2-hop record traversal at run
time. The run script can do this live, but the cache makes runs much cheaper and repeatable.
This step calls the *same* code the runtime uses (`GroupChatMem._identify_query_nodes`,
`RelationalGraph.combined_until`), so the cache equals what a live run would compute.

Requires steps 1-2 for the chosen LLM (records). Output, per (show, name setting, trial):
    .../per-trial/t{N}/per-llm/{llm}/b06000/query_node_raw_nq3.json            {"<i>": [node, ...]}
    .../per-trial/t{N}/per-llm/{llm}/b06000/query_node_raw_nq3_attempted.json  {"<i>": 1}

`_attempted` lists the questions whose LLM calls succeeded; an empty list there is a genuine
"no matching node" answer. A question that failed or hit `--per-question-timeout` is *not*
written at all, so a later run retries it (the loader treats missing entries as "identify live").
Re-running the same command resumes.

The LLM is built from the experiment config (`--config`, the `llms` entry named `--llm`) with
the cache-time reasoning setting used in the paper: thinking_token_budget 6000 for budgeted
models, reasoning_effort=medium for gpt-oss.

Usage:
  PYTHONPATH=src python preprocess/step3_build_query_nodes.py --data-root data --shows friends \
      --config configs/groupchat_mem/groupchat_mem.qwen.friends.newname.t0.yaml \
      --llm qwen3.5-35b-a3b-on --workers 12
"""
from __future__ import annotations

import argparse
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml

from common import (BUDGET_SEGMENT, DATA_TYPES, SHOWS, TRIALS, load_json, parse_list,
                    per_llm_dir, query_nodes_path, save_json, show_root)

from groupchat_qa.memory.groupchat_mem import GroupChatMem, RelationalGraph
from groupchat_qa.pipeline.data_loader import load_converted_dataset
from groupchat_qa.utils.filesys_utils import load_prompt
from groupchat_qa.utils.types import CallMetrics

CACHE_THINKING_BUDGET = 6000


def load_llm_cfg(config: Path, llm: str) -> tuple[dict, int | None]:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    # GroupChat-Mem configs wrap everything in one top-level method block; baseline configs are flat.
    method_cfg = cfg if "llms" in cfg else next(iter(cfg.values()))
    for entry in method_cfg.get("llms", []):
        if llm in (entry.get("name"), entry.get("model_name")):
            seed = (method_cfg.get("run") or {}).get("seed")
            return copy.deepcopy(entry), seed
    raise SystemExit(f"--llm {llm!r} not found in the llms list of {config}")


def cache_reasoning(llm_cfg: dict) -> dict:
    """Apply the cache-time reasoning setting (the experiment configs hold the QA setting)."""
    eb = dict(llm_cfg.get("extra_body") or {})
    name = str(llm_cfg.get("model_name") or llm_cfg.get("name") or "")
    if "gpt-oss" in name:
        ctk = dict(eb.get("chat_template_kwargs") or {})
        ctk["reasoning_effort"] = "medium"
        eb["chat_template_kwargs"] = ctk
        eb.pop("thinking_token_budget", None)
    elif "thinking_token_budget" in eb:
        eb["thinking_token_budget"] = CACHE_THINKING_BUDGET
        llm_cfg["max_tokens"] = max(int(llm_cfg.get("max_tokens") or 0),
                                    CACHE_THINKING_BUDGET + 6384)
    llm_cfg["extra_body"] = eb
    return llm_cfg


def make_identifier(llm, k: int, batch_size: int):
    """The attributes `GroupChatMem._identify_query_nodes` reads, without building a full memory
    (no embeddings / data caches are needed for this step)."""
    return SimpleNamespace(
        llm=llm, num_query_nodes=k, query_node_batch_size=batch_size,
        query_node_batch_workers=1, precomputed_query_nodes={},
        _p_query_node_id=load_prompt("query_node_id.txt"),
        _p_query_node_select=load_prompt("query_node_id_select.txt"),
    )


def run_one(ident, data_root: Path, show: str, dt: str, trial: int, llm: str, k: int,
            workers: int, timeout: float, save_every: int, overwrite: bool) -> None:
    out_path = query_nodes_path(data_root, show, dt, trial, llm, k)
    att_path = out_path.with_name(out_path.stem + "_attempted.json")
    bundle = load_converted_dataset(
        dataset_root=show_root(data_root, show), data_type=dt, trial_idx=trial,
        ie_type="openie", normalization="raw", per_llm_name=llm,
        cache_budget=int(BUDGET_SEGMENT.lstrip("b")))
    sessions, questions = bundle["sessions"], bundle["questions"]

    result: dict[str, list[str]] = {}
    attempted: set[str] = set()
    if not overwrite and out_path.exists() and att_path.exists():
        result = load_json(out_path)
        attempted = set(load_json(att_path))
        result = {i: v for i, v in result.items() if i in attempted}
    todo = [j for j, s in enumerate(sessions) if str(s.idx) not in attempted]
    print(f"[{show}/{dt}/t{trial}] {len(attempted)}/{len(sessions)} done, {len(todo)} to identify")
    if not todo:
        return

    # R accumulates the entire-session records with their session index; each question then sees
    # only sessions < i (+ its own partial records) via combined_until (no future information).
    graph = RelationalGraph()
    for s in sessions:
        for r in s.entire_records or ():
            graph.insert_record(r.head, r.relation, r.tail, r.start_date, s.idx)

    lock = threading.Lock()
    state = {"since_save": 0, "fail": 0}

    def _flush() -> None:  # caller holds the lock
        save_json(result, out_path)
        save_json({i: 1 for i in sorted(attempted, key=int)}, att_path)
        state["since_save"] = 0

    def _compute(j: int, box: dict) -> None:
        s, q = sessions[j], questions[j]
        nodes = graph.combined_until(s.idx, s.partial_records).nodes()
        box["nodes"], _, _ = GroupChatMem._identify_query_nodes(ident, q, nodes, CallMetrics())

    def _one(j: int) -> None:
        box: dict = {}
        err: list = []

        def _target() -> None:
            try:
                _compute(j, box)
            except Exception as e:  # noqa: BLE001
                err.append(e)

        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout)
        key = str(sessions[j].idx)
        if th.is_alive() or err or "nodes" not in box:
            why = "timeout" if th.is_alive() else f"{type(err[0]).__name__}" if err else "no result"
            print(f"    question {key} not cached ({why}); it will be retried on the next run")
            with lock:
                state["fail"] += 1
            return
        with lock:
            result[key] = list(box["nodes"])
            attempted.add(key)
            state["since_save"] += 1
            if state["since_save"] >= save_every:
                _flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_one, todo))
    with lock:
        _flush()
    print(f"[{show}/{dt}/t{trial}] {len(attempted)}/{len(sessions)} done, "
          f"{state['fail']} not cached, {time.time() - t0:.0f}s -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path,
                    help="experiment config whose `llms` entry defines the endpoint/sampling")
    ap.add_argument("--llm", required=True, help="llm name (same as used in step 2)")
    ap.add_argument("--base-url", default=None,
                    help="override the endpoint in the config (e.g. http://localhost:8000/v1)")
    ap.add_argument("--shows", default=",".join(SHOWS))
    ap.add_argument("--data-types", default=",".join(DATA_TYPES))
    ap.add_argument("--trials", default=",".join(map(str, TRIALS)))
    ap.add_argument("--num-query-nodes", type=int, default=3, help="K (paper: 3)")
    ap.add_argument("--batch-size", type=int, default=100, help="nodes per stage-1 prompt")
    ap.add_argument("--workers", type=int, default=12, help="questions processed in parallel")
    ap.add_argument("--per-question-timeout", type=float, default=7200.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from run.run_baseline import _make_llm_from_cfg  # same LLM construction as the run scripts
    llm_cfg, seed = load_llm_cfg(args.config, args.llm)
    llm_cfg = cache_reasoning(llm_cfg)
    if args.base_url:
        llm_cfg["base_url"] = args.base_url
        llm_cfg.pop("host", None)
        llm_cfg.pop("port", None)
        llm_cfg["extra_base_urls"] = []
    print(f"llm={args.llm} extra_body={llm_cfg.get('extra_body')} max_tokens={llm_cfg.get('max_tokens')}")
    ident = make_identifier(_make_llm_from_cfg(llm_cfg, seed=seed),
                            args.num_query_nodes, args.batch_size)

    for show in parse_list(args.shows):
        for dt in parse_list(args.data_types):
            for t in parse_list(args.trials, int):
                if not per_llm_dir(args.data_root, show, dt, t, args.llm).exists():
                    print(f"[{show}/{dt}/t{t}] skipped: no records for {args.llm} (run step 2)")
                    continue
                run_one(ident, args.data_root, show, dt, t, args.llm, args.num_query_nodes,
                        args.workers, args.per_question_timeout, args.save_every, args.overwrite)


if __name__ == "__main__":
    main()
