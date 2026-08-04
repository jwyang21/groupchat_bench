"""Converted (json/npy/pickle) dataset loader - shared by v1_2 + v1_3.

Converted tree structure (v1_3 is identical):

    {dataset_root}/                # data/v1_2_2/friends or data/v1_3/friends
        date.json
        {data_type}/               # normal / newname / shuffled
            entire_session.json
            entire_session_embedding.npy
            precomputed/
                common/{ie_type}/  # openie / closedie / gt
                    {llm_name}/[bNNNNN/]                  # budget-branch cache (optional)
                        entire_session_records_{en,raw}.json
                        entire_session_records_embedding_{en,raw}_{temporal,non_temporal}.pickle
                per-trial/t{N}/
                    qa.json
                    partial_session.json
                    partial_session_embedding.npy          # budget-independent (text-embedding-3-small)
                    question_embeddings_{type}_cache.pickle # budget-independent
                    per-llm/{llm_name}/[bNNNNN/]          # v1_3 assistant==QA + budget branch (optional)
                        route.json                       # {idx: "RAG"|"R"} - per LLM, ie_type-independent
                        query_node.json                  # {idx: [candidate,...]} (optional)
                        {ie_type}/                       # openie / closedie / gt
                            partial_session_records_{en,raw}.json
                            partial_session_records_embedding_{en,raw}_{type}.pickle

cache budget branch (`bNNNNN` segment):
    Within the same dataset tree, extraction caches per budget (e.g. b06000/b08000/b10000) coexist.
    The segment is a single layer directly under the LLM folder (per-llm/{llm}, common/{ie}/{llm}).
    cache_budget=None -> no segment (the original single layout). dialogue/question embeddings are not
    outputs of the extraction LLM, so the budget branch does not affect them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from groupchat_qa.utils import (
    Question,
    Record,
    Session,
    load_pickle,
    local_first,  # read embeddings local-first (reduce NAS load)
)


# cache budget directory branch
# A directory segment so that extraction caches per budget (route/query_node/triple/record + record
# embeddings) can coexist within the same dataset tree. Instead of keeping a separate dataset_root
# copy per budget, insert a single `bNNNNN` layer directly under the LLM folder of per-llm/common.
#   - dialogue/question embeddings (text-embedding-3-small) are independent of the reasoning budget
#     (not outputs of the extraction LLM), so they stay above the LLM folder -> not affected by the
#     budget branch (avoids duplication).
#   - cache_budget=None/unset -> no segment = the original single cache layout (backward-compat).
def budget_segment(cache_budget: int | str | None) -> str:
    """cache_budget -> directory segment string.

    None / empty -> "" (no segment, original layout). Positive int (or int string) -> "bNNNNN"
    (zero-padded to 5 digits, e.g. 8000 -> "b08000"). <= 0 or non-int -> ValueError.
    """
    if cache_budget is None or cache_budget == "":
        return ""
    try:
        n = int(cache_budget)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"cache_budget must be None or a positive integer, got {cache_budget!r}"
        ) from e
    if n <= 0:
        raise ValueError(f"cache_budget must be a positive integer, got {n}")
    return f"b{n:05d}"


def _apply_budget(base: Path, cache_budget: int | str | None) -> Path:
    """Insert the budget segment under the given base path (per-llm/{llm} or common/{ie}/{llm}).

    If cache_budget is absent, return base unchanged (original behavior).
    """
    seg = budget_segment(cache_budget)
    return base / seg if seg else base


# query node cache - file separation per K (num_query_nodes)
def query_node_filename(normalization: str, num_query_nodes: int | None = None) -> str:
    """query node cache file name. With K (num_query_nodes), split by K with a `_nq{K}` suffix.

    None -> original behavior (`query_node_{norm}.json`). int -> `query_node_{norm}_nq{K}.json`.
    Must be **exactly identical** to the save file name in the query-node precompute step (read/write paths in sync).
    """
    if num_query_nodes is not None:
        return f"query_node_{normalization}_nq{int(num_query_nodes)}.json"
    return f"query_node_{normalization}.json"


def load_query_node_cache(
    base: Path | str, normalization: str, num_query_nodes: int | None = None
) -> dict[int, list]:
    """Load the query node cache under per-llm base (with budget segment) -> {int idx: [candidate,...]}.

    With K (num_query_nodes), prefer `query_node_{norm}_nq{K}.json` ->
    fall back to `query_node_{norm}.json` if absent -> {} if neither exists (runtime live extraction).
    Parse failure / non-dict is handled safely by trying the next candidate or returning {}.
    """
    base = Path(base)
    # With K given, prefer nq{K}, then always fall back to legacy (query_node_{norm}.json).
    rel_names: list[str] = []
    if num_query_nodes is not None:
        rel_names.append(query_node_filename(normalization, num_query_nodes))
    rel_names.append(query_node_filename(normalization, None))
    for rel in rel_names:
        p = base / rel
        if not p.exists():
            continue
        try:
            raw = json.loads(local_first(p).read_text(encoding="utf-8"))  # local-first (existence check on p = NAS canonical)
        except Exception:
            continue  # corrupt file -> try the next candidate
        if not isinstance(raw, dict):
            continue
        out: dict[int, list] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = v
            except (ValueError, TypeError):
                continue  # ignore keys that cannot be converted to int
        return out
    return {}


def load_query_nodes_for(
    dataset_root: Path | str,
    data_type: str,
    trial_idx: int,
    llm_name: str,
    normalization: str = "raw",
    cache_budget: int | str | None = None,
    num_query_nodes: int | None = None,
) -> dict[int, list]:
    """Single entry point that builds the per-llm path (with budget segment) and loads the per-K query node cache.

    Used when the ablation grid loads `query_node_{norm}_nq{K}.json` per num_query_nodes (K).
    (The bundle loads once per (llm, trial) -> K-independent, so per-K loading is done via this
    function within the grid.)
    """
    base = _apply_budget(
        Path(dataset_root) / data_type / "precomputed" / "per-trial"
        / f"t{int(trial_idx)}" / "per-llm" / llm_name,
        cache_budget,
    )
    return load_query_node_cache(base, normalization, num_query_nodes)


def _normalize_qa(qa: Any) -> dict:
    """Handle the two formats of qa.json:
       - dict of lists  : {"question": [...], "option": [...], "true_answer_choice": [...], ...}
       - list of records: [{"question": ..., "options": ..., "true_answer": ..., ...}, ...]
    """
    if isinstance(qa, dict):
        return qa
    if isinstance(qa, list):
        # list of records -> dict of lists
        def col(rec, *keys, default=None):
            for k in keys:
                if k in rec:
                    return rec[k]
            return default
        return {
            "question":           [col(r, "question") for r in qa],
            "option":             [col(r, "option", "options") for r in qa],
            # Prefer true_answer_choice, which holds the letter ('(A)'~'(E)').
            #   true_answer is the answer 'text' and does not match scoring (letter comparison) ->
            #   prefer choice if present, otherwise fall back.
            "true_answer_choice": [col(r, "true_answer_choice", "true_answer") for r in qa],
            # list-of-records has no derived true_answer - derive from option if needed
            "questioner":         [col(r, "questioner", default="default") for r in qa],
            "hop_count":          [col(r, "hop_count") for r in qa],
            "route_label":        [col(r, "route_label") for r in qa],
        }
    raise TypeError(f"Unsupported qa.json type: {type(qa)}")


def _load_json_strict(path: Path, what: str) -> object:
    """JSON load with a friendly error message. Clearly reports missing / 0-byte / parse failure.

    Args:
        path: JSON file path to load
        what: a human-readable label such as "date.json in data/v1_2"
    """
    # Prefer the local copy (fall back to the NAS original if absent). Large JSON (entire_session/qa/
    #   partial_dialogues/date) is loaded from local NVMe to reduce NFS load/latency. Existence/size
    #   validation and reading use the actual file to read (lp=local_first) - so a 0-byte/truncated
    #   local copy is caught - while error messages point to the original path (path).
    lp = local_first(path)
    if not lp.exists():
        raise FileNotFoundError(
            f"{what} file missing: {path}\n"
            f"  -> run the conversion notebook (preprocess/v1_2_2/ or preprocess/v1_3_1/) first."
        )
    size = lp.stat().st_size
    if size == 0:
        raise RuntimeError(
            f"{what} file is empty (0 bytes): {path}\n"
            f"  -> the conversion notebook did not finish, or it is a placeholder.\n"
            f"  -> run the notebook (preprocess/v1_2_2/260523_convert-v1_2-data-to-json-npy_pickle.ipynb\n"
            f"     or preprocess/v1_3_1/260524_convert-v1_3-data-to-json-npy_pickle.ipynb) and retry."
        )
    import json
    try:
        # Specify UTF-8 - prevent decode failure of non-ASCII (Korean / smart quote / character name)
        # under a locale=POSIX/ASCII environment.
        return json.loads(lp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        # Surface line/col + snippet info.
        try:
            raw = lp.read_text(encoding="utf-8", errors="replace")
            snippet_start = max(0, e.pos - 100)
            snippet = raw[snippet_start:e.pos + 100]
        except Exception:
            snippet = "(failed to read snippet)"
        raise RuntimeError(
            f"{what} file JSON parse failed: {path}\n"
            f"  cause: {e.msg} at line {e.lineno} col {e.colno} (pos {e.pos})\n"
            f"  context: ...{snippet}...\n"
            f"  -> suspected truncation / incomplete conversion notebook. Re-run the notebook."
        ) from e


def _load_emb(directory: Path, basename: str) -> np.ndarray | None:
    """Prefer `{basename}.npy`; if absent, fall back to `{basename}.npz` (key 'embeddings' or 'vectors').

    The v1_2 notebook saves as `.npy`. Some older outputs or v1_3 raw are in `.npz` format.
    Both are handled for compatibility.
    """
    npy = directory / f"{basename}.npy"
    if npy.exists():
        return np.load(local_first(npy))
    npz = directory / f"{basename}.npz"
    if npz.exists():
        d = np.load(local_first(npz))
        for k in ("embeddings", "vectors"):
            if k in d.files:
                return d[k]
        return d[d.files[0]] if d.files else None
    return None


def _flatten_record_emb(raw: Any) -> dict | None:
    """Normalize a record embedding pickle into a flat map that the relational memory (R) uses directly.

    Storage format (conversion notebook):
        {session_idx: {"temporal": {key: vec}, "non_temporal": {key: vec}}}
    where key is a space-joined string:
        temporal     = f"{head} {relation} {tail} on {start_date}"
        non_temporal = f"{head} {relation} {tail}"

    Returns:
        {record_string_key: vec}  (temporal + non_temporal merged into one dict)

    If already in flat {key: vec} form, return as-is.
    """
    if not isinstance(raw, dict) or not raw:
        return raw if isinstance(raw, dict) else None
    flat: dict = {}
    for _sess_idx, sub in raw.items():
        # nested form (session -> temporal/non_temporal)
        if isinstance(sub, dict) and ("temporal" in sub or "non_temporal" in sub):
            for kind in ("temporal", "non_temporal"):
                inner = sub.get(kind) or {}
                if isinstance(inner, dict):
                    flat.update(inner)
        else:
            # already in flat {key: vec} form - take as-is.
            flat[_sess_idx] = sub
    return flat


def load_per_llm_precomputed(
    dataset_root: Path | str,
    data_type: str,
    trial_idx: int,
    llm_name: str,
    ie_type: str = "openie",
    normalization: str = "raw",
    cache_budget: int | str | None = None,
) -> dict[str, Any]:
    """Load the per-LLM precomputed cache (assistant==QA policy).

    Path convention (raw/en suffix separated; budget branch via bNNNNN):
        {dataset_root}/{data_type}/precomputed/per-trial/t{N}/per-llm/{llm_name}/[bNNNNN/]
            route.json                                                 # {idx: "RAG"|"R"} (required)
            query_node.json                                            # {idx: [candidate,...]} (optional)
            {ie_type}/                                                 # openie / closedie / gt
                partial_session_records_{normalization}.json           # raw or en (required)
                entire_session_records_{normalization}.json            # raw or en (optional)
                # legacy compat: partial_session_records.json (no suffix) also loads as raw fallback

    If cache_budget is given (None means no segment = the original single layout), a `bNNNNN` segment
    is inserted directly under the LLM folder of per-llm/{llm}/ and common/{ie}/{llm}/. dialogue/question
    embeddings (not extraction-LLM outputs) are unaffected.

    Cache miss -> explicitly raises FileNotFoundError.

    Returns:
        {"route": dict[int,str], "partial_records": list, "entire_records": list|None}
    """
    import re as _re
    root = Path(dataset_root)
    # Insert the budget segment directly under the LLM folder of per-llm (unchanged if absent).
    base = _apply_budget(
        root / data_type / "precomputed" / "per-trial" / f"t{trial_idx}" / "per-llm" / llm_name,
        cache_budget,
    )
    route_path = base / "route.json"
    ie_dir = base / ie_type

    def _load_split_route(parent: Path) -> dict | None:
        """Auto-merge split files (`route_split_S_E.json`)."""
        # A glob matching a stray file without the `_S_E` pattern (e.g. route_split_tmp.json) would
        #   make int(stem.split("_")[-2]) fail with ValueError and kill the whole load -> handle it
        #   robustly with regex + 0 fallback, like _load_split_records.
        def _rt_sort_key(p: Path) -> int:
            m = _re.search(r"_split_(\d+)_", p.name)
            return int(m.group(1)) if m else 0
        shards = sorted(parent.glob("route_split_*.json"), key=_rt_sort_key)
        if not shards:
            return None
        merged: dict[int, str] = {}
        for shard in shards:
            m = _re.search(r"_split_(\d+)_(\d+)\.json$", shard.name)
            if not m:
                continue
            s = int(m.group(1))
            data = json.loads(local_first(shard).read_text(encoding="utf-8"))  # local-first
            if isinstance(data, dict):
                for k, v in data.items():
                    merged[int(k) + s] = v
        return merged or None

    def _load_split_records(parent: Path, stem_prefix: str, full_n: int | None = None) -> list | None:
        """Merge split files (`{stem_prefix}_split_S_E.json`).
        full_n has no hardcoded default - the caller specifies it or it is auto-detected.
        Auto-detection: estimate from the shards' max(e). Prevents silent misalignment when other
        datasets (bigbang 805, theoffice 2347) are used."""
        def _shard_sort_key(p: Path) -> int:
            m = _re.search(r"_split_(\d+)_", p.name)
            return int(m.group(1)) if m else 0  # fall back to 0 on match failure (avoid AttributeError)

        shards = sorted(parent.glob(f"{stem_prefix}_split_*.json"), key=_shard_sort_key)
        if not shards:
            return None
        # auto-detect full_n (if caller did not specify): the max end index.
        if full_n is None:
            full_n = 0
            for shard in shards:
                m = _re.search(r"_split_(\d+)_(\d+)\.json$", shard.name)
                if m:
                    full_n = max(full_n, int(m.group(2)))
            if full_n == 0:
                return None
        merged: list = [[] for _ in range(full_n)]
        for shard in shards:
            m = _re.search(r"_split_(\d+)_(\d+)\.json$", shard.name)
            if not m:
                continue
            s, e = int(m.group(1)), int(m.group(2))
            data = json.loads(local_first(shard).read_text(encoding="utf-8"))  # local-first
            if isinstance(data, list):
                for i, item in enumerate(data):
                    gi = s + i
                    if 0 <= gi < full_n:
                        merged[gi] = item
        return merged

    # route - explicit file -> split fallback.
    p_named = ie_dir / f"partial_session_records_{normalization}.json"
    p_legacy = ie_dir / "partial_session_records.json"
    if p_named.exists():
        partial_path = p_named
    elif p_legacy.exists() and normalization == "raw":
        partial_path = p_legacy
    else:
        partial_path = p_named

    if not route_path.exists() and not list(base.glob("route_split_*.json")):
        raise FileNotFoundError(
            f"per-LLM precomputed route missing: {route_path}\n"
            f"  -> cache miss under the assistant==QA policy. Run per-LLM precompute (`--llms {llm_name}`) first, then retry."
        )
    if not partial_path.exists() and not list(ie_dir.glob(f"partial_session_records_{normalization}_split_*.json")):
        raise FileNotFoundError(
            f"per-LLM precomputed partial records missing: {partial_path}\n"
            f"  -> cache miss under the assistant==QA policy. Run per-LLM precompute "
            f"(`--llms {llm_name} --ie-types {ie_type}`) first, then retry.\n"
            f"  -> if normalization={normalization!r} is needed, the en file generation step must also be included."
        )

    # route load: prefer the merged file, fall back to merging split files.
    if route_path.exists():
        route_raw = json.loads(local_first(route_path).read_text(encoding="utf-8"))  # local-first
        route = {int(k): v for k, v in route_raw.items()} if isinstance(route_raw, dict) else route_raw
    else:
        route = _load_split_route(base) or {}

    # query node load (produced by the query-node precompute step, optional - if absent, an empty
    #   dict -> the ablation runtime live-computes only those idx as a fallback; the result is
    #   identical, only slower). key=session_idx (=current_session_idx), val=query node candidate
    #   entity list.
    # query nodes depend on records (raw/en) -> loaded per normalization (same convention as the
    #   query-node precompute step). route (above)/question/embedding are normalization-independent
    #   and shared. query nodes can be split per K (num_query_nodes) with `_nq{K}` -> use
    #   load_query_node_cache. The bundle loads once per (llm, trial), so it holds only the
    #   K-independent default (query_node_{norm}.json) as a fallback. Per-K loading is done separately
    #   by the ablation grid / simulate via load_query_nodes_for (reflecting num_query_nodes).
    query_nodes = load_query_node_cache(base, normalization, num_query_nodes=None)

    # partial_records load: prefer the merged file, fall back to merging split files.
    if partial_path.exists():
        partial_records = json.loads(local_first(partial_path).read_text(encoding="utf-8"))  # local-first (theoffice 1.5M+)
    else:
        partial_records = _load_split_records(ie_dir, f"partial_session_records_{normalization}") or []

    # entire records are trial-independent -> read from the common path.
    #   (entire records = datasetxdata_typexLLM, trial-independent. Match the save path in
    #    per_llm_precompute.py (common/{ie}/{llm}/) - write/read paths in sync.)
    #   The budget segment sits directly under the LLM folder (same convention as per-llm).
    common_ie_dir = _apply_budget(
        root / data_type / "precomputed" / "common" / ie_type / llm_name, cache_budget
    )
    # entire session count (N) - fixed as full_n during split merge to prevent silent truncation when
    #   only some shards are complete. If entire_session.json is missing/non-list -> None (auto-detect fallback).
    try:
        _es = json.loads(local_first(root / data_type / "entire_session.json").read_text(encoding="utf-8"))  # local-first
        n_entire = len(_es) if (isinstance(_es, list) and len(_es) > 0) else None  # empty list -> None (auto-detect fallback)
    except Exception:
        n_entire = None
    e_named = common_ie_dir / f"entire_session_records_{normalization}.json"
    e_legacy = common_ie_dir / "entire_session_records.json"
    e_split = list(common_ie_dir.glob(f"entire_session_records_{normalization}_split_*.json"))
    if e_split:
        # prefer session-shard results - so a stale single file does not shadow the split (prevents stale-read).
        entire_records = _load_split_records(common_ie_dir, f"entire_session_records_{normalization}",
                                             full_n=n_entire)
    elif e_named.exists():
        entire_records = json.loads(local_first(e_named).read_text(encoding="utf-8"))  # local-first (theoffice 2.1M+)
    elif e_legacy.exists() and normalization == "raw":
        entire_records = json.loads(local_first(e_legacy).read_text(encoding="utf-8"))  # local-first
    else:
        entire_records = _load_split_records(common_ie_dir, f"entire_session_records_{normalization}",
                                             full_n=n_entire)

    return {
        "route": route,
        "query_nodes": query_nodes,
        "partial_records": partial_records,
        "entire_records": entire_records,
        # Include trial_idx for consistency. run_baseline.py's bundle is built by
        # load_converted_dataset / load_v1_2_1_trial, but this guarantees the same key when a future
        # caller invokes this directly.
        "trial_idx": trial_idx,
    }


def load_converted_dataset(
    dataset_root: Path | str,
    data_type: str = "normal",
    trial_idx: int = 0,
    ie_type: str = "openie",
    normalization: str = "raw",
    per_llm_name: str | None = None,
    cache_budget: int | str | None = None,
) -> dict[str, Any]:
    """Load the resources for one (data_type, trial, ie_type | per_llm_name) from the converted tree.

    Args:
        dataset_root : converted root (e.g. `data/v1_2_2/friends` or `data/v1_3/friends`).
        data_type    : 'normal' / 'newname' / 'shuffled'. (user-selectable for both v1_2/v1_3)
        trial_idx    : 0 / 1 / 2.
        ie_type      : 'openie' / 'closedie' / 'gt'. (user-selectable for both v1_2/v1_3)
                       Also used in per_llm mode: load records from per-llm/{llm_name}/{ie_type}/.
        normalization: 'raw' / 'en'. For gt, 'raw' only is recommended. (ignored when per_llm_name is
                       active - the per-llm cache has no separate normalization branch. Extend the
                       path convention if needed.)
        per_llm_name : the assistant==QA policy from v1_3 onward. When set, both triple/route are
                       loaded from per-llm/{llm_name}/, and qa.json's route_label is ignored.
                       Cache miss -> FileNotFoundError (no live fallback; explicit precompute required).
        cache_budget : reasoning-budget branch of the extraction cache (route/query_node/triple/record).
                       None -> no segment (original single layout). Positive int -> insert a `bNNNNN`
                       segment under per-llm/{llm}/ and common/{ie}/{llm}/. Only valid in per_llm_name
                       mode (the v1_2 non-per_llm path does not support the budget branch).
                       dialogue/question embeddings are budget-independent, so they are unaffected.

    Returns dict:
        sessions               : list[Session]
        questions              : list[Question | None]
        entire_dialogue_emb    : np.ndarray | None
        partial_dialogue_emb   : np.ndarray | None
        entire_records         : list[list[dict]] | None   (per-session record list)
        partial_records        : list[list[dict]] | None
        entire_record_emb      : dict | None               (record embeddings; None in per-llm mode)
        partial_record_emb     : dict | None
    """
    root = Path(dataset_root)

    # Layout handling (v1_2 entire dialogue == v1_3 entire dialogue)
    # For both v1_2 / v1_3:
    #   - root/date.json : list[str], len=N (per-session timestamp)
    #   - root/{data_type}/entire_session.json : list[str], len=N (per-session dialogue)
    #   - root/{data_type}/precomputed/per-trial/t{trial}/qa.json : dict or list, len=N (questions)
    # partial_session.json is usually absent in both versions -> reuse entire_session.json (list) as partial.
    dates = _load_json_strict(root / "date.json", "date.json")
    entire_dialogues = _load_json_strict(
        root / data_type / "entire_session.json", f"{data_type}/entire_session.json"
    )
    # Only wrap into a list when entire_session.json is a (legacy) single string - new layouts are always lists.
    if isinstance(entire_dialogues, str):
        entire_dialogues = [entire_dialogues]
        # a single string may mismatch the session count -> warn
        if len(entire_dialogues) != len(dates):
            import warnings as _w
            _w.warn(
                f"entire_dialogues legacy single-string, "
                f"len({len(entire_dialogues)}) != session count ({len(dates)})"
            )
    n = len(dates)
    partial_dialogues_v12 = entire_dialogues if isinstance(entire_dialogues, list) else None

    entire_emb = _load_emb(root / data_type, "entire_session_embedding")
    # If embedding length mismatches n -> a silent live-embed fallback -> biased dense retrieval.
    if entire_emb is not None and len(entire_emb) != n:
        import warnings as _warnings
        _warnings.warn(
            f"entire_session_embedding length ({len(entire_emb)}) != session count ({n}) - "
            f"some sessions' dialogue_embedding=None triggers a live embed fallback. Re-run the conversion notebook."
        )

    # per-trial resources
    trial_dir = root / data_type / "precomputed" / "per-trial" / f"t{trial_idx}"

    qa_raw = _load_json_strict(trial_dir / "qa.json", f"{data_type}/precomputed/per-trial/t{trial_idx}/qa.json")
    qa = _normalize_qa(qa_raw)

    # partial dialogue priority:
    #   1) partial_dialogues.json (conversion output of v1_2 raw's partial_session_dialogue) - true partial
    #   2) partial_session.json (legacy)
    #   3) entire_dialogues fallback (old v1_2 handling)
    #   4) empty
    partial_json_new = trial_dir / "partial_dialogues.json"
    partial_path_legacy = trial_dir / "partial_session.json"
    if partial_json_new.exists():
        partial_dialogues = _load_json_strict(partial_json_new, f"{data_type}/.../t{trial_idx}/partial_dialogues.json")
    elif partial_path_legacy.exists():
        partial_dialogues = _load_json_strict(partial_path_legacy, f"{data_type}/.../t{trial_idx}/partial_session.json")
    elif partial_dialogues_v12 is not None:
        partial_dialogues = partial_dialogues_v12
    else:
        partial_dialogues = [""] * n

    # partial embedding priority:
    #   1) partial_dialogues_embedding.npy (openai-embedding conversion of raw)
    #   2) partial_session_embedding.npy (legacy)
    import numpy as _np
    partial_emb_new_path = trial_dir / "partial_dialogues_embedding.npy"
    if partial_emb_new_path.exists():
        partial_emb = _np.load(local_first(partial_emb_new_path))
    else:
        partial_emb = _load_emb(trial_dir, "partial_session_embedding")

    # Triple / route : per_llm_name branch
    per_llm_route: dict[int, str] | None = None
    per_llm_query_nodes: dict[int, list] = {}  # produced by the query-node precompute step (filled only in per_llm mode)
    if per_llm_name is not None:
        # v1_3+ assistant==QA policy - the ie_type subdir is also branched inside per-llm (user choice).
        # pass normalization (raw/en) - to compare the two variants in ablation.
        per_llm = load_per_llm_precomputed(
            dataset_root, data_type, trial_idx, per_llm_name,
            ie_type=ie_type, normalization=normalization,
            cache_budget=cache_budget,
        )
        entire_records = per_llm["entire_records"]
        partial_records = per_llm["partial_records"]
        per_llm_route = per_llm["route"]
        per_llm_query_nodes = per_llm.get("query_nodes") or {}
        entire_record_emb = None
        partial_record_emb = None
    else:
        # cache_budget is per_llm-mode only - raise explicitly rather than silently ignoring it on the non-per_llm (v1_2) path.
        if budget_segment(cache_budget):
            raise ValueError(
                f"cache_budget={cache_budget!r} is only supported in per_llm_name mode. "
                f"The non-per_llm (v1_2) path has no budget-branch cache - use data.per_llm_precomputed: true."
            )
        # v1_2 (original) - IE pre-computed (OpenIE/GT) + qa.json route_label
        common_ie_dir = root / data_type / "precomputed" / "common" / ie_type
        trial_ie_dir = trial_dir / ie_type

        entire_records_path = common_ie_dir / f"entire_session_records_{normalization}.json"
        partial_records_path = trial_ie_dir / f"partial_session_records_{normalization}.json"
        entire_records = (
            _load_json_strict(entire_records_path, f"common/{ie_type}/entire_session_records_{normalization}.json") if entire_records_path.exists() else None
        )
        partial_records = (
            _load_json_strict(partial_records_path, f"per-trial/t{trial_idx}/{ie_type}/partial_session_records_{normalization}.json") if partial_records_path.exists() else None
        )

        entire_record_emb_path = (
            common_ie_dir / f"entire_session_records_embedding_{normalization}.pickle"
        )
        partial_record_emb_path = (
            trial_ie_dir / f"partial_session_records_embedding_{normalization}.pickle"
        )
        # Load record embeddings (theoffice is the heaviest) local-first too.
        #   .exists() checks the NAS original (unchanged); only load uses local_first (the local copy
        #   if present, otherwise NAS). Previously only dialogue/question were local-first, so records
        #   were always loaded from NAS -> theoffice was slow.
        entire_record_emb = (
            _flatten_record_emb(load_pickle(local_first(entire_record_emb_path))) if entire_record_emb_path.exists() else None
        )
        partial_record_emb = (
            _flatten_record_emb(load_pickle(local_first(partial_record_emb_path))) if partial_record_emb_path.exists() else None
        )

    # Session / Question mapping
    sessions: list[Session] = []
    questions: list[Question | None] = []

    questioners = qa.get("questioner", ["default"] * n)
    # Guard against KeyError on qa["question"]/qa["option"] - on conversion loss in list-of-records,
    # fall back to an empty list (all questions become None, graceful degradation).
    # Use a list comprehension (not `[[]] * n`) to avoid a shared-reference bug.
    questions_text = qa.get("question", [""] * n)
    options_per_session = qa.get("option", [[] for _ in range(n)])
    true_answer_choice = qa.get("true_answer_choice", [None] * n)
    # route : per-llm cache in per_llm mode, otherwise qa.json's route_label
    if per_llm_route is not None:
        route_labels = [per_llm_route.get(i) for i in range(n)]
    else:
        route_labels = qa.get("route_label", [None] * n)
    hop_counts = qa.get("hop_count", [None] * n)

    for i in range(n):
        partial_d = partial_dialogues[i] if i < len(partial_dialogues) else ""
        # Whether partial records are the *cumulative memory up to that point* or *that session alone*
        # depends on the data.
        # entire->Session.entire_records, partial->Session.partial_records delivered separately.
        #   Memory (R) assigns meaning at commit (cumulative entire memory) / retrieve (merges the current session's partial).
        # record json format (dual schema):
        #   v1_2 conversion notebook -> list-of-list (sessions[i] = [[head, rel, tail, date], ...])
        #   v1_3 conversion notebook -> list-of-dict  (sessions[i] = [{"head":..., "relation":..., "tail":..., "start_date":...}, ...])
        # Both formats are accepted and converted to the Record dataclass.
        # R/RAG symmetry - entire/partial records as separate Session fields.
        #   _to_record_tuple: converts both list-of-dict (v1_3) / list-of-list (v1_2) to Record tuples.
        def _to_record_tuple(raw_list) -> tuple[Record, ...] | None:
            rl: list[Record] = []
            for r in raw_list:
                if isinstance(r, dict):
                    rl.append(Record(
                        head=r["head"], relation=r["relation"],
                        tail=r["tail"], start_date=r["start_date"],
                    ))
                else:
                    rl.append(Record(head=r[0], relation=r[1], tail=r[2], start_date=r[3]))
            return tuple(rl)

        partial_records_i: tuple[Record, ...] | None = None
        if partial_records is not None and i < len(partial_records):
            partial_records_i = _to_record_tuple(partial_records[i])
        entire_records_i: tuple[Record, ...] | None = None
        if entire_records is not None and i < len(entire_records):
            entire_records_i = _to_record_tuple(entire_records[i])

        # Inject the precomputed dialogue embedding (avoids a duplicate call inside commit()).
        d_emb = entire_emb[i] if entire_emb is not None and i < len(entire_emb) else None
        p_emb = partial_emb[i] if partial_emb is not None and i < len(partial_emb) else None
        # treat falsy non-None questioners[i] (0, False, empty string) as "default" too
        q_val = questioners[i] if i < len(questioners) else None
        q_str = str(q_val) if q_val is not None and str(q_val).strip() else "default"
        sessions.append(
            Session(
                idx=i,
                date=dates[i],
                questioner=q_str,
                dialogue=entire_dialogues[i] if i < len(entire_dialogues) else "",
                partial_dialogue=partial_d,
                entire_records=entire_records_i,
                partial_records=partial_records_i,
                dialogue_embedding=d_emb,
                partial_embedding=p_emb,
            )
        )

        # If questions_text[i] is None, `.strip()` raises AttributeError.
        # Force to a string with `or ""` + add a len guard.
        text = (questions_text[i] or "") if i < len(questions_text) else ""
        if not text.strip():
            questions.append(None)
            continue
        questions.append(
            Question(
                text=text.strip(),
                questioner=q_str,  # reuse the same q_str as Session
                # If options_per_session[i] is None, tuple(None) raises TypeError.
                # Fall back to an empty tuple with `or []`. Also add a len guard for true_answer_choice.
                options=tuple(options_per_session[i] or []) if i < len(options_per_session) else (),
                timestamp=dates[i],
                true_answer=true_answer_choice[i] if i < len(true_answer_choice) else None,
                # converted data has no pre-built prompts (live-retrieval mode only).
                prebuilt_prompt_sessions=None,
                prebuilt_prompt_records=None,
                prebuilt_prompt_no_history=None,
                route_label=route_labels[i] if i < len(route_labels) else None,
                hop_count=hop_counts[i] if i < len(hop_counts) else None,
            )
        )

    return {
        "sessions": sessions,
        "questions": questions,
        "query_nodes": per_llm_query_nodes,  # {session_idx: candidate list} (per_llm mode only, else {})
        "entire_dialogue_emb": entire_emb,
        "partial_dialogue_emb": partial_emb,
        "entire_records": entire_records,
        "partial_records": partial_records,
        "entire_record_emb": entire_record_emb,
        "partial_record_emb": partial_record_emb,
        # run_baseline.py's shared_bundle trial-mismatch warning was permanently silent dead code
        # (the returned dict had no trial_idx). Made explicit.
        "trial_idx": trial_idx,
    }


# v1.2.1 raw pickle loader - reproduce/data/{data_type}/{ie_type}/trial{N}.pickle
def load_v1_2_1_trial(
    pickle_path: Path | str,
) -> dict[str, Any]:
    """Load a v1.2.1 raw pickle -> return a bundle in the same format as load_converted_dataset.

    v1.2.1 pickle structure (reproduce/data/{data_type}/{ie_type}/trial{N}.pickle):
        {
          'questioner':              list[str],           # len=N
          'question':                list[str],           # len=N
          'option':                  list[list[str]],     # len=N
          'true_answer':             list[str],           # e.g. '(A)'
          'hop_count':               list[int],           # -1/1/2
          'answer':                  list[str],           # (unused)
          'partial_session_dialogue': list[str],          # len=N
          'session_dialogue':        list[str],           # len=N
          'idx':                     list[int],           # len=N
          'date':                    list[str],           # len=N
          'route_label':             list[str],           # 'RAG'/'R'
          'document_embedding': {
              'sentence-transformer': np.ndarray (N, 384),
              'openai-embedding':     dict{dialogue_text: list[1536-dim]},
          },
          'question_embedding': {
              'sentence-transformer': np.ndarray (N, 384),
              'openai-embedding':     list[list[1536-dim]],  # index-aligned
          },
          ... (qa_user_prompt_* - pre-built prompts, ignored by this loader)
        }

    Returned bundle (same format as load_converted_dataset):
        sessions               : list[Session]
        questions              : list[Question | None]
        entire_dialogue_emb    : np.ndarray | None   (document openai-embedding, Nx1536)
        partial_dialogue_emb   : None                (v1.2.1 has no separate partial embedding)
        entire_records         : None                (v1.2.1 raw pickle has no record list)
        partial_records        : None
        entire_record_emb      : None
        partial_record_emb     : None

    NOTE: Since this legacy loader provides no extracted records, only the raw session
    repository (S) view carries evidence; the relational memory (R) falls back to empty.
    """
    import numpy as _np

    path = Path(pickle_path)
    d = load_pickle(path)
    if not isinstance(d, dict):
        raise TypeError(f"load_v1_2_1_trial: top-level pickle type must be dict, got {type(d)}")

    n = len(d["idx"])

    # embedding extraction
    doc_emb_block = d.get("document_embedding") or {}
    q_emb_block = d.get("question_embedding") or {}

    # document (session dialogue) openai-embedding: {text: vec} dict -> index-aligned ndarray
    doc_oe = doc_emb_block.get("openai-embedding")
    dialogues = d["session_dialogue"]  # len=N
    entire_emb: _np.ndarray | None = None
    if isinstance(doc_oe, dict) and doc_oe:
        vecs = []
        for dia in dialogues:
            vec = doc_oe.get(dia)
            if vec is not None:
                vecs.append(_np.array(vec, dtype=_np.float32))
            else:
                # miss -> zero-vec (the live embed fallback is handled by run_baseline)
                vecs.append(_np.zeros(1536, dtype=_np.float32))
        entire_emb = _np.stack(vecs)  # (N, 1536)

    # sentence-transformer embedding (384-dim) fallback
    doc_st = doc_emb_block.get("sentence-transformer")
    if entire_emb is None and isinstance(doc_st, _np.ndarray) and len(doc_st) == n:
        entire_emb = doc_st.astype(_np.float32)

    # Session / Question mapping
    questioners = d.get("questioner", ["default"] * n)
    questions_text = d["question"]
    options_list = d["option"]
    true_answers = d.get("true_answer", [None] * n)
    dates = d["date"]
    # route_label: the pickle may contain values with embedded JSON quotes like '"R"' -> strip
    route_labels = [
        str(v).strip().strip('"') if v is not None else None
        for v in d.get("route_label", [None] * n)
    ]
    hop_counts = d.get("hop_count", [None] * n)
    partial_dialogues = d.get("partial_session_dialogue", dialogues)

    sessions: list[Session] = []
    questions: list[Question | None] = []

    for i in range(n):
        d_emb = entire_emb[i] if entire_emb is not None and i < len(entire_emb) else None
        q_str = str(questioners[i]).strip() if questioners[i] else "default"
        sessions.append(
            Session(
                idx=d["idx"][i],
                date=dates[i],
                questioner=q_str,
                dialogue=dialogues[i] if i < len(dialogues) else "",
                partial_dialogue=partial_dialogues[i] if i < len(partial_dialogues) else "",
                entire_records=None,   # v1.2.1 raw pickle has no record list
                partial_records=None,
                dialogue_embedding=d_emb,
                partial_embedding=None,
            )
        )

        text = questions_text[i] if i < len(questions_text) else ""
        if not text or not text.strip():
            questions.append(None)
            continue

        # true_answer format: a string like '(A)' - same as true_answer_choice in load_converted_dataset
        ta = true_answers[i] if i < len(true_answers) else None
        questions.append(
            Question(
                text=text.strip(),
                questioner=q_str,
                options=tuple(options_list[i]) if i < len(options_list) else (),
                timestamp=dates[i],
                true_answer=ta,
                prebuilt_prompt_sessions=None,
                prebuilt_prompt_records=None,
                prebuilt_prompt_no_history=None,
                route_label=route_labels[i] if i < len(route_labels) else None,
                hop_count=hop_counts[i] if i < len(hop_counts) else None,
            )
        )

    # Extract the trial from the pickle_path stem (e.g. trial0.pickle -> 0), so run_baseline.py's
    # trial-mismatch warning works.
    import re as _re
    _stem = Path(pickle_path).stem  # e.g., "trial0"
    _trial_match = _re.match(r"trial(\d+)$", _stem)
    _trial_idx = int(_trial_match.group(1)) if _trial_match else None
    return {
        "sessions": sessions,
        "questions": questions,
        "entire_dialogue_emb": entire_emb,
        "partial_dialogue_emb": None,
        "entire_records": None,
        "partial_records": None,
        "entire_record_emb": None,
        "partial_record_emb": None,
        # None on parse failure (e.g. a different naming convention) - simulate.py's warning skips when None.
        "trial_idx": _trial_idx,
    }


# Deprecated - old direct pickle loader (v1_2 raw pickle). Alias kept for compatibility.
def load_trial(*args, **kwargs):
    """[DEPRECATED] Direct load of a v1_2 raw pickle. New code should use `load_converted_dataset`."""
    raise NotImplementedError(
        "load_trial(raw pickle) is deprecated. Use load_converted_dataset(converted_root, ...)."
    )
