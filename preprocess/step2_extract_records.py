"""Step 2 - extract relational records (head, relation, tail, timestamp) with an LLM.

GroupChat-Mem keeps a relational memory R built from records. For every name setting we extract
records from that setting's own dialogue (never by renaming another setting's records), for two
views of each session:

  * entire  : the whole session                    (trial-independent, accumulated into R)
  * partial : the session up to the question turn  (one file per trial, the current session's
              contribution at question time)

The LLM sees only the dialogue and returns `{"triples": [{"head", "relation", "tail"}, ...]}`
(prompt: `prompts/record_openie_extraction.txt`). The timestamp is *not* asked from the model:
every record receives its session date from `date.json` (`start_date`), which keeps record dates
exact and lets date-based retrieval work.

Outputs (raw records, no entity normalization):
    {data_root}/{show}/{dt}/precomputed/common/openie/{llm}/b06000/entire_session_records_raw.json
    {data_root}/{show}/{dt}/precomputed/per-trial/t{N}/per-llm/{llm}/b06000/openie/partial_session_records_raw.json
    list[session][{head, relation, tail, start_date}]

Resume: next to each output a `<file>.done` list records the session indices whose call
succeeded (an empty record list is a valid result). Failed calls are not marked done, so simply
re-running the same command retries only the failed / missing sessions.

Serving: any OpenAI-compatible endpoint (e.g. `vllm serve ... --served-model-name <llm>`).
Sampling presets used in the paper (`--preset auto` picks one from the model name):
  * thinking  (Qwen3.5 / Gemma-4 with reasoning on): temperature 0.6, top_p 0.95,
               max_tokens 16384, enable_thinking=True, thinking_token_budget=6000
               (the server needs `--reasoning-parser` and `--reasoning-config`).
  * gpt-oss   : temperature 1.0, top_p 1.0, max_tokens 16384, reasoning_effort=medium.
  * plain     : temperature 0.0, top_p 1.0, max_tokens 4096.

Usage:
  python preprocess/step2_extract_records.py --data-root data --shows friends \
      --llm qwen3.5-35b-a3b-on --base-url http://localhost:8000/v1 --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from common import (DATA_TYPES, SHOWS, TRIALS, entire_records_path, load_json, parse_list,
                    partial_records_path, save_json, show_root, trial_dir)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "record_openie_extraction.txt"


class Triple(BaseModel):
    head: str
    relation: str
    tail: str


class TripleList(BaseModel):
    triples: List[Triple] = Field(default_factory=list)


PRESETS = {
    "thinking": dict(temperature=0.6, top_p=0.95, max_tokens=16384,
                     extra_body={"chat_template_kwargs": {"enable_thinking": True},
                                 "thinking_token_budget": 6000}),
    "gpt-oss": dict(temperature=1.0, top_p=1.0, max_tokens=16384,
                    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}}),
    "plain": dict(temperature=0.0, top_p=1.0, max_tokens=4096, extra_body=None),
}


def pick_preset(name: str, llm: str) -> dict:
    if name != "auto":
        return PRESETS[name]
    low = llm.lower()
    if "gpt-oss" in low:
        return PRESETS["gpt-oss"]
    if low.endswith("-on") or "qwen3" in low or "gemma-4" in low:
        return PRESETS["thinking"]
    return PRESETS["plain"]


# --------------------------------------------------------------------------- LLM call
class Extractor:
    """One structured extraction call per dialogue (user message only, no system prompt)."""

    def __init__(self, llm: str, base_url: str, api_key: str, preset: dict, timeout: float):
        from openai import OpenAI
        self.llm = llm
        self.preset = preset
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.template = PROMPT_PATH.read_text(encoding="utf-8")
        if "{input_text}" not in self.template:
            raise ValueError(f"{PROMPT_PATH} must contain the {{input_text}} placeholder")

    def _create(self, prompt: str, structured: bool):
        kw = dict(model=self.llm, messages=[{"role": "user", "content": prompt}],
                  temperature=self.preset["temperature"], top_p=self.preset["top_p"],
                  max_tokens=self.preset["max_tokens"])
        if self.preset.get("extra_body"):
            kw["extra_body"] = self.preset["extra_body"]
        if structured:
            kw["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "TripleList", "schema": TripleList.model_json_schema()}}
        return self.client.chat.completions.create(**kw)

    @staticmethod
    def _parse(text: str) -> TripleList | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return TripleList.model_validate_json(text)
        except Exception:
            pass
        # Free-form fallback: take the outermost JSON object in the reply.
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return TripleList.model_validate(json.loads(m.group(0)))
            except Exception:
                return None
        return None

    def extract(self, dialogue: str) -> list[dict]:
        """Return [{head, relation, tail}]. Raises if no parsable answer was obtained."""
        prompt = self.template.replace("{input_text}", dialogue)
        parsed = None
        try:  # 1) schema-constrained decoding
            parsed = self._parse(self._create(prompt, structured=True).choices[0].message.content)
        except Exception:
            parsed = None
        if parsed is None:  # 2) free-form generation, then parse the JSON in the reply
            parsed = self._parse(self._create(prompt, structured=False).choices[0].message.content)
        if parsed is None:
            raise ValueError("could not parse a TripleList from the model output")
        return [t.model_dump() for t in parsed.triples]


# --------------------------------------------------------------------------- one output file
def run_file(extractor: Extractor, dialogues: list[str], dates: list[str], out_path: Path,
             workers: int, save_every: int, overwrite: bool) -> None:
    done_path = Path(str(out_path) + ".done")
    out: list[list[dict]] = [[] for _ in dialogues]
    done: set[int] = set()
    if not overwrite and out_path.exists() and done_path.exists():
        prev = load_json(out_path)
        if len(prev) == len(dialogues):
            out = prev
            done = {int(i) for i in load_json(done_path)}
    # Empty dialogues have nothing to extract: they are done with no records.
    for i, d in enumerate(dialogues):
        if not (d or "").strip():
            out[i] = []
            done.add(i)
    todo = [i for i in range(len(dialogues)) if i not in done]
    print(f"  {out_path.name}: {len(done)}/{len(dialogues)} done, {len(todo)} to extract")
    if not todo:
        save_json(out, out_path)
        save_json(sorted(done), done_path)
        return

    lock = threading.Lock()
    state = {"since_save": 0, "fail": 0}

    def _flush() -> None:  # caller holds the lock
        save_json(out, out_path)
        save_json(sorted(done), done_path)
        state["since_save"] = 0

    def _one(i: int) -> None:
        try:
            recs = extractor.extract(dialogues[i])
        except Exception as e:  # not marked done -> retried on the next run
            with lock:
                state["fail"] += 1
            print(f"    session {i} failed ({type(e).__name__}: {str(e)[:120]})")
            return
        for r in recs:
            r["start_date"] = dates[i]  # the record date is always the session date
        with lock:
            out[i] = recs
            done.add(i)
            state["since_save"] += 1
            if state["since_save"] >= save_every:
                _flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_one, todo))
    with lock:
        _flush()
    n_rec = sum(len(x) for x in out)
    print(f"  {out_path.name}: done {len(done)}/{len(dialogues)} sessions, {n_rec} records, "
          f"{state['fail']} failed, {time.time() - t0:.0f}s")
    if state["fail"]:
        print("  -> re-run the same command to retry the failed sessions")


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--shows", default=",".join(SHOWS))
    ap.add_argument("--data-types", default=",".join(DATA_TYPES))
    ap.add_argument("--trials", default=",".join(map(str, TRIALS)))
    ap.add_argument("--scopes", default="entire,partial", help="entire and/or partial")
    ap.add_argument("--llm", required=True, help="served model name (also the output folder name)")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--preset", default="auto", choices=["auto", *PRESETS])
    ap.add_argument("--workers", type=int, default=16, help="parallel requests")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout (s)")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true", help="ignore previous progress")
    args = ap.parse_args()

    preset = pick_preset(args.preset, args.llm)
    print(f"model={args.llm} endpoint={args.base_url} preset={preset}")
    extractor = Extractor(args.llm, args.base_url, args.api_key, preset, args.timeout)
    scopes = parse_list(args.scopes)
    for show in parse_list(args.shows):
        dates = load_json(show_root(args.data_root, show) / "date.json")
        for dt in parse_list(args.data_types):
            if "entire" in scopes:
                print(f"[{show}/{dt}] entire sessions")
                dlg = load_json(show_root(args.data_root, show) / dt / "entire_session.json")
                run_file(extractor, dlg, dates, entire_records_path(args.data_root, show, dt, args.llm),
                         args.workers, args.save_every, args.overwrite)
            if "partial" in scopes:
                for t in parse_list(args.trials, int):
                    print(f"[{show}/{dt}/t{t}] partial sessions")
                    dlg = load_json(trial_dir(args.data_root, show, dt, t) / "partial_dialogues.json")
                    run_file(extractor, dlg, dates,
                             partial_records_path(args.data_root, show, dt, t, args.llm),
                             args.workers, args.save_every, args.overwrite)


if __name__ == "__main__":
    main()
