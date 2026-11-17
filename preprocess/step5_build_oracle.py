"""Step 5 - oracle bundles (the evidence each question actually needs).

For every question, the raw oracle pickles name the dates of the sessions that contain the
answer ("oracle timestamps"). This step collects, per question:

  * the full dialogues of those sessions  -> {data_root}/{show}/{dt}/oracle/trial{N}.oracle.dialogue.json
  * the records of those sessions         -> {data_root}/{show}/{dt}/oracle/{llm}/trial{N}.oracle_raw.json

`llm` is `gt` (human-annotated records from step 4, used by the paper's oracle runs) and/or any
model whose records were extracted in step 2.

Dialogue orderings stored per question:
  oracle_dialogues_time_ordered   session (= time) order, no ranking   <- used by the paper
  oracle_dialogues_by_bm25        Okapi BM25 (k1=1.5, b=0.75) w.r.t. the question
  oracle_dialogues_by_cosine      cosine similarity of OpenAI text-embedding-3-small vectors
  oracle_dialogues_by_alternating BM25 #1, cosine #1, BM25 #2, ... (duplicates skipped)
Records are stored de-duplicated and cosine-sorted (`oracle_records_by_cosine`); a record is
embedded as "{head} {relation} {tail} on {start_date}".

Timestamps are looked up with the *normal* question (names in newname/shuffled questions would
not match the raw pickles); the sessions are identical across name settings.

`--no-embedding` skips the OpenAI calls: cosine orderings then keep session order (the
time-ordered dialogues and BM25 orderings are unaffected, so the paper's oracle runs still work).

Usage:
  python preprocess/step5_build_oracle.py --raw-dir /path/to/groupchat_bench --data-root data \
      --record-llms gt
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from common import (DATA_TYPES, SHOWS, TRIALS, entire_records_path, load_json, load_pickle,
                    parse_list, save_json, show_root, trial_dir)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100


# --------------------------------------------------------------------------- timestamps
def _sorted_unique_dates(dates: list[str]) -> list[str]:
    def key(d: str):
        try:
            return (0, datetime.strptime(d, "%B %d, %Y").toordinal(), "")
        except (ValueError, TypeError):
            return (1, 0, d)
    return sorted(set(dates), key=key)


def oracle_timestamps(prov: dict, oracle_tkg: dict, oracle_fan: dict, raw: dict,
                      question: str) -> list[str]:
    """Dates of the sessions that answer one question.

    fan-quiz : oracle_fan[episode][session][subbucket][raw_key]
    TKG list : find the raw question variant equal to `question` (case/space-insensitive) in
               raw[episode][session]['hard_q'][subbucket]; its index j selects oracle_tkg[...][j]
               (the last matching j wins)
    TKG dict : oracle_tkg[...][subbucket][question]
    'future' markers are dropped; dates are de-duplicated and sorted.
    """
    ep, sn, sub, rk = prov["episode"], prov["session_num"], prov["subbucket"], prov["raw_key"]
    ts: Any = []
    if prov["question_type"] == "fan":
        entry = ((oracle_fan.get(ep) or {}).get(sn) or {}).get(sub)
        if isinstance(entry, dict):
            ts = entry.get(rk, [])
        elif isinstance(entry, (list, tuple)):
            try:
                ts = entry[int(rk)]
            except (IndexError, TypeError, ValueError):
                ts = []
    else:
        entry = ((oracle_tkg.get(ep) or {}).get(sn) or {}).get(sub)
        if isinstance(entry, list):
            variants = (((raw.get(ep) or {}).get(sn) or {}).get("hard_q") or {}).get(sub)
            hit = None
            if isinstance(variants, list):
                target = question.lower().replace(" ", "")
                for j, item in enumerate(variants):
                    qs = item.get("questions") if isinstance(item, dict) else None
                    if qs and j < len(entry) and any(
                            isinstance(v, str) and target == v.lower().replace(" ", "")
                            for v in qs.values()):
                        hit = entry[j]
            ts = hit if hit is not None else []
        elif isinstance(entry, dict):
            ts = entry.get(question, [])
    if not ts:
        return []
    if isinstance(ts, str):
        ts = [ts]
    return _sorted_unique_dates([t for t in ts if isinstance(t, str)
                                 and t.strip().lower() != "future"])


# --------------------------------------------------------------------------- ranking
_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bm25_order(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    if not docs:
        return []
    q_terms = _tokens(query)
    doc_terms = [_tokens(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(d) for d in doc_terms) / n
    df: dict[str, int] = {}
    for d in doc_terms:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    scores = []
    for d in doc_terms:
        tf, dl, s = Counter(d), len(d), 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f:
                idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / (avgdl or 1.0)))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])


def cosine_order(q_vec: np.ndarray, doc_vecs: np.ndarray) -> list[int]:
    qn = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    dn = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return [int(i) for i in np.argsort(-(dn @ qn), kind="stable")]  # stable: deterministic ties


def alternating(bm25_sorted: list[str], cos_sorted: list[str]) -> list[str]:
    out, seen = [], set()
    for k in range(max(len(bm25_sorted), len(cos_sorted))):
        for lst in (bm25_sorted, cos_sorted):
            if k < len(lst) and lst[k] not in seen:
                seen.add(lst[k])
                out.append(lst[k])
    return out


def record_string(r: dict) -> str:
    return (f"{r.get('head') or ''} {r.get('relation') or ''} "
            f"{r.get('tail') or ''} on {r.get('start_date') or ''}")


class Embeddings:
    """Embeds each distinct string once; `None` model -> no embedding (session order kept)."""

    def __init__(self, enabled: bool):
        self.vecs: dict[str, np.ndarray] = {}
        self.embedder = None
        if enabled:
            from groupchat_qa.embedder.openai_embedder import OpenAIEmbedder
            self.embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    def add(self, texts) -> None:
        if self.embedder is None:
            return
        todo = [t for t in dict.fromkeys(texts) if t not in self.vecs]
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            for t, v in zip(batch, self.embedder.embed(batch)):
                self.vecs[t] = np.asarray(v, dtype="float32")

    def order(self, query: str, docs: list[str]) -> list[int]:
        if self.embedder is None or not docs:
            return list(range(len(docs)))
        return cosine_order(self.vecs[query], np.stack([self.vecs[d] for d in docs]))


# --------------------------------------------------------------------------- build
def load_raw(raw_dir: Path, show: str) -> tuple:
    """The three raw pickles of a show (loaded once per show)."""
    return (load_pickle(raw_dir / f"{show}_groupchat_bench.pickle"),
            load_pickle(raw_dir / f"{show}_oracle_tkg.pickle"),
            load_pickle(raw_dir / f"{show}_oracle_fan.pickle"))


def build(raw_pickles: tuple, data_root: Path, show: str, dt: str, trial: int,
          record_llms: list[str], emb: Embeddings) -> None:
    root = show_root(data_root, show)
    prov = load_json(trial_dir(data_root, show, "normal", trial) / "qa_provenance.json")
    normal_q = load_json(trial_dir(data_root, show, "normal", trial) / "qa.json")["question"]
    qa = load_json(trial_dir(data_root, show, dt, trial) / "qa.json")
    dates = load_json(root / "date.json")
    entire = load_json(root / dt / "entire_session.json")
    assert len(prov) == len(qa["question"]) == len(normal_q) and len(dates) == len(entire)

    raw, oracle_tkg, oracle_fan = raw_pickles

    base, sess_idx = [], []
    for i, p in enumerate(prov):
        ts = oracle_timestamps(p, oracle_tkg, oracle_fan, raw, normal_q[i])
        tset = set(ts)
        sess_idx.append([j for j, d in enumerate(dates) if d in tset])
        base.append({"session_idx": i, "question": qa["question"][i],
                     "questioner": p.get("questioner", ""), "option": list(qa["option"][i]),
                     "true_answer": qa["true_answer"][i], "epi": p.get("episode"),
                     "sid": p.get("session_num"), "q_kind": p.get("oracle_kind"),
                     "q_type": p.get("subbucket"), "hop": p.get("tkg_hop"), "timestamp": ts})

    # dialogue bundle
    emb.add(r["question"] for r in base)
    emb.add(entire[j] for idx in sess_idx for j in idx)
    dial = []
    for rec, idx in zip(base, sess_idx):
        dials = [entire[j] for j in idx]
        b_sorted = [dials[k] for k in bm25_order(rec["question"], dials)]
        c_sorted = [dials[k] for k in emb.order(rec["question"], dials)]
        dial.append({**rec, "oracle_dialogues_by_cosine": c_sorted,
                     "oracle_dialogues_by_bm25": b_sorted,
                     "oracle_dialogues_by_alternating": alternating(b_sorted, c_sorted),
                     "oracle_dialogues_time_ordered": dials})
    out = root / dt / "oracle" / f"trial{trial}.oracle.dialogue.json"
    save_json(dial, out)
    print(f"[{show}/{dt}/t{trial}] with-timestamp={sum(1 for r in base if r['timestamp'])}/"
          f"{len(base)} dialogues={sum(map(len, sess_idx))} -> {out}")

    # record bundles
    for llm in record_llms:
        path = entire_records_path(data_root, show, dt, llm)
        if not path.exists():
            print(f"  [skip] {llm}: {path} not found (run step 2 or step 4 first)",
                  file=sys.stderr)
            continue
        per_session = load_json(path)
        assert len(per_session) == len(dates), f"{path}: session count mismatch"
        recs_per_q = []
        for idx in sess_idx:
            seen, recs = set(), []
            for j in idx:
                for r in per_session[j]:
                    sig = tuple(r.get(k) or "" for k in ("head", "relation", "tail", "start_date"))
                    if sig not in seen:
                        seen.add(sig)
                        recs.append(r)
            recs_per_q.append(recs)
        emb.add(record_string(r) for recs in recs_per_q for r in recs)
        bundle = []
        for rec, recs in zip(base, recs_per_q):
            strings = [record_string(r) for r in recs]
            bundle.append({**rec, "oracle_records_by_cosine":
                           [recs[k] for k in emb.order(rec["question"], strings)]})
        out = root / dt / "oracle" / llm / f"trial{trial}.oracle_raw.json"
        save_json(bundle, out)
        print(f"  [{llm}] records={sum(map(len, recs_per_q))} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="folder holding the raw pickles (questions + oracle_tkg/oracle_fan)")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--shows", default=",".join(SHOWS))
    ap.add_argument("--data-types", default=",".join(DATA_TYPES))
    ap.add_argument("--trials", default=",".join(map(str, TRIALS)))
    ap.add_argument("--record-llms", default="gt",
                    help="comma list of record sources: 'gt' and/or model names from step 2")
    ap.add_argument("--no-embedding", action="store_true",
                    help="skip OpenAI embeddings (cosine orderings keep session order)")
    args = ap.parse_args()
    emb = Embeddings(enabled=not args.no_embedding)
    for show in parse_list(args.shows):
        raw_pickles = load_raw(args.raw_dir, show)
        for dt in parse_list(args.data_types):
            for t in parse_list(args.trials, int):
                build(raw_pickles, args.data_root, show, dt, t, parse_list(args.record_llms), emb)


if __name__ == "__main__":
    main()
