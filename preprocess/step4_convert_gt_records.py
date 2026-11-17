"""Step 4 - human-annotated (ground-truth) records -> oracle record files.

The oracle setting gives the model the *gold* records of the sessions that hold the answer.
Those gold records are human annotations shipped as `gt_triples/{show}.json`:

    list[session] of {"date": str, "triples": [[head, relation, tail], ...]}

in session order (lower-case, normal character names). This step only changes the format and the
character names; it never edits the triples.

Output (one file per show x name setting):
    {data_root}/{show}/{dt}/precomputed/common/openie/gt/b06000/entire_session_records_raw.json
    list[session][{head, relation, tail, start_date}]

Renaming (newname / shuffled) is applied to whole lower-case words only (`\\bross\\b`), so common
words that contain a name (e.g. "across") stay intact. It runs in a single regex pass, so swapped
names do not chain.

Usage:
  python preprocess/step4_convert_gt_records.py --gt-dir /path/to/gt_triples --data-root data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import (DATA_TYPES, NAME_MAPS, SHOWS, entire_records_path, load_json, parse_list,
                    save_json, show_root)


def _rename_lower(show: str, data_type: str):
    if data_type == "normal":
        return lambda s: s
    mapping = {k.lower(): v.lower() for k, v in NAME_MAPS[data_type][show].items()}
    pattern = re.compile(r"\b(" + "|".join(
        re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")\b")
    return lambda s: pattern.sub(lambda m: mapping[m.group(0)], s) if isinstance(s, str) else s


def convert(gt_dir: Path, data_root: Path, show: str, data_type: str) -> None:
    sessions = load_json(gt_dir / f"{show}.json")
    dates = load_json(show_root(data_root, show) / "date.json")
    if len(sessions) != len(dates):
        raise ValueError(f"{show}: {len(sessions)} annotated sessions != {len(dates)} in date.json")
    fn = _rename_lower(show, data_type)
    out, n = [], 0
    for i, sess in enumerate(sessions):
        if sess["date"] != dates[i]:
            raise ValueError(f"{show} session {i}: date {sess['date']!r} != {dates[i]!r}")
        recs = []
        for h, r, t in sess["triples"]:
            recs.append({"head": fn(h), "relation": fn(r), "tail": fn(t),
                         "start_date": sess["date"]})
        out.append(recs)
        n += len(recs)
    path = entire_records_path(data_root, show, data_type, "gt")
    save_json(out, path)
    print(f"[{show}/{data_type}] sessions={len(out)} records={n} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-dir", required=True, type=Path, help="folder holding {show}.json")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--shows", default=",".join(SHOWS))
    ap.add_argument("--data-types", default=",".join(DATA_TYPES))
    args = ap.parse_args()
    for show in parse_list(args.shows):
        for dt in parse_list(args.data_types):
            convert(args.gt_dir, args.data_root, show, dt)


if __name__ == "__main__":
    main()
