"""Read-only access to the per-sample logs of a finished LEAF rollout run."""

from __future__ import annotations

import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    """One sample log, read-only: torn lines dropped, the last row per step kept.

    (``leaf.explore.leaf_rollout.read_sample_records`` heals the file in place; a
    reader of someone else's run must not write to it.)
    """
    by_step: dict[int, dict] = {}
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def run_rows(run: Path) -> dict[str, list[dict]]:
    """``{problem_uid: rows by step}`` for every sample log of a rollout run."""
    out: dict[str, list[dict]] = {}
    for path in sorted((run / "samples").glob("*.jsonl")):
        rows = read_rows(path)
        if rows:
            out[rows[0]["problem_uid"]] = rows
    return out
