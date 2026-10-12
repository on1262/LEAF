"""Grade one exploration cell, complete or stopped by its 5 h time limit.

Every formal cell is one single-GPU job capped at 5 GPU hours; a cell that has not
reached 24 aligned steps by then is stopped by its time limit and must still be graded
on the steps it completed. This CPU job runs after the GPU job, whatever its exit state:

* LEAF / LEAF with interleaved updates (``--kind leaf``): when the driver did not finish
  (``generations.jsonl`` short of the manifest's steps), the steps every problem
  completed are assembled from ``samples/*.jsonl`` (``leaf.explore.leaf_rollout.assemble_steps``)
  and written to ``generations.jsonl``; then ``leaf.grading.grade_leaf_rollout --steps K``
  (public grade + MBPP+ private).
* baselines (``--kind baseline``): ``leaf.grading.grade_baseline --mode formal
  --allow-partial`` (the attempt-aligned window ends at the last step every problem
  reached; MBPP+ private inside).

A complete run is graded by the same two commands without the partial-run handling. LCB private runs as the next job
(``leaf.grading.grade_lcb_private``, which already grades min(24, summary steps)).

    python -m leaf.grading.grade_explore_cell --kind leaf --output DIR --workers 24
    python -m leaf.grading.grade_explore_cell --kind baseline --output DIR --condition grpo --benchmark lcb
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

from leaf.core.project_config import atomic_jsonl, read_jsonl


def complete_steps(output: Path, problems: list[dict]) -> int:
    """Largest K with steps 1..K recorded for every problem."""
    from leaf.explore.leaf_rollout import read_sample_records, sample_record_path

    best = None
    for index, problem in enumerate(problems):
        steps = {int(r["step"]) for r in read_sample_records(sample_record_path(output, index, problem["uid"]))}
        k = 0
        while k + 1 in steps:
            k += 1
        best = k if best is None else min(best, k)
    return best or 0


def finalize_leaf(output: Path) -> int:
    from leaf.explore.leaf_rollout import assemble_steps

    manifest = json.loads((output / "manifest.json").read_text())
    planned = int(manifest["steps"])
    problems = read_jsonl(Path(manifest["dataset"]))
    rows = read_jsonl(output / "generations.jsonl")
    if len(rows) == planned * len(problems):
        return planned  # the driver finished and assembled its own run
    steps = min(planned, complete_steps(output, problems))
    if steps < 1:
        raise RuntimeError(f"{output}: no aligned step completed by every problem")
    assemble_steps(output, problems, steps)
    finals = [row for step in range(1, steps + 1)
              for row in read_jsonl(output / "steps" / f"step_{step:04d}" / "final.jsonl")]
    atomic_jsonl(finals, output / "generations.jsonl")
    print(json.dumps({"event": "partial_leaf_assembled", "output": str(output),
                      "planned_steps": planned, "complete_steps": steps,
                      "generations": len(finals)}), flush=True)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("leaf", "baseline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--benchmark")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.kind == "leaf":
        steps = finalize_leaf(args.output)
        command = ["leaf.grading.grade_leaf_rollout", "--generations", str(args.output / "generations.jsonl"),
                   "--output", str(args.output / "grades_public.jsonl"),
                   "--workers", str(args.workers), "--steps", str(steps)]
    else:
        command = ["leaf.grading.grade_baseline", "--output", str(args.output),
                   "--condition", args.condition, "--benchmark", args.benchmark,
                   "--mode", "formal", "--workers", str(args.workers), "--allow-partial"]
    raise SystemExit(subprocess.call([sys.executable, "-m", *command]))


if __name__ == "__main__":
    main()
