"""Grade one LEAF rollout condition and summarize its aligned steps.

One aligned step is one sequence for every
problem in the frozen dataset; every aggregate is also reported per `source`.
Steps are grouped by 4 so the 24 formal steps divide evenly.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from leaf.verifiers import mbppplus
from leaf.core.io_utils import atomic_json, atomic_jsonl
from leaf.verifiers.answer_grader import grade_records

GROUP_WIDTH = 4


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_config() -> dict:
    from leaf import paths
    from leaf.core.project_config import CONFIG_PATH

    return paths.load_json(CONFIG_PATH)


def summarize_subset(records: list[dict], grade_by_key: dict, steps_total: int) -> dict:
    """Per-step accuracy and cumulative pass rate over one slice of the dataset."""
    uids = {row["problem_uid"] for row in records}
    by_step = collections.defaultdict(list)
    for row in records:
        by_step[int(row["step"])].append(row)
    discovered: set[str] = set()
    steps = []
    for step in range(1, steps_total + 1):
        current = by_step.get(step, [])
        if not current:
            continue
        correct = [bool(grade_by_key[row["key"]]["correct"]) for row in current]
        discovered.update(
            row["problem_uid"]
            for row, value in zip(current, correct, strict=True)
            if value
        )
        steps.append({
            "step": step,
            "sequences": len(current),
            "correct": sum(correct),
            "accuracy": sum(correct) / len(current),
            "passed_problems_cumulative": len(discovered),
            "pass_rate_cumulative": len(discovered) / len(uids),
        })
    groups = []
    for start in range(1, len(steps) + 1, GROUP_WIDTH):
        selected = steps[start - 1 : start - 1 + GROUP_WIDTH]
        if len(selected) != GROUP_WIDTH:
            break
        correct = sum(row["correct"] for row in selected)
        sequences = sum(row["sequences"] for row in selected)
        groups.append({
            "start_step": start,
            "end_step": start + GROUP_WIDTH - 1,
            "accuracy": correct / sequences,
            "pass_rate_cumulative_at_end": selected[-1]["pass_rate_cumulative"],
        })
    total = sum(row["sequences"] for row in steps)
    return {
        "problems": len(uids),
        "candidates": total,
        "accuracy": sum(row["correct"] for row in steps) / total if total else 0.0,
        "final_pass_rate": steps[-1]["pass_rate_cumulative"] if steps else 0.0,
        "steps": steps,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--steps", type=int, default=None,
                        help="grade only the first STEPS aligned steps (a run stopped by its "
                             "time limit; leaf.grading.grade_explore_cell); default: the manifest's")
    args = parser.parse_args()
    config = load_config()
    records = read_jsonl(args.generations)
    manifest = json.loads((args.generations.parent / "manifest.json").read_text())
    if args.steps is not None:
        records = [row for row in records if int(row["step"]) <= args.steps]
        manifest = {**manifest, "steps": args.steps,
                    "candidates": args.steps * int(manifest["problems_per_step"])}

    # Server-rollout runs write steps and problems_per_step but no candidates.
    expected = int(manifest.get("candidates",
                                int(manifest["steps"]) * int(manifest["problems_per_step"])))
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(records)}")
    keys = [row["key"] for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Generation keys are not unique")

    steps_total = int(manifest["steps"])
    problem_uids = {row["problem_uid"] for row in records if int(row["step"]) == 1}
    if len(problem_uids) != int(manifest["problems_per_step"]):
        raise RuntimeError("Step 1 does not contain the frozen problem set")
    by_step = collections.defaultdict(set)
    for row in records:
        by_step[int(row["step"])].add(row["problem_uid"])
    for step in range(1, steps_total + 1):
        if by_step[step] != problem_uids:
            raise RuntimeError(f"Step {step} does not contain the same problem set")

    grades = grade_records(records, config, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(grades, args.output)
    grade_by_key = {row["key"]: row for row in grades}

    overall = summarize_subset(records, grade_by_key, steps_total)
    by_source = {}
    for source in sorted({row["source"] for row in records}):
        subset = [row for row in records if row["source"] == source]
        by_source[source] = summarize_subset(subset, grade_by_key, steps_total)

    summary = {
        "condition": manifest["condition"],
        "model": manifest.get("model_key", manifest.get("model")),
        "candidates": len(grades),
        "problems": len(problem_uids),
        "accuracy": overall["accuracy"],
        "final_pass_rate": overall["final_pass_rate"],
        "failure_types": dict(collections.Counter(row["failure_type"] for row in grades)),
        "failure_types_by_source": {
            source: dict(collections.Counter(
                row["failure_type"] for row in grades if row["source"] == source
            ))
            for source in by_source
        },
        "steps": overall["steps"],
        "groups_of_four": overall["groups"],
        "by_source": by_source,
    }
    if args.steps is not None:
        summary["aligned_steps_found"] = args.steps
    atomic_json(summary, args.output.with_name("rollout_summary.json"))
    # MBPP+: the offline private test, written beside the public
    # summary as rollout_summary_mbppplus.json (leaf.verifiers.mbppplus).
    if mbppplus.enabled(config) and "mbpp" in by_source:
        mbppplus.grade_cell(
            args.output.parent, kind="leaf", config=config, workers=args.workers,
            records=records, grades=grades,
        )
    print(json.dumps({
        "condition": summary["condition"],
        "accuracy": summary["accuracy"],
        "final_pass_rate": summary["final_pass_rate"],
        "by_source": {
            source: {
                "accuracy": value["accuracy"],
                "final_pass_rate": value["final_pass_rate"],
            }
            for source, value in by_source.items()
        },
    }), flush=True)


if __name__ == "__main__":
    main()
