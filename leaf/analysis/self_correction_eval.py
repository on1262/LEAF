"""Train-set self-correction metrics of a LEAF rollout (CPU).

A LEAF rollout (condition ``leaf``, restart every 5 steps: 1, 6, 11, 16) on the training set -- the
base model's is the rollout behind the LEAF update data itself, a trained model's comes from
``leaf.explore.leaf_rollout --run-dir`` -- is read through its per-problem sample logs. Over steps
1..S, every pair (k, k+1) whose step k+1 is not a restart is a transition: step k+1 saw the
final of step k and its feedback. With before = ``final.public_correct`` of step k and after of
step k+1:

* wrong -> right (wrong-to-right rate): P(after correct | before wrong)
* right -> wrong (right-to-wrong rate): P(after wrong | before correct)
* repeat error (repeat-error rate): P(after wrong and the same answer as before | before wrong)

"Same answer": the whitespace-normalized extracted answer of both finals is equal and non-empty
-- ``carry.grade.extracted_answer`` (MATH: the boxed answer, also when the feedback hides it;
LCB: the graded program); SciKnowEval stores none there, so its answer is the text from the
last "Final answer:" of the response, lower-cased. Also: turn-1 accuracy (the restart steps'
finals) and the accuracy of every step. Each metric is given over all problems and over the
update problems (``--uids-file``: the problems with a correct rollout solution).

    python -m leaf.analysis.self_correction_eval summarize --run RUN --steps 10 \\
        --uids-file $RESULTS_ROOT/general/seed1/sft/qwen3_8b/qwen3_8b/formal/update_data/math500/successful_contexts.jsonl \\
        --output out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESTART_INTERVAL = 5
STEPS = 10


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines also breaks at U+2028 inside JSON strings
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def is_restart(step: int) -> bool:
    return (int(step) - 1) % RESTART_INTERVAL == 0


def normalize(text: str | None) -> str:
    return " ".join(str(text or "").split())


def answer_key(row: dict) -> str:
    """The comparable answer of a step-final ('' when none)."""
    final = row["final"]
    if final.get("source") == "sciknoweval" or final.get("task_type") == "sciqa":
        from leaf.explore.leaf_rollout import final_answer_text

        return normalize(final_answer_text(final.get("response") or "")).lower()
    grade = (row.get("carry") or {}).get("grade") or {}
    return normalize(grade.get("extracted_answer"))


def run_rows(run: Path, steps: int = STEPS) -> dict[str, dict[int, dict]]:
    """``{uid: {step: row}}`` of steps 1..``steps`` from ``samples/*.jsonl``."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted((Path(run) / "samples").glob("*.jsonl")):
        for row in read_jsonl(path):
            step = int(row["step"])
            if step <= steps:
                out.setdefault(row["problem_uid"], {})[step] = row
    return out


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def transition_metrics(rows: dict[str, dict[int, dict]], steps: int = STEPS,
                       uids: set[str] | None = None) -> dict:
    counts = {"wrong": 0, "right": 0, "wrong_to_right": 0, "right_to_wrong": 0,
              "wrong_to_wrong": 0, "repeat_error": 0, "transitions": 0}
    turn1, per_step = [], {}
    problems = complete = 0
    for uid, by_step in rows.items():
        if uids is not None and uid not in uids:
            continue
        problems += 1
        complete += all(s in by_step for s in range(1, steps + 1))
        for step, row in by_step.items():
            per_step.setdefault(step, []).append(bool(row["final"]["public_correct"]))
            if is_restart(step):
                turn1.append(bool(row["final"]["public_correct"]))
        for step in range(1, steps):
            nxt = step + 1
            if is_restart(nxt) or step not in by_step or nxt not in by_step:
                continue
            before, after = by_step[step], by_step[nxt]
            was, now = bool(before["final"]["public_correct"]), bool(after["final"]["public_correct"])
            counts["transitions"] += 1
            if was:
                counts["right"] += 1
                counts["right_to_wrong"] += not now
            else:
                counts["wrong"] += 1
                if now:
                    counts["wrong_to_right"] += 1
                else:
                    counts["wrong_to_wrong"] += 1
                    key = answer_key(before)
                    counts["repeat_error"] += bool(key) and key == answer_key(after)
    return {
        "problems": problems, "complete_problems": complete, **counts,
        "wrong_to_right_rate": ratio(counts["wrong_to_right"], counts["wrong"]),
        "right_to_wrong_rate": ratio(counts["right_to_wrong"], counts["right"]),
        "repeat_error_rate": ratio(counts["repeat_error"], counts["wrong"]),
        "wrong_to_wrong_rate": ratio(counts["wrong_to_wrong"], counts["wrong"]),
        "turn1_accuracy": ratio(sum(turn1), len(turn1)),
        "accuracy": ratio(sum(sum(v) for v in per_step.values()), sum(len(v) for v in per_step.values())),
        "accuracy_by_step": {str(s): ratio(sum(v), len(v)) for s, v in sorted(per_step.items())},
    }


def summarize_run(run: Path, steps: int = STEPS, uids_file: Path | None = None) -> dict:
    rows = run_rows(run, steps)
    out = {"run": str(run), "steps": steps, "all": transition_metrics(rows, steps)}
    if uids_file is not None:
        uids = {row["uid"] for row in read_jsonl(uids_file)}
        out["uids_file"] = str(uids_file)
        out["update_uids"] = transition_metrics(rows, steps, uids)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("summarize",))
    parser.add_argument("--run", type=Path, required=True, help="leaf.explore.leaf_rollout output directory")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--uids-file", type=Path, default=None, help="successful_contexts.jsonl of the training set")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = summarize_run(args.run, args.steps, args.uids_file)
    text = json.dumps(summary, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
