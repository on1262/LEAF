"""The successful contexts of the LEAF update from a finished LEAF rollout (CPU only).

``successful_contexts.jsonl``: one row per problem with >= 1 correct step-final that is
not cut by the length cap (exact duplicates dropped; :func:`leaf.update.trace_selection.
correct_traces`): ``teacher_messages_list`` = the chat messages each of those
solutions was generated from (its round prompt: earlier attempt + feedback, the
privileged context of the update), in step order. The trainer draws one per
(step, problem). Written into the update data directory of the training set
(``manifest.json`` there names the dataset).

    python -m leaf.update.build_contexts --model qwen3_8b --benchmark math500 \\
        --rollout $RESULTS_ROOT/general/seed1/leaf/qwen3_8b/qwen3_8b/formal/leaf/leaf/math500 \\
        --output $RESULTS_ROOT/general/seed1/leaf/qwen3_8b/qwen3_8b/formal/update_data/math500
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from leaf.update.trace_selection import correct_traces

MODELS = ("olmo3_7b", "qwen3_8b")
CONTEXTS = "successful_contexts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from leaf.core.project_config import atomic_json, atomic_jsonl, read_jsonl
    from leaf.prompts.task_prompts import opening_messages
    from leaf.update.rollout_rows import run_rows

    run = args.rollout.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    dataset = read_jsonl(Path(manifest["dataset"]))
    rows_by_uid = run_rows(run)
    out = []
    for index, problem in enumerate(dataset):
        traces = correct_traces(rows_by_uid.get(problem["uid"], []))
        traces = [t for t in traces if t["messages"] and t["messages"][-1]["role"] == "user"]
        if not traces:
            continue
        contexts = [t["messages"] for t in traces]
        out.append({"uid": problem["uid"], "problem_index": index,
                    "prompt_messages": opening_messages(problem, args.model, ""),
                    "teacher_messages_list": contexts,
                    "teacher_steps": [t["step"] for t in traces]})
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(out, args.output / f"{CONTEXTS}.jsonl")
    target = args.output / "manifest.json"
    if not target.exists():
        atomic_json({"dataset": manifest["dataset"], "rollout": str(run), "model": args.model,
                     "benchmark": args.benchmark}, target)
    summary = {"rollout": str(run), "problems": len(out),
               "contexts_mean": statistics.mean(len(r["teacher_messages_list"]) for r in out),
               "contexts_max": max(len(r["teacher_messages_list"]) for r in out)}
    atomic_json(summary, args.output / f"{CONTEXTS}_summary.json")
    print(json.dumps({"event": "prepared", **summary}), flush=True)


if __name__ == "__main__":
    main()
