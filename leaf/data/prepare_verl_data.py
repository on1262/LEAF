"""Export each benchmark to verl parquet and freeze its budget.

Every experiment is split into independent jobs, one per
benchmark, so each of the training baselines trains **separate
policies per base model**.  The mixed 1,331-problem dataset exists only as the
reporting aggregate; nothing trains on it.  Consequently every artefact here is
per benchmark and lives under ``<data_root>/rl_data/<benchmark>/``.

Per benchmark this writes:

``train.parquet`` / ``smoke.parquet``
    One verl ``RLHFDataset`` row per problem.  ``data_source`` is the row's
    *source*, so the reward function can dispatch on it, and
    ``reward_model.ground_truth`` is the whole ``verifier`` payload as JSON, so
    the reward function needs nothing else at rollout time.

``trial_schedule.jsonl``
    The random-sampling allocation: ``n x 24`` single generations balanced to within
    one trial per problem, seeded from ``decode_seed``.

``question_schedule.jsonl``
    The batched-method allocation: ``U x 32`` question occurrences balanced to
    within one occurrence per problem, seeded from ``schedule_seed``.

``train_formal.parquet`` / ``score_*.parquet`` / ``seed_*.parquet``
    That schedule materialised in update order for each method.

``budget_manifest.json``
    Every number below.  ``<data_root>/rl_data/budget_manifest.json`` holds the
    same numbers for all four benchmarks plus their totals.

Budget arithmetic
-----------------
For a benchmark with ``n`` problems the public-verifier allowance is ``n x 24``
nominal trials, and an *aligned rollout step* is one sequence for every one of
that benchmark's problems, so the target is **24 aligned steps** everywhere.

* Random Sampling draws one sequence per aligned-step position: ``24 x n``.
* rollout-N=8 methods (SDPO, GRPO, SEED-RL) expand each question occurrence into
  8 sequences, i.e. 8 aligned steps, so 24 aligned steps = 3 complete epochs =
  ``3n`` occurrences.  At ``question_batch_size=32`` that is
  ``U = ceil(3n / 32)`` updates::

      math500      n=500  U=47  1,504 occ  12,032 trials  vs 12,000 nominal (+32)
      lcb          n=131  U=13    416 occ   3,328 trials  vs  3,144 nominal (+184)
      mbpp         n=500  U=47  1,504 occ  12,032 trials  vs 12,000 nominal (+32)
      sciknoweval  n=200  U=19    608 occ   4,864 trials  vs  4,800 nominal (+64)
      ------------------------------------------------------------------------
      total              U=126  4,032 occ  32,256 trials  vs 31,944 nominal (+312)

  The overshoot is recorded in the budget manifest.  The
  aligned-step window is still exactly 24 for every benchmark, because
  ``floor(occurrences / n) = 3`` complete epochs x 8 rollouts = 24.  ``lcb``
  overshoots most (+5.9%) because 393 occurrences is only 12.3 batches of 32.
* SCoRe samples n=4 trajectories of 2 turns per occurrence, the same 8 sequences
  per occurrence, so it runs to the same U per benchmark.
* SEED spends the first ``round(seed_collection_fraction x U)`` whole updates on
  hindsight collection - a 4/25 = 16% share, taken
  proportionally rather than as a fixed 4 updates, which at ``lcb``'s U=13 would
  have eaten 31% of the run.  Collection trials plus RL trials equal the
  benchmark's total either way.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from leaf.prompts import task_prompts
from leaf.core.project_config import (
    BENCHMARKS, atomic_json, atomic_jsonl, balanced_schedule,
    benchmark_dataset_path, load_config, read_jsonl, rl_data_dir,
)

ABILITY = {
    "math": "math",
    "code_lcb": "code",
    "code_mbpp": "code",
    "sciqa": "science",
}
STYLE = {
    "math": "rule",
    "code_lcb": "code",
    "code_mbpp": "code",
    "sciqa": "rule",
}


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #


def budget_values(config: dict, benchmark: str, problems: int) -> dict:
    """Freeze every count for one benchmark from its problem count alone."""
    if problems < 1:
        raise ValueError(f"{benchmark} has no problems")
    trials_per_problem = int(config["trials_per_problem"])
    batch = int(config["question_batch_size"])
    group = int(config["rollout_group_size"])
    nominal = problems * trials_per_problem

    aligned_steps_target = int(config["aligned_steps_target"])
    if aligned_steps_target != trials_per_problem:
        raise RuntimeError("An aligned step is one sequence per problem; the two must agree")
    # 24 aligned steps / 8 rollouts per occurrence = 3 complete epochs. The
    # generalization runs' 20 steps are 2.5 epochs, so the occurrence count is taken before rounding;
    # when the step count is a multiple of the group size this equals epochs * problems.
    epochs = aligned_steps_target // group
    exact_occurrences = math.ceil(aligned_steps_target * problems / group)
    updates = math.ceil(exact_occurrences / batch)
    occurrences = updates * batch
    trials = occurrences * group

    collection_updates = max(
        1, round(float(config["seed_collection_fraction"]) * updates)
    )
    if updates - collection_updates < 1:
        raise RuntimeError(f"{benchmark}: SEED has no RL update left after collection")
    save_freq, optimizer_save_freq = checkpoint_cadence(config, updates)

    return {
        "benchmark": benchmark,
        "problems": problems,
        "trials_per_problem": trials_per_problem,
        "nominal_trials": nominal,
        "aligned_steps_target": aligned_steps_target,
        "random_sampling_generations": nominal,
        "question_batch_size": batch,
        "rollout_group_size": group,
        "complete_epochs": epochs,
        "exact_occurrences": exact_occurrences,
        "exact_updates": exact_occurrences / batch,
        "updates": updates,
        "question_occurrences": occurrences,
        "rollout_trials": trials,
        "trial_overshoot": trials - nominal,
        "trial_overshoot_fraction": (trials - nominal) / nominal,
        "aligned_steps": (occurrences // problems) * group,
        "checkpoint_save_freq": save_freq,
        "optimizer_save_freq": optimizer_save_freq,
        "score_rollout_group_size": int(config["score_rollout_group_size"]),
        "score_turns": 2,
        "score_updates": updates,
        "score_sequences": occurrences * int(config["score_rollout_group_size"]) * 2,
        "seed_collection_fraction": float(config["seed_collection_fraction"]),
        "seed_collection_updates": collection_updates,
        "seed_collection_trials": collection_updates * batch * group,
        "seed_rl_updates": updates - collection_updates,
        "seed_rl_trials": (updates - collection_updates) * batch * group,
        "seed_total_trials": trials,
    }


def checkpoint_cadence(config: dict, updates: int) -> tuple[int, int]:
    """Checkpoint intervals that still bound lost work on a 13-update run.

    The default is a LoRA checkpoint every 5 updates and an optimizer copy
    every 25.  At ``lcb``'s U=13 a 5-update interval can lose 38% of the run to
    one preempted chunk, so the interval is capped at ``U // 5`` and the
    optimizer keeps the same 5:1 ratio to it.  Long benchmarks are unaffected:
    ``math500``'s U=47 keeps 5 / 25.
    """
    interval = int(config["checkpoint_interval"])
    ratio = max(1, int(config["optimizer_checkpoint_interval"]) // interval)
    save_freq = max(1, min(interval, updates // 5))
    optimizer_save_freq = min(
        int(config["optimizer_checkpoint_interval"]), save_freq * ratio
    )
    return save_freq, max(save_freq, optimizer_save_freq)


# --------------------------------------------------------------------------- #
# parquet
# --------------------------------------------------------------------------- #


def verl_row(row: dict, split: str, model: str | None = None) -> dict:
    task_type = row["task_type"]
    if task_type not in task_prompts.TASK_TYPES:
        raise ValueError(f"Unknown task_type: {task_type}")
    messages = task_prompts.opening_messages(row, model)
    prompt = messages[-1]["content"]
    return {
        # The reward function dispatches on data_source, so it is the source.
        "data_source": row["source"],
        "prompt": messages,
        "ability": ABILITY[task_type],
        "reward_model": {
            "style": STYLE[task_type],
            # The whole verifier payload travels with the row: the reward
            # function never reads the dataset back from disk.
            "ground_truth": json.dumps(row["verifier"], ensure_ascii=False, sort_keys=True),
        },
        "extra_info": {
            "split": split,
            "index": str(row["index"]),
            "problem_uid": row["uid"],
            "source": row["source"],
            "task_type": task_type,
            "problem": prompt,
        },
    }


def write_parquet(rows: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)
    return {"path": str(path), "rows": len(rows)}


def check_balance(uids: list[str], label: str) -> tuple[int, int]:
    counts = collections.Counter(uids)
    low, high = min(counts.values()), max(counts.values())
    if high - low > 1:
        raise RuntimeError(f"{label} schedule is not balanced to within one trial")
    return low, high


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #

# Seed blocks are spaced far enough apart that no two sequences, in any
# benchmark or stage, ever share a decode seed.
SEED_STRIDE = 10_000_000
BENCHMARK_SEED_OFFSET = {name: index * SEED_STRIDE for index, name in enumerate(BENCHMARKS)}
SMOKE_SEED_OFFSET = 900_000_000


def trial_schedule(uids: list[str], config: dict, budget: dict, offset: int) -> list[dict]:
    order = balanced_schedule(
        uids, budget["random_sampling_generations"], int(config["decode_seed"]) + offset
    )
    check_balance(order, "random-sampling trial")
    attempts: collections.Counter = collections.Counter()
    base = int(config["decode_seed"]) + offset
    rows = []
    for index, uid in enumerate(order):
        rows.append({
            "trial_index": index,
            "problem_uid": uid,
            "attempt_index": attempts[uid],
            # One aligned step = one pass over the benchmark, so the step a
            # trial belongs to is its attempt index.
            "aligned_step": attempts[uid],
            "seed": base + index,
        })
        attempts[uid] += 1
    return rows


def question_schedule(uids: list[str], config: dict, budget: dict, offset: int) -> list[dict]:
    batch = budget["question_batch_size"]
    group = budget["rollout_group_size"]
    order = balanced_schedule(
        uids, budget["question_occurrences"], int(config["schedule_seed"]) + offset
    )
    check_balance(order, "question")
    base = int(config["decode_seed"]) + offset + budget["random_sampling_generations"] + 1
    return [
        {
            "question_position": index,
            "update_index": index // batch,
            "batch_index": index % batch,
            "problem_uid": uid,
            "seed": base + index * group,
            "rollout_trials": group,
        }
        for index, uid in enumerate(order)
    ]


def smoke_schedule(
    uids: list[str], config: dict, updates: int, batch: int, offset: int
) -> list[dict]:
    order = balanced_schedule(
        uids, updates * batch, int(config["schedule_seed"]) + offset
    )
    base = int(config["decode_seed"]) + SMOKE_SEED_OFFSET + offset
    return [
        {
            "question_position": index,
            "update_index": index // batch,
            "batch_index": index % batch,
            "problem_uid": uid,
            "seed": base + index * int(config["rollout_group_size"]),
            "rollout_trials": int(config["rollout_group_size"]),
        }
        for index, uid in enumerate(order)
    ]


def rows_in_schedule_order(
    schedule: list[dict], by_uid: dict[str, dict], extra=None
) -> list[dict]:
    output = []
    for scheduled in schedule:
        row = json.loads(json.dumps(by_uid[scheduled["problem_uid"]]))
        row["extra_info"].update({
            "question_position": str(scheduled["question_position"]),
            "update_index": str(scheduled["update_index"]),
            "batch_index": str(scheduled["batch_index"]),
        })
        if extra is not None:
            extra(row, scheduled)
        row["rollout_seed"] = int(scheduled["seed"])
        output.append(row)
    return output


# --------------------------------------------------------------------------- #


def prepare_benchmark(
    config: dict, benchmark: str, out: Path, model: str | None = None
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    offset = BENCHMARK_SEED_OFFSET[benchmark]
    dataset = read_jsonl(benchmark_dataset_path(config, benchmark, "formal"))
    if not dataset:
        raise RuntimeError(f"{benchmark}: no frozen dataset file")
    expected = int(config["source_sizes"][benchmark])
    if len(dataset) != expected:
        raise RuntimeError(f"{benchmark}: expected {expected} problems, found {len(dataset)}")
    if any(row["source"] != benchmark for row in dataset):
        raise RuntimeError(f"{benchmark}: dataset file contains another source")
    smoke = read_jsonl(benchmark_dataset_path(config, benchmark, "smoke"))
    uids = [row["uid"] for row in dataset]
    if len(set(uids)) != len(uids):
        raise RuntimeError(f"{benchmark}: duplicate problem uid")

    budget = budget_values(config, benchmark, len(dataset))
    files: dict[str, dict] = {}

    train_rows = [verl_row(row, "train", model) for row in dataset]
    files["train"] = write_parquet(train_rows, out / "train.parquet")
    if smoke:
        files["smoke"] = write_parquet(
            [verl_row(row, "train", model) for row in smoke],
            out / "smoke.parquet",
        )
    by_uid = {row["extra_info"]["problem_uid"]: row for row in train_rows}

    trials = trial_schedule(uids, config, budget, offset)
    atomic_jsonl(trials, out / "trial_schedule.jsonl")
    questions = question_schedule(uids, config, budget, offset)
    atomic_jsonl(questions, out / "question_schedule.jsonl")

    # --- SDPO / GRPO ---------------------------------------------------------
    files["train_formal"] = write_parquet(
        rows_in_schedule_order(questions, by_uid), out / "train_formal.parquet"
    )

    # --- SCoRe ---------------------------------------------------------------
    stage1_updates = max(1, math.floor(
        float(config["score_stage1_fraction"]) * budget["updates"]
    ))

    def score_extra(row: dict, scheduled: dict) -> None:
        row["extra_info"]["score_stage"] = (
            "stage1" if scheduled["update_index"] < stage1_updates else "stage2"
        )
        row["extra_info"]["score_correction_bonus"] = str(config["score_correction_bonus"])
        row["rollout_seed_stride"] = 2
        row["agent_name"] = "score_agent"

    files["score_formal"] = write_parquet(
        rows_in_schedule_order(questions, by_uid, score_extra), out / "score_formal.parquet"
    )

    # --- SEED ----------------------------------------------------------------
    reserved = budget["seed_collection_updates"]
    collection = [row for row in questions if row["update_index"] < reserved]
    seed_rl = [row for row in questions if row["update_index"] >= reserved]
    if len(collection) * budget["rollout_group_size"] != budget["seed_collection_trials"]:
        raise RuntimeError(f"{benchmark}: SEED collection slice is the wrong size")
    if len(seed_rl) != budget["seed_rl_updates"] * budget["question_batch_size"]:
        raise RuntimeError(f"{benchmark}: SEED RL slice is the wrong size")
    files["seed_collection_formal"] = write_parquet(
        rows_in_schedule_order(collection, by_uid), out / "seed_collection_formal.parquet"
    )
    files["seed_rl_formal"] = write_parquet(
        rows_in_schedule_order(seed_rl, by_uid), out / "seed_rl_formal.parquet"
    )

    # --- smoke variants ------------------------------------------------------
    if smoke:
        smoke_rows = [verl_row(row, "train", model) for row in smoke]
        smoke_by_uid = {row["extra_info"]["problem_uid"]: row for row in smoke_rows}
        smoke_uids = [row["uid"] for row in smoke]
        score_batch = int(config["score_smoke_question_batch_size"])
        score_smoke = smoke_schedule(
            smoke_uids, config, int(config["smoke_updates"]), score_batch, offset
        )

        def score_smoke_extra(row: dict, scheduled: dict) -> None:
            row["extra_info"]["score_stage"] = (
                "stage1" if scheduled["update_index"] < 1 else "stage2"
            )
            row["extra_info"]["score_correction_bonus"] = str(config["score_correction_bonus"])
            row["rollout_seed_stride"] = 2
            row["agent_name"] = "score_agent"

        files["score_smoke"] = write_parquet(
            rows_in_schedule_order(score_smoke, smoke_by_uid, score_smoke_extra),
            out / "score_smoke.parquet",
        )
        seed_batch = int(config["seed_smoke_question_batch_size"])
        seed_smoke = smoke_schedule(smoke_uids, config, 1, seed_batch, offset)
        files["seed_collection_smoke"] = write_parquet(
            rows_in_schedule_order(seed_smoke[:1], smoke_by_uid),
            out / "seed_collection_smoke.parquet",
        )
        files["seed_rl_smoke"] = write_parquet(
            rows_in_schedule_order(seed_smoke, smoke_by_uid), out / "seed_rl_smoke.parquet"
        )

    trial_low, trial_high = check_balance([row["problem_uid"] for row in trials], "trial")
    q_low, q_high = check_balance([row["problem_uid"] for row in questions], "question")
    budget.update({
        "schedule_seed": int(config["schedule_seed"]) + offset,
        "decode_seed": int(config["decode_seed"]) + offset,
        "score_stage1_updates": stage1_updates,
        "score_stage2_updates": budget["updates"] - stage1_updates,
        "per_problem_trial_min": trial_low,
        "per_problem_trial_max": trial_high,
        "per_problem_question_min": q_low,
        "per_problem_question_max": q_high,
    })
    atomic_json(budget, out / "budget_manifest.json")
    atomic_json({
        "revision": config["revision"],
        "benchmark": benchmark,
        "schema": "SDPO/verl RLHFDataset parquet",
        "dataset": str(benchmark_dataset_path(config, benchmark, "formal")),
        "files": files,
        "prompt_model": model,
    }, out / "manifest.json")
    return budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), action="append",
                        help="repeatable; defaults to all four")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to <data_root>/rl_data[/<model>]")
    parser.add_argument("--model", default=None,
                        help="write the tree a variant_instruction_cells model "
                             "needs; "
                             "omit for the shared, prompt-neutral tree")
    args = parser.parse_args()
    config = load_config()
    benchmarks = args.benchmark or list(BENCHMARKS)
    if args.model:
        cells = (config.get("variant_instruction_cells") or {}).get(args.model, ())
        extra = [name for name in benchmarks if name not in cells]
        if extra:
            raise SystemExit(
                f"{args.model} takes the shared prompt on {', '.join(extra)}; "
                "those benchmarks read the shared tree, so do not rebuild them "
                "per model"
            )
    out = args.output or (
        rl_data_dir(config) / args.model if args.model else rl_data_dir(config)
    )

    budgets = {}
    for benchmark in benchmarks:
        budgets[benchmark] = prepare_benchmark(
            config, benchmark, out / benchmark, args.model
        )

    # Rebuilding one benchmark keeps the others' entries in the aggregate, so a
    # benchmark added to a root later does not drop the ones already there.
    aggregate_path = out / "budget_manifest.json"
    if aggregate_path.exists():
        previous = json.loads(aggregate_path.read_text()).get("benchmarks", {})
        budgets = {**{name: value for name, value in previous.items()
                      if name not in budgets and (out / name / "budget_manifest.json").exists()},
                   **budgets}
    totals = {
        key: sum(budget[key] for budget in budgets.values())
        for key in (
            "problems", "nominal_trials", "random_sampling_generations", "updates",
            "question_occurrences", "rollout_trials", "trial_overshoot",
            "seed_collection_trials", "seed_rl_trials",
        )
    }
    aggregate = {
        "revision": config["revision"],
        "split": "one independent training job per benchmark",
        "benchmarks": budgets,
        "totals": totals,
    }
    atomic_json(aggregate, out / "budget_manifest.json")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
