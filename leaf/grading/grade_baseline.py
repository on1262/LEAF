"""Summarize one *baseline* run onto the aligned-step grid.

``leaf.grading.grade_leaf_rollout`` fits the LEAF rollout driver, whose generations already sit on
the aligned-step grid: one sequence per problem per step.  The baselines do not.

* **Random Sampling** writes one ungraded trial per line into ``generations.jsonl``,
  already carrying the ``aligned_step`` its frozen trial schedule assigned it.
  Those records still have to go through the public verifier.
* the **verl** baselines (SDPO, GRPO, SCoRe, SEED-RL) write one
  ``rollouts/<update>.jsonl`` per policy update, and every record there was
  already graded inline by ``leaf.baselines.verl_reward`` while training ran.  Grading
  them again would cost hours and could disagree with the reward the policy
  actually saw, so their ``acc`` is taken as given.  What they lack is the
  alignment: a batch is 32 *question occurrences*, each expanded into 8
  sequences, which is 8 aligned steps' worth of work for 32 of the benchmark's
  problems.  One complete epoch of occurrences therefore expands into 8 aligned
  steps, and the 3 epochs every exploration budget buys expand into the 24 steps
  the report compares.

The output is exactly the ``rollout_summary.json`` schema
``leaf.grading.grade_leaf_rollout`` writes so the report code consumes both
without a special case.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from leaf.core.project_config import (
    BENCHMARKS, MODES, atomic_json, atomic_jsonl, benchmark_rl_data_dir,
    load_config, parquet_mode, read_jsonl,
)
from leaf.verifiers import mbppplus

GROUP_WIDTH = 4
CANDIDATES_PER_OCCURRENCE = 8
ALIGNED_STEPS_TARGET = 24

VERL_CONDITIONS = ("sdpo", "grpo", "score", "seed_rl")
CONDITIONS = ("random_sampling", *VERL_CONDITIONS)

# The parquet each runner actually trains on; its row order *is* the question
# schedule, and every row carries the question position in ``extra_info``.
# smoke_full rolls out against the formal parquets, so it grades against them.
SCHEDULE_PARQUET = {
    "sdpo": {"formal": "train_formal.parquet", "smoke": "smoke.parquet",
             "smoke_full": "train_formal.parquet"},
    "grpo": {"formal": "train_formal.parquet", "smoke": "smoke.parquet",
             "smoke_full": "train_formal.parquet"},
    "score": {"formal": "score_formal.parquet", "smoke": "score_smoke.parquet",
              "smoke_full": "score_formal.parquet"},
    "seed_rl": {"formal": "seed_rl_formal.parquet", "smoke": "seed_rl_smoke.parquet",
                "smoke_full": "seed_rl_formal.parquet"},
}
# The prompt -> uid map only needs a file that holds every problem of the
# benchmark, which the schedule parquets of a truncated stage (SEED-RL) do not
# promise.
MAPPING_PARQUET = {"formal": "train_formal.parquet", "smoke": "smoke.parquet",
                   "smoke_full": "train_formal.parquet"}

# SCoRe samples 4 trajectories of 2 turns, so it still spends 8 sequences on a
# question occurrence; the two turns of one record are two of those sequences.
RECORDS_PER_OCCURRENCE = {"sdpo": 8, "grpo": 8, "seed_rl": 8, "score": 4}


class PromptMappingError(RuntimeError):
    """A rollout prompt could not be traced back to exactly one problem."""


# --------------------------------------------------------------------------- #
# prompt -> problem_uid
# --------------------------------------------------------------------------- #


def normalise(text: str) -> str:
    """Line-ending normalisation only, so the prompt stays a literal substring."""
    return str(text).replace("\r\n", "\n")


def prompt_uid_map(parquet: Path) -> dict[str, str]:
    """``{rendered problem prompt: problem_uid}`` for one benchmark."""
    table = pq.read_table(parquet, columns=["prompt", "extra_info"])
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in table.to_pylist():
        # The user turn: an OLMES LiveCodeBench prompt opens on a system
        # turn that every problem shares.
        prompt = normalise(row["prompt"][-1]["content"])
        uid = str(row["extra_info"]["problem_uid"])
        if result.setdefault(prompt, uid) != uid:
            # MBPP 76 and 347 are the same task; with the asserts out of the
            # OLMES prompt they render identically. Their verifier payloads
            # still differ, so they resolve through ground_truth_uid_map, and a
            # record that neither map resolves still fails in the resolver.
            ambiguous.add(prompt)
    for prompt in ambiguous:
        result.pop(prompt)
    if not result:
        raise PromptMappingError(f"{parquet} holds no prompts")
    return result


def ground_truth_uid_map(parquet: Path) -> dict[str, str]:
    """``{verifier payload: problem_uid}`` for one benchmark.

    The rollout records carry the verifier payload verbatim, so this maps a
    record to its problem without going through the rendered prompt. That
    matters because the prompt in a rollout record is a tokenizer round trip of
    the parquet text, and it is not always byte-identical: a tokenizer can give back
    ``characters'std'`` where the parquet holds ``characters 'std'``.
    """
    table = pq.read_table(parquet, columns=["reward_model", "extra_info"])
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in table.to_pylist():
        payload = row["reward_model"]["ground_truth"]
        if not isinstance(payload, str):
            continue
        uid = str(row["extra_info"]["problem_uid"])
        if result.setdefault(payload, uid) != uid:
            # Two problems can carry the same verifier payload -- MBPP holds a
            # few problems whose entry point and asserts coincide. Those cannot
            # be told apart this way, so they fall back to the prompt text,
            # which does distinguish them; the rest keep the exact path.
            ambiguous.add(payload)
    for payload in ambiguous:
        result.pop(payload, None)
    return result


class PromptResolver:
    """Map a rendered rollout prompt back to its problem uid.

    The rollout records store the whole chat-templated prompt, so the parquet
    prompt is a substring of it framed by a chat prefix and suffix that are
    constant for a given model and stage.  The first record pays a linear scan
    over the benchmark's problems to learn that framing; every later record is
    one dictionary lookup.  Anything the learned framing does not explain falls
    back to the scan, and a record that matches no problem — or more than one —
    is a hard failure rather than a silent mis-assignment.
    """

    def __init__(self, mapping: dict[str, str],
                 ground_truths: dict[str, str] | None = None) -> None:
        self.mapping = mapping
        self.ground_truths = ground_truths or {}
        self.affixes: list[tuple[str, str]] = []
        self.scans = 0
        self.squeezed_scans = 0
        self.by_ground_truth = 0

    def resolve_row(self, row: dict) -> str:
        """The record's problem: by verifier payload first, prompt text after.

        The payload survives the chat template and the tokenizer round trip;
        the prompt text does not always.
        """
        payload = row.get("gts")
        if isinstance(payload, str):
            uid = self.ground_truths.get(payload)
            if uid is not None:
                self.by_ground_truth += 1
                return uid
        return self.resolve(row["input"])

    def resolve(self, text: str) -> str:
        value = normalise(text)
        for prefix, suffix in self.affixes:
            uid = self._by_affix(value, prefix, suffix)
            if uid is not None:
                return uid
        for body in self._guesses(value):
            uid = self.mapping.get(body)
            if uid is not None:
                return uid
        return self._scan(value)

    def _by_affix(self, value: str, prefix: str, suffix: str) -> str | None:
        if not value.startswith(prefix) or not value.endswith(suffix):
            return None
        body = value[len(prefix) : len(value) - len(suffix)] if suffix else value[len(prefix):]
        return self.mapping.get(body)

    @staticmethod
    def _guesses(value: str) -> list[str]:
        """Cheap candidate bodies, best first."""
        guesses = [value]
        if value.startswith("user\n"):
            inner = value[len("user\n") :]
            for body in (
                inner.split("\nassistant\n", 1)[0],
                inner.rsplit("\nassistant\n", 1)[0],
                inner,
            ):
                if body not in guesses:
                    guesses.append(body)
        return guesses

    @staticmethod
    def _squeeze(text: str) -> str:
        """The text with every whitespace character removed.

        A rollout record's prompt is a tokenizer round trip of the parquet text
        and is not always byte-identical: a tokenizer can return ``characters'std'``
        where the parquet holds ``characters 'std'``. Dropping whitespace on
        both sides makes the containment test survive that, and the ambiguity
        check below still refuses a match that fits two problems.
        """
        return "".join(text.split())

    def _scan(self, value: str) -> str:
        self.scans += 1
        found: list[tuple[str, str]] = [
            (prompt, uid) for prompt, uid in self.mapping.items() if prompt in value
        ]
        if not found:
            squeezed = self._squeeze(value)
            found = [
                (prompt, uid) for prompt, uid in self.mapping.items()
                if self._squeeze(prompt) in squeezed
            ]
            if found:
                self.squeezed_scans += 1
        if not found:
            raise PromptMappingError(
                "Rollout prompt matches no problem in the frozen parquet; "
                f"prompt starts with {value[:120]!r}"
            )
        if len({uid for _, uid in found}) > 1:
            raise PromptMappingError(
                "Rollout prompt matches several problems: "
                + ", ".join(sorted({uid for _, uid in found})[:4])
            )
        # Longest match wins when one prompt is a prefix of another.
        prompt, uid = max(found, key=lambda item: len(item[0]))
        start = value.find(prompt)
        if start >= 0:
            # Only an exact match frames a reusable affix; a whitespace-
            # insensitive one says nothing about where the prompt starts.
            affix = (value[:start], value[start + len(prompt) :])
            if affix not in self.affixes:
                self.affixes.append(affix)
        return uid


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #


def stable_rng(seed: int, *parts: object) -> random.Random:
    text = ":".join([str(seed), *(str(part) for part in parts)])
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def numbered_jsonl(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError(f"No rollout JSONL files in {directory}")
    return paths


def parquet_schedule(path: Path, batch_hint: int) -> list[dict]:
    """The question occurrences a verl run consumed, in update order.

    The formal parquets are materialised straight from the frozen question
    schedule, so every row already carries its ``question_position`` and
    ``update_index``.  The smoke parquets for SDPO and GRPO are the plain
    one-row-per-problem files the runner clamps its batch to, so the positions
    are synthesized from that same clamped batch and the rows' order, which is
    what ``data.shuffle=False`` makes the dataloader walk.
    """
    table = pq.read_table(path, columns=["extra_info"]).to_pylist()
    rows = [row["extra_info"] for row in table]
    if rows and rows[0].get("question_position") is not None:
        schedule = [
            {
                "question_position": int(row["question_position"]),
                "update_index": int(row["update_index"]),
                "problem_uid": str(row["problem_uid"]),
            }
            for row in rows
        ]
        schedule.sort(key=lambda row: row["question_position"])
        return schedule
    batch = max(1, min(batch_hint, len(rows)))
    return [
        {
            "question_position": index,
            "update_index": index // batch,
            "problem_uid": str(row["problem_uid"]),
        }
        for index, row in enumerate(rows)
    ]


def repeat_schedule(schedule: list[dict], passes: int) -> list[dict]:
    """Extend a single-pass schedule over the updates that actually ran.

    verl wraps its dataloader when the parquet is shorter than the run, which
    only happens on the smoke parquets: they hold one pass over the smoke
    subset while the smoke run does two updates.
    """
    if passes <= 1:
        return schedule
    updates_per_pass = len({row["update_index"] for row in schedule})
    result = list(schedule)
    for index in range(1, passes):
        for row in schedule:
            result.append({
                "question_position": row["question_position"] + index * len(schedule),
                "update_index": row["update_index"] + index * updates_per_pass,
                "problem_uid": row["problem_uid"],
            })
    return result


# --------------------------------------------------------------------------- #
# candidate recovery
# --------------------------------------------------------------------------- #


def candidate_outcomes(row: dict, condition: str) -> list[tuple[bool, str]]:
    """The (correct, failure_type) pairs one rollout record contributes."""
    if condition == "score":
        # One SCoRe record is a 2-turn trajectory: both turns are sequences.
        return [
            (float(row[key]) > 0.5, "pass" if float(row[key]) > 0.5 else "incorrect")
            for key in ("score_r1", "score_r2")
        ]
    correct = float(row["acc"]) > 0.5
    failure = str(row.get("failure_type") or ("pass" if correct else "incorrect"))
    return [(correct, failure)]


def update_candidates(
    *, rows: list[dict], scheduled: list[dict], resolver: PromptResolver,
    condition: str, seed: int,
) -> dict[int, list[tuple[bool, str]]]:
    """Attach one update's sequences to the question occurrences it served."""
    by_uid: dict[str, list[tuple[bool, str]]] = collections.defaultdict(list)
    for row in rows:
        by_uid[resolver.resolve_row(row)].extend(candidate_outcomes(row, condition))

    positions_by_uid: dict[str, list[int]] = collections.defaultdict(list)
    for item in scheduled:
        positions_by_uid[item["problem_uid"]].append(int(item["question_position"]))
    if set(by_uid) != set(positions_by_uid):
        missing = sorted(set(positions_by_uid) - set(by_uid))[:3]
        extra = sorted(set(by_uid) - set(positions_by_uid))[:3]
        raise RuntimeError(
            f"{condition}: update {scheduled[0]['update_index']} rollout and schedule "
            f"problem sets differ (missing {missing}, unexpected {extra})"
        )

    result: dict[int, list[tuple[bool, str]]] = {}
    for uid, positions in positions_by_uid.items():
        candidates = by_uid[uid]
        expected = len(positions) * CANDIDATES_PER_OCCURRENCE
        if len(candidates) != expected:
            raise RuntimeError(
                f"{condition}: {uid} has {len(candidates)} candidates in update "
                f"{scheduled[0]['update_index']}, expected {expected}"
            )
        # A balanced batch can straddle two epochs and hold the same problem
        # twice; the rollout keeps no occurrence id, so split its candidates
        # with a frozen partition instead of an arbitrary one.
        stable_rng(seed, condition, scheduled[0]["update_index"], uid).shuffle(candidates)
        for occurrence, position in enumerate(sorted(positions)):
            start = occurrence * CANDIDATES_PER_OCCURRENCE
            result[position] = candidates[start : start + CANDIDATES_PER_OCCURRENCE]
    return result


def rollout_occurrences(
    *, rollout_dir: Path, schedule: list[dict], resolver: PromptResolver,
    condition: str, seed: int,
) -> dict[int, list[tuple[bool, str]]]:
    by_update: dict[int, list[dict]] = collections.defaultdict(list)
    for row in schedule:
        by_update[int(row["update_index"])].append(row)
    offset = min(by_update)
    result: dict[int, list[tuple[bool, str]]] = {}
    for path in numbered_jsonl(rollout_dir):
        update = int(path.stem) - 1 + offset
        scheduled = by_update.get(update)
        if not scheduled:
            raise RuntimeError(f"{condition}: no schedule rows for update {update}")
        rows = read_jsonl(path)
        expected = len(scheduled) * RECORDS_PER_OCCURRENCE[condition]
        if len(rows) != expected:
            raise RuntimeError(
                f"{path}: {len(rows)} records, expected {expected}"
            )
        current = update_candidates(
            rows=rows, scheduled=scheduled, resolver=resolver,
            condition=condition, seed=seed,
        )
        overlap = sorted(set(result) & set(current))
        if overlap:
            raise RuntimeError(f"{condition}: duplicate question positions {overlap[:3]}")
        result.update(current)
    return result


def collection_occurrences(directory: Path) -> dict[int, list[tuple[bool, str]]]:
    """SEED's hindsight-collection stage, which spends the run's first updates.

    Its trials are ``question_position * 8 + rollout_index``, so they recover
    the occurrences the RL stage never saw.  Without them SEED-RL alone starts
    mid-epoch and can never show 24 complete aligned steps.
    """
    path = directory / "grades_public.json"
    if not path.is_file():
        return {}
    grades = json.loads(path.read_text())
    result: dict[int, list[tuple[bool, str]]] = collections.defaultdict(list)
    for row in sorted(grades, key=lambda value: int(value["trial_index"])):
        correct = bool(row["correct"])
        result[int(row["trial_index"]) // CANDIDATES_PER_OCCURRENCE].append(
            (correct, str(row.get("failure_type") or ("pass" if correct else "incorrect")))
        )
    for position, values in result.items():
        if len(values) != CANDIDATES_PER_OCCURRENCE:
            raise RuntimeError(
                f"SEED collection position {position} has {len(values)} candidates"
            )
    return dict(result)


# --------------------------------------------------------------------------- #
# aligned-step expansion
# --------------------------------------------------------------------------- #


def steps_from_occurrences(
    *, occurrences: dict[int, list[tuple[bool, str]]], schedule: list[dict],
    uids: set[str], condition: str, seed: int,
) -> list[dict[str, tuple[bool, str]]]:
    """Expand complete epochs of occurrences into 8 aligned steps each."""
    schedule = [row for row in schedule if int(row["question_position"]) in occurrences]
    schedule.sort(key=lambda row: int(row["question_position"]))
    problems = len(uids)
    steps: list[dict[str, tuple[bool, str]]] = []
    for epoch in range(len(schedule) // problems):
        block = schedule[epoch * problems : (epoch + 1) * problems]
        if {row["problem_uid"] for row in block} != uids:
            break  # a partial epoch ends the aligned window
        expanded: list[dict[str, tuple[bool, str]]] = [
            {} for _ in range(CANDIDATES_PER_OCCURRENCE)
        ]
        for row in block:
            uid = row["problem_uid"]
            candidates = list(occurrences[int(row["question_position"])])
            if len(candidates) != CANDIDATES_PER_OCCURRENCE:
                raise RuntimeError(
                    f"{condition}: occurrence {row['question_position']} has "
                    f"{len(candidates)} candidates"
                )
            stable_rng(seed, condition, "expand", epoch, uid).shuffle(candidates)
            for offset, outcome in enumerate(candidates):
                expanded[offset][uid] = outcome
        steps.extend(expanded)
    return steps


def random_sampling_steps(
    grades: list[dict], records: list[dict], uids: set[str]
) -> list[dict[str, tuple[bool, str]]]:
    """Random Sampling already carries its aligned step; only validate and group."""
    grade_by_key = {row["key"]: row for row in grades}
    by_step: dict[int, dict[str, tuple[bool, str]]] = collections.defaultdict(dict)
    for record in records:
        # The trial schedule numbers aligned steps from 0; the report is 1-based.
        step = int(record["aligned_step"]) + 1
        uid = str(record["problem_uid"])
        if uid in by_step[step]:
            raise RuntimeError(f"random-sampling step {step} repeats problem {uid}")
        grade = grade_by_key[record["key"]]
        by_step[step][uid] = (
            bool(grade["correct"]),
            str(grade.get("failure_type") or ("pass" if grade["correct"] else "incorrect")),
        )
    steps = []
    for step in sorted(by_step):
        values = by_step[step]
        if set(values) != uids:
            break  # the trailing partial step is outside the aligned window
        if step != len(steps) + 1:
            raise RuntimeError(f"random-sampling aligned steps are not contiguous at {step}")
        steps.append(values)
    return steps


# --------------------------------------------------------------------------- #
# summary, in the schema leaf.grading.grade_leaf_rollout writes
# --------------------------------------------------------------------------- #


def summarize_steps(steps: list[dict[str, tuple[bool, str]]], uids: set[str]) -> dict:
    discovered: set[str] = set()
    rows = []
    for index, values in enumerate(steps, 1):
        if set(values) != uids:
            raise RuntimeError(f"Step {index} is not one sequence per problem")
        discovered.update(uid for uid, (correct, _) in values.items() if correct)
        correct = sum(1 for _, (value, _) in values.items() if value)
        rows.append({
            "step": index,
            "sequences": len(values),
            "correct": correct,
            "accuracy": correct / len(values),
            "passed_problems_cumulative": len(discovered),
            "pass_rate_cumulative": len(discovered) / len(uids),
        })
    groups = []
    for start in range(1, len(rows) + 1, GROUP_WIDTH):
        selected = rows[start - 1 : start - 1 + GROUP_WIDTH]
        if len(selected) != GROUP_WIDTH:
            break
        sequences = sum(row["sequences"] for row in selected)
        groups.append({
            "start_step": start,
            "end_step": start + GROUP_WIDTH - 1,
            "accuracy": sum(row["correct"] for row in selected) / sequences,
            "pass_rate_cumulative_at_end": selected[-1]["pass_rate_cumulative"],
        })
    total = sum(row["sequences"] for row in rows)
    return {
        "problems": len(uids),
        "candidates": total,
        "accuracy": sum(row["correct"] for row in rows) / total if total else 0.0,
        "final_pass_rate": rows[-1]["pass_rate_cumulative"] if rows else 0.0,
        "steps": rows,
        "groups": groups,
    }


def build_summary(
    *, condition: str, model: str, source: str,
    steps: list[dict[str, tuple[bool, str]]], uids: set[str],
) -> dict:
    overall = summarize_steps(steps, uids)
    failures = collections.Counter(
        failure for values in steps for _, failure in values.values()
    )
    # A baseline run is one benchmark, so by_source holds a single entry; the
    # key stays present because the report and plot code index it.
    by_source = {source: overall}
    return {
        "condition": condition,
        "model": model,
        "candidates": overall["candidates"],
        "problems": overall["problems"],
        "accuracy": overall["accuracy"],
        "final_pass_rate": overall["final_pass_rate"],
        "failure_types": dict(failures),
        "failure_types_by_source": {source: dict(failures)},
        "steps": overall["steps"],
        "groups_of_four": overall["groups"],
        "by_source": by_source,
    }


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #


def run_model_key(output: Path, fallback: str = "") -> str:
    path = output / "launch_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text())
        return str(manifest.get("model") or fallback)
    return fallback


def grade_random_sampling(
    output: Path, config: dict, workers: int
) -> tuple[list[dict[str, tuple[bool, str]]], set[str], str]:
    from leaf.verifiers.answer_grader import grade_records

    records = read_jsonl(output / "generations.jsonl")
    if not records:
        raise RuntimeError(f"{output}/generations.jsonl is empty")
    keys = [row["key"] for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Random Sampling generation keys are not unique")
    grades = grade_records(records, config, workers=workers)
    atomic_jsonl(grades, output / "grades_public.jsonl")
    uids = {str(row["problem_uid"]) for row in records}
    source = str(records[0]["source"])
    return random_sampling_steps(grades, records, uids), uids, source


def verl_steps(
    output: Path, config: dict, condition: str, benchmark: str, mode: str, seed: int,
    model: str | None = None,
) -> tuple[list[dict[str, tuple[bool, str]]], set[str]]:
    # The mapping is keyed by prompt text, so it has to come from the same
    # parquet tree the run rolled out against -- per-model for the cells whose
    # prompt differs from the shared one.
    data = benchmark_rl_data_dir(config, benchmark, model)
    mapping = prompt_uid_map(data / MAPPING_PARQUET[mode])
    ground_truths = ground_truth_uid_map(data / MAPPING_PARQUET[mode])
    # Either map can drop a problem it cannot tell apart (MBPP 76 / 347 share
    # an OLMES prompt); the benchmark's problems are those of both.
    uids = set(mapping.values()) | set(ground_truths.values())
    resolver = PromptResolver(mapping, ground_truths)

    schedule = parquet_schedule(
        data / SCHEDULE_PARQUET[condition][mode], int(config["question_batch_size"])
    )
    rollout_dir = output / "rollouts"
    updates = len(numbered_jsonl(rollout_dir))
    batches = len({row["update_index"] for row in schedule})
    if updates > batches:
        if mode == "formal":
            raise RuntimeError(
                f"{condition}/{benchmark}: {updates} rollout files but the frozen "
                f"schedule only has {batches} updates"
            )
        schedule = repeat_schedule(schedule, -(-updates // batches))

    occurrences = rollout_occurrences(
        rollout_dir=rollout_dir, schedule=schedule, resolver=resolver,
        condition=condition, seed=seed,
    )
    if condition == "seed_rl":
        collection = collection_occurrences(
            output.parent.parent / "seed_collection" / output.name
        )
        # The RL stage is authoritative wherever the two overlap, which they
        # only do in the degenerate smoke schedule.
        occurrences = {**collection, **occurrences}
        schedule = merge_seed_schedule(config, benchmark, mode, schedule, model)
    print(json.dumps({
        "event": "prompt_mapping", "condition": condition,
        "records_needing_a_scan": resolver.scans,
        "resolved_by_verifier_payload": resolver.by_ground_truth,
        "records_matched_without_whitespace": resolver.squeezed_scans,
        "learned_chat_affixes": len(resolver.affixes),
        "occurrences": len(occurrences),
    }), flush=True)
    return steps_from_occurrences(
        occurrences=occurrences, schedule=schedule, uids=uids,
        condition=condition, seed=seed,
    ), uids


def merge_seed_schedule(
    config: dict, benchmark: str, mode: str, rl_schedule: list[dict],
    model: str | None = None,
) -> list[dict]:
    """The whole question schedule SEED spent, collection stage included."""
    data = benchmark_rl_data_dir(config, benchmark, model)
    path = data / f"seed_collection_{parquet_mode(mode)}.parquet"
    if not path.is_file():
        return rl_schedule
    merged = {
        int(row["question_position"]): row
        for row in parquet_schedule(path, int(config["question_batch_size"]))
    }
    merged.update({int(row["question_position"]): row for row in rl_schedule})
    return [merged[position] for position in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--allocation-seed", type=int, default=0,
                        help="freezes the candidate-to-occurrence assignment")
    parser.add_argument("--allow-partial", action="store_true",
                        help="formal run stopped by its time limit (exploration): summarise the "
                             "complete aligned steps it reached instead of requiring 24")
    args = parser.parse_args()
    config = load_config()
    output = args.output

    model = run_model_key(output)
    if args.condition == "random_sampling":
        steps, uids, source = grade_random_sampling(output, config, args.workers)
    else:
        steps, uids = verl_steps(
            output, config, args.condition, args.benchmark, args.mode,
            args.allocation_seed, model,
        )
        source = args.benchmark

    found = len(steps)
    if args.mode == "formal":
        expected_problems = int(config["source_sizes"][args.benchmark])
        if len(uids) != expected_problems:
            raise RuntimeError(
                f"{args.benchmark}: {len(uids)} problems, expected {expected_problems}"
            )
        if found < ALIGNED_STEPS_TARGET and not args.allow_partial:
            raise RuntimeError(
                f"{args.condition}/{args.benchmark}: only {found} complete aligned "
                f"steps, need {ALIGNED_STEPS_TARGET}"
            )
    steps = steps[:ALIGNED_STEPS_TARGET]

    summary = build_summary(
        condition=args.condition,
        model=model,
        source=source, steps=steps, uids=uids,
    )
    summary["aligned_steps_found"] = found
    atomic_json(summary, output / "rollout_summary.json")
    # MBPP+: the offline private test, written beside the public
    # summary as rollout_summary_mbppplus.json (leaf.verifiers.mbppplus).
    if mbppplus.enabled(config) and args.benchmark == "mbpp":
        random_sampling = args.condition == "random_sampling"
        mbppplus.grade_cell(
            output, kind="random_sampling" if random_sampling else "verl", config=config,
            workers=args.workers,
            records=read_jsonl(output / "generations.jsonl") if random_sampling else None,
            grades=read_jsonl(output / "grades_public.jsonl") if random_sampling else None,
            condition=args.condition, mode=args.mode, seed=args.allocation_seed,
            model=model,
        )
    print(json.dumps({
        "condition": summary["condition"],
        "benchmark": args.benchmark,
        "aligned_steps_found": found,
        "aligned_steps_reported": len(summary["steps"]),
        "accuracy": summary["accuracy"],
        "final_pass_rate": summary["final_pass_rate"],
    }), flush=True)


if __name__ == "__main__":
    main()
