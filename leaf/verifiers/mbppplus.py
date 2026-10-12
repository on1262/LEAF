"""MBPP+: problem set, public tests, and offline private grading.

The MBPP benchmark is all 378 tasks of EvalPlus
MbppPlus v0.2.0 (``data/benchmarks/MBPPPlus.jsonl``).

* **uid / entry point.** ``Mbpp/<n>`` becomes ``mbpp/<n>``; the entry point is
  MBPP+'s ``entry_point``; source ``mbpp``, task_type ``code_mbpp``.
* **Public tests** (:func:`public_tests`): the first 3 assert lines of MBPP+'s
  ``assertion`` field -- MBPP+'s own version of the original MBPP asserts. The
  non-assert lines of that field (``import math`` for the 10 ``math.isclose``
  tasks) become ``test_setup_code``, and so does ``import math`` for Mbpp/120,
  whose asserts call ``math.isclose`` without importing it. They are shown in
  the prompt and are all the rollout verifier (``leaf.verifiers.answer_grader``, route
  ``mbpp``, unchanged) ever runs or reports.
* **Private tests** (:func:`build_ground_truth`): MBPP+'s ``plus_input``
  (``base_input`` for Mbpp/793, which has none), graded the EvalPlus way --
  ``evalplus.eval.untrusted_check`` with the task's ``atol``, EvalPlus's
  set-equality / not-None / custom oracles, per-input time limit max(4 s, 4x the
  canonical solution's time), expected outputs and times computed from the
  canonical solution on the grading node. Every candidate runs inside the warm
  **spawn** pool of :class:`leaf.verifiers.answer_grader.Grader` (EvalPlus forks its
  sandbox child from that CUDA-free worker), never inside ``evalplus.evaluate``'s
  own pool and never from a vLLM process. Private grading is offline only.

:func:`grade_cell` is the post-step ``leaf.grading.grade_leaf_rollout`` (LEAF rollouts)
and ``leaf.grading.grade_baseline`` (Random Sampling, verl) run on an MBPP cell when
the config sets ``mbpp_private_tests = "mbppplus"``. It rebuilds the cell's
aligned 24-step grid (checked against ``rollout_summary.json``), grades every
extracted program on the private set, and writes

* ``rollout_summary_mbppplus.json`` -- the rollout_summary schema whose main
  numbers (``accuracy``, ``final_pass_rate``, ``steps``, ``by_source.mbpp``)
  are public AND private; ``mbppplus.private_only`` / ``mbppplus.public_only``
  hold the same numbers for the private ("IID") set alone and the 3 public
  asserts alone;
* ``grades_mbppplus.jsonl`` -- one line per grid entry.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

from leaf import paths

PROJECT = Path(__file__).resolve().parents[2]
# MbppPlus v0.2.0 (the EvalPlus release asset), under $DATA_ROOT/raw unless MBPPPLUS_PATH is set.
MBPP_PLUS_PATH = Path(os.environ.get("MBPPPLUS_PATH") or paths.MBPPPLUS_RAW)
EVALPLUS_ROOT = PROJECT / "third_party" / "EvalPlus"
CONFIG_FLAG = "mbpp_private_tests"
CONFIG_VALUE = "mbppplus"
PUBLIC_TESTS = 3
ALIGNED_STEPS = 24
GROUP_WIDTH = 4
CANDIDATE_MEMORY_BYTES = 4 * 2**30  # EvalPlus default EVALPLUS_MAX_MEMORY_BYTES
MIN_TIME_LIMIT = 4.0  # evalplus.config.DEFAULT_MIN_TIME_LIMIT
GT_TIME_LIMIT_FACTOR = 4.0  # evalplus.config.DEFAULT_GT_TIME_LIMIT_FACTOR
# A task whose private inputs + expected outputs pickle to more than this is
# graded as one batch, so no two workers hold its ground truth at once.
BIG_TASK_BYTES = 64 * 2**20
BATCH_PROGRAMS = 16


def enabled(config: dict) -> bool:
    return config.get(CONFIG_FLAG) == CONFIG_VALUE


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# EvalPlus import
# --------------------------------------------------------------------------- #


def evalplus_path() -> None:
    """Make the vendored EvalPlus importable, in this process or a pool worker.

    ``evalplus.data`` imports ``tempdir`` / ``wget`` / ``appdirs`` at module
    scope, only for downloading a release; the reference environment lacks
    them. The dataset is read from ``MBPP_OVERRIDE_PATH``, so those are
    stubbed when missing and never called.
    """
    if str(EVALPLUS_ROOT) not in sys.path:
        sys.path.insert(0, str(EVALPLUS_ROOT))
    os.environ.setdefault("MBPP_OVERRIDE_PATH", str(MBPP_PLUS_PATH))
    for name in ("tempdir", "wget"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            sys.modules[name] = types.ModuleType(name)
    if "appdirs" not in sys.modules and importlib.util.find_spec("appdirs") is None:
        stub = types.ModuleType("appdirs")
        stub.user_cache_dir = lambda name: str(Path.home() / ".cache" / name)
        sys.modules["appdirs"] = stub


# --------------------------------------------------------------------------- #
# problem set and public tests
# --------------------------------------------------------------------------- #


def check_dataset(path: Path | None = None) -> Path:
    """The MbppPlus v0.2.0 file."""
    path = Path(path or MBPP_PLUS_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (run run/data_fetch.sh)")
    return path


def read_plus_rows(path: Path | None = None) -> list[dict]:
    """Raw MBPP+ rows (inputs still serialized), in task-number order."""
    with check_dataset(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return sorted(rows, key=lambda row: int(row["task_id"].split("/")[1]))


def plus_uid(task_id: str) -> str:
    return "mbpp/" + task_id.split("/", 1)[1]


def plus_task_id(uid: str) -> str:
    return "Mbpp/" + uid.split("/", 1)[1]


def description(row: dict) -> str:
    """The EvalPlus ``prompt`` docstring without its trailing example assert.

    Every MBPP+ prompt is ``\"\"\"\\n<description>\\nassert ...\\n\"\"\"``. The
    result equals the HF ``evalplus/mbppplus`` ``prompt`` field on 360 of the 378
    tasks; on the other 18 the HF field also carries the source URL (and, for
    Mbpp/752 / 753, a note after it), which EvalPlus dropped.
    """
    text = row["prompt"].strip()
    if not (text.startswith('"""') and text.endswith('"""')):
        raise ValueError(f"{row['task_id']}: prompt is not a docstring")
    lines = text[3:-3].strip("\n").split("\n")
    asserts = [index for index, line in enumerate(lines) if line.startswith("assert")]
    if not asserts or asserts[-1] != len(lines) - 1:
        raise ValueError(f"{row['task_id']}: prompt does not end on an assert")
    return "\n".join(lines[: asserts[-1]]).strip()


def public_tests(row: dict) -> tuple[list[str], str]:
    """(the 3 public asserts, their setup code) from MBPP+'s ``assertion``."""
    lines = [line for line in row["assertion"].split("\n") if line.strip()]
    asserts = [line for line in lines if line.startswith("assert")]
    setup = [line for line in lines if not line.startswith("assert")]
    if len(asserts) < PUBLIC_TESTS:
        raise ValueError(f"{row['task_id']}: only {len(asserts)} asserts")
    tests = asserts[:PUBLIC_TESTS]
    for test in tests:
        compile(test, "<test>", "exec")  # one assert per line
    # Mbpp/120's asserts use math.isclose with no import line.
    if any("math." in test for test in tests) and "import math" not in setup:
        setup.append("import math")
    return tests, "\n".join(setup)


def olmes_header(code: str, entry_point: str) -> str:
    """mbppplus::olmo3:adapt's function header from MBPP+'s canonical solution.

    OLMES takes ``code.split(":")[0] + ":"`` (HF ``code`` opens on one newline,
    which becomes the newline after the description). Two refinements, both
    keeping the entry point's full ``def`` line in view:

    * when a helper comes first (8 tasks, e.g. Mbpp/6, 123, 301), the line that
      defines the entry point is used instead (the OLMES-prompt fallback);
    * when the first colon is a type annotation inside the signature (Mbpp/4
      ``def heap_queue_largest(nums: list,n: int) -> list:``), the def line is
      completed up to its own closing colon.
    """
    code = code.replace("\r\n", "\n").lstrip("\n")
    match = re.search(rf"^def {re.escape(entry_point)}\s*\(.*$", code, re.MULTILINE)
    if not match:
        raise ValueError(f"No def {entry_point} line in the canonical solution")
    def_line = match.group(0).rstrip()
    if not def_line.endswith(":"):
        raise ValueError(f"def {entry_point} does not fit on one line")
    header = code.split(":")[0] + ":"
    last = header.rsplit("\n", 1)[-1]
    if not re.match(rf"def {re.escape(entry_point)}\b", last):
        return def_line
    if last.count("(") != last.count(")"):
        return header[: len(header) - len(last)] + def_line
    return header


def tests_block(tests: list[str], setup: str) -> str:
    """The default prompt's wording of the tests (``leaf.data.prepare_test_sets.mbpp_problem_text``)."""
    block = "Your code should pass these tests:\n\n" + "\n".join(tests)
    if setup.strip():
        block += f"\n\nThe tests run after this setup code:\n\n{setup.strip()}"
    return block


def build_rows(prompt_style: str, path: Path | None = None) -> list[dict]:
    """One frozen dataset row per MBPP+ task (unshuffled, no ``index`` yet).

    ``prompt_style`` "chat": the row's ``problem`` is description + tests, to
    which ``leaf.prompts.task_prompts.problem_prompt`` prepends ``MBPP_INSTRUCTION``
    (Qwen3-8B). "olmes": the row also carries ``prompt_user``, the
    ``mbppplus::olmo3:adapt`` template on description + function header + tests
    (``leaf.prompts.olmes_prompts.MBPP_TEMPLATE``, Olmo-3-7B), and its verifier
    payload ``answer_format = "olmes"``.
    """
    from leaf.prompts.olmes_prompts import ANSWER_FORMAT, MBPP_TEMPLATE

    if prompt_style not in ("chat", "olmes"):
        raise ValueError(f"Unknown MBPP+ prompt style {prompt_style}")
    rows = []
    for raw in read_plus_rows(path):
        tests, setup = public_tests(raw)
        text = description(raw)
        entry_point = raw["entry_point"]
        row = {
            "uid": plus_uid(raw["task_id"]),
            "source": "mbpp",
            "task_type": "code_mbpp",
            "problem": f"{text}\n\n{tests_block(tests, setup)}",
            "verifier": {
                "kind": "mbpp",
                "test_list": tests,
                "test_setup_code": setup,
                "entry_point": entry_point,
            },
            "meta": {
                "task_id": int(raw["task_id"].split("/")[1]),
                "mbppplus_task_id": raw["task_id"],
                "mbppplus_version": "v0.2.0",
                "private_split": "plus_input" if raw["plus_input"] else "base_input",
                "private_inputs": len(raw["plus_input"] or raw["base_input"]),
                "atol": raw["atol"],
            },
        }
        if prompt_style == "olmes":
            code_prompt = (f"{text}\n{olmes_header(raw['canonical_solution'], entry_point)}"
                           f"\n\n{tests_block(tests, setup)}")
            row["prompt_system"] = None
            row["prompt_user"] = MBPP_TEMPLATE.format(code_prompt=code_prompt)
            row["verifier"]["answer_format"] = ANSWER_FORMAT
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# private ground truth
# --------------------------------------------------------------------------- #


def _ground_truth_job(directory: str, task_ids: list[str] | None) -> dict:
    """Canonical outputs + times on the private inputs, one pickle per task.

    Runs in a throwaway spawn process (see :func:`build_ground_truth`), serially,
    as ``evalplus.evaluate.get_groundtruth`` does: the times set the limits.
    """
    evalplus_path()
    check_dataset(Path(os.environ["MBPP_OVERRIDE_PATH"]))
    from evalplus.data import get_mbpp_plus
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    problems = get_mbpp_plus()
    index = {}
    started = time.monotonic()
    for task_id in task_ids or sorted(problems):
        problem = problems[task_id]
        split = "plus_input" if problem["plus_input"] else "base_input"
        inputs = problem[split]
        expected, ref_time = trusted_exec(
            problem["prompt"] + problem["canonical_solution"], inputs,
            problem["entry_point"], record_time=True,
            output_not_none=problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS)
        path = Path(directory) / f"{task_id.split('/')[1]}.pkl"
        with path.open("wb") as handle:
            pickle.dump({"task_id": task_id, "entry_point": problem["entry_point"],
                         "atol": problem["atol"], "split": split, "inputs": inputs,
                         "expected": expected, "ref_time": ref_time}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        index[task_id] = {"path": str(path), "bytes": path.stat().st_size,
                          "split": split, "inputs": len(inputs),
                          "max_limit": max(MIN_TIME_LIMIT, GT_TIME_LIMIT_FACTOR
                                           * max(ref_time or [0.0]))}
    return {"tasks": index, "seconds": time.monotonic() - started}


def build_ground_truth(directory: Path, task_ids: list[str] | None = None) -> dict:
    """Ground truth under ``directory``, computed in a fresh spawn process.

    The caller never executes canonical code or holds the ~3 GB of MBPP+
    outputs itself, and the pool workers only ever load one task's pickle.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(_ground_truth_job, str(directory), task_ids).result()


# --------------------------------------------------------------------------- #
# private check (pool-worker side)
# --------------------------------------------------------------------------- #

_TASK_CACHE: dict = {}


def _vm_bytes() -> int:
    with open("/proc/self/status") as handle:
        return next(int(line.split()[1]) for line in handle if line.startswith("VmSize")) * 1024


def _private_batch(path: str, programs: list[str]) -> list[dict]:
    """EvalPlus-check each program against one task's private inputs.

    Runs in a :class:`leaf.verifiers.answer_grader.Grader` spawn worker; EvalPlus's
    ``untrusted_check`` forks its sandbox from here. ``RLIMIT_AS`` counts the
    address space that child inherits from this worker (its interpreter plus the
    task's inputs and expected outputs), so each candidate gets EvalPlus's 4 GiB
    on top of the worker's current size.
    """
    evalplus_path()
    from evalplus.eval import untrusted_check

    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)
    if _TASK_CACHE.get("path") != path:
        _TASK_CACHE.clear()
        with open(path, "rb") as handle:
            _TASK_CACHE.update(path=path, data=pickle.load(handle))
    data = _TASK_CACHE["data"]
    results = []
    for code in programs:
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(CANDIDATE_MEMORY_BYTES + _vm_bytes())
        started = time.monotonic()
        status, details = untrusted_check(
            "mbpp", code, data["inputs"], data["entry_point"], expected=data["expected"],
            atol=data["atol"], ref_time=data["ref_time"], fast_check=True,
            min_time_limit=MIN_TIME_LIMIT, gt_time_limit_factor=GT_TIME_LIMIT_FACTOR)
        details = [bool(value) for value in details]
        results.append({
            "status": status,
            "passed": status == "pass",
            "inputs": len(data["inputs"]),
            "inputs_run": len(details),
            "first_failed_input": next((i for i, ok in enumerate(details) if not ok), None),
            "seconds": time.monotonic() - started,
        })
    return results


def run_private(jobs: dict[tuple[str, str], tuple[str, str]], ground_truth: dict,
                grader, retry: set[tuple[str, str]] | None = None) -> dict:
    """``{(uid, code_sha): outcome}`` for ``jobs = {(uid, sha): (uid, code)}``.

    One pool task holds up to ``BATCH_PROGRAMS`` programs of one task, so a
    worker loads that task's ground truth once per batch; a big task goes as
    one batch. EvalPlus limits are per input, and under a full pool a slow but
    correct program can miss one, so every failing program in ``retry`` (keys
    whose public test passed) or that timed out is run once more, a quarter of
    the pool at a time, and the second verdict is kept.
    """
    tasks = ground_truth["tasks"]
    by_task: dict[str, list[tuple[tuple[str, str], str]]] = collections.defaultdict(list)
    for key, (uid, code) in jobs.items():
        by_task[plus_task_id(uid)].append((key, code))
    batches = []
    for task_id, items in by_task.items():
        info = tasks[task_id]
        size = len(items) if info["bytes"] > BIG_TASK_BYTES else BATCH_PROGRAMS
        for start in range(0, len(items), size):
            batches.append((info["bytes"], task_id, items[start:start + size]))
    batches.sort(key=lambda item: -item[0])  # big tasks first, shorter tail
    started = time.monotonic()
    outcomes = grader._map([(_private_batch, (tasks[task_id]["path"], [c for _, c in items]))
                            for _, task_id, items in batches])
    results = {}
    for (_, _, items), batch in zip(batches, outcomes, strict=True):
        for (key, _), outcome in zip(items, batch, strict=True):
            results[key] = outcome
    first_pass = time.monotonic() - started

    retry = retry or set()
    again = [key for key, outcome in results.items()
             if not outcome["passed"] and (key in retry or outcome["status"] == "timeout")]
    flipped = 0
    width = max(1, grader.workers // 4)
    for start in range(0, len(again), width):
        wave = again[start:start + width]
        batch = grader._map([(_private_batch, (tasks[plus_task_id(key[0])]["path"],
                                               [jobs[key][1]])) for key in wave])
        for key, (outcome,) in zip(wave, batch, strict=True):
            outcome["retried"] = True
            outcome["first_status"] = results[key]["status"]
            flipped += outcome["passed"]
            results[key] = outcome
    print(json.dumps({"event": "mbppplus_private", "unique_programs": len(jobs),
                      "batches": len(batches), "first_pass_seconds": round(first_pass, 1),
                      "retried": len(again), "retry_flipped_to_pass": flipped,
                      "total_seconds": round(time.monotonic() - started, 1)}), flush=True)
    return results


# --------------------------------------------------------------------------- #
# aligned grids: one entry per (step, problem)
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def leaf_grid(records: list[dict], grades: list[dict]) -> list[dict]:
    """The LEAF rollout driver (and the server-rollout runner): one record per cell."""
    by_key = {grade["key"]: grade for grade in grades}
    return [dict(step=int(r["step"]), problem_uid=r["problem_uid"],
                 public=bool(by_key[r["key"]]["correct"]),
                 public_failure=by_key[r["key"]].get("failure_type"),
                 code=by_key[r["key"]].get("extracted_answer") or "", ref=r["key"])
            for r in records if r["source"] == "mbpp"]


def random_sampling_grid(records: list[dict], grades: list[dict], steps: int) -> list[dict]:
    by_key = {grade["key"]: grade for grade in grades}
    uids = {r["problem_uid"] for r in records}
    by_step: dict[int, dict] = collections.defaultdict(dict)
    for r in records:
        grade = by_key[r["key"]]
        step = int(r["aligned_step"]) + 1
        by_step[step][r["problem_uid"]] = dict(
            step=step, problem_uid=r["problem_uid"], public=bool(grade["correct"]),
            public_failure=grade.get("failure_type"),
            code=grade.get("extracted_answer") or "", ref=r["key"])
    grid = []
    for step in range(1, steps + 1):
        if set(by_step[step]) != uids:
            raise RuntimeError(f"best-of-k step {step} incomplete")
        grid.extend(by_step[step].values())
    return grid


def split_score_output(output: str, correction: str) -> tuple[str, str]:
    """The two turns of one SCoRe record (decoded with special tokens skipped)."""
    at = output.find(correction)
    if at < 0:
        raise RuntimeError("SCoRe output holds no correction turn")
    first = re.sub(r"\n(?:system\n[^\n]*\n)?user\n$", "", output[:at])
    second = re.sub(r"^\nassistant\n", "", output[at + len(correction):])
    return first, second


@contextlib.contextmanager
def _carrying_code(gb, condition: str):
    """Let ``leaf.grading.grade_baseline`` carry each candidate's code.

    Its aligned-step expansion shuffles opaque candidate tuples, so extending
    the tuples with the code (and restoring the originals afterwards) makes the
    same frozen shuffles hand back the program behind every (step, problem).
    """
    from leaf.prompts import task_prompts
    from leaf.verifiers.answer_grader import extract_mbpp_code, strip_think

    def code_of(row: dict, turn: int | None) -> str:
        if turn is None and "extracted_answer" in row:
            return str(row["extracted_answer"] or "")
        text = row["output"]
        if turn is not None:
            correction = task_prompts.correction_instruction_for_task(
                row.get("task_type") or "code_mbpp")
            text = split_score_output(text, correction)[turn]
        payload = json.loads(row["gts"])
        text, _ = strip_think(text, "mbpp", payload)
        return extract_mbpp_code(text, payload)

    def outcomes(row: dict, cond: str) -> list[tuple]:
        if cond == "score":
            return [(float(row[k]) > 0.5, "pass" if float(row[k]) > 0.5 else "incorrect",
                     {"code": code_of(row, turn), "ref": f"turn{turn + 1}"})
                    for turn, k in enumerate(("score_r1", "score_r2"))]
        correct = float(row["acc"]) > 0.5
        failure = str(row.get("failure_type") or ("pass" if correct else "incorrect"))
        return [(correct, failure, {"code": code_of(row, None), "ref": ""})]

    def collection(directory: Path) -> dict:
        path = directory / "grades_public.json"
        if not path.is_file():
            return {}
        result = collections.defaultdict(list)
        for row in sorted(json.loads(path.read_text()), key=lambda v: int(v["trial_index"])):
            correct = bool(row["correct"])
            result[int(row["trial_index"]) // gb.CANDIDATES_PER_OCCURRENCE].append(
                (correct, str(row.get("failure_type") or ("pass" if correct else "incorrect")),
                 {"code": str(row.get("extracted_answer") or ""), "ref": row["key"]}))
        return dict(result)

    saved = gb.candidate_outcomes, gb.collection_occurrences
    gb.candidate_outcomes, gb.collection_occurrences = outcomes, collection
    try:
        yield
    finally:
        gb.candidate_outcomes, gb.collection_occurrences = saved


def verl_grid(cell: Path, config: dict, condition: str, mode: str, seed: int,
              model: str | None, steps: int) -> list[dict]:
    import leaf.grading.grade_baseline as gb

    with _carrying_code(gb, condition):
        aligned, _ = gb.verl_steps(cell, config, condition, "mbpp", mode, seed, model)
    grid = []
    for index, values in enumerate(aligned[:steps], 1):
        for uid, (correct, failure, extra) in values.items():
            grid.append(dict(step=index, problem_uid=uid, public=bool(correct),
                             public_failure=failure, code=extra["code"], ref=extra["ref"]))
    return grid


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #


def summarize(grid: list[dict], uids: set[str], key: str) -> dict:
    by_step = collections.defaultdict(list)
    for entry in grid:
        by_step[entry["step"]].append(entry)
    discovered: set[str] = set()
    rows = []
    for step in sorted(by_step):
        current = by_step[step]
        correct = [e for e in current if e[key]]
        discovered.update(e["problem_uid"] for e in correct)
        rows.append({"step": step, "sequences": len(current), "correct": len(correct),
                     "accuracy": len(correct) / len(current),
                     "passed_problems_cumulative": len(discovered),
                     "pass_rate_cumulative": len(discovered) / len(uids)})
    groups = []
    for start in range(1, len(rows) + 1, GROUP_WIDTH):
        selected = rows[start - 1: start - 1 + GROUP_WIDTH]
        if len(selected) != GROUP_WIDTH:
            break
        groups.append({"start_step": start, "end_step": start + GROUP_WIDTH - 1,
                       "accuracy": sum(r["correct"] for r in selected)
                       / sum(r["sequences"] for r in selected),
                       "pass_rate_cumulative_at_end": selected[-1]["pass_rate_cumulative"]})
    total = sum(r["sequences"] for r in rows)
    return {"problems": len(uids), "candidates": total,
            "accuracy": sum(r["correct"] for r in rows) / total if total else 0.0,
            "final_pass_rate": rows[-1]["pass_rate_cumulative"] if rows else 0.0,
            "steps": rows, "groups": groups}


def _check_public(grid: list[dict], summary: dict) -> None:
    """The rebuilt grid's public verdicts must be the cell's rollout_summary."""
    source = summary["by_source"].get("mbpp") or summary
    mine = summarize(grid, {e["problem_uid"] for e in grid}, "public")
    same = len(mine["steps"]) == len(source["steps"]) and all(
        a["correct"] == b["correct"]
        and a["passed_problems_cumulative"] == b["passed_problems_cumulative"]
        for a, b in zip(mine["steps"], source["steps"]))
    if not same:
        raise RuntimeError("MBPP+ grid does not reproduce rollout_summary.json "
                           f"({mine['accuracy']} vs {source['accuracy']})")


def _recheck_score_turns(grid: list[dict], dataset: dict[str, dict], grader, config) -> dict:
    """SCoRe turns are re-extracted from the joined output: re-run their public asserts."""
    records = [{"key": str(i), "problem_uid": e["problem_uid"], "source": "mbpp",
                "task_type": "code_mbpp",
                "verifier": {k: v for k, v in dataset[e["problem_uid"]]["verifier"].items()
                             if k != "answer_format"},
                "response": f"```python\n{e['code']}\n```", "finish_reason": "stop"}
               for i, e in enumerate(grid)]
    grades = grader.grade(records, config)
    disagree = sum(bool(g["correct"]) != e["public"] for g, e in zip(grades, grid))
    if disagree > 0.01 * len(grid):
        raise RuntimeError(f"SCoRe turn split disagrees with the inline reward on {disagree}")
    return {"entries": len(grid), "disagree": disagree}


def grade_cell(cell: Path, *, kind: str, config: dict, workers: int,
               records: list[dict] | None = None, grades: list[dict] | None = None,
               condition: str | None = None, mode: str = "formal", seed: int = 0,
               model: str | None = None) -> dict:
    """Private-grade one MBPP cell and write its MBPP+ summary next to it.

    ``kind``: "leaf" (``records`` / ``grades`` = generations + public grades of
    the LEAF rollout driver or the server-rollout runner), "random_sampling" (same, with
    ``aligned_step``), or "verl" (``condition`` in sdpo / grpo / score /
    seed_rl, read back from the cell's rollouts like
    ``leaf.grading.grade_baseline``).
    """
    from leaf.core.project_config import atomic_json, atomic_jsonl, root
    from leaf.verifiers.answer_grader import default_grader

    started = time.monotonic()
    cell = Path(cell)
    summary = json.loads((cell / "rollout_summary.json").read_text())
    steps = min(ALIGNED_STEPS, len(summary["steps"]))
    grader = default_grader(config, workers)
    dataset = {r["uid"]: r for r in read_jsonl(root(config) / "dataset_mbpp.jsonl")}
    if kind == "leaf":
        grid = leaf_grid(records, grades)
    elif kind == "random_sampling":
        grid = random_sampling_grid(records, grades, steps)
    elif kind == "verl":
        grid = verl_grid(cell, config, condition, mode, seed, model, steps)
    else:
        raise ValueError(f"Unknown cell kind {kind}")
    uids = {e["problem_uid"] for e in grid}
    if uids - set(dataset):
        raise RuntimeError(f"{cell}: problems outside dataset_mbpp.jsonl")
    _check_public(grid, summary)
    score_check = (_recheck_score_turns(grid, dataset, grader, config)
                   if condition == "score" else None)

    scratch = Path(tempfile.mkdtemp(prefix="mbppplus_gt_", dir=os.environ.get("TMPDIR")))
    try:
        ground_truth = build_ground_truth(scratch, sorted(plus_task_id(u) for u in uids))
        print(json.dumps({"event": "mbppplus_ground_truth", "tasks": len(ground_truth["tasks"]),
                          "seconds": round(ground_truth["seconds"], 1),
                          "bytes": sum(t["bytes"] for t in ground_truth["tasks"].values())}),
              flush=True)
        jobs, retry = {}, set()
        for e in grid:
            e["sha"] = sha(e["code"])
            if e["code"].strip():
                key = (e["problem_uid"], e["sha"])
                jobs.setdefault(key, (e["problem_uid"], e["code"]))
                if e["public"]:
                    retry.add(key)
        results = run_private(jobs, ground_truth, grader, retry)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    lines = []
    for e in grid:
        outcome = results.get((e["problem_uid"], e["sha"]))
        e["private"] = bool(outcome and outcome["passed"])
        e["correct"] = e["public"] and e["private"]
        if not e["public"]:
            failure = e["public_failure"] or "incorrect"
        elif e["private"]:
            failure = "pass"
        else:
            failure = "private_timeout" if outcome and outcome["status"] == "timeout" \
                else "private_fail"
        e["failure_type"] = failure
        lines.append({"step": e["step"], "problem_uid": e["problem_uid"], "ref": e["ref"],
                      "public_correct": e["public"], "private_correct": e["private"],
                      "private_status": outcome["status"] if outcome else "no_code",
                      "private_inputs_run": outcome["inputs_run"] if outcome else 0,
                      "private_retried": bool(outcome and outcome.get("retried")),
                      "correct": e["correct"], "failure_type": failure,
                      "code_hash": e["sha"]})
    both = summarize(grid, uids, "correct")
    private = summarize(grid, uids, "private")
    public = summarize(grid, uids, "public")
    failures = dict(collections.Counter(e["failure_type"] for e in grid))
    splits = collections.Counter(t["split"] for t in ground_truth["tasks"].values())
    result = {
        "condition": summary["condition"],
        "model": summary.get("model"),
        "candidates": both["candidates"],
        "problems": both["problems"],
        "accuracy": both["accuracy"],
        "final_pass_rate": both["final_pass_rate"],
        "failure_types": failures,
        "failure_types_by_source": {"mbpp": failures},
        "steps": both["steps"],
        "groups_of_four": both["groups"],
        "by_source": {"mbpp": both},
        "mbppplus": {
            "definition": "MbppPlus v0.2.0, 378 tasks. public = first 3 asserts of MBPP+ "
                          "'assertion' (shown in the prompt, the only rollout feedback); "
                          "private = EvalPlus check on plus_input (base_input when a task "
                          "has none), offline. Main numbers: public AND private.",
            "private_only": {k: private[k] for k in ("accuracy", "final_pass_rate")},
            "private_only_steps": private["steps"],
            "private_only_groups": private["groups"],
            "public_only": {k: public[k] for k in ("accuracy", "final_pass_rate")},
            "public_only_steps": public["steps"],
            "public_only_groups": public["groups"],
            "private_split_tasks": dict(splits),
            "private_inputs": sum(t["inputs"] for t in ground_truth["tasks"].values()),
            "unique_programs": len(jobs),
            "public_pass_records": sum(e["public"] for e in grid),
            "private_pass_records": sum(e["private"] for e in grid),
            "public_pass_private_fail_records": sum(e["public"] and not e["private"]
                                                    for e in grid),
            "public_fail_private_pass_records": sum(e["private"] and not e["public"]
                                                    for e in grid),
            "retried_programs": sum(bool(o.get("retried")) for o in results.values()),
            "score_public_recheck": score_check,
            "ground_truth_seconds": ground_truth["seconds"],
            "seconds": time.monotonic() - started,
        },
    }
    atomic_jsonl(lines, cell / "grades_mbppplus.jsonl")
    atomic_json(result, cell / "rollout_summary_mbppplus.json")
    print(json.dumps({"event": "mbppplus_cell", "cell": str(cell),
                      "public_private": {k: both[k] for k in ("accuracy", "final_pass_rate")},
                      "private_only": result["mbppplus"]["private_only"],
                      "public_only": result["mbppplus"]["public_only"],
                      "seconds": round(result["mbppplus"]["seconds"], 1)}), flush=True)
    return result
