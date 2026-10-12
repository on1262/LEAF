"""LiveCodeBench private tests: offline regrade of existing LCB cells.

The frozen 131-problem LCB subset splits LiveCodeBench's
own ``private_test_cases`` of every problem by index:

* **public** = ``verifier.public_test_indices`` -- what the rollout verifier
  (``leaf.verifiers.answer_grader`` -> ``leaf.verifiers.lcb_public`` public route, SDPO
  executor) runs and reports; its ground truth
  (``data/lcb/train.parquet``) is exactly those test cases;
* **private** = ``verifier.heldout_test_indices`` -- the rest of
  ``private_test_cases``, never run during rollout.

So public and private are disjoint by index and together are all of LCB's
``private_test_cases``; "public AND private" means the program passes every one
of them. The tests are read from the LiveCodeBench ``code_generation_lite``
release_v6 HF snapshot (config ``lcb_hf_snapshot``), matched by ``question_id``
and checked against the frozen row (test count, public tests = parquet ground
truth). LCB's ``public_test_cases`` (the statement examples) are not part of
either set.

Private grading follows the official ``lcb_runner`` grader
(``third_party/LiveCodeBench``): ``compute_code_generation_metrics.check_correctness``
(one forked child per program with the global timeout
``(timeout + 1) * n_tests + 5``) around ``testing_util.run_test`` (stdin programs
wrapped by ``make_function`` with line-wise / Decimal output comparison;
call-based programs through ``Solution()`` or the top-level function, one JSON
argument per line, ``==`` after tuple->list; fail-fast), per-test timeout
``verifier_timeout`` (6 s, the official ``--timeout`` default). The program is the
one the public verifier executed (the grade's ``extracted_answer``: the longest
fenced block of the text after ``</think>``, else of the whole response). Every
program runs inside the warm **spawn** pool of
:class:`leaf.verifiers.answer_grader.Grader`; ``reliability_guard`` gets an address-space
limit of 4 GiB on top of the pool worker's current size (the child inherits the
worker's interpreter and the problem's tests), as in the MBPP+ memory fix.

:func:`grade_cell` rebuilds a cell's aligned 24-step grid (checked against its
``rollout_summary.json``), private-grades every extracted program, and writes

* ``rollout_summary_lcbprivate.json`` -- the rollout_summary schema whose main
  numbers are public AND private, with ``lcbprivate.public_only`` /
  ``lcbprivate.private_only`` (and per-step / per-group versions) and counts;
* ``grades_lcbprivate.jsonl`` -- one line per grid entry.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from leaf.verifiers import mbppplus
from leaf import paths

ALIGNED_STEPS = 24
CANDIDATE_MEMORY_BYTES = 4 * 2**30
BATCH_PROGRAMS = 8
PUBLIC_GROUND_TRUTH = paths.LCB_PUBLIC_GROUND_TRUTH


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #


def _snapshot_row(snapshot: Path, ref: dict, question_id: str, index_cache: dict) -> tuple[dict, int]:
    """The snapshot row of ``question_id``: at the frozen offset if it is there, else scanned."""
    path = snapshot / Path(ref["path"]).name
    with path.open("rb") as handle:
        handle.seek(int(ref["offset"]))
        line = handle.readline()
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        row = {}
    if row.get("question_id") == question_id:
        return row, int(ref["offset"])
    if path not in index_cache:
        offsets, offset = {}, 0
        with path.open("rb") as handle:
            for raw in handle:
                offsets[json.loads(raw)["question_id"]] = offset
                offset += len(raw)
        index_cache[path] = offsets
    offset = index_cache[path][question_id]
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline()), offset


def build_ground_truth(dataset: list[dict], snapshot: Path, directory: Path | None,
                       public_ground_truth: Path = PUBLIC_GROUND_TRUTH) -> dict:
    """One pickle per problem holding the public / private / full official samples.

    Refuses a problem whose snapshot row does not match the frozen row: same
    ``question_id``, ``full_test_count`` test cases, index split covering every
    test exactly once, and public indices equal to the public verifier's
    ground truth.
    """
    import pyarrow.parquet as pq

    from leaf.core.io_utils import decode_private_tests, function_name, load_raw

    table = pq.read_table(public_ground_truth, columns=["reward_model", "extra_info"]).to_pylist()
    public_gt = {row["extra_info"]["problem_uid"]: json.loads(row["reward_model"]["ground_truth"])
                 for row in table}
    if directory is not None:  # None: counts only, no samples written
        directory.mkdir(parents=True, exist_ok=True)
    tasks, index_cache = {}, {}
    counts = collections.Counter()
    started = time.monotonic()
    for row in dataset:
        verifier, uid = row["verifier"], row["uid"]
        question_id = str(row["meta"]["question_id"])
        raw, offset = _snapshot_row(snapshot, verifier["evaluation_ref"], question_id, index_cache)
        tests = decode_private_tests(raw["private_test_cases"])
        frozen = load_raw(verifier["evaluation_ref"])
        if frozen["private_test_cases"] != raw["private_test_cases"]:
            raise RuntimeError(f"{uid}: snapshot private_test_cases differ from the frozen export")
        public_idx = list(verifier["public_test_indices"])
        private_idx = list(verifier["heldout_test_indices"])
        if len(tests) != int(verifier["full_test_count"]):
            raise RuntimeError(f"{uid}: {len(tests)} private_test_cases, frozen row says "
                               f"{verifier['full_test_count']}")
        if sorted(public_idx + private_idx) != list(range(len(tests))):
            raise RuntimeError(f"{uid}: public + heldout indices do not partition the tests")
        gt = public_gt[verifier["problem_uid"]]
        if (gt["inputs"] != [tests[i]["input"] for i in public_idx]
                or gt["outputs"] != [tests[i]["output"] for i in public_idx]):
            raise RuntimeError(f"{uid}: public verifier tests are not private_test_cases[public]")
        fn_name = function_name(raw)
        if (gt.get("fn_name") or None) != fn_name:
            raise RuntimeError(f"{uid}: fn_name {fn_name!r} vs public {gt.get('fn_name')!r}")

        def sample(indices: list[int]) -> dict:
            return {"input_output": json.dumps({
                "inputs": [tests[i]["input"] for i in indices],
                "outputs": [tests[i]["output"] for i in indices],
                "fn_name": fn_name}, ensure_ascii=False)}

        pair = lambda t: (t["input"], t["output"])  # noqa: E731
        public_pairs = {pair(tests[i]) for i in public_idx}
        statement = json.loads(raw["public_test_cases"]) if raw.get("public_test_cases") else []
        all_pairs = {pair(t) for t in tests}
        info = {
            "uid": uid, "question_id": question_id, "snapshot_offset": offset,
            "testtype": tests[0]["testtype"], "fn_name": fn_name,
            "public": len(public_idx), "private": len(private_idx), "full": len(tests),
            "private_duplicating_public": sum(pair(tests[i]) in public_pairs for i in private_idx),
            "statement_examples": len(statement),
            "statement_examples_in_private_test_cases": sum(pair(t) in all_pairs for t in statement),
        }
        if directory is not None:
            path = directory / f"{sha(uid)[:16]}.pkl"
            with path.open("wb") as handle:
                pickle.dump({"uid": uid, "samples": {"public": sample(public_idx),
                                                     "private": sample(private_idx),
                                                     "full": sample(list(range(len(tests))))}},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
            info["path"], info["bytes"] = str(path), path.stat().st_size
        tasks[uid] = info
        for key in ("public", "private", "full", "private_duplicating_public",
                    "statement_examples", "statement_examples_in_private_test_cases"):
            counts[key] += info[key]
        counts["testtype_" + info["testtype"]] += 1
    counts["problems"] = len(tasks)
    counts["problems_without_private_tests"] = sum(t["private"] == 0 for t in tasks.values())
    return {"tasks": tasks, "counts": dict(counts), "seconds": time.monotonic() - started}


# --------------------------------------------------------------------------- #
# official check (pool-worker side)
# --------------------------------------------------------------------------- #

_TASK_CACHE: dict = {}
_WORKER: dict = {}

STATUS = {-1: "timeout", -2: "wrong_answer", -3: "timeout", -4: "runtime_error",
          -5: "test_runner_error"}


def _worker_setup() -> None:
    """Once per pool worker: fork children, keep the original guard, mute candidate output."""
    if _WORKER:
        return
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)
    from lcb_runner.evaluation import testing_util

    _WORKER["guard"] = testing_util.reliability_guard
    # Call-based candidates print straight to the worker's stdout, and run_test's
    # alarm handler prints on every timeout; keep them out of the job log.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    sys.stderr = sys.stdout


def _check_batch(path: str, suite: str, programs: list[str], timeout: int) -> list[dict]:
    """Official ``check_correctness`` of each program on one problem's ``suite`` tests."""
    import functools

    import numpy as np
    from lcb_runner.evaluation import testing_util
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    from leaf.verifiers.lcb_memory import candidate_memory_limit

    _worker_setup()
    if _TASK_CACHE.get("path") != path:
        _TASK_CACHE.clear()
        with open(path, "rb") as handle:
            _TASK_CACHE.update(path=path, data=pickle.load(handle))
    sample = _TASK_CACHE["data"]["samples"][suite]
    tests = len(json.loads(sample["input_output"])["inputs"])
    results = []
    for code in programs:
        # RLIMIT_AS counts the address space the forked child inherits from this
        # worker (interpreter + this problem's tests): 4 GiB on top of it.
        testing_util.reliability_guard = functools.partial(
            _WORKER["guard"], maximum_memory_bytes=candidate_memory_limit(CANDIDATE_MEMORY_BYTES))
        started = time.monotonic()
        try:
            res, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
        except Exception as error:  # the official evaluate_generations_by_problem rule
            res, metadata = [-2], {"error": repr(error), "error_code": -5,
                                   "error_message": "TestRunnerError"}
        fixed = []
        for value in res:
            if isinstance(value, np.ndarray):
                value = value.item(0)
            if isinstance(value, np.bool_):
                value = bool(value)
            fixed.append(value)
        passed = len(fixed) == tests and all(value is True or (
            not isinstance(value, bool) and value > 0) for value in fixed)
        code_value = next((v for v in fixed if not isinstance(v, bool) and v < 0), None)
        if passed:
            status = "pass"
        elif code_value is not None:
            status = STATUS.get(int(code_value), "fail")
        else:
            status = STATUS.get(int(metadata.get("error_code", -2)), "wrong_answer")
        results.append({
            "status": status, "passed": passed, "tests": tests, "tests_run": len(fixed),
            "error_message": str(metadata.get("error_message", ""))[:200],
            "seconds": time.monotonic() - started,
        })
    return results


def run_checks(jobs: dict, ground_truth: dict, grader, suite: str, timeout: int,
               retry: set | None = None, label: str = "lcb_private") -> dict:
    """``{key: outcome}`` for ``jobs = {key: (uid, code)}`` on one suite.

    Up to ``BATCH_PROGRAMS`` programs of one problem per pool task, so a worker
    loads that problem's tests once per batch. Failing programs whose key is in
    ``retry`` (public-passing entries) are run once more at a quarter of the pool
    and the second verdict is kept: a per-test limit can be missed under a
    full pool.
    """
    tasks = ground_truth["tasks"]
    by_uid: dict[str, list] = collections.defaultdict(list)
    for key, (uid, code) in jobs.items():
        by_uid[uid].append((key, code))
    batches = []
    for uid, items in by_uid.items():
        for start in range(0, len(items), BATCH_PROGRAMS):
            batches.append((tasks[uid]["bytes"], uid, items[start:start + BATCH_PROGRAMS]))
    batches.sort(key=lambda item: -item[0])
    started = time.monotonic()
    outcomes = grader._map([(_check_batch, (tasks[uid]["path"], suite, [c for _, c in items],
                                            timeout)) for _, uid, items in batches])
    results = {}
    for (_, _, items), batch in zip(batches, outcomes, strict=True):
        for (key, _), outcome in zip(items, batch, strict=True):
            results[key] = outcome
    first_pass = time.monotonic() - started
    again = [key for key, outcome in results.items()
             if not outcome["passed"] and key in (retry or set())]
    flipped = 0
    width = max(1, grader.workers // 4)
    for start in range(0, len(again), width):
        wave = again[start:start + width]
        batch = grader._map([(_check_batch, (tasks[jobs[key][0]]["path"], suite, [jobs[key][1]],
                                             timeout)) for key in wave])
        for key, (outcome,) in zip(wave, batch, strict=True):
            outcome["retried"] = True
            outcome["first_status"] = results[key]["status"]
            flipped += outcome["passed"]
            results[key] = outcome
    print(json.dumps({"event": label, "suite": suite, "unique_programs": len(jobs),
                      "batches": len(batches), "first_pass_seconds": round(first_pass, 1),
                      "retried": len(again), "retry_flipped_to_pass": flipped,
                      "statuses": dict(collections.Counter(o["status"] for o in results.values())),
                      "total_seconds": round(time.monotonic() - started, 1)}), flush=True)
    return results


# --------------------------------------------------------------------------- #
# aligned grids
# --------------------------------------------------------------------------- #


def leaf_grid(records: list[dict], grades: list[dict]) -> list[dict]:
    """The LEAF rollout driver and the server-rollout runner: one record per (step, problem)."""
    by_key = {grade["key"]: grade for grade in grades}
    return [dict(step=int(r["step"]), problem_uid=r["problem_uid"],
                 public=bool(by_key[r["key"]]["correct"]),
                 public_failure=by_key[r["key"]].get("failure_type"),
                 code=by_key[r["key"]].get("extracted_answer") or "", ref=r["key"])
            for r in records if r["source"] == "lcb"]


def lcb_code(text: str, payload: dict) -> str:
    """What the public verifier executes for one response: SDPO's longest fenced block
    of the text after ``</think>`` (whole response if nothing gradable follows it)."""
    from leaf.verifiers.answer_grader import strip_think

    text, _ = strip_think(text, "lcb", payload)
    blocks = re.findall(r"```(\w*)\n(.*?)```", text, re.DOTALL)
    return max((body for _, body in blocks), key=len) if blocks else ""


@contextlib.contextmanager
def _carrying_code(gb):
    """``leaf.grading.grade_baseline`` with each candidate carrying its program.

    Same trick as the MBPP+ regrade: the aligned-step expansion shuffles opaque
    candidate tuples, so extending them with the code makes the same frozen
    shuffles return the program behind every (step, problem).
    """
    from leaf.prompts import task_prompts

    def code_of(row: dict, turn: int | None) -> str:
        if turn is None:
            return str(row.get("extracted_answer") or "")
        correction = task_prompts.correction_instruction_for_task(
            row.get("task_type") or "code_lcb")
        text = mbppplus.split_score_output(row["output"], correction)[turn]
        return lcb_code(text, json.loads(row["gts"]))

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

    with _carrying_code(gb):
        aligned, _ = gb.verl_steps(cell, config, condition, "lcb", mode, seed, model)
    grid = []
    for index, values in enumerate(aligned[:steps], 1):
        for uid, (correct, failure, extra) in values.items():
            grid.append(dict(step=index, problem_uid=uid, public=bool(correct),
                             public_failure=failure, code=extra["code"], ref=extra["ref"]))
    return grid


def check_public(grid: list[dict], summary: dict) -> None:
    """The rebuilt grid's public verdicts must reproduce the cell's rollout_summary."""
    source = summary["by_source"].get("lcb") or summary
    mine = mbppplus.summarize(grid, {e["problem_uid"] for e in grid}, "public")
    same = len(mine["steps"]) == len(source["steps"]) and all(
        a["sequences"] == b["sequences"] and a["correct"] == b["correct"]
        and a["passed_problems_cumulative"] == b["passed_problems_cumulative"]
        for a, b in zip(mine["steps"], source["steps"]))
    if not same:
        raise RuntimeError("LCB grid does not reproduce rollout_summary.json "
                           f"({mine['accuracy']} vs {source['accuracy']})")


def recheck_score_turns(grid: list[dict], dataset: dict[str, dict], grader, config) -> dict:
    """SCoRe turns are re-extracted from the joined output: re-run their public tests."""
    records = [{"key": str(i), "problem_uid": e["problem_uid"], "source": "lcb",
                "task_type": "code_lcb", "verifier": dataset[e["problem_uid"]]["verifier"],
                "response": f"```python\n{e['code']}\n```" if e["code"] else "",
                "finish_reason": "stop"}
               for i, e in enumerate(grid)]
    grades = grader.grade(records, config)
    disagree = sum(bool(g["correct"]) != e["public"] for g, e in zip(grades, grid))
    print(json.dumps({"event": "score_public_recheck", "entries": len(grid),
                      "disagree": disagree}), flush=True)
    if disagree > 0.01 * len(grid):
        raise RuntimeError(f"SCoRe turn split disagrees with the inline reward on {disagree}")
    return {"entries": len(grid), "disagree": disagree}


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #


def build_grid(cell: Path, *, kind: str, config: dict, condition: str | None = None,
               mode: str = "formal", seed: int = 0, model: str | None = None) -> list[dict]:
    summary = json.loads((cell / "rollout_summary.json").read_text())
    steps = min(ALIGNED_STEPS, len(summary["steps"]))
    if kind == "leaf":
        grid = leaf_grid(read_jsonl(cell / "generations.jsonl"),
                         read_jsonl(cell / "grades_public.jsonl"))
    elif kind == "random_sampling":
        grid = mbppplus.random_sampling_grid(read_jsonl(cell / "generations.jsonl"),
                                     read_jsonl(cell / "grades_public.jsonl"), steps)
    elif kind == "verl":
        grid = verl_grid(cell, config, condition, mode, seed, model, steps)
    else:
        raise ValueError(f"Unknown cell kind {kind}")
    check_public(grid, summary)
    return grid


def grade_cell(cell: Path, grid: list[dict], ground_truth: dict, grader, config: dict,
               dataset: dict[str, dict], condition: str | None = None) -> dict:
    """Private-grade one rebuilt grid and write the cell's LCB-private summary."""
    started = time.monotonic()
    uids = {e["problem_uid"] for e in grid}
    if uids - set(ground_truth["tasks"]):
        raise RuntimeError(f"{cell}: problems outside the 131-problem ground truth")
    score_check = (recheck_score_turns(grid, dataset, grader, config)
                   if condition == "score" else None)
    timeout = int(config["verifier_timeout"])
    jobs, retry = {}, set()
    for e in grid:
        e["sha"] = sha(e["code"])
        e["no_private"] = ground_truth["tasks"][e["problem_uid"]]["private"] == 0
        if e["code"].strip() and not e["no_private"]:
            key = (e["problem_uid"], e["sha"])
            jobs.setdefault(key, (e["problem_uid"], e["code"]))
            if e["public"]:
                retry.add(key)
    results = run_checks(jobs, ground_truth, grader, "private", timeout, retry)
    return write_cell(cell, grid, results, ground_truth, timeout, score_check, started)


def write_cell(cell: Path, grid: list[dict], results: dict, ground_truth: dict, timeout: int,
               score_check: dict | None, started: float) -> dict:
    """Combine each grid entry's public verdict with its private outcome and write both files."""
    from leaf.core.project_config import atomic_json, atomic_jsonl

    summary = json.loads((cell / "rollout_summary.json").read_text())
    uids = {e["problem_uid"] for e in grid}
    for e in grid:
        e["sha"] = sha(e["code"])
        e["no_private"] = ground_truth["tasks"][e["problem_uid"]]["private"] == 0
    lines = []
    for e in grid:
        outcome = results.get((e["problem_uid"], e["sha"]))
        if e["no_private"]:
            # Problems whose single private_test_case is the public test have an
            # empty private set; their private verdict is the public one.
            outcome = {"status": "no_private_tests", "passed": e["public"], "tests_run": 0,
                       "error_message": ""}
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
                      "private_tests_run": outcome["tests_run"] if outcome else 0,
                      "private_tests": ground_truth["tasks"][e["problem_uid"]]["private"],
                      "private_error": outcome["error_message"] if outcome else "",
                      "private_retried": bool(outcome and outcome.get("retried")),
                      "correct": e["correct"], "failure_type": failure,
                      "code_hash": e["sha"]})
    both = mbppplus.summarize(grid, uids, "correct")
    private = mbppplus.summarize(grid, uids, "private")
    public = mbppplus.summarize(grid, uids, "public")
    failures = dict(collections.Counter(e["failure_type"] for e in grid))
    counts = ground_truth["counts"]
    result = {
        "condition": summary["condition"],
        "model": summary.get("model"),
        "candidates": both["candidates"],
        "problems": both["problems"],
        "accuracy": both["accuracy"],
        "final_pass_rate": both["final_pass_rate"],
        "failure_types": failures,
        "failure_types_by_source": {"lcb": failures},
        "steps": both["steps"],
        "groups_of_four": both["groups"],
        "by_source": {"lcb": both},
        "lcbprivate": {
            "definition": "LiveCodeBench code_generation_lite release_v6 private_test_cases of "
                          "the 131 frozen problems. public = public_test_indices (the rollout "
                          "verifier's tests); private = heldout_test_indices (the rest), graded "
                          "offline with the official lcb_runner check_correctness/run_test, "
                          f"per-test timeout {timeout} s, 4 GiB candidate memory. Problems whose "
                          "only private_test_case is the public test have no private tests; "
                          "their private verdict equals the public one. Main numbers: "
                          "public AND private.",
            "private_only": {k: private[k] for k in ("accuracy", "final_pass_rate")},
            "private_only_steps": private["steps"],
            "private_only_groups": private["groups"],
            "public_only": {k: public[k] for k in ("accuracy", "final_pass_rate")},
            "public_only_steps": public["steps"],
            "public_only_groups": public["groups"],
            "test_counts": counts,
            "timeout_seconds": timeout,
            "unique_programs": len({(e["problem_uid"], e["sha"]) for e in grid
                                    if e["code"].strip() and not e["no_private"]}),
            "public_pass_records": sum(e["public"] for e in grid),
            "private_pass_records": sum(e["private"] for e in grid),
            "public_pass_private_fail_records": sum(e["public"] and not e["private"] for e in grid),
            "public_fail_private_pass_records": sum(e["private"] and not e["public"] for e in grid),
            "private_status_public_pass": dict(collections.Counter(
                line["private_status"] for line in lines if line["public_correct"])),
            "problems_without_private_tests": sorted(
                uid for uid, t in ground_truth["tasks"].items() if t["private"] == 0),
            "retried_programs": len({(line["problem_uid"], line["code_hash"])
                                     for line in lines if line["private_retried"]}),
            "retry_flipped_to_pass": len({(line["problem_uid"], line["code_hash"])
                                          for line in lines if line["private_retried"]
                                          and line["private_correct"]
                                          and line["private_status"] == "pass"}),
            "score_public_recheck": score_check,
            "seconds": time.monotonic() - started,
        },
    }
    atomic_jsonl(lines, cell / "grades_lcbprivate.jsonl")
    atomic_json(result, cell / "rollout_summary_lcbprivate.json")
    print(json.dumps({"event": "lcbprivate_cell", "cell": str(cell),
                      "public_private": {k: both[k] for k in ("accuracy", "final_pass_rate")},
                      "private_only": result["lcbprivate"]["private_only"],
                      "public_only": result["lcbprivate"]["public_only"],
                      "public_pass_private_fail_records":
                          result["lcbprivate"]["public_pass_private_fail_records"],
                      "seconds": round(result["lcbprivate"]["seconds"], 1)}), flush=True)
    return result


def ground_truth_scratch() -> Path:
    return Path(tempfile.mkdtemp(prefix="lcbprivate_gt_", dir=os.environ.get("TMPDIR")))


def remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def resummarize_cell(cell: Path, grid: list[dict], ground_truth: dict, timeout: int) -> dict:
    """Rewrite a graded cell's files from its own ``grades_lcbprivate.jsonl`` outcomes.

    No program is executed: the private outcome of every (problem, program) is
    read back from the previous grading of the same grid (checked entry by
    entry), and only the combination rules are applied again.
    """
    old_summary = json.loads((cell / "rollout_summary_lcbprivate.json").read_text())
    old_lines = read_jsonl(cell / "grades_lcbprivate.jsonl")
    if len(old_lines) != len(grid):
        raise RuntimeError(f"{cell}: {len(old_lines)} graded lines for {len(grid)} entries")
    results = {}
    for e, line in zip(grid, old_lines, strict=True):
        if (line["step"], line["problem_uid"], line["code_hash"]) != (
                e["step"], e["problem_uid"], sha(e["code"])):
            raise RuntimeError(f"{cell}: graded line does not match grid entry {e['step']}")
        if line["private_status"] in ("no_code", "no_private_tests"):
            continue
        results[(line["problem_uid"], line["code_hash"])] = {
            "status": line["private_status"], "passed": bool(line["private_correct"]),
            "tests_run": line["private_tests_run"], "error_message": line["private_error"],
            "retried": bool(line["private_retried"])}
    started = time.monotonic() - float(old_summary["lcbprivate"].get("seconds", 0.0))
    return write_cell(cell, grid, results, ground_truth, timeout,
                      old_summary["lcbprivate"].get("score_public_recheck"), started)
