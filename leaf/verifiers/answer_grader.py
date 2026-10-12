"""Public verifier for the four benchmarks.

It dispatches on the frozen ``verifier["kind"]`` payload carried by every
dataset row, so a caller never has to know which benchmark a record came from:

* ``lcb``   — the LiveCodeBench public grader (``leaf.verifiers.lcb_public``) verbatim,
              including the SDPO ``native_feedback`` string that goes back into the prompt;
* ``mbpp``  — executed here; bare MBPP has no in-repo grader.  Each of the three
              asserts runs separately so the feedback turn can name the failing one;
* ``math``  — the vendored ``math-verify`` route (:mod:`leaf.verifiers.math_verify_official`);
* ``sciqa`` — boxed-answer extraction plus per-``answer_type`` normalisation;
* ``sciqa_judge`` — open-answer SciKnowEval: reads the inline LLM-judge
              verdict stored on the record (:mod:`leaf.verifiers.llm_judge`).

Only the two code tasks produce ``native_feedback``; MATH500 and SciKnowEval have
no execution environment, and :func:`leaf.prompts.task_prompts.feedback_section` builds
their feedback turn from ``correct`` + ``extracted_answer`` instead.

Shape of the implementation.  A batch verifier between generation rounds leaves the
GPU idle, and the rollout driver grades every sample the instant its
rollout finishes — thousands of one-record calls from many rollout threads.  So
everything expensive is a warm singleton:

* one module-level :class:`Grader` owning a **spawn** ``ProcessPoolExecutor``
  (spawn, so no CUDA state is inherited from the vLLM process it runs inside);
* MBPP and MATH candidates are fanned out over that pool, MBPP forking one short
  lived child per candidate inside the worker so a wedged program can be killed
  without losing the warm interpreter;
* the LiveCodeBench ground-truth parquet and the SDPO functional patch are loaded
  once per process instead of once per call.

MATH also has to run in a pool worker rather than a caller thread: ``math-verify``
arms its parsing timeout with ``SIGALRM``, which is only valid on a main thread.

The module imports no torch/vllm/sympy at module scope.  Callers that use it must
keep their ``__main__`` import-safe (``if __name__ == "__main__":``), as any spawn
pool requires.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import re
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from leaf.verifiers.extraction import extract_last_boxed, unwrap

# A candidate may print or return an integer with more than 4,300 digits;
# formatting it for feedback must not raise (Python 3.11 int->str limit).
import sys as _sys
if hasattr(_sys, "set_int_max_str_digits"):
    _sys.set_int_max_str_digits(0)

# Fields the caller is contracted to supply; everything else a record carries
# (step, trial_index, condition, ...) is copied into the grade untouched.
_INPUT_FIELDS = frozenset(
    {"key", "problem_uid", "source", "task_type", "verifier", "response", "finish_reason",
     "judge"}
)

MBPP_FEEDBACK_MAX_CHARS = 2000
MBPP_EXCEPTION_MAX_CHARS = 300
FORK_TIMEOUT = 30  # only covers fork + interpreter bookkeeping, not the candidate

_FENCE = re.compile(r"```([^\n`]*)\r?\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = {"", "python", "py", "python3"}
_OPTION = re.compile(
    r"(?i)^(?:the\s+)?(?:option|answer|choice)?\s*[:\-]?\s*\(?\s*([A-Za-z])\s*\)?\s*[.):]?$"
)


def _passthrough(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in _INPUT_FIELDS}


def _base_grade(record: dict) -> dict:
    return {
        **_passthrough(record),
        "key": record["key"],
        "problem_uid": record["problem_uid"],
        "source": record["source"],
        "task_type": record["task_type"],
    }


# --------------------------------------------------------------------------- #
# LiveCodeBench
# --------------------------------------------------------------------------- #

_LCB_LOCK = threading.Lock()
_LCB_BACKEND: tuple = ()


def _lcb_backend(config: dict):
    """Load the SDPO grader and the 34 MB public ground truth once per process.

    ``leaf.verifiers.lcb_public.grade_records`` re-reads that parquet on every call,
    which is invisible in a single large batch and ruinous when the
    driver grades one sample at a time.  Its per-candidate scoring function is
    reused verbatim so the reward and the feedback string stay identical.
    """
    global _LCB_BACKEND
    with _LCB_LOCK:
        if _LCB_BACKEND:
            return _LCB_BACKEND
        # Both are read when the SDPO module is imported / called, so set first.
        os.environ["LEAF_CANDIDATE_MEMORY_GIB"] = str(
            config["verifier_candidate_memory_gib"])
        os.environ.setdefault(
            "LEAF_VERIFIER_WORKERS",
            str(config.get("rollout_verifier_test_workers", 1)),
        )
        from pathlib import Path
        import pyarrow.parquet as pq

        from leaf.verifiers.lcb_functional_patch import apply as apply_functional_patch
        from leaf.verifiers.lcb_functional_patch import set_stop_at_first_failure
        from leaf.verifiers.lcb_public import PUBLIC_GROUND_TRUTH

        code = apply_functional_patch()
        # Training on the LCB training problems grades every test but stops at the
        # first failure (config lcb_stop_at_first_failure).
        set_stop_at_first_failure(code, bool(config.get("lcb_stop_at_first_failure")))
        # The generalization runs train on the LCB training problems with every test
        # as feedback, so their config names that export; the env override still wins.
        path = Path(os.environ.get("LEAF_LCB_PUBLIC_GROUND_TRUTH")
                    or config.get("lcb_ground_truth") or PUBLIC_GROUND_TRUTH)
        if config.get("lcb_ground_truth"):
            # 6 GB of test JSON: read per problem on demand instead of per process.
            ground_truth = LazyGroundTruth(path)
        else:
            table = pq.read_table(path, columns=["reward_model", "extra_info"]).to_pylist()
            ground_truth = {
                row["extra_info"]["problem_uid"]: row["reward_model"]["ground_truth"]
                for row in table
            }
        _LCB_BACKEND = (code, ground_truth)
        return _LCB_BACKEND


class LazyGroundTruth:
    """``problem_uid -> SDPO ground-truth JSON``, read from its row group on first use.

    The all-tests export of the training problems holds 6 GB of JSON (77 problems above 20 MB),
    so every grading process loading all of it would not fit beside the
    trainer. Only the uid index is read up front; a problem's string is read
    when first graded and kept in an LRU bounded by ``budget_bytes``, and the
    same string object is returned while it stays cached, so the parse cache of
    :mod:`leaf.verifiers.lcb_functional_patch` hits on it.
    """

    def __init__(self, path, budget_bytes: int = 3 * 2**30):
        import collections

        import pyarrow.parquet as pq

        self._file = pq.ParquetFile(path)
        self._index = {}
        for group in range(self._file.num_row_groups):
            infos = self._file.read_row_group(group, columns=["extra_info"]).column(
                "extra_info").to_pylist()
            for position, info in enumerate(infos):
                self._index[info["problem_uid"]] = (group, position)
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._bytes = 0
        self._budget = int(budget_bytes)
        self._lock = threading.Lock()

    def __iter__(self):
        return iter(self._index)

    def __contains__(self, uid) -> bool:
        return uid in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, uid: str) -> str:
        with self._lock:
            if uid in self._cache:
                self._cache.move_to_end(uid)
                return self._cache[uid]
            group, position = self._index[uid]
            value = self._file.read_row_group(group, columns=["reward_model"]).column(
                "reward_model")[position].as_py()["ground_truth"]
            self._cache[uid] = value
            self._bytes += len(value)
            while self._bytes > self._budget and len(self._cache) > 1:
                _, dropped = self._cache.popitem(last=False)
                self._bytes -= len(dropped)
            return value


def lcb_valid_solution(response: str) -> str:
    """The code the LCB verifier grades from ``response`` ("" when it finds none).

    Same view (:func:`strip_think`) and extractor (SDPO ``extract_code`` behind
    the functional patch) as :func:`_grade_lcb`, so what a later turn is shown is
    exactly what was scored.
    """
    from leaf.verifiers.lcb_functional_patch import apply as apply_functional_patch

    text, _ = strip_think(response, "lcb")
    return apply_functional_patch().extract_code(text) or ""


_LCB_SHARDS: list = []
_LCB_SHARDS_LOCK = threading.Lock()
_SHARD_KEYS = ("verifier_candidate_memory_gib", "rollout_verifier_test_workers",
               "lcb_ground_truth", "lcb_stop_at_first_failure")


def _shard_config(config: dict) -> dict:
    return {key: config[key] for key in _SHARD_KEYS if key in config}


def _lcb_shard_pools(count: int) -> list:
    """``count`` single-process spawn pools; a problem always goes to the same one,
    so each worker's ground-truth and parse caches hold only its own problems."""
    with _LCB_SHARDS_LOCK:
        if not _LCB_SHARDS:
            _LCB_SHARDS.extend(
                ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                    initializer=_pool_init)
                for _ in range(count))
        return list(_LCB_SHARDS)


def _lcb_shard_score(config: dict, record: dict) -> dict:
    """One candidate in a shard worker: the same scorer and ground truth as in-process."""
    from leaf.verifiers.lcb_public import _public_feedback_grade

    # A spawn worker inherits the spawn start method, which would make every
    # per-test sandbox re-import this module; fork them, as in-process grading does.
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)
    code, ground_truth = _lcb_backend(config)
    return _public_feedback_grade(record, ground_truth[record["problem_uid"]], code)


def lcb_workers(config: dict, workers: int) -> int:
    """LiveCodeBench gets its own, larger pool.

    An LCB candidate that times out holds its slot for the full per-test
    timeout, and 11-13% of them do, so the shared 12-way pool spent most of its
    slots waiting. The other routes finish in milliseconds and do not need it.
    """
    return int(config.get("lcb_verifier_workers", workers))


def _grade_lcb(records: list[dict], config: dict, workers: int) -> list[dict]:
    workers = lcb_workers(config, workers)
    """Score LCB candidates with the LiveCodeBench public grader, then rename fields."""
    from leaf.verifiers.lcb_public import _public_feedback_grade

    code, ground_truth = _lcb_backend(config)
    missing = {record["verifier"]["problem_uid"] for record in records} - set(ground_truth)
    if missing:
        raise ValueError(f"Public verifier has unknown problem UIDs: {sorted(missing)[:3]}")

    def score(record: dict) -> dict:
        return _public_feedback_grade(
            {
                "key": record["key"],
                "problem_uid": record["verifier"]["problem_uid"],
                "response": record["response"],
                "finish_reason": record.get("finish_reason"),
            },
            ground_truth[record["verifier"]["problem_uid"]],
            code,
        )

    shards = int(config.get("lcb_grading_processes") or 0)
    if shards:
        # config lcb_grading_processes: grade in small spawn workers instead of this
        # (trainer-sized) process, whose every per-test fork holds the GIL.
        pools = _lcb_shard_pools(shards)
        futures = [
            pools[zlib.crc32(record["verifier"]["problem_uid"].encode()) % shards].submit(
                _lcb_shard_score, _shard_config(config), {
                    "key": record["key"],
                    "problem_uid": record["verifier"]["problem_uid"],
                    "response": record["response"],
                    "finish_reason": record.get("finish_reason"),
                })
            for record in records
        ]
        sub_grades = [future.result() for future in futures]
    # The SDPO scorer forks its own candidate processes, so threads are enough
    # here and keep the warm parquet in this process.
    elif len(records) == 1:
        sub_grades = [score(records[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(records)), thread_name_prefix="lcb-grader",
        ) as executor:
            sub_grades = list(executor.map(score, records))
    grades = []
    for record, sub in zip(records, sub_grades, strict=True):
        grades.append({
            **_base_grade(record),
            "correct": bool(sub["correct"]),
            "failure_type": sub["failure_type"],
            "extracted_answer": sub.get("extracted_code", "") or "",
            "native_feedback": sub.get("native_feedback", ""),
            "tests": sub.get("tests", []),
            "metadata": {
                "suite": "public",
                "lcb_problem_uid": record["verifier"]["problem_uid"],
                "lcb_metadata": sub.get("metadata", []),
                "verifier_candidate_seconds": sub.get("verifier_candidate_seconds"),
            },
        })
    return grades


# --------------------------------------------------------------------------- #
# MBPP
# --------------------------------------------------------------------------- #

def extract_code(response: str, entry_point: str | None = None) -> str:
    """The fenced block holding the answer, else the whole response.

    MBPP answers are short, so SDPO's "longest block wins" heuristic tends to
    pick an illustrative snippet over the final program, and plain "last block
    wins" picks up the trailing block of `assert` statements that models like to
    append after the solution -- which grades as `NameError: name 'f' is not
    defined` on about a fifth of MBPP candidates. So when the task names an
    entry point, prefer the last block that actually defines it.
    """
    blocks = _FENCE.findall(response)
    if not blocks:
        return response.strip()
    python_blocks = [body for tag, body in blocks if tag.strip().lower() in _PYTHON_TAGS]
    candidates = python_blocks or [body for _, body in blocks]
    if entry_point:
        defining = [
            body for body in candidates
            if re.search(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(entry_point)}\b",
                         body, re.MULTILINE)
        ]
        if defining:
            return defining[-1].strip()
        # No single block defines it: the answer may be split across blocks
        # (helper first, entry point later), so fall back to all of them.
        if len(candidates) > 1:
            joined = "\n\n".join(body.strip() for body in candidates)
            if re.search(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(entry_point)}\b",
                         joined, re.MULTILINE):
                return joined
        # Still nothing: the model named its function something else. Prefer the
        # last block that defines anything, so the verdict reads as the missing
        # entry point rather than a NameError inside the model's own test block.
        defines_something = [
            body for body in candidates
            if re.search(r"^\s*(?:async\s+def|def|class)\s+\w+", body, re.MULTILINE)
        ]
        if defines_something:
            return defines_something[-1].strip()
    return candidates[-1].strip()


def extract_mbpp_code(response: str, payload: dict) -> str:
    """The program to run for one MBPP answer.

    An OLMES-prompted answer was asked to close on ``Here is the completed function:``
    and a code block, so the last such block is the answer, as in OLMES; one
    that never wrote that phrase falls back to :func:`extract_code`, a superset
    of OLMES's own fallback (the last code block).
    """
    if payload.get("answer_format") == OLMES_FORMAT:
        blocks = MBPP_OLMES_FORMAT.findall(response)
        if blocks and blocks[-1].strip():
            return blocks[-1].strip()
    return extract_code(response, payload.get("entry_point"))


def _mbpp_child(connection, code: str, setup: str, tests: list[str], memory_gib: int):
    """Run one candidate under an RLIMIT cap and report per-assert outcomes."""
    import contextlib
    import io
    import resource
    import traceback

    from leaf.verifiers.lcb_memory import candidate_memory_limit

    limit = candidate_memory_limit(memory_gib * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    # Lets the parent time the candidate itself rather than process start-up.
    connection.send({"stage": "ready"})

    def short(error: BaseException) -> str:
        return traceback.format_exception_only(
            type(error), error)[-1].strip()[:MBPP_EXCEPTION_MAX_CHARS]

    namespace: dict = {"__name__": "__candidate__"}
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                compiled = compile(code, "<candidate>", "exec")
            except SyntaxError as error:
                connection.send({"stage": "syntax", "detail": short(error)})
                return
            try:
                exec(compiled, namespace)
                if setup.strip():
                    exec(compile(setup, "<setup>", "exec"), namespace)
            except BaseException as error:  # noqa: BLE001 - candidate code
                connection.send({"stage": "program", "detail": short(error)})
                return
            results = []
            for test in tests:
                try:
                    exec(compile(test, "<test>", "exec"), namespace)
                except AssertionError as error:
                    results.append({"passed": False, "assertion": True,
                                    "detail": short(error) or "AssertionError"})
                except BaseException as error:  # noqa: BLE001 - candidate code
                    results.append({"passed": False, "assertion": False,
                                    "detail": short(error)})
                else:
                    results.append({"passed": True, "assertion": False, "detail": ""})
        connection.send({"stage": "tests", "results": results})
    except BaseException as error:  # noqa: BLE001 - e.g. MemoryError from the cap
        connection.send({"stage": "program", "detail": repr(error)[:MBPP_EXCEPTION_MAX_CHARS]})
    finally:
        os._exit(0)


def _mbpp_task(code: str, setup: str, tests: list[str], timeout: float,
               memory_gib: int) -> dict:
    """Pool-worker side: fork one throwaway child so a wedged candidate is killable.

    The fork is from the (small, CUDA-free) pool worker, so it costs milliseconds
    and never touches the vLLM parent's address space.
    """
    started = time.monotonic()
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_mbpp_child, args=(child, code, setup, tests, memory_gib), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(FORK_TIMEOUT) or parent.recv()["stage"] != "ready":
            outcome = {"stage": "program", "detail": "The candidate process never started."}
        elif parent.poll(timeout):
            outcome = parent.recv()
        else:
            outcome = {"stage": "timeout", "detail": f"{timeout:g}"}
    except EOFError:
        outcome = {"stage": "program",
                   "detail": "The candidate process died without a result."}
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    outcome["seconds"] = time.monotonic() - started
    return outcome


def _render_mbpp_feedback(tests: list[str], outcome: dict) -> str:
    stage = outcome["stage"]
    if stage == "syntax":
        text = f"The submitted program does not compile.\n{outcome['detail']}"
    elif stage == "timeout":
        text = f"Execution timed out after {outcome['detail']} seconds."
    elif stage == "program":
        text = f"The submitted program raised before any test could run.\n{outcome['detail']}"
    else:
        results = outcome["results"]
        passed = sum(1 for item in results if item["passed"])
        lines = [f"Public test results: {passed}/{len(results)} asserts passed."]
        for index, (test, item) in enumerate(zip(tests, results, strict=True), start=1):
            if item["passed"]:
                continue
            lines.append(f"Failed assert {index}: {test}")
            lines.append(f"  {item['detail']}")
        text = "\n".join(lines)
    return text[:MBPP_FEEDBACK_MAX_CHARS]


def _mbpp_failure_type(outcome: dict) -> str:
    stage = outcome["stage"]
    if stage == "syntax":
        return "syntax_error"
    if stage == "timeout":
        return "timeout"
    if stage == "program":
        return "runtime_error"
    results = outcome["results"]
    if all(item["passed"] for item in results):
        return "pass"
    if any(not item["passed"] and not item["assertion"] for item in results):
        return "runtime_error"
    return "wrong_answer"


def _mbpp_grade(record: dict, code: str, outcome: dict | None, seconds: float) -> dict:
    payload = record["verifier"]
    tests = list(payload["test_list"])
    grade = {
        **_base_grade(record),
        "extracted_answer": code,
        "tests": [False] * len(tests),
        "metadata": {
            "entry_point": payload.get("entry_point", ""),
            "verifier_candidate_seconds": seconds,
        },
    }
    if outcome is None:
        grade.update(correct=False, failure_type="extraction_failure",
                     native_feedback="No Python program could be extracted from the answer.")
        return grade
    category = _mbpp_failure_type(outcome)
    results = outcome.get("results")
    grade.update(
        correct=category == "pass",
        failure_type=category,
        tests=[bool(item["passed"]) for item in results] if results else [False] * len(tests),
        native_feedback=("All public asserts passed." if category == "pass"
                         else _render_mbpp_feedback(tests, outcome)),
    )
    grade["metadata"]["stage"] = outcome["stage"]
    return grade


# --------------------------------------------------------------------------- #
# MATH500
# --------------------------------------------------------------------------- #

_MATH_ROUTE: tuple = ()


def _math_route():
    """Import (and keep) the vendored math-verify route inside whatever process runs it."""
    global _MATH_ROUTE
    if not _MATH_ROUTE:
        from leaf.verifiers.math_grader import extract_boxed_or_final_answer
        from leaf.verifiers.math_verify_official import verify_math

        _MATH_ROUTE = (verify_math, extract_boxed_or_final_answer)
    return _MATH_ROUTE


def _math_task(response: str, answer: str) -> dict:
    started = time.monotonic()
    verify_math, fallback_answer = _math_route()
    boxed = extract_last_boxed(response)
    extracted = boxed if boxed is not None else fallback_answer(response)
    return {
        "correct": bool(verify_math(response, answer)),
        "extracted": extracted,
        "boxed": boxed is not None,
        "seconds": time.monotonic() - started,
    }


def _math_grade(record: dict, outcome: dict, seconds: float) -> dict:
    correct = outcome["correct"]
    if correct:
        category = "pass"
    elif outcome["extracted"] is None:
        category = "extraction_failure"
    else:
        category = "wrong_answer"
    return {
        **_base_grade(record),
        "correct": correct,
        "failure_type": category,
        "extracted_answer": (outcome["extracted"] or "").strip(),
        "native_feedback": "",
        "tests": [correct],
        "metadata": {"boxed": outcome["boxed"], "verifier_candidate_seconds": seconds},
    }


# --------------------------------------------------------------------------- #
# SciKnowEval
# --------------------------------------------------------------------------- #

def _normalise_space(text: str) -> str:
    return " ".join(str(text).split())


def normalise_mcq(text: str, choice_labels=None) -> str:
    """Reduce an answer to a bare option letter when it unambiguously is one."""
    stripped = _normalise_space(unwrap(text)).strip("*` ")
    match = _OPTION.match(stripped)
    if match:
        return match.group(1).upper()
    # Some answers give the option text; map it back only on an exact match.
    for index, label in enumerate(choice_labels or []):
        label = str(label)
        if _normalise_space(label).casefold() == stripped.casefold():
            return (label.strip().upper() if len(label.strip()) == 1
                    else chr(ord("A") + index))
    return stripped


def normalise_yes_no(text: str) -> str:
    stripped = _normalise_space(unwrap(text)).strip("*`.  ").casefold()
    if stripped in {"yes", "true", "y"}:
        return "Yes"
    if stripped in {"no", "false", "n"}:
        return "No"
    return _normalise_space(unwrap(text))


# --------------------------------------------------------------------------- #
# OLMES answer formats (Olmo-3-7B-Instruct prompts)
# --------------------------------------------------------------------------- #

# A verifier payload with ``answer_format == "olmes"`` was prompted with an
# OLMES olmo3:adapt template (leaf.prompts.olmes_prompts) and is read the way OLMES
# reads it; every other payload keeps the default extraction.
OLMES_FORMAT = "olmes"

# oe_eval/configs/regexes.py OLMO_3_REGEXES (allenai/olmes 5a51f502), tried in
# order with $ANS$ replaced by each answer regex.
OLMO_3_REGEXES = (
    r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?($ANS$)\b",
    r"(?i)so\s+the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)the\s+correct\s+answer\s+is:?\s*($ANS$)",
    r"(?i)the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)a:\s*($ANS$)",
    r"(?i)answer:\s*($ANS$)",
    r"(?i)\b($ANS$)\)?\s+is\s+correct",
    r"(?i)\(($ANS$)\)",
    r"(?i)\b($ANS$)\b",
    r"(?i)\b($ANS$)\b",
    r"(?i).*\b($ANS$)\b",
)
# mbppplus::olmo3:adapt metric_kwargs.answer_format_regex.
MBPP_OLMES_FORMAT = re.compile(
    r"(?s)Here is the completed function:\n*```(?:python\n)?\n*(.*?)\s*```"
)


def olmes_extract(text: str, format_regex: str, answer_regexes: list[str]) -> tuple[str, str]:
    """oe_eval.tasks.utils.extract_answer followed by gpqa's ``_clean_answer``.

    Returns the answer (``""`` when nothing matched) and which rule produced it:
    ``format`` (the exact closing phrase), ``template_<i>`` (OLMO_3_REGEXES
    entry i), or ``raw`` (a bare answer anywhere). Each rule takes its last
    match, as OLMES does with ``use_last_prefix_match`` / ``use_last_raw_match``.
    """
    answer, how = "", ""
    matches = re.findall(format_regex, text)
    if matches:
        answer, how = matches[-1], "format"
    if not answer:
        for index, template in enumerate(OLMO_3_REGEXES):
            for answer_regex in answer_regexes:
                found = list(re.finditer(template.replace("$ANS$", answer_regex), text))
                if found:
                    groups = found[-1].groups()
                    answer = next((g for g in reversed(groups) if g), groups[0]) if groups \
                        else found[-1].group(0)
                    if answer:
                        how = f"template_{index}"
                        break
            if answer:
                break
    if not answer:
        for answer_regex in answer_regexes:
            found = re.findall(answer_regex, text)
            if found:
                answer, how = found[-1], "raw"
                break
    if isinstance(answer, tuple):  # a pattern with several groups
        answer = next((g for g in reversed(answer) if g), "")
    return re.sub(r"\(|\)", "", answer), how


def sciqa_olmes_regexes(answer_type: str, labels) -> tuple[str, list[str]]:
    """The format regex and answer regex for an OLMES-prompted SciKnowEval problem.

    mcq is gpqa::olmo3:adapt with the letter class narrowed to the options the
    question has; yes/no swaps the letters for Yes / No.
    """
    if answer_type == "mcq":
        options = "".join(str(label) for label in labels)
        return rf"Therefore, the answer is \(([{options}])\)", [rf"\(?([{options}])\)?"]
    if answer_type == "yes_no":
        return r"Therefore, the answer is \((Yes|No)\)", [r"\(?(Yes|No)\)?"]
    raise ValueError(f"No OLMES answer phrase for answer_type {answer_type}")


def _olmes_sciqa_answer(record: dict) -> tuple[str, str]:
    payload = record["verifier"]
    format_regex, answer_regexes = sciqa_olmes_regexes(
        payload["answer_type"], payload.get("choice_labels") or [])
    return olmes_extract(record["response"], format_regex, answer_regexes)


_LATEX_WRAPPERS = re.compile(r"\\(?:text|mathrm|textrm|rm|ce|mathbf)\s*\{([^{}]*)\}")
_LATEX_ARROWS = re.compile(
    r"\\(?:xrightarrow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}|longrightarrow|rightarrow|"
    r"rightleftharpoons|leftrightarrow|Longrightarrow|Rightarrow|to)(?![A-Za-z])"
    r"|->|<=>|→|⟶|⇌|⇒"
)


def plain_equation(text: str) -> str:
    """A SciKnowEval filling answer with its LaTeX markup removed.

    The gold is plain text (``Na2SO4 + 2HBr = H2SO4 + 2NaBr``) while boxed
    answers are usually LaTeX (``\text{Na}_2\text{SO}_4 + ... \rightarrow``);
    without it, every one of 312 filling answers was scored wrong,
    correct ones included. Only notation goes: wrappers, sub/superscript marks,
    braces, spacing commands and dollar signs, and every reaction arrow reads as
    ``=``. Coefficients, formulas and state labels still have to match.
    """
    value = str(text)
    for _ in range(3):  # nested wrappers, e.g. \text{\mathrm{Na}}
        value = _LATEX_WRAPPERS.sub(r"\1", value)
    value = _LATEX_ARROWS.sub("=", value)
    value = re.sub(r"\\[,;:! ]|~|\$", "", value)
    value = value.replace("_", "").replace("^", "").replace("{", "").replace("}", "")
    return "".join(value.split())


def _grade_sciqa_one(record: dict) -> dict:
    payload = record["verifier"]
    answer_type = payload["answer_type"]
    gold = str(payload["gold"])
    labels = payload.get("choice_labels")
    grade = {
        **_base_grade(record),
        "native_feedback": "",
        "metadata": {"answer_type": answer_type},
    }
    if payload.get("answer_format") == OLMES_FORMAT and answer_type != "filling":
        extracted, how = _olmes_sciqa_answer(record)
        grade["metadata"]["olmes_rule"] = how
        if not extracted:
            grade.update(correct=False, failure_type="extraction_failure",
                         extracted_answer="", tests=[False])
            return grade
        if answer_type == "mcq":
            # OLMES compares the cleaned string with the gold letter as is, so a
            # lower-case article caught by a fallback rule does not count as A.
            correct = extracted == normalise_mcq(gold, labels)
        else:
            extracted = normalise_yes_no(extracted)
            correct = extracted == normalise_yes_no(gold)
        grade.update(correct=correct, failure_type="pass" if correct else "wrong_answer",
                     extracted_answer=extracted, tests=[correct])
        return grade
    boxed = extract_last_boxed(record["response"])
    if boxed is None:
        grade.update(correct=False, failure_type="extraction_failure",
                     extracted_answer="", tests=[False])
        return grade
    if answer_type == "mcq":
        extracted = normalise_mcq(boxed, labels)
        correct = extracted.casefold() == normalise_mcq(gold, labels).casefold()
    elif answer_type == "yes_no":
        extracted = normalise_yes_no(boxed)
        correct = extracted == normalise_yes_no(gold)
    elif answer_type == "filling":
        extracted = _normalise_space(unwrap(boxed))
        target = _normalise_space(gold)
        correct = (
            extracted == target
            or extracted.replace(" ", "") == target.replace(" ", "")
            or plain_equation(extracted) == plain_equation(gold)
        )
    else:
        raise ValueError(f"Unknown SciKnowEval answer_type: {answer_type}")
    grade.update(
        correct=correct,
        failure_type="pass" if correct else "wrong_answer",
        extracted_answer=extracted,
        tests=[correct],
    )
    return grade


# --------------------------------------------------------------------------- #
# SciKnowEval open-answer problems: the inline LLM-judge verdict
# --------------------------------------------------------------------------- #


def _grade_judged_one(record: dict) -> dict:
    """Read the verdict the rollout's own judge stored (leaf.verifiers.llm_judge).

    This route never judges: every condition judges its responses inline on the
    GPU and stores ``record["judge"]``, so grading here -- in the rollout's
    verifier thread or a CPU grading job -- reproduces that verdict exactly. A
    record without one is a pipeline bug, not an incorrect answer.
    """
    judge = record.get("judge")
    if not isinstance(judge, dict) or "verdict" not in judge:
        raise ValueError(
            f"{record.get('key')}: sciqa_judge record carries no inline judge verdict"
        )
    correct = bool(judge["verdict"])
    if correct:
        failure = "pass"
    elif not judge.get("parsed", False):
        failure = "judge_unparsed"
    else:
        failure = "wrong_answer"
    return {
        **_base_grade(record),
        "correct": correct,
        "failure_type": failure,
        "extracted_answer": "",
        "native_feedback": "",
        "tests": [correct],
        "metadata": {
            "answer_type": "open",
            "criterion": record["verifier"].get("criterion"),
            "judge_parsed": bool(judge.get("parsed", False)),
            "judge_output": str(judge.get("output", "")),
            "judge_completion_tokens": judge.get("completion_tokens"),
            "judge_response_truncated": bool(judge.get("truncated", False)),
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

_ROUTES = ("lcb", "mbpp", "math", "sciqa", "sciqa_judge")


def _pool_init() -> None:
    """Keep each sandbox worker single-threaded; the pool itself is the parallelism."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_THINK_CLOSE = "</think>"


def strip_think(response: str, kind: str, payload: dict | None = None) -> tuple[str, str]:
    """The text to grade, and how it was chosen: none / stripped / fallback.

    A response that reasons inside ``<think> ... </think>`` is graded on what
    follows the last ``</think>``, so a draft inside the reasoning cannot stand
    in for the answer -- the LCB extractor takes the longest code block in the
    whole response, which is often a draft. If nothing gradable follows the tag
    (no fenced block for code, no ``\\boxed{}`` for math / sciqa), the whole
    response is graded as before rather than failing on extraction. Responses
    without ``</think>`` -- every run before the <think> prompt -- are untouched.
    """
    if _THINK_CLOSE not in response or kind == "sciqa_judge":
        # The judge reads its own view of the response (leaf.verifiers.llm_judge).
        return response, "none"
    after = response.rsplit(_THINK_CLOSE, 1)[1]
    payload = payload or {}
    if (kind == "sciqa" and payload.get("answer_format") == OLMES_FORMAT
            and payload.get("answer_type") != "filling"):
        gradable = bool(_olmes_sciqa_answer({"response": after, "verifier": payload})[0])
    elif kind in ("math", "sciqa"):
        gradable = "\\boxed" in after
    else:
        gradable = bool(_FENCE.search(after))
    return (after, "stripped") if gradable else (response, "fallback")


class Grader:
    """Warm verifier front-end: one spawn pool reused across every call.

    Created lazily and shared, because the rollout driver grades one sample per
    call, thousands of times, from concurrent rollout threads.
    """

    def __init__(self, config: dict, workers: int | None = None):
        self.config = config
        requested = int(workers or config.get("rollout_verifier_workers")
                        or config.get("verifier_workers", 8))
        if requested < 1:
            raise ValueError("Verifier workers must be positive")
        self.workers = max(1, min(requested, os.cpu_count() or requested))
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def _executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_pool_init,
                )
            return self._pool

    def _map(self, tasks: list[tuple]) -> list:
        """Run one sandboxed (function, args) task per entry, restarting a broken pool."""
        for attempt in (1, 2):
            executor = self._executor()
            try:
                futures = [executor.submit(function, *args) for function, args in tasks]
                return [future.result() for future in futures]
            except BrokenProcessPool:
                with self._lock:
                    if self._pool is executor:
                        self._pool = None
                executor.shutdown(wait=False)
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def grade(self, records: list[dict], config: dict | None = None) -> list[dict]:
        """Grade one batch; `config` overrides the grader's own for this call only."""
        config = config or self.config
        if not records:
            return []
        # Grade what follows </think> when there is something gradable there;
        # every route below reads record["response"], so swap it in a copy.
        views = [
            strip_think(r["response"], r["verifier"]["kind"], r["verifier"]) for r in records
        ]
        records = [
            record if how == "none" else {**record, "response": text}
            for record, (text, how) in zip(records, views, strict=True)
        ]
        grades = self._grade(records, config)
        for grade, (_, how) in zip(grades, views, strict=True):
            grade["think"] = how
        return grades

    def _grade(self, records: list[dict], config: dict) -> list[dict]:
        positions: dict[str, list[int]] = {kind: [] for kind in _ROUTES}
        for index, record in enumerate(records):
            kind = record["verifier"]["kind"]
            if kind not in positions:
                raise ValueError(f"Unknown verifier kind: {kind}")
            positions[kind].append(index)
        grades: list[dict | None] = [None] * len(records)

        if positions["lcb"]:
            subset = [records[index] for index in positions["lcb"]]
            for index, grade in zip(
                positions["lcb"], _grade_lcb(subset, config, self.workers), strict=True
            ):
                grades[index] = grade
        for index in positions["sciqa"]:
            grades[index] = _grade_sciqa_one(records[index])
        for index in positions["sciqa_judge"]:
            grades[index] = _grade_judged_one(records[index])

        mbpp_codes = [
            extract_mbpp_code(records[index]["response"], records[index]["verifier"])
            for index in positions["mbpp"]
        ]
        runnable = [
            (index, code) for index, code in zip(positions["mbpp"], mbpp_codes, strict=True)
            if code.strip()
        ]
        tasks = [
            (_mbpp_task,
             (code, records[index]["verifier"].get("test_setup_code", "") or "",
              list(records[index]["verifier"]["test_list"]),
              float(config["mbpp_timeout"]),
              int(config["verifier_candidate_memory_gib"])))
            for index, code in runnable
        ]
        tasks += [
            (_math_task, (records[index]["response"], records[index]["verifier"]["answer"]))
            for index in positions["math"]
        ]
        if tasks:
            # One wave over the shared pool so MBPP and MATH share the cores.
            outcomes = self._map(tasks)
            cursor = 0
            for index, code in runnable:
                outcome = outcomes[cursor]
                grades[index] = _mbpp_grade(
                    records[index], code, outcome, outcome["seconds"])
                cursor += 1
            for index in positions["math"]:
                outcome = outcomes[cursor]
                grades[index] = _math_grade(records[index], outcome, outcome["seconds"])
                cursor += 1
        for index, code in zip(positions["mbpp"], mbpp_codes, strict=True):
            if grades[index] is None:
                grades[index] = _mbpp_grade(records[index], code, None, 0.0)
        return grades

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_GRADER: Grader | None = None


def default_grader(config: dict, workers: int | None = None) -> Grader:
    """The process-wide warm grader; the pool is built on its first real call."""
    global _DEFAULT_GRADER
    with _DEFAULT_LOCK:
        if _DEFAULT_GRADER is None:
            _DEFAULT_GRADER = Grader(config, workers)
        return _DEFAULT_GRADER


def grade_records(records: list[dict], config: dict, *, workers: int | None = None) -> list[dict]:
    """Grade one batch of candidates.  Returns grades aligned 1:1 to `records`.

    Delegates to the shared :class:`Grader`, so repeated single-record calls reuse
    one warm process pool.  `workers` only takes effect on the first call of a
    process; build a private :class:`Grader` when a different degree is needed.
    """
    return default_grader(config, workers).grade(records, config)


@atexit.register
def _shutdown_default_grader() -> None:
    if _DEFAULT_GRADER is not None:
        _DEFAULT_GRADER.close()
