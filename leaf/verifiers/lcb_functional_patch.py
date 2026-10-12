"""LiveCodeBench-parity entry-point resolution for the SDPO functional runner.

`third_party/SDPO/verl/utils/reward_score/feedback/code.py::run_test_func` looks
the callable up in the *top-level* namespace only.  LeetCode problems ship a
`class Solution:` starter signature, so every class-wrapped answer raised
`KeyError: '<fn_name>'` and every flattened `def fn(self, ...)` answer raised
`TypeError: missing 1 required positional argument`.  On the frozen 131-problem
LCB split that silenced all 49 functional problems: 2,393 rollouts scored
2 passes while the same responses reach ~25% under the official runner.

LiveCodeBench instantiates `Solution()` whenever the source mentions the class
(`third_party/LiveCodeBench/lcb_runner/evaluation/testing_util.py:192-204`) and
splits call-based arguments on newlines rather than on arbitrary whitespace.
This module reproduces both rules inside the SDPO sandbox so the training reward
agrees with the evaluation grader.  SDPO is vendored outside version control, so
the fix lives here and is applied by monkeypatch; `compute_score` forks its
candidate processes, which inherits a patch applied in the parent.
"""

from __future__ import annotations

import io
import json
import sys

_PATCHED_FLAG = "_lcb_functional_patch_applied"


def parse_functional_args(test_input):
    """Decode call-based arguments the way LiveCodeBench does.

    One JSON value per line.  The upstream `str.split()` broke every argument
    that contains whitespace, e.g. `[1, 2]` or `"a b"`.
    """
    if isinstance(test_input, dict):
        return None
    text = str(test_input)
    lines = [line for line in text.split("\n") if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except (ValueError, TypeError):
        return [json.loads(token) for token in text.split()]


def resolve_entry_point(completion, fn_name, namespace):
    """Return the callable a functional test should invoke, or None."""
    if "class Solution" in completion:
        solution_cls = namespace.get("Solution")
        if callable(solution_cls):
            instance = solution_cls()
            if fn_name and callable(getattr(instance, fn_name, None)):
                return getattr(instance, fn_name)
            methods = [
                name for name in dir(instance)
                if not name.startswith("_") and callable(getattr(instance, name, None))
            ]
            if len(methods) == 1:
                return getattr(instance, methods[0])
    if fn_name and callable(namespace.get(fn_name)):
        return namespace[fn_name]
    return None


def _build_run_test_func(sdpo_code):
    import __future__

    def run_test_func(completion, test_input, test_output, fn_name, namespace=None):
        if namespace is None:
            namespace = sdpo_code._create_sandbox_namespace()
        code_obj = compile(
            completion,
            sdpo_code.FILENAME,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        sdpo_code._exec_with_isolated_locals(code_obj, namespace)

        func = resolve_entry_point(completion, fn_name, namespace)
        if func is None:
            # The upstream fallback guessed a name out of the source text and
            # could call an unrelated object; fail explicitly instead.
            return False, (
                f"{sdpo_code.ERROR_PREFIX}NameError: no callable entry point "
                f"{fn_name!r} was defined by the submitted program"
            )

        output = io.StringIO()
        sys.stdout = output
        old_stderr = sdpo_code._capture_stderr(namespace)
        try:
            args = parse_functional_args(test_input)
            result = func(**test_input) if args is None else func(*args)
            try:
                left = sdpo_code._to_safe_jsonable(result)
                right = sdpo_code._to_safe_jsonable(json.loads(test_output))
                dumped = lambda value: json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
                if dumped(left) != dumped(right):
                    return False, result
                return True, result
            except Exception as serialization_error:
                return False, f"{sdpo_code.ERROR_PREFIX}{serialization_error}"
        except BaseException as error:
            return False, f"{sdpo_code.ERROR_PREFIX}{sdpo_code._short_trace(error)}"
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = old_stderr

    return run_test_func


def _build_run_tests(sdpo_code, original):
    """SDPO's ``run_tests`` that can stop at the first failing test.

    With ``sdpo_code.LEAF_STOP_AT_FIRST_FAILURE`` unset this is the original
    function. Set (training on the LCB training problems, config
    ``lcb_stop_at_first_failure``), the
    tests still run in index order, in the same waves and under the same
    per-test deadline, but the first test that does not pass -- wrong answer,
    timeout or runtime error -- ends the run: the rest of its wave is killed and
    only the records up to it are returned. The binary reward is unchanged (any
    failure fails the candidate); the feedback shows that first failure.
    """
    import multiprocessing
    import os
    import time

    def run_tests(test_cases, solution, sparse_rewards, max_test_cases):
        if not getattr(sdpo_code, "LEAF_STOP_AT_FIRST_FAILURE", False):
            return original(test_cases, solution, sparse_rewards, max_test_cases)
        completion = sdpo_code.extract_code(solution)
        if completion is None:
            return [{"test_idx": 0, "input": None, "expected": None,
                     "actual": sdpo_code.INCORRECT_FORMAT, "passed": False, "debug": "",
                     "time": float("inf")}]
        num_test_cases = (min(max_test_cases, len(test_cases["inputs"])) if max_test_cases
                          else len(test_cases["inputs"]))
        timeout_per_test_case = (float(test_cases["time_limit"])
                                 if test_cases["time_limit"] is not None
                                 else sdpo_code.DEFAULT_TIMEOUT)
        max_parallel = max(1, int(os.environ.get("LEAF_VERIFIER_WORKERS", str(num_test_cases))))
        records = []
        for batch_start in range(0, num_test_cases, max_parallel):
            process_data = []
            for test_idx in range(batch_start, min(batch_start + max_parallel, num_test_cases)):
                parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
                process = multiprocessing.Process(
                    target=sdpo_code.run_tests_for_one_example,
                    args=(test_cases, completion, child_conn, sparse_rewards, test_idx),
                )
                process.start()
                child_conn.close()
                process_data.append({"test_idx": test_idx, "process": process,
                                     "parent_conn": parent_conn})
            start_time = time.time()
            failed = False
            for index, data in enumerate(process_data):
                test_idx, process, parent_conn = data["test_idx"], data["process"], data["parent_conn"]
                deadline = max(0, timeout_per_test_case * sdpo_code.TIMEOUT_SCALER + 1
                               - (time.time() - start_time))
                if parent_conn.poll(deadline):
                    try:
                        result = parent_conn.recv()
                    except Exception as error:
                        result = {"test_idx": test_idx, "input": test_cases["inputs"][test_idx],
                                  "expected": test_cases["outputs"][test_idx],
                                  "actual": f"Process Error: {sdpo_code._short_trace(error)}",
                                  "passed": False, "debug": "", "time": float("inf")}
                else:
                    result = {"test_idx": test_idx, "input": test_cases["inputs"][test_idx],
                              "expected": test_cases["outputs"][test_idx],
                              "actual": sdpo_code.TIMEOUT, "passed": False, "debug": "",
                              "time": float("inf")}
                records.append(result)
                parent_conn.close()
                process.join(timeout=0)
                if process.is_alive():
                    process.kill()
                    process.join()
                if not result["passed"]:
                    for pending in process_data[index + 1:]:
                        pending["parent_conn"].close()
                        if pending["process"].is_alive():
                            pending["process"].kill()
                        pending["process"].join()
                    failed = True
                    break
            if failed:
                break
        assert records
        return records

    return run_tests


class _CachedJson:
    """``json`` for the SDPO module, with ``loads`` cached for big ground truths.

    ``compute_score`` parses the whole ground-truth string for every candidate,
    and an all-tests training problem is up to 190 MB of JSON, parsed under the
    GIL while every other grading thread waits. Strings above ``MIN_BYTES`` are
    parsed once and the parse is reused (the sandboxed tests deep-copy what
    they read, so it is never mutated); the cache is an LRU bounded by source
    bytes. Everything else is the real ``json`` module.
    """

    MIN_BYTES = 2**20

    def __init__(self, budget_bytes: int = 3 * 2**30):
        import collections
        import threading

        self._cache = collections.OrderedDict()
        self._bytes = 0
        self._budget = budget_bytes
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(json, name)

    def loads(self, text, *args, **kwargs):
        if args or kwargs or not isinstance(text, str) or len(text) < self.MIN_BYTES:
            return json.loads(text, *args, **kwargs)
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        value = json.loads(text)
        with self._lock:
            if text not in self._cache:
                self._cache[text] = value
                self._bytes += len(text)
                while self._bytes > self._budget and len(self._cache) > 1:
                    dropped, _ = self._cache.popitem(last=False)
                    self._bytes -= len(dropped)
        return value


def set_stop_at_first_failure(sdpo_code, enabled: bool) -> None:
    """LCB training-problem grading settings for this process.

    ``enabled`` switches :func:`_build_run_tests`'s early stop on and installs the
    ground-truth parse cache (:class:`_CachedJson`); off, both are the originals.
    """
    sdpo_code.LEAF_STOP_AT_FIRST_FAILURE = bool(enabled)
    if enabled and not isinstance(sdpo_code.json, _CachedJson):
        sdpo_code.json = _CachedJson()


def apply():
    """Install the patched functional runner.  Idempotent."""
    from verl.utils.reward_score.feedback import code as sdpo_code

    if getattr(sdpo_code, _PATCHED_FLAG, False):
        return sdpo_code
    sdpo_code.run_test_func = _build_run_test_func(sdpo_code)
    sdpo_code.run_tests = _build_run_tests(sdpo_code, sdpo_code.run_tests)
    setattr(sdpo_code, _PATCHED_FLAG, True)
    return sdpo_code
