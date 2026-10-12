"""Suite-aware LiveCodeBench verifier with public-only rich feedback."""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

from leaf.verifiers.lcb_memory import candidate_memory_limit
from leaf import paths
from leaf.core.io_utils import evaluation_sample


# Public-suite ground truth for the frozen 131 training questions.  Override with
# LEAF_LCB_PUBLIC_GROUND_TRUTH when grading against a different frozen export.
PUBLIC_GROUND_TRUTH = str(paths.LCB_PUBLIC_GROUND_TRUTH)

GRADE_METADATA_FIELDS = (
    "condition", "mode", "split", "problem_index", "trial_index",
    "attempt_index", "rollout_index", "seed", "turn", "trajectory_id",
    "skill_used",
)


def generation_metadata(record):
    """Preserve the frozen pairing/trajectory identifiers in every grade."""
    return {field: record[field] for field in GRADE_METADATA_FIELDS if field in record}


def failure_type(code, verdict, metadata, finish_reason):
    if not code.strip():
        return "extraction_failure"
    try:
        ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    if finish_reason == "length":
        return "generation_truncated"
    codes = {int(value) for value in verdict if not isinstance(value, bool)}
    detail = json.dumps(metadata).lower()
    if -3 in codes or "timeout" in detail or "time limit exceeded" in detail:
        return "timeout"
    if -4 in codes or "error during testing" in detail:
        return "runtime_error"
    if -5 in codes or "testrunnererror" in detail:
        return "infrastructure_error"
    return "wrong_answer"


def _public_feedback_grade(record, ground_truth, code):
    """Use exactly the rich-feedback executor used by SDPO training."""
    started = time.monotonic()
    result = code.compute_score(
        record["response"], ground_truth,
        {
            "split": "train",
            "truncated": record.get("finish_reason") == "length",
        },
        sparse_rewards=False,
        max_test_cases=None,
    )
    correct = float(result["acc"]) == 1.0
    extracted = code.extract_code(record["response"]) or ""
    if correct:
        category = "pass"
    elif result.get("incorrect_format"):
        category = "extraction_failure"
    else:
        try:
            ast.parse(extracted)
        except SyntaxError:
            category = "syntax_error"
        else:
            if result.get("timed_out"):
                category = "timeout"
            elif result.get("error_in_test_cases"):
                category = "runtime_error"
            else:
                category = "wrong_answer"
    return {
        "key": record["key"],
        "problem_uid": record["problem_uid"],
        "suite": "public",
        "correct": correct,
        "failure_type": category,
        "tests": [correct],
        "metadata": [{
            "accuracy": float(result["acc"]),
            "timed_out_fraction": float(result.get("timed_out", 0.0)),
            "error_in_test_cases": int(result.get("error_in_test_cases", 0)),
        }],
        "native_feedback": result.get("feedback", ""),
        "extracted_code": extracted,
        "attempt": 1,
        "verifier_candidate_seconds": time.monotonic() - started,
        **generation_metadata(record),
    }


def _grade_public_feedback(records, workers, candidate_memory_gib, test_workers):
    os.environ["LEAF_VERIFIER_WORKERS"] = str(test_workers)
    os.environ["LEAF_CANDIDATE_MEMORY_GIB"] = str(candidate_memory_gib)
    from leaf.verifiers.lcb_functional_patch import apply as apply_functional_patch
    import pyarrow.parquet as pq

    # Keeps the training reward on the same functional entry-point rules as the
    # LiveCodeBench grader used for the heldout and full suites.
    code = apply_functional_patch()
    table = pq.read_table(
        Path(os.environ.get("LEAF_LCB_PUBLIC_GROUND_TRUTH", PUBLIC_GROUND_TRUTH)),
        columns=["reward_model", "extra_info"],
    ).to_pylist()
    ground_truth = {
        row["extra_info"]["problem_uid"]: row["reward_model"]["ground_truth"]
        for row in table
    }
    missing = {row["problem_uid"] for row in records} - set(ground_truth)
    if missing:
        raise ValueError(f"Public verifier has unknown problem UIDs: {sorted(missing)[:3]}")
    evaluate = lambda row: _public_feedback_grade(
        row, ground_truth[row["problem_uid"]], code
    )
    with ThreadPoolExecutor(
        max_workers=min(workers, len(records)),
        thread_name_prefix="lcb-public-verifier",
    ) as executor:
        return list(executor.map(evaluate, records))


def grade_records(
    records, suite, workers=8, timeout=6, candidate_memory_gib=4,
    test_workers=1,
):
    if suite not in {"public", "heldout", "full"}:
        raise ValueError(f"Unknown LCB test suite: {suite}")
    if suite == "public" and any(record.get("split") != "train" for record in records):
        raise ValueError("Public verifier accepts only the frozen LCB split's questions")
    if workers < 1:
        raise ValueError("Verifier workers must be positive")
    if test_workers < 1:
        raise ValueError("Verifier per-candidate workers must be positive")
    if suite == "public":
        return _grade_public_feedback(
            records, workers, candidate_memory_gib, test_workers
        )
    from lcb_runner.evaluation import testing_util
    testing_util.reliability_guard = partial(
        testing_util.reliability_guard,
        maximum_memory_bytes=candidate_memory_limit(candidate_memory_gib * 2**30),
    )
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.utils.extraction_utils import extract_code

    grades = []
    # Keep one full worker wave per LCB executor so every worker stays busy.
    # The batch size does not affect scores or output order.
    batch_size = max(8, workers)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        codes = [extract_code(record["response"], None) for record in batch]
        samples = [evaluation_sample(record, suite) for record in batch]
        result = codegen_metrics(
            samples, [[code] for code in codes], k_list=[1],
            num_process_evaluate=workers, timeout=timeout,
        )
        retry = []
        for offset, record in enumerate(batch):
            verdict = result[1][offset][0]
            metadata = result[2][offset]
            correct = bool(verdict) and all(value is True or value == 1 for value in verdict)
            category = "pass" if correct else failure_type(
                codes[offset], verdict, metadata, record.get("finish_reason"))
            grade = {
                "key": record["key"],
                "problem_uid": record["problem_uid"],
                "suite": suite,
                "correct": correct,
                "failure_type": category,
                "tests": verdict,
                "metadata": metadata,
                "extracted_code": codes[offset],
                "attempt": 1,
                **generation_metadata(record),
            }
            grades.append(grade)
            if category == "infrastructure_error":
                retry.append((len(grades) - 1, record, codes[offset]))
        for grade_index, record, code in retry:
            sample = evaluation_sample(record, suite)
            for attempt in (2, 3):
                again = codegen_metrics(
                    [sample], [[code]], k_list=[1], num_process_evaluate=1,
                    timeout=timeout,
                )
                verdict = again[1][0][0]
                metadata = again[2][0]
                correct = bool(verdict) and all(value is True or value == 1 for value in verdict)
                category = "pass" if correct else failure_type(
                    code, verdict, metadata, record.get("finish_reason"))
                grades[grade_index].update(
                    correct=correct, failure_type=category, tests=verdict,
                    metadata=metadata, attempt=attempt,
                )
                if category != "infrastructure_error":
                    break
            if grades[grade_index]["failure_type"] == "infrastructure_error":
                raise RuntimeError(
                    f"LCB infrastructure failure after retries: {record['key']}")
    return grades


def _parsed_metadata(values):
    result = []
    for value in values:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result.append(value)
    return result


def rich_feedback(grade, tokenizer, max_tokens):
    if grade["suite"] != "public":
        raise ValueError("Rich feedback may only be constructed from public-test grades")
    if grade["correct"]:
        full_text = "Outcome: correct"
    else:
        lines = [
            "Outcome: incorrect",
            f"Failure type: {grade['failure_type']}",
            "Per-test verdicts: " + json.dumps(grade.get("tests", []), ensure_ascii=False),
        ]
        code = grade.get("extracted_code", "")
        if grade["failure_type"] == "extraction_failure":
            lines.append("No executable Python code could be extracted from the answer.")
        elif grade["failure_type"] == "syntax_error":
            try:
                ast.parse(code)
            except SyntaxError as exc:
                lines.append(
                    f"Syntax error: {exc.msg}; line={exc.lineno}; offset={exc.offset}")
        metadata = _parsed_metadata(grade.get("metadata", []))
        if metadata:
            lines.append("Environment details (failed public test/compiler/runtime report):")
            lines.append(json.dumps(metadata, ensure_ascii=False, indent=2))
        full_text = "\n".join(lines)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    shown_tokens = full_tokens[:max_tokens]
    shown_text = tokenizer.decode(shown_tokens, skip_special_tokens=True)
    return {
        "feedback": shown_text,
        "feedback_tokens_before_truncation": len(full_tokens),
        "feedback_tokens": len(shown_tokens),
        "feedback_truncated": len(full_tokens) > len(shown_tokens),
    }


def worker(connection, cfg):
    try:
        mp.set_start_method("fork", force=True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
        while True:
            command = connection.recv()
            if command["action"] == "close":
                break
            suite = command["suite"]
            grades = grade_records(
                command["records"], suite,
                cfg.get("verifier_service_workers", cfg["verifier_workers"]),
                cfg["verifier_timeout"], cfg["verifier_candidate_memory_gib"],
                cfg["verifier_workers"],
            )
            if command.get("include_feedback", False):
                if suite != "public":
                    raise ValueError("Feedback cannot be requested for a non-public suite")
                for grade in grades:
                    if "native_feedback" in grade:
                        full_text = grade["native_feedback"]
                        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                        shown_tokens = full_tokens[:cfg["augmented_prompt_max_tokens"]]
                        grade.update({
                            "feedback": tokenizer.decode(
                                shown_tokens, skip_special_tokens=True
                            ),
                            "feedback_tokens_before_truncation": len(full_tokens),
                            "feedback_tokens": len(shown_tokens),
                            "feedback_truncated": len(full_tokens) > len(shown_tokens),
                        })
                    else:
                        grade.update(rich_feedback(
                            grade, tokenizer, cfg["augmented_prompt_max_tokens"]))
            path = Path(command["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text("".join(
                json.dumps(grade, ensure_ascii=False) + "\n" for grade in grades))
            temporary.replace(path)
            connection.send({"n": len(grades)})
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        connection.send({"error": repr(exc)})
        raise


class Verifier:
    def __init__(self, cfg):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, cfg))
        self.process.start()
        child.close()

    def score(self, records, path, suite="public", include_feedback=False):
        self.connection.send({
            "action": "score", "records": records, "path": str(path),
            "suite": suite, "include_feedback": include_feedback,
        })
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f"Verifier exited: {self.process.exitcode}")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.process.is_alive():
            self.connection.send({"action": "close"})
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
