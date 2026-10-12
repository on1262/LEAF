"""verl ``custom_reward_function`` entry point of the verl baselines.

The vendored fork calls this once per rolled-out sequence, from
``verl/workers/reward_manager/naive.py``::

    self.compute_score(
        data_source=..., solution_str=..., ground_truth=..., extra_info=...,
    )

so the signature below matches that call exactly.  ``**kwargs`` absorbs future
keyword arguments rather than crashing a multi-hour run.

Dispatch
--------
There is no ``if data_source == ...`` ladder here.  Everything the public
verifier needs travels with the row: ``ground_truth`` is the JSON-encoded
``verifier`` payload and ``extra_info`` carries ``task_type`` / ``source`` /
``problem_uid`` / ``problem``.  This module rebuilds the canonical record and
hands it to :func:`leaf.verifiers.answer_grader.grade_records`, which owns all four
verifier routes.  ``data_source`` is kept only for cross-checking and logging.

Feedback
--------
SDPO and SEED train on the ``feedback`` string, so it is the point of this
module rather than a by-product.  It is always
:func:`leaf.prompts.task_prompts.feedback_section` applied to the grade, which renders
the executor's own report for ``code_lcb`` / ``code_mbpp`` and the verdict-only
block (verdict + extracted answer, never the reference answer) for ``math`` /
``sciqa``.

Cost
----
The fork gives a custom reward function no batch hook, so one call grades one
sequence, and with per-benchmark jobs it is called on many small batches.
Everything expensive is therefore module level and paid once per worker
process: the config is parsed at import and the verifier entry point is
resolved on the first call and cached in ``_GRADER``, so no call reimports or
re-reads anything.  Concurrency comes from the fork's naive reward manager,
which overlaps candidates across a thread pool sized by
``LEAF_VERIFIER_CANDIDATE_WORKERS`` (the runners export it from
``rollout_verifier_workers``, and the jobs request 16 cores).  Per call the cost
is whatever the route costs: a string / ``math_verify`` comparison for ``math``
and ``sciqa``, and a sandboxed test execution bounded by ``verifier_timeout`` /
``mbpp_timeout`` for the two code routes.
"""

from __future__ import annotations

from leaf.prompts import task_prompts
from leaf.verifiers import llm_judge
from leaf.core.project_config import load_config, record_from_reward_call

# Parsed once per worker process, not per call.
_CONFIG = load_config()
_GRADER = None

# Grade keys that are useful to log but are not part of the reward itself.
_PASSTHROUGH = ("failure_type", "extracted_answer")


def _grade_records(records, config):
    """Resolve the verifier once and keep it; also the seam a test fakes."""
    global _GRADER
    if _GRADER is None:
        from leaf.verifiers.answer_grader import grade_records

        _GRADER = grade_records
    return _GRADER(records, config)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    extra_info = dict(extra_info or {})
    record = record_from_reward_call(ground_truth, extra_info, solution_str)
    if record["task_type"] not in task_prompts.TASK_TYPES:
        raise ValueError(
            f"reward received an unknown task_type: {record['task_type']!r}"
        )
    if data_source and record["source"] and data_source != record["source"]:
        raise ValueError(
            f"data_source {data_source!r} disagrees with extra_info source "
            f"{record['source']!r}"
        )
    grades = _grade_records([record], _CONFIG)
    if len(grades) != 1:
        raise RuntimeError("verifier returned the wrong number of grades")
    grade = grades[0]
    correct = bool(grade["correct"])
    truncated = bool(extra_info.get("truncated", False))
    feedback = task_prompts.feedback_section(record, grade)
    if truncated:
        feedback += (
            "\n\nYour response was truncated because it exceeded the maximum length."
        )
    failure = str(grade.get("failure_type") or ("pass" if correct else "unknown"))
    result = {
        # Binary public-verifier reward.
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "pred": str(grade.get("extracted_answer") or "")[:5000],
        "feedback": feedback,
        "source": record["source"],
        "task_type": record["task_type"],
        "failure_type": failure,
        "incorrect_format": 1 if failure == "extraction_failure" else 0,
        "truncated": 1 if truncated else 0,
        "truncated_and_missing_answer": 1 if truncated and not grade.get(
            "extracted_answer"
        ) else 0,
        "error_in_test_cases": 1 if failure in {
            "infrastructure_error", "runtime_error", "syntax_error"
        } else 0,
        "timed_out": 1 if failure == "timeout" else 0,
    }
    for key in _PASSTHROUGH:
        if key in grade and key not in result:
            result[key] = grade[key]
    if llm_judge.needs_judge(record["verifier"]):
        # The judge prompt / output / verdict go into the rollout log beside
        # the reward they produced.
        result.update(llm_judge.reward_fields(record.get("judge")))
    return result


def compute_score_batch(records: list[dict]) -> list[dict]:
    """Grade many records in one verifier call.

    The fork has no batched reward hook today, but the offline graders do, and
    keeping the batched path here means both use exactly one implementation.
    """
    if not records:
        return []
    grades = _grade_records(list(records), _CONFIG)
    if len(grades) != len(records):
        raise RuntimeError("verifier returned the wrong number of grades")
    return [
        {
            **grade,
            "feedback": task_prompts.feedback_section(record, grade),
        }
        for record, grade in zip(records, grades, strict=True)
    ]


__all__ = ["compute_score", "compute_score_batch"]
