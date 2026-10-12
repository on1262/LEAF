"""Public-verifier reward for the two-turn SCoRe adaptation.

The SCoRe agent loop (``leaf/baselines/score_agent.py``, which is task-agnostic
and reused unchanged) emits both turns of a trajectory in ``extra_info``.  This
module grades them and applies SCoRe's reward shaping:

* stage 1 rewards only the second turn, so the first turn stays near the base
  policy;
* stage 2 rewards both turns plus ``score_correction_bonus`` times the
  improvement, which is what pushes incorrect -> correct transitions.

Both turns go through :func:`leaf.verifiers.answer_grader.grade_records` in a **single**
batched call, so a SCoRe sequence costs one verifier invocation, not two.
"""

from __future__ import annotations

from leaf.prompts import task_prompts
from leaf.verifiers import llm_judge
from leaf.core.project_config import load_config, record_from_reward_call

_CONFIG = load_config()
_GRADER = None


def _grade_records(records, config):
    """Resolve the verifier once and keep it; also the seam a test fakes."""
    global _GRADER
    if _GRADER is None:
        from leaf.verifiers.answer_grader import grade_records

        _GRADER = grade_records
    return _GRADER(records, config)


def _is_truncated(stop_reason) -> bool:
    return str(stop_reason).lower() in {"length", "max_tokens"}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    del data_source, solution_str
    if extra_info is None:
        raise ValueError("SCoRe reward requires trajectory metadata")
    first_response = extra_info.get("score_first_response")
    second_response = extra_info.get("score_second_response")
    if not isinstance(first_response, str) or not isinstance(second_response, str):
        raise ValueError("SCoRe reward is missing one of its two response texts")

    common = dict(extra_info)
    first_extra = {**common, "truncated": _is_truncated(common.get("score_first_stop_reason"))}
    second_extra = {**common, "truncated": _is_truncated(common.get("score_second_stop_reason"))}
    records = [
        record_from_reward_call(ground_truth, first_extra, first_response),
        record_from_reward_call(ground_truth, second_extra, second_response),
    ]
    records[0]["key"] = f"{records[0]['key']}:turn1"
    records[1]["key"] = f"{records[1]['key']}:turn2"
    # open-answer SciKnowEval: each turn carries its own inline verdict
    # from the fork's judge step (leaf.verifiers.llm_judge).
    judged = llm_judge.needs_judge(records[0]["verifier"])
    if judged:
        for record, key in zip(records, ("score_first_judge", "score_second_judge"), strict=True):
            record.pop("judge", None)
            if common.get(key):
                record["judge"] = dict(common[key])
    grades = _grade_records(records, _CONFIG)
    if len(grades) != 2:
        raise RuntimeError("verifier returned the wrong number of SCoRe grades")

    r1 = 1.0 if bool(grades[0]["correct"]) else 0.0
    r2 = 1.0 if bool(grades[1]["correct"]) else 0.0
    stage = common.get("score_stage")
    if stage == "stage1":
        shaped = r2
    elif stage == "stage2":
        bonus = float(common["score_correction_bonus"])
        shaped = r1 + r2 + bonus * (r2 - r1)
    else:
        raise ValueError(f"Unknown SCoRe stage: {stage!r}")
    result = {
        "score": shaped,
        "acc": r2,
        "score_r1": r1,
        "score_r2": r2,
        "score_stage1": stage == "stage1",
        "score_incorrect_to_correct": r1 == 0.0 and r2 == 1.0,
        "score_correct_to_incorrect": r1 == 1.0 and r2 == 0.0,
        "source": records[0]["source"],
        "task_type": records[0]["task_type"],
        "score_first_feedback": task_prompts.feedback_section(records[0], grades[0]),
        "score_second_feedback": task_prompts.feedback_section(records[1], grades[1]),
    }
    if judged:
        result.update(llm_judge.reward_fields(records[0].get("judge"), "judge1"))
        result.update(llm_judge.reward_fields(records[1].get("judge"), "judge2"))
    return result


__all__ = ["compute_score"]
