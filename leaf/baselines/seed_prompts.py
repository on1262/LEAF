"""Task-agnostic SEED hindsight-analyzer prompt.

The programming analyzer prompt (:func:`leaf.baselines.seed_analyzer.analyzer_content_from_prompt`)
opens with "a completed programming trajectory" and asks for algorithmic
invariants and implementation checks. The benchmarks include MATH500 and
SciKnowEval, so the analyzer needs wording that fits a proof or a chemistry
question as well as a program.

The vendored fork imports the analyzer from ``leaf.baselines.seed_analyzer`` by name
(``verl/trainer/ppo/ray_trainer.py``, ``loss_mode == "seed"``).  Rather than
patch the fork, ``leaf/baselines/seed_analyzer.py`` delegates here when
``LEAF_SEED_ANALYZER=1`` is set in the environment, which the SEED runners do
and nothing else does.  With the variable unset, the programming prompt is used
byte for byte.
"""

from __future__ import annotations

ENV_FLAG = "LEAF_SEED_ANALYZER"


def analyzer_content_from_prompt(
    problem_prompt: str, response: str, public_feedback: str, correct: bool
) -> str:
    return f"""You are a hindsight-skill analyzer for a completed problem-solving trajectory.

Problem:
{problem_prompt}

Completed response:
{response}

Public-verifier outcome: {"correct" if correct else "incorrect"}

Public-verifier feedback:
{public_feedback}

Extract one concise hindsight Skill of at most 500 tokens that would improve a future attempt.
State actionable reasoning, the invariants or facts that decide the answer, and the checks that
would have caught the mistake. Do not reproduce the complete response and do not output a
ready-to-submit answer. Output only the Skill."""
