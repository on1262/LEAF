"""SEED hindsight-Skill analyzer prompt and the Skill format check."""
import os
import re

from leaf.baselines import seed_prompts


def analyzer_content_from_prompt(problem_prompt, response, public_feedback, correct):
    # The vendored fork imports this name directly; the SEED runners swap in a
    # task-agnostic wording (the benchmarks are not only code) through an env flag
    # instead of patching the fork.  Unset, the programming wording below is used.
    if os.environ.get(seed_prompts.ENV_FLAG) == "1":
        return seed_prompts.analyzer_content_from_prompt(
            problem_prompt, response, public_feedback, correct
        )
    return f"""You are a hindsight-skill analyzer for a completed programming trajectory.

Problem:
{problem_prompt}

Completed response:
{response}

Public-test outcome: {"correct" if correct else "incorrect"}

Public-test feedback:
{public_feedback}

Extract one concise hindsight Skill of at most 500 tokens that would improve a future attempt.
State actionable reasoning, algorithmic invariants, complexity, and implementation checks.
Do not reproduce the complete response or output a submit-ready program. Output only the Skill."""


def validate_seed_skill(skill, response, tokenizer, max_tokens=500):
    """Apply the frozen, response-independent SEED Skill format boundary."""
    skill = "\n".join(line.rstrip() for line in str(skill).strip().splitlines()).strip()
    if not skill:
        return False, "empty", skill
    if len(tokenizer.encode(skill, add_special_tokens=False)) > max_tokens:
        return False, "token_limit", skill
    if "```" in skill:
        return False, "code_block", skill

    # Reject near-verbatim complete programs while retaining short identifiers
    # and pseudocode that are useful as a Skill.
    def tokens(text):
        return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]", text.lower())

    code_blocks = "\n".join(
        re.findall(r"```(?:python)?\s*(.*?)```", str(response), flags=re.DOTALL | re.IGNORECASE)
    )
    code_tokens = tokens(code_blocks)
    skill_tokens = tokens(skill)
    width = 5
    if len(code_tokens) >= width and len(skill_tokens) >= 20:
        code_ngrams = {
            tuple(code_tokens[index:index + width])
            for index in range(len(code_tokens) - width + 1)
        }
        skill_ngrams = [
            tuple(skill_tokens[index:index + width])
            for index in range(len(skill_tokens) - width + 1)
        ]
        containment = sum(value in code_ngrams for value in skill_ngrams) / len(skill_ngrams)
        if containment >= 0.5:
            return False, "response_overlap", skill
    return True, None, skill
