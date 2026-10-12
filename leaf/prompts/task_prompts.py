"""Task abstraction over the four benchmarks.

Every method needs the same four hooks per problem:

* ``problem_prompt`` / ``skill_prompt`` — the opening user message;
* ``correction_instruction`` — the blind self-correction turn (shared);
* ``feedback_section`` — what a verifier verdict looks like in a user message;
* the verifier itself, which lives in :mod:`leaf.verifiers.answer_grader`.

The four task types and what their public verifier can say:

| task_type    | source       | signal                                    |
|--------------|--------------|-------------------------------------------|
| ``code_lcb`` | LiveCodeBench| per-test verdicts + SDPO rich feedback    |
| ``code_mbpp``| MBPP         | which asserts failed + the exception      |
| ``math``     | MATH500      | correct / incorrect only                  |
| ``sciqa``    | SciKnowEval  | correct / incorrect only                  |

The open-answer SciKnowEval problems keep task_type ``sciqa``; their verifier
payload is ``kind = "sciqa_judge"`` (an LLM judge, :mod:`leaf.verifiers.llm_judge`)
and their feedback is the verdict alone.

MATH500 and SciKnowEval have no execution environment, so their feedback turn
carries the verdict and the answer that was extracted, and never the reference
answer.
"""

from __future__ import annotations

import functools

from leaf.core.io_utils import BASE_INSTRUCTION
from leaf.prompts.correction_prompt import CORRECTION_INSTRUCTION  # noqa: F401  (re-export)

# The code correction instruction (CORRECTION_INSTRUCTION) asks the model to look
# for "algorithmic, complexity, and implementation" mistakes. That is code
# vocabulary, so the math and science task types get wording that matches what
# they can actually go wrong at.
MATH_CORRECTION_INSTRUCTION = (
    "Review your previous answer carefully for errors in reasoning, algebra, "
    "arithmetic, and the handling of special or boundary cases. Then produce a "
    "corrected complete answer to the original problem. Do not assume that any "
    "external correctness signal is available."
)
SCIQA_CORRECTION_INSTRUCTION = (
    "Review your previous answer carefully for errors in scientific reasoning, "
    "the facts and relationships you relied on, unit and magnitude handling, "
    "and whether you addressed what the question actually asks. Then produce a "
    "corrected complete answer to the original problem. Do not assume that any "
    "external correctness signal is available."
)

TASK_TYPES = ("math", "code_lcb", "code_mbpp", "sciqa")

MBPP_INSTRUCTION = (
    "You are a coding expert. You will be given a Python programming task and "
    "the tests your code must satisfy. Write a correct Python program that "
    "matches the specification and passes all tests. Keep the function names "
    "and signatures the tests require. You may start by outlining your thought "
    "process. In the end, please provide the complete code in a code block "
    "enclosed with ``` ```."
)
MATH_INSTRUCTION = (
    "You are a mathematics expert. You will be given a competition mathematics "
    "problem. Solve it and give the final answer. You may start by outlining "
    "your thought process. In the end, please put the final answer inside "
    "\\boxed{}."
)
SCIQA_INSTRUCTION = (
    "You are a scientific reasoning expert. You will be given an "
    "undergraduate-level science question. Reason about it step by step, then "
    "give the final answer. In the end, please put the final answer inside "
    "\\boxed{}."
)
# Open-answer SciKnowEval problems (leaf.data.sciknoweval_pool). Same opening
# as SCIQA_INSTRUCTION -- so the SCoRe prompt matcher below still reads it as
# sciqa -- but no \boxed{}: an LLM judge (leaf.verifiers.llm_judge) reads the answer,
# and the task's own instruction in the problem body states its format.
SCIQA_OPEN_INSTRUCTION = (
    "You are a scientific reasoning expert. You will be given a science task. "
    "Reason about it step by step, then give the final answer. In the end, "
    'write "Final answer:" followed by your complete final answer, in the '
    "format the task asks for."
)
# Olmo-3-7B averages 4,767 tokens on a LiveCodeBench attempt and runs into the
# 8,192-token cap on 39% of them, almost all of it spent narrating the reasoning
# before the code. The cap stays where it is, so the prompt is what gives: this
# variant keeps every requirement of BASE_INSTRUCTION byte for byte and replaces
# only the invitation to outline a thought process with a bound on it.
#
# Deleting that sentence and putting nothing in its place was measured first,
# on the same 262 best-of-k trials the formal run spends: truncation went from
# 40.1% to 24.8% and the median attempt from 5,128 tokens to 2,055, which still
# extrapolated to more wall clock than the schedule allows. The bound is back.
#
# Which (model, benchmark) cells use it is config rather than code, and every
# method resolves it the same way, so a cell's prompt does not depend on which
# condition is generating for it.
LCB_BRIEF_INSTRUCTION = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and "
    "passes all tests. The time limit is 1 second. Keep any analysis brief: at "
    "most a few sentences naming the approach, with no worked examples and no "
    "step-by-step traces. In the end, please provide the complete code in a "
    "code block enclosed with ``` ```."
)

# LiveCodeBench's own chat prompt, the messages its OpenAIChat / LLaMa3 styles
# send (lcb_runner/prompts/code_generation.py: PromptConstants and
# get_generic_question_template_answer). The strings are copied verbatim so that
# building the prompt needs neither LCB on sys.path nor the few-shot files LCB
# opens relative to cwd at import; tests pin the copies to the originals.
LCB_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
LCB_FORMAT_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
LCB_FORMAT_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)
# The one line added to LCB's Format section. It asks for <think> tags even
# though Olmo-3-7B-Instruct is not a reasoning model -- deliberately.
LCB_THINK_INSTRUCTION = (
    "Provide CONCISE reasoning on how to arrive at the answer in the <think> "
    "</think> tag."
)


def lcb_think_messages(question_content: str, starter_code: str) -> list[dict[str, str]]:
    """LCB's generic chat prompt with a concise-<think> line opening its Format.

    ``question_content`` and ``starter_code`` are LCB's raw fields, not the
    frozen ``problem`` text: for call-based problems that text appends a
    "Your solution should have the following signature" line instead of LCB's
    ``class Solution`` starter code.
    """
    if starter_code:
        fmt = f"{LCB_FORMAT_WITH_STARTER_CODE}\n```python\n{starter_code}\n```\n\n"
    else:
        fmt = f"{LCB_FORMAT_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n"
    user = (
        f"### Question:\n{question_content}\n\n"
        f"### Format:\n{LCB_THINK_INSTRUCTION}\n{fmt}"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    return [
        {"role": "system", "content": LCB_SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": user},
    ]


SCIQA_FORMAT = {
    "mcq": "The final answer must be the letter of the correct option.",
    "yes_no": 'The final answer must be exactly "Yes" or "No".',
    "filling": "The final answer must be the completed expression only.",
}

IMPROVE_INSTRUCTION = (
    "Use this execution feedback to improve your answer. Re-check the original "
    "problem yourself and produce a corrected complete answer."
)
JUDGED_IMPROVE_INSTRUCTION = (
    "Use this outcome to improve your answer. Re-check the original problem "
    "yourself and produce a corrected complete answer."
)
# Ablation "w/o last response" (wo_last_response): the
# feedback reaches the model without the answer it is about, so it is worded
# for an attempt that is not shown, and it shares the opening user turn.
UNSEEN_IMPROVE_INSTRUCTION = (
    "Use this execution feedback to write an improved answer. Re-check the "
    "original problem yourself and produce a corrected complete answer."
)
UNSEEN_JUDGED_IMPROVE_INSTRUCTION = (
    "Use this outcome to write an improved answer. Re-check the original "
    "problem yourself and produce a corrected complete answer."
)


@functools.cache
def variant_instruction_cells() -> frozenset[tuple[str, str]]:
    """The (model, benchmark) cells that open on a variant of the shared prompt.

    The name says nothing about the wording on purpose: which variant a listed
    cell gets is the constant above, and that has already been rewritten once.

    Read from the config so the set is an experiment setting, not a code edit,
    and cached because every prompt built during a rollout consults it.
    """
    from leaf.core.project_config import load_config

    cells = load_config().get("variant_instruction_cells") or {}
    return frozenset(
        (model, benchmark)
        for model, benchmarks in cells.items()
        for benchmark in benchmarks
    )


def wants_variant_instruction(row: dict, model: str | None) -> bool:
    return bool(model) and (model, row["source"]) in variant_instruction_cells()


def base_instruction(row: dict, model: str | None = None) -> str:
    """The task-appropriate opening instruction (BASE_INSTRUCTION for LiveCodeBench).

    ``model`` is optional: without it every cell gets the shared instruction,
    which is what the cells outside :func:`variant_instruction_cells` use anyway.
    """
    task_type = row["task_type"]
    if task_type == "code_lcb":
        if wants_variant_instruction(row, model):
            return LCB_BRIEF_INSTRUCTION
        return BASE_INSTRUCTION
    if task_type == "code_mbpp":
        return MBPP_INSTRUCTION
    if task_type == "math":
        return MATH_INSTRUCTION
    if task_type == "sciqa":
        answer_type = row["verifier"].get("answer_type", "mcq")
        if answer_type == "open":
            return SCIQA_OPEN_INSTRUCTION
        return f"{SCIQA_INSTRUCTION} {SCIQA_FORMAT[answer_type]}"
    raise ValueError(f"Unknown task_type: {task_type}")


def problem_prompt(row: dict, model: str | None = None) -> str:
    """Opening user message: instruction + the frozen problem body."""
    return f"{base_instruction(row, model)}\n\n{row['problem']}"


def skill_prompt(row: dict, notes: str, model: str | None = None) -> str:
    """Same opening message with accumulated notes injected before the problem."""
    if not notes:
        return problem_prompt(row, model)
    return (
        f"{base_instruction(row, model)}\n\n"
        "Experience distilled from earlier attempts:\n"
        f"<skill>\n{notes}\n</skill>\n\n"
        f"{row['problem']}"
    )


def opening_messages(
    row: dict, model: str | None = None, notes: str = ""
) -> list[dict[str, str]]:
    """The opening turn(s) of an attempt, as chat messages.

    An Olmo row carries its prompt frozen in ``prompt_system`` /
    ``prompt_user`` (the OLMES template, see :mod:`leaf.prompts.olmes_prompts`); the
    system message is present only where that template has one (LiveCodeBench).
    Accumulated notes go in front of the user message there, since the OLMES
    templates put their instructions on both sides of the problem. Rows without
    those fields keep :func:`skill_prompt` byte for byte.
    """
    if "prompt_user" not in row:
        return [{"role": "user", "content": skill_prompt(row, notes, model)}]
    user = row["prompt_user"]
    if notes:
        user = (
            "Experience distilled from earlier attempts:\n"
            f"<skill>\n{notes}\n</skill>\n\n{user}"
        )
    messages = []
    if row.get("prompt_system"):
        messages.append({"role": "system", "content": row["prompt_system"]})
    messages.append({"role": "user", "content": user})
    return messages


def opening_user_text(row: dict, model: str | None = None) -> str:
    """The user turn of :func:`opening_messages`, e.g. for the SEED analyzer."""
    return opening_messages(row, model)[-1]["content"]


def correction_instruction(row: dict) -> str:
    """The blind self-correction turn, worded for this problem's task type."""
    task_type = row["task_type"]
    if task_type == "math":
        return MATH_CORRECTION_INSTRUCTION
    if task_type == "sciqa":
        return SCIQA_CORRECTION_INSTRUCTION
    return CORRECTION_INSTRUCTION


def has_execution_feedback(row: dict) -> bool:
    """True when the public verifier can say more than correct / incorrect."""
    return row["task_type"] in {"code_lcb", "code_mbpp"}


def feedback_section(row: dict, grade: dict) -> str:
    """Render one public-verifier verdict as a user-visible feedback block.

    Code tasks get the executor's own report. MATH500 and SciKnowEval have no
    execution environment, so they get the verdict plus the answer that was
    extracted from the response — never the reference answer.
    """
    correct = bool(grade["correct"])
    if has_execution_feedback(row):
        verdict = (
            "Your previous answer passed every public test."
            if correct
            else "Your previous answer failed the public tests."
        )
        native = str(grade.get("native_feedback", "")).strip()
        if not native:
            native = (
                "No failing test details were produced."
                if correct
                else "No failing test details are available."
            )
        return (
            "Your previous answer was executed against the public tests.\n\n"
            f"{verdict}\n\n{native}"
        )
    verdict = (
        "Your previous answer was judged correct."
        if correct
        else "Your previous answer was judged incorrect."
    )
    if row["verifier"].get("kind") == "sciqa_judge":
        # An LLM judge compared the whole answer with the reference; there is
        # no extracted answer to quote, and the judge's reasoning would leak
        # the reference, so the verdict is all the feedback says.
        return f"Your previous answer was checked against the reference.\n\n{verdict}"
    if grade.get("hide_answer"):
        # LEAF on MATH: the verdict only, no extracted answer quoted back
        return f"Your previous answer was checked against the reference.\n\n{verdict}"
    extracted = str(grade.get("extracted_answer", "") or "").strip()
    if grade.get("extracted_answer_too_long"):
        # math_answer_max_tokens: a degenerate answer is not quoted back.
        detail = (
            "No valid final answer could be read from your response (the extracted "
            f"answer was {int(grade['extracted_answer_tokens'])} tokens long)."
        )
    elif extracted:
        detail = f"The final answer read from your response was: {extracted}"
    else:
        detail = "No final answer could be read from your response."
    return f"Your previous answer was checked against the reference.\n\n{verdict}\n\n{detail}"


def improve_prompt(row: dict, grade: dict) -> str:
    """The feedback-carrying correction turn of a carrying step."""
    instruction = (
        IMPROVE_INSTRUCTION
        if has_execution_feedback(row)
        else JUDGED_IMPROVE_INSTRUCTION
    )
    return f"{feedback_section(row, grade)}\n\n{instruction}"


def unseen_feedback_section(row: dict, grade: dict) -> str:
    """:func:`feedback_section` for an attempt the model is not shown (w/o last response).

    The same verdict, executor report and answer-quoting rules, worded so that
    nothing points at an answer above it in the conversation.
    """
    correct = bool(grade["correct"])
    subject = "Your previous answer to this problem (not shown here)"
    if has_execution_feedback(row):
        verdict = "It passed every public test." if correct else "It failed the public tests."
        native = str(grade.get("native_feedback", "")).strip()
        if not native:
            native = (
                "No failing test details were produced."
                if correct
                else "No failing test details are available."
            )
        return f"{subject} was executed against the public tests.\n\n{verdict}\n\n{native}"
    verdict = "It was judged correct." if correct else "It was judged incorrect."
    head = f"{subject} was checked against the reference.\n\n{verdict}"
    if row["verifier"].get("kind") == "sciqa_judge" or grade.get("hide_answer"):
        return head
    extracted = str(grade.get("extracted_answer", "") or "").strip()
    if grade.get("extracted_answer_too_long"):
        detail = (
            "No valid final answer could be read from it (the extracted answer was "
            f"{int(grade['extracted_answer_tokens'])} tokens long)."
        )
    elif extracted:
        detail = f"The final answer read from it was: {extracted}"
    else:
        detail = "No final answer could be read from it."
    return f"{head}\n\n{detail}"


def unseen_improve_prompt(row: dict, grade: dict) -> str:
    """The feedback turn of a carrying step that does not show the previous answer."""
    instruction = (
        UNSEEN_IMPROVE_INSTRUCTION
        if has_execution_feedback(row)
        else UNSEEN_JUDGED_IMPROVE_INSTRUCTION
    )
    return f"{unseen_feedback_section(row, grade)}\n\n{instruction}"


def correction_instruction_for_task(task_type: str) -> str:
    return correction_instruction({"task_type": task_type})


def correction_instruction_for_prompt(prompt_text: str) -> str:
    """Task-appropriate correction turn recovered from a rendered prompt.

    Used when the caller has only the rendered messages; the SCoRe agent loop
    prefers the row's ``extra_info.task_type`` (the OLMES templates do not
    tell sciqa filling apart from math). A prompt that matches none of them
    falls back to the code wording.
    """
    text = prompt_text or ""
    if MATH_INSTRUCTION[:40] in text:
        return MATH_CORRECTION_INSTRUCTION
    if SCIQA_INSTRUCTION[:40] in text:
        return SCIQA_CORRECTION_INSTRUCTION
    return CORRECTION_INSTRUCTION
