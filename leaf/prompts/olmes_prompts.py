"""Olmo-3-7B-Instruct opening prompts: the OLMES ``::olmo3:adapt`` task templates.

The shared instruction family (leaf.prompts.task_prompts) invites the model to
outline its thought process and never asks it to be brief; with it Olmo's
attempts ran long and 39% of its LiveCodeBench attempts hit the cap. Olmo
therefore opens each benchmark on the template OLMES evaluates Olmo 3
post-trained models with (allenai/olmes
``5a51f502``, suite ``olmo3:adapt``). Each benchmark maps to its template as
follows:

| benchmark / type        | OLMES template                          | answer format                  |
|-------------------------|-----------------------------------------|--------------------------------|
| MATH500                 | ``minerva_math_{subject}::olmo3:adapt`` | last ``\\boxed{}``              |
| LiveCodeBench           | ``livecodebench_codegeneration::olmo3:adapt`` (system + user) | code block |
| MBPP                    | ``mbppplus::olmo3:adapt``               | ``Here is the completed function:`` block |
| SciKnowEval mcq         | ``gpqa::olmo3:adapt``                   | ``Therefore, the answer is (X)`` |
| SciKnowEval yes/no      | ``gpqa::olmo3:adapt`` with (Yes)/(No)   | ``Therefore, the answer is (Yes)`` |
| SciKnowEval filling     | MATH's ``final_description``            | last ``\\boxed{}``              |
| SciKnowEval open        | yes/no shape, ``Final answer:`` line    | LLM judge (leaf.verifiers.llm_judge) |

The prompts are frozen into the Olmo dataset rows (``prompt_system`` /
``prompt_user``) by ``leaf/data/prepare_olmes_test_sets.py`` rather than rebuilt at
rollout time: LiveCodeBench and MBPP need raw upstream fields the frozen rows do
not keep, and a frozen string is the same for every condition by construction.
"""

from __future__ import annotations

import re

from leaf.prompts.task_prompts import lcb_think_messages

# The value every OLMES-prompt verifier payload carries in ``answer_format``; the
# verifier dispatches its extraction on it, and rows without it keep the default one.
ANSWER_FORMAT = "olmes"

# minerva_math::olmo3:adapt, context_kwargs.final_description.
MATH_FINAL_DESCRIPTION = "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

# mbppplus::olmo3:adapt, context_kwargs.template with {{code_prompt}} filled in.
MBPP_TEMPLATE = (
    "{code_prompt}\n\nProvide CONCISE reasoning on how to arrive at the answer, "
    "and make sure to finish the response with the following:\n\n"
    "Here is the completed function:\n\n```python\n(CODE)\n```\n"
    "where (CODE) is the code for the complete function."
)


def _letters(labels: list[str]) -> str:
    return ", ".join(f"({label})" for label in labels)


def gpqa_description(labels: list[str]) -> str:
    """gpqa::olmo3:adapt ``description``; the letter list follows the options."""
    return (
        "Answer the following multiple-choice question by giving the correct "
        "answer letter in parentheses. Provide CONCISE reasoning for the answer, "
        'and make sure to finish the response with "Therefore, the answer is '
        f'(ANSWER_LETTER)" where (ANSWER_LETTER) is one of {_letters(labels)}.\n\n'
    )


def gpqa_final_description(labels: list[str]) -> str:
    return (
        "\n\nAnswer the above question and REMEMBER to finish your response with "
        'the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where '
        f"(ANSWER_LETTER) is one of {_letters(labels)}."
    )


YES_NO_DESCRIPTION = (
    "Answer the following question with Yes or No. Provide CONCISE reasoning for "
    'the answer, and make sure to finish the response with "Therefore, the answer '
    'is (ANSWER)" where (ANSWER) is one of (Yes), (No).\n\n'
)
YES_NO_FINAL_DESCRIPTION = (
    "\n\nAnswer the above question and REMEMBER to finish your response with the "
    'exact phrase "Therefore, the answer is (ANSWER)" where (ANSWER) is one of '
    "(Yes), (No)."
)

# Open-answer SciKnowEval problems (leaf.data.sciknoweval_pool): the yes/no
# template's shape with the answer phrase replaced by a "Final answer:" line the
# LLM judge (leaf.verifiers.llm_judge) reads; the task's own instruction states the
# answer's format.
OPEN_DESCRIPTION = (
    "Answer the following science task. Provide CONCISE reasoning for the answer, "
    'and make sure to finish the response with "Final answer:" followed by your '
    "complete answer in the format the task asks for.\n\n"
)
OPEN_FINAL_DESCRIPTION = (
    "\n\nAnswer the above task and REMEMBER to finish your response with "
    '"Final answer:" followed by your complete answer.'
)


def sciqa_open_user(problem: str) -> str:
    return f"{OPEN_DESCRIPTION}Task: {problem.strip()}{OPEN_FINAL_DESCRIPTION}"


_OPTION_LINE = re.compile(r"^([A-Z])\. (.*)$")


def split_sciknoweval_mcq(problem: str, labels: list[str]) -> tuple[str, list[str]]:
    """Question and option texts from SciKnowEval's official mcq rendering.

    The frozen text is ``question`` + ``\\n\\nOptions:\\n`` + one ``X. text``
    line per option; anything else is a hard error rather than a guess.
    """
    question, sep, options = problem.rpartition("\n\nOptions:\n")
    if not sep:
        raise ValueError("SciKnowEval mcq has no Options block")
    parsed = [_OPTION_LINE.match(line) for line in options.split("\n")]
    if not all(parsed) or [m.group(1) for m in parsed] != list(labels):
        raise ValueError(f"SciKnowEval options do not match labels {labels}")
    return question, [m.group(2) for m in parsed]


def gpqa_user(question: str, labels: list[str], choices: list[str]) -> str:
    """gpqa.py ``_process_doc`` query between the description and final line."""
    query = f"Question: {question}\nChoices:\n"
    query += "".join(f" ({label}) {choice}\n" for label, choice in zip(labels, choices, strict=True))
    return f"{gpqa_description(labels)}{query}{gpqa_final_description(labels)}"


def mbpp_header(code: str, entry_point: str) -> str:
    """The OLMES MBPP task definition's ``code.split(":")[0] + ":"`` on the reference program.

    Twenty MBPP references define something else first -- a helper class
    (tasks 342, 367) or a helper function (e.g. task 152's ``merge`` before
    ``merge_sort``) -- so that header would not name the function the tests
    call. Those take the reference line that defines the entry point instead.
    """
    code = code.replace("\r\n", "\n")
    header = code.split(":")[0] + ":"
    if re.search(rf"^def {re.escape(entry_point)}\b", header, re.MULTILINE):
        return header
    match = re.search(rf"^def {re.escape(entry_point)}\s*\(.*$", code, re.MULTILINE)
    if not match:
        raise ValueError(f"No def {entry_point} line in the reference code")
    line = match.group(0)
    return line.split(":")[0] + ":"


def mbpp_tests_block(test_list: list[str], setup: str = "") -> str:
    """MBPP's graded asserts, worded as the default prompt gives them (``mbpp_problem_text``).

    The asserts are put back into the Olmo MBPP prompt so every
    condition sees the tests it is graded on -- not only the ones whose verifier
    feedback quotes them. Task 367's asserts need its setup code, which follows.
    """
    block = "\n\nYour code should pass these tests:\n\n" + "\n".join(test_list)
    setup = setup.replace("\r\n", "\n").strip()
    if setup:
        block += f"\n\nThe tests run after this setup code:\n\n{setup}"
    return block


def mbpp_user(
    text: str, code: str, entry_point: str,
    test_list: list[str] | None = None, setup: str = "",
) -> str:
    """mbppplus::olmo3:adapt on raw MBPP: description, newline, function header.

    With ``test_list`` the asserts follow the header, before the template's
    closing instruction; without it the prompt is the plain template.
    """
    code_prompt = f"{text.strip()}\n{mbpp_header(code, entry_point)}"
    if test_list:
        code_prompt += mbpp_tests_block(test_list, setup)
    return MBPP_TEMPLATE.format(code_prompt=code_prompt)


def math_user(problem: str) -> str:
    return f"{problem.strip()}{MATH_FINAL_DESCRIPTION}"


def build_prompt(
    row: dict, *, lcb_raw: dict[str, dict], mbpp_raw: dict[int, dict],
    mbpp_asserts: bool = False,
) -> dict:
    """``{"prompt_system": str | None, "prompt_user": str}`` for one frozen row.

    ``mbpp_asserts`` (config ``mbpp_prompt_asserts``) adds MBPP's
    three graded asserts to the MBPP prompt; every other benchmark is unchanged.
    """
    task_type = row["task_type"]
    system = None
    if task_type == "math":
        user = math_user(row["problem"])
    elif task_type == "code_lcb":
        raw = lcb_raw[row["meta"]["question_id"]]
        system_message, user_message = lcb_think_messages(
            raw["question_content"], raw.get("starter_code") or ""
        )
        system, user = system_message["content"], user_message["content"]
    elif task_type == "code_mbpp":
        raw = mbpp_raw[int(row["meta"]["task_id"])]
        payload = row["verifier"]
        if mbpp_asserts:
            user = mbpp_user(
                raw["text"], raw["code"], payload["entry_point"],
                test_list=list(payload["test_list"]),
                setup=str(payload.get("test_setup_code") or ""),
            )
        else:
            user = mbpp_user(raw["text"], raw["code"], payload["entry_point"])
    elif task_type == "sciqa":
        payload = row["verifier"]
        answer_type = payload["answer_type"]
        if answer_type == "mcq":
            labels = list(payload["choice_labels"])
            question, choices = split_sciknoweval_mcq(row["problem"], labels)
            user = gpqa_user(question, labels, choices)
        elif answer_type == "yes_no":
            user = f"{YES_NO_DESCRIPTION}Question: {row['problem']}\n{YES_NO_FINAL_DESCRIPTION}"
        elif answer_type == "filling":
            user = f"{row['problem']}{MATH_FINAL_DESCRIPTION}"
        elif answer_type == "open":
            user = sciqa_open_user(row["problem"])
        else:
            raise ValueError(f"Unknown SciKnowEval answer_type: {answer_type}")
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return {"prompt_system": system, "prompt_user": user}
