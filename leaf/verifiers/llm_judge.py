"""LLM-judge verifier for the open-answer SciKnowEval problems.

The judge is each experiment's own base model **without LoRA** (Qwen3-8B for
Qwen runs, Olmo-3-7B-Instruct for Olmo runs), decoding greedily on the engine
that generates the rollouts. It reads the task, the reference answer and the
response, and ends on ``\\boxed{yes}`` or ``\\boxed{no}``; anything else counts
as no.

Where the judge runs (always inline, on the GPU, right after a response is
generated), and where its verdict is stored:

========================  ===================================================
LEAF rollout              ``leaf.explore.leaf_rollout.AsyncRollout`` -- same AsyncLLM
                          engine; ``final["judge"]``
LEAF + interleaved update ``leaf.explore.server_rollout.ServerRollout`` -- the
                          verl vLLM server with ``use_lora=False``;
                          ``final["judge"]``
Random Sampling / SEED    offline ``vllm.LLM`` after each chunk
                          (:func:`judge_records_offline`); ``record["judge"]``
verl (SDPO, GRPO, SCoRe,  a judge step in the fork's ``fit`` loop
SEED-RL)                  (``RayPPOTrainer._judge_phase``) through
                          ``AgentLoopManager.generate_raw(use_lora=False)``;
                          ``extra_info`` -> reward -> ``judge_*`` rollout keys
========================  ===================================================

The verifier route ``sciqa_judge`` (:mod:`leaf.verifiers.answer_grader`) only reads
``record["judge"]["verdict"]``; it never judges. So the offline grading jobs
reproduce the inline verdicts on a CPU node, and a record without an inline
verdict is a hard error, not a silent "incorrect".
"""

from __future__ import annotations

import re

KIND = "sciqa_judge"
# The request phase number the LEAF rollout drivers give a judge request (0-29 are
# solution turns, 30 is compression).
JUDGE_PHASE = 40
JUDGE_SEED_OFFSET = 700_000_000
MAX_TOKENS = 512
# A response longer than this many characters is shown as its head and tail:
# the final answer sits at the end, the approach at the start.
RESPONSE_MAX_CHARS = 12_000
RESPONSE_HEAD_CHARS = 3_000
THINK_CLOSE = "</think>"

CRITERIA = {
    "relation_set": (
        "This is a relation extraction task. The response is correct only if its "
        "final list contains the same set of relations as the reference: every "
        "relation in the reference appears in the response, and the response adds "
        "no relation that the reference does not contain. Differences in case, "
        "formatting, ordering, synonyms or abbreviations of the same entity do not "
        "matter. If the reference states that there is no such relation, the "
        "response must state the same."
    ),
    "composition": (
        "This is a composition extraction task. The response is correct only if it "
        "lists the same elements as the reference with the same composition values "
        "or ranges; table formatting and notation may differ."
    ),
    "summary": (
        "This is a summarization task. The response is correct if its summary "
        "states the same main findings and conclusions as the reference. Wording "
        "may differ, but missing a central finding of the reference or stating "
        "something that contradicts it makes the response incorrect."
    ),
    "key_content": (
        "This is an experimental design task. The response is correct if it covers "
        "the key content of the reference -- its essential materials, reagents, "
        "equipment or procedure steps -- without contradicting it. Wording, "
        "ordering and additional reasonable details do not matter, but omitting "
        "several essential items of the reference makes the response incorrect."
    ),
    "derivation": (
        "This is a derivation task. The response is correct if it reaches the "
        "target result through a valid derivation that contains the key steps of "
        "the reference. A chain of equations that goes from the given relations "
        "to the target result counts as a derivation even without prose. A "
        "derivation that does not reach the target result, or that relies on an "
        "invalid step, is incorrect."
    ),
    "final_result": (
        "The response is correct if its final result is equivalent to the "
        "reference answer: the same formula, value, expression, structure or "
        "material, allowing equivalent notation, units and rounding. The reasoning "
        "does not need to match the reference. If the reference answers several "
        "parts, every part must match."
    ),
}

TEMPLATE = """You are an expert grader for science tasks. Decide whether the response to the task below is correct, using the reference answer as the ground truth.

### Task
{question}

### Reference answer
{reference}

### Response
{response}

### Grading instructions
{criterion}
Judge the final answer the response gives; its style and length do not matter. Be strict but fair. A response that gives no final answer, hedges between several answers, or is cut off before answering is incorrect.

Explain your judgment in at most three sentences, then end with \\boxed{{yes}} if the response is correct or \\boxed{{no}} if it is not."""

_VERDICT = re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?(yes|no)\s*\}?\s*\}", re.IGNORECASE)


def needs_judge(payload: dict | None) -> bool:
    return bool(payload) and payload.get("kind") == KIND


def response_view(response: str) -> tuple[str, dict]:
    """What the judge sees of a response: after ``</think>``, head + tail if long."""
    text = str(response or "")
    info = {"response_chars": len(text), "think_stripped": False, "truncated": False}
    if THINK_CLOSE in text:
        after = text.rsplit(THINK_CLOSE, 1)[1]
        if after.strip():
            text = after
            info["think_stripped"] = True
    text = text.strip()
    if len(text) > RESPONSE_MAX_CHARS:
        tail = RESPONSE_MAX_CHARS - RESPONSE_HEAD_CHARS
        omitted = len(text) - RESPONSE_MAX_CHARS
        text = (f"{text[:RESPONSE_HEAD_CHARS]}\n\n[... {omitted} characters omitted ...]\n\n"
                f"{text[-tail:]}")
        info["truncated"] = True
    if not text:
        text = "(empty response)"
    return text, info


def judge_content(payload: dict, response: str) -> tuple[str, dict]:
    """The judge's user message for one response, and how the response was cut."""
    view, info = response_view(response)
    content = TEMPLATE.format(
        question=str(payload["question"]).strip(),
        reference=str(payload["reference"]).strip(),
        response=view,
        criterion=CRITERIA[payload["criterion"]],
    )
    return content, info


def judge_messages(payload: dict, response: str) -> tuple[list[dict], dict]:
    content, info = judge_content(payload, response)
    return [{"role": "user", "content": content}], info


def prompt_ids(tokenizer, payload: dict, response: str,
               template_kwargs: dict | None = None) -> tuple[list[int], str, dict]:
    messages, info = judge_messages(payload, response)
    kwargs = {"enable_thinking": False} if template_kwargs is None else dict(template_kwargs)
    ids = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs))
    return ids, messages[0]["content"], info


def parse_verdict(text: str) -> bool | None:
    """The last ``\\boxed{yes|no}`` of the judge output; ``None`` if there is none."""
    found = _VERDICT.findall(str(text or ""))
    if not found:
        return None
    return found[-1].lower() == "yes"


def sampling_params(config: dict | None = None, seed: int = 0) -> dict:
    """Greedy decoding with a short cap (plain dict; each engine wraps it)."""
    max_tokens = int((config or {}).get("judge_max_tokens", MAX_TOKENS))
    return {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "max_tokens": max_tokens, "seed": int(seed)}


def verdict_record(*, prompt: str, output: str, info: dict, prompt_tokens: int,
                   completion_tokens: int, finish_reason: str | None,
                   seconds: float | None = None) -> dict:
    """What every condition stores beside a judged response."""
    parsed = parse_verdict(output)
    record = {
        "verdict": bool(parsed),
        "parsed": parsed is not None,
        "output": output,
        "prompt": prompt,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "finish_reason": finish_reason,
        **info,
    }
    if seconds is not None:
        record["seconds"] = float(seconds)
    return record


def judge_records_offline(llm, tokenizer, records: list[dict], config: dict,
                          seed_base: int = JUDGE_SEED_OFFSET) -> float:
    """Judge every ``sciqa_judge`` record in place with an offline ``vllm.LLM``.

    Used by best-of-k and SEED collection right after a chunk is generated, on
    the same (base-weight) engine. Returns the seconds spent.
    """
    import time

    from vllm import SamplingParams

    pending = [record for record in records if needs_judge(record.get("verifier"))]
    if not pending:
        return 0.0
    prompts, contents, infos = [], [], []
    for record in pending:
        ids, content, info = prompt_ids(tokenizer, record["verifier"], record["response"])
        prompts.append(ids)
        contents.append(content)
        infos.append(info)
    params = [
        SamplingParams(**sampling_params(config, seed_base + int(record.get("trial_index", 0))))
        for record in pending
    ]
    started = time.monotonic()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params,
                           use_tqdm=False)
    seconds = time.monotonic() - started
    for record, ids, content, info, output in zip(
            pending, prompts, contents, infos, outputs, strict=True):
        answer = output.outputs[0]
        record["judge"] = verdict_record(
            prompt=content, output=answer.text, info=info, prompt_tokens=len(ids),
            completion_tokens=len(answer.token_ids), finish_reason=answer.finish_reason,
        )
    return seconds


def reward_fields(judge: dict | None, prefix: str = "judge") -> dict:
    """The judge columns a verl reward result carries into the rollout logs."""
    judge = judge or {}
    return {
        f"{prefix}_verdict": int(bool(judge.get("verdict"))) if judge else -1,
        f"{prefix}_parsed": int(bool(judge.get("parsed"))) if judge else -1,
        f"{prefix}_output": str(judge.get("output", "")),
        f"{prefix}_prompt": str(judge.get("prompt", "")),
    }
