"""LEAF exploration rollout driver: one asyncio task per dataset sample.

A synchronous rollout would make every turn one ``llm.generate`` barrier across
all problems, with the public verifier running between those barriers, so the
GPU would idle while the CPU grades. This driver has neither barrier:

* the engine is the vLLM V1 :class:`~vllm.v1.engine.async_llm.AsyncLLM`, driven
  through one ``asyncio.Task`` per dataset sample, each running its own full
  step loop ``1..N``; samples are free to sit on different steps at the same
  time, so a long-tail sample never stalls the rest;
* the public verifier runs off the event loop through ``run_in_executor``, so
  one sample's verification overlaps every other sample's generation.

The conditions update no parameters and are dispatched from the data-driven
:data:`SPECS` table. A step either restarts (plain prompt -> initial solution ->
one blind correction, whose answer is the step's final) or carries (plain prompt
+ the previous step's final + its public-verifier feedback -> one correction);
restarts are every 5 steps (1, 6, 11, 16, 21).

History compression (the ``compress_*`` conditions) changes only the text a later
turn is shown in place of a previous response (``carry_transform``, see
:meth:`AsyncRollout._carry_text`): the solution carried into the next step and
the initial solution a restart step's correction turn reads.

| condition                   | carry_transform  | previous response shown when > 2,000 tokens |
|-----------------------------|------------------|------------------------------------------------|
| ``compress_none``           | (none)           | the whole response |
| ``compress_llm_summary``    | ``llm_summary``  | the model's own <= 1,000-token summary (a second turn of the same conversation) |
| ``compress_tail``           | ``tail``         | its last 2,000 tokens |
| ``compress_head_tail``      | ``head_tail``    | its first 1,000 + last 1,000 tokens |
| ``compress_summary_block``  | ``summary_block``| always: its own <summary> block (<= 500 tokens) + final code / final answer |
| ``compress_lowest_logprob`` | ``logprob``      | the 1,000 tokens with the lowest sampled log-prob, in original order |

:data:`LEAF` (``leaf``) is ``compress_tail`` with the quoted MATH answer capped at
``math_answer_max_tokens``, every generation's prompt messages recorded (the policy
update rebuilds its contexts from them) and MATH feedback stating only the verdict.
The exploration ablations change one thing of LEAF each (restart steps are never
changed):

| condition             | spec key                        | carry step                                         |
|-----------------------|---------------------------------|----------------------------------------------------|
| ``wo_last_response``  | ``show_previous_response=False``| one user turn: opening + feedback on the unseen attempt |
| ``wo_feedback``       | ``show_feedback=False``         | previous final + the blind correction instruction (verifier still runs) |
| ``restart_every_2`` / ``restart_every_10`` / ``no_restart`` | ``restart_interval`` 2 / 10 / 0 | (0: only step 1 restarts) |
| ``last_3_rounds``     | ``history_window=3``            | the last <= 3 rounds since the restart (final + feedback each); only the last user turn asks to improve |
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import inspect
import json
import os
import re
import time
from pathlib import Path

from leaf.core.request_seeds import GpuMemoryMonitor, read_jsonl, request_seed
from leaf.core.io_utils import atomic_json, atomic_jsonl, json_line
from leaf.prompts.task_prompts import (
    correction_instruction,
    feedback_section,
    improve_prompt,
    opening_messages,
    unseen_improve_prompt,
)
from leaf import paths
from leaf.core.project_config import CONFIG_PATH

PROJECT = Path(__file__).resolve().parents[2]
# The models the config knows about, so adding one there is enough.
MODELS = tuple(paths.load_json(CONFIG_PATH)["models"])

# History-compression ablation (see the module docstring).
SPECS = {
    "compress_none": {
        "carry": "solution_feedback", "restart_interval": 5,
        "restart_corrections": 1, "carry_corrections": 1,
    },
}
_BASE_SPEC = SPECS["compress_none"]
SPECS.update({
    "compress_llm_summary": {**_BASE_SPEC, "carry_transform": "llm_summary"},
    "compress_tail": {**_BASE_SPEC, "carry_transform": "tail"},
    "compress_head_tail": {**_BASE_SPEC, "carry_transform": "head_tail"},
    "compress_summary_block": {**_BASE_SPEC, "carry_transform": "summary_block"},
    "compress_lowest_logprob": {**_BASE_SPEC, "carry_transform": "logprob"},
})
# LEAF: the tail compression; MATH feedback quotes the extracted answer only when
# it is at most ``math_answer_max_tokens`` tokens long (a degenerate answer is
# reported by its length instead) -- and with ``math_hide_answer`` it states only the
# verdict --, and every generation records its prompt messages so the policy update
# can rebuild the privileged context.
LEAF = "leaf"
SPECS[LEAF] = {**SPECS["compress_tail"], "math_answer_max_tokens": 100, "record_messages": True,
               "math_hide_answer": True}
# Exploration ablations of LEAF (see the module docstring)
SPECS.update({
    "wo_last_response": {**SPECS[LEAF], "show_previous_response": False},
    "wo_feedback": {**SPECS[LEAF], "show_feedback": False},
    "restart_every_2": {**SPECS[LEAF], "restart_interval": 2},
    "restart_every_10": {**SPECS[LEAF], "restart_interval": 10},
    "no_restart": {**SPECS[LEAF], "restart_interval": 0},
    "last_3_rounds": {**SPECS[LEAF], "history_window": 3},
})
CONDITIONS = tuple(SPECS)
# Carry summary request (compress_llm_summary) for the answer of solution turn t: request slot 50 + t
# (judge is 40), so its seed and request id never collide with another request.
CARRY_SUMMARY_PHASE = 50
# Compression lengths, in tokens of the response being carried.
CARRY_LONG_TOKENS = 2000        # llm_summary / tail / head_tail / logprob leave shorter responses as is
CARRY_SUMMARY_TOKENS = 1000     # llm_summary cap
CARRY_HEAD_TAIL_TOKENS = 1000   # head_tail: head and tail each
CARRY_LOGPROB_KEEP = 1000       # logprob: kept tokens
SUMMARY_BLOCK_TOKENS = 500      # summary_block: <summary> cap
CARRY_SUMMARY_INSTRUCTION = (
    "Summarize your previous response in at most 1000 tokens. Keep the key "
    "intermediate steps and results (including the final code or final answer "
    "if there is one) and remove redundant sentences. Output only the summary "
    "itself."
)
SUMMARY_BLOCK_INSTRUCTION = (
    "At the very end of your response, after your final answer, add a "
    "<summary>...</summary> block of at most 500 tokens that summarizes the "
    "approaches you tried, the key results, and what you are still unsure about."
)
_SUMMARY_BLOCK = re.compile(r"<summary>(.*?)(?:</summary>|$)", re.DOTALL | re.IGNORECASE)


def strip_summary_block(response: str) -> str:
    """The response with its (last, possibly unterminated) <summary> block removed."""
    matches = list(_SUMMARY_BLOCK.finditer(response))
    if not matches:
        return response
    last = matches[-1]
    return (response[:last.start()] + response[last.end():]).rstrip()


def summary_block_text(response: str) -> str:
    matches = list(_SUMMARY_BLOCK.finditer(response))
    return matches[-1].group(1).strip() if matches else ""


def final_answer_text(response: str) -> str:
    """SciKnowEval: what follows the last "Final answer:" ("" when absent)."""
    index = response.lower().rfind("final answer:")
    return response[index:].strip() if index >= 0 else ""


def logprob_keep_indices(logprobs: list[float], keep: int) -> list[int]:
    """Positions of the ``keep`` least predictable tokens, in original order."""
    order = sorted(range(len(logprobs)), key=lambda i: (logprobs[i], i))
    return sorted(order[:keep])


def load_config() -> dict:
    return paths.load_json(CONFIG_PATH)


def benchmark_of(problems: list[dict]) -> str | None:
    """The single source a per-benchmark run covers, for per-benchmark caps."""
    sources = {problem["source"] for problem in problems}
    return next(iter(sources)) if len(sources) == 1 else None


def model_max_num_seqs(config: dict, model: str,
                       benchmark: str | None = None,
                       key: str = "max_num_seqs",
                       condition: str | None = None) -> int:
    """vLLM concurrency, sized per cell against what the KV cache holds.

    Per-token KV is only half the story: what the cache admits is
    `capacity / (tokens a resident sequence holds on average)`, and the second
    factor differs per benchmark and per condition. Llama-3.1 costs 128 KiB per
    token against Olmo-3's 512 KiB, but its MATH500 sequences under a carrying
    condition average 4,122 resident tokens, so only ~101 of the configured 256
    fit in the 414,528-token cache. The surplus does not queue for free: it
    preempts and recomputes, and it leaves no spare blocks for the prefix cache
    that makes the correction turns cheap. The comparable Qwen3-8B cell ran 128
    against 237 that fit and reached 1.9x the token throughput.

    ``key`` selects which pair of config entries to read, so Random Sampling, which
    runs its own engine at its own utilization, sizes the same way off
    ``random_sampling_max_num_seqs`` / ``random_sampling_max_num_seqs_by_model``.
    """
    default = int(config[key])
    # a per-condition value measured by that condition's own smoke run.
    tuned = ((config.get(f"{key}_by_condition") or {}).get(model, {})
             .get(condition or "", {}).get(benchmark or ""))
    if tuned:
        return int(tuned)
    by_model = config.get(f"{key}_by_model") or {}
    value = by_model.get(model, default)
    if isinstance(value, dict):
        value = value.get(benchmark, value.get("default", default))
    return int(value)


def model_response_max_tokens(
    config: dict, model: str, benchmark: str | None = None
) -> int:
    """Generation cap, per model and with per-benchmark overrides.

    Every cell currently sits at the shared 8,192. Olmo's LiveCodeBench, which
    truncated 39% of its attempts there, is held under the cap from the prompt
    side instead -- see ``variant_instruction_cells`` in the config.
    """
    base = int(
        (config.get("response_max_tokens_by_model") or {}).get(
            model, config["response_max_tokens"]
        )
    )
    override = (config.get("response_max_tokens_overrides") or {}).get(model, {})
    return int(override.get(benchmark, base)) if benchmark else base


def model_max_model_len(config: dict, model: str) -> int:
    """Per-model context window.

    Olmo-3 writes roughly three times as much per turn as Qwen3, so a carrying
    step that inherits a previous solution plus feedback can approach the
    window. Each value stays within that model's own max_position_embeddings.
    """
    by_model = config.get("max_model_len_by_model") or {}
    return int(by_model.get(model, config["max_model_len"]))


def model_path(config: dict, model: str) -> Path:
    return Path(config["models"][model])


# --------------------------------------------------------------------------
# Condition semantics
# --------------------------------------------------------------------------
def is_restart(step: int, spec: dict) -> bool:
    """Steps that build their own first solution instead of inheriting one
    (``restart_interval`` 0: only step 1)."""
    if step == 1:
        return True
    interval = int(spec["restart_interval"])
    return bool(interval) and (step - 1) % interval == 0


def history_window(spec: dict) -> int:
    """Rounds (step final + its feedback) a carrying step is shown; 1 unless last_3_rounds."""
    return int(spec.get("history_window", 1))


def carry_history(previous: dict | None, round_: dict, *, restart: bool, window: int) -> list[dict]:
    """The rounds the next step is shown: this step's ``round_`` after the earlier
    rounds of the same restart segment (none when this step restarted), newest last."""
    earlier = [] if restart or not previous else list(previous.get("history") or [])
    return [*earlier, round_][-window:]


def seeded_messages(
    problem: dict, carry: dict, spec: dict, model: str | None = None
) -> list[dict[str, str]]:
    """Open a carrying step from what the previous step handed over."""
    opening = opening_messages(problem, model, carry.get("notes", ""))
    if not spec.get("show_previous_response", True):
        # wo_last_response: one user turn, the problem followed by the feedback
        if not spec.get("show_feedback", True):
            raise ValueError("a carry step with neither response nor feedback carries nothing")
        last = dict(opening[-1])
        last["content"] = f"{last['content']}\n\n{unseen_improve_prompt(problem, carry['grade'])}"
        return [*opening[:-1], last]
    if not spec.get("show_feedback", True):
        # wo_feedback: the previous final, then the blind restart-step correction turn
        return [
            *opening,
            {"role": "assistant", "content": carry["solution"]},
            {"role": "user", "content": correction_instruction(problem)},
        ]
    window = history_window(spec)
    if window > 1:
        # last_3_rounds: earlier rounds show their feedback only, the newest asks to improve
        rounds = carry["history"][-window:]
        messages = [*opening]
        for index, round_ in enumerate(rounds):
            newest = index == len(rounds) - 1
            messages.append({"role": "assistant", "content": round_["solution"]})
            messages.append({"role": "user", "content": (
                improve_prompt(problem, round_["grade"]) if newest
                else feedback_section(problem, round_["grade"]))})
        return messages
    return [
        *opening,
        {"role": "assistant", "content": carry["solution"]},
        {"role": "user", "content": improve_prompt(problem, carry["grade"])},
    ]


# --------------------------------------------------------------------------
# Chat templating across model families
# --------------------------------------------------------------------------
def thinking_template_kwargs(tokenizer) -> tuple[dict, str]:
    """Decide the chat-template kwargs once, and report what the template did.

    ``enable_thinking`` is a Qwen3 template variable: Qwen3-8B honours it (the
    rendering gains an empty ``<think>`` block), while Olmo-3-7B-Instruct's
    Jinja template silently ignores the unknown variable and renders
    byte-identically with and without it (it also always injects its own default
    system message). The flag is therefore passed unconditionally -- it is never
    dropped for Qwen3 -- and the probe only classifies the template so the
    manifest records which behaviour the run actually got.
    """
    kwargs = {"enable_thinking": False}
    probe = [{"role": "user", "content": "ping"}]
    try:
        off = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        on = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except Exception:  # a template that rejects the keyword outright
        return {}, "rejected_enable_thinking"
    if off != on:
        return kwargs, "enable_thinking_false_honoured"
    return kwargs, "enable_thinking_false_ignored_by_template"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def final_record(
    *,
    condition: str,
    step: int,
    problem_index: int,
    problem: dict,
    answer: dict,
    problem_count: int,
) -> dict:
    """One step's metric solution, shaped for ``answer_grader.grade_records``."""
    return {
        "key": f"leaf:{condition}:{step}:{problem['uid']}",
        "condition": condition,
        "mode": "leaf_rollout",
        "split": "train",
        "problem_uid": problem["uid"],
        "source": problem["source"],
        "task_type": problem["task_type"],
        "verifier": problem["verifier"],
        "problem_index": problem_index,
        "step": step,
        "trial_index": (step - 1) * problem_count + problem_index,
        "attempt_index": step - 1,
        "seed": answer["seed"],
        "solution_turn": answer["solution_turn"],
        "prompt_tokens": answer["prompt_tokens"],
        "completion_tokens": answer["completion_tokens"],
        "response": answer["response"],
        "finish_reason": answer["finish_reason"],
    }


def request_identifier(condition: str, step: int, problem_index: int, phase: int) -> str:
    return f"leaf:{condition}:s{step:04d}:i{problem_index:05d}:t{phase:02d}"


def verifier_worker_count(config: dict) -> int:
    """Size the verify executor from the config, capped by the cores we actually own.

    The jobs request 8-16 cores, so a configured 16 workers would otherwise
    starve the event loop that drives the engine client. One core is always left
    for the loop plus the vLLM client threads.
    """
    requested = int(config["rollout_verifier_workers"])
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count() or 1
    return max(1, min(requested, cores - 1))


def default_grade(records: list[dict], config: dict) -> list[dict]:
    """Grade through the shared public verifier.

    Imported lazily so that importing this driver (in tests, or on a CPU-only
    login node) does not require the verifier's own heavy dependencies.
    """
    from leaf.verifiers.answer_grader import grade_records

    return grade_records(records, config)


# --------------------------------------------------------------------------
# Per-sample append-only store and per-sample state
# --------------------------------------------------------------------------
def sample_record_path(output: Path, problem_index: int, uid: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:80]
    return output / "samples" / f"{problem_index:05d}_{slug}.jsonl"


def read_sample_records(path: Path) -> list[dict]:
    """Read one sample's append-only log, healing it if a write was torn.

    A chunk killed mid-append leaves an incomplete final line. Tolerating it
    only in memory is not enough: the next chunk appends after those bytes, so
    the torn text becomes an interior line and every later resume refuses the
    file, wedging the run for good. So drop unparseable lines, keep the last
    record for any step that appears twice, and rewrite the log compacted
    whenever anything was dropped.
    """
    if not path.exists():
        return []
    # split("\n"), not splitlines(): the latter also breaks on U+0085,
    # U+2028 and U+2029, which turns one valid record into fragments that
    # the loop below then discards as unparseable.
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    rows: list[dict] = []
    dropped = 0
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            dropped += 1
    if dropped:
        by_step: dict[int, dict] = {}
        for row in rows:
            by_step[int(row["step"])] = row
        rows = [by_step[step] for step in sorted(by_step)]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json_line(row) + "\n" for row in rows))
        temporary.replace(path)
        print(json.dumps({
            "event": "healed_sample_log", "path": str(path),
            "dropped_lines": dropped, "records_kept": len(rows),
        }), flush=True)
    return rows


def append_sample_record(path: Path, record: dict) -> None:
    """Append one step's record as a single write.

    A plain append is safe again now that the record can no longer contain a
    character the reader treats as a line break (`json_line` escapes U+0085,
    U+2028 and U+2029) and the reader splits on "\n" only. The read-modify-write
    that replaced it while the cause was still unknown re-serialised the whole
    log on every step, synchronously on the event loop, which at 1-2 MB per
    sample was a tax on every completed step for no benefit.

    A chunk killed mid-write can still leave a line with no terminator; that
    line is closed here so the reader drops only the torn record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json_line(record) + "\n").encode("utf-8")
    if path.exists() and path.stat().st_size:
        with path.open("rb") as probe:
            probe.seek(-1, os.SEEK_END)
            if probe.read(1) != b"\n":
                data = b"\n" + data
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def initial_state(problems: list[dict]) -> dict:
    return {
        "samples": {
            problem["uid"]: {"next_step": 1, "carry": None, "notes": ""}
            for problem in problems
        }
    }


def load_state(output: Path, problems: list[dict]) -> dict:
    """Resume from ``state.json``, reconciled against the per-sample logs.

    ``state.json`` is flushed on an interval, the per-sample logs are fsynced on
    every completed sample step, so a log is the ground truth for what that
    sample actually finished -- including when it holds *fewer* records than the
    snapshot claims, which is what a healed log looks like. A sample continues
    from its own next step, which is what makes a mixed-step kill resumable.
    """
    state = initial_state(problems)
    path = output / "state.json"
    if path.exists():
        stored = json.loads(path.read_text()).get("samples", {})
        for uid, entry in stored.items():
            if uid in state["samples"]:
                state["samples"][uid] = {
                    "next_step": int(entry["next_step"]),
                    "carry": entry.get("carry"),
                    "notes": entry.get("notes", ""),
                }
    for index, problem in enumerate(problems):
        records = read_sample_records(sample_record_path(output, index, problem["uid"]))
        if not records:
            continue
        # Resume from the first gap, not from the last record. The log is
        # authoritative in both directions -- a healed log holds fewer records
        # than the snapshot remembers -- and healing usually removes a step
        # from the *middle*, so `last + 1` would report the sample finished
        # while a step was still missing. The run then failed at assembly with
        # "expected N candidates, got N-1" and nothing queued to fix it.
        done = {int(row["step"]) for row in records}
        next_step = 1
        while next_step in done:
            next_step += 1
        resume_from = max(
            (row for row in records if int(row["step"]) < next_step),
            key=lambda row: int(row["step"]),
            default=None,
        )
        state["samples"][problem["uid"]] = {
            "next_step": next_step,
            "carry": resume_from.get("carry") if resume_from else None,
            "notes": resume_from.get("notes", "") if resume_from else "",
        }
    return state


def assemble_steps(output: Path, problems: list[dict], steps: int) -> tuple[int, int]:
    """Rebuild ``steps/step_NNNN/{final,intermediate}.jsonl`` from the sample logs."""
    finals: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    intermediate: dict[int, dict[int, list[dict]]] = collections.defaultdict(dict)
    for index, problem in enumerate(problems):
        for record in read_sample_records(
            sample_record_path(output, index, problem["uid"])
        ):
            step = int(record["step"])
            if step > steps:
                continue
            # A (uid, step) pair can only repeat if a kill landed between the
            # append and the state flush; the later record wins.
            finals[step][index] = record["final"]
            intermediate[step][index] = record["intermediate"]
    written_finals = 0
    written_intermediate = 0
    for step in range(1, steps + 1):
        directory = output / "steps" / f"step_{step:04d}"
        rows = [finals[step][index] for index in sorted(finals[step])]
        middles = [
            row
            for index in sorted(intermediate[step])
            for row in intermediate[step][index]
        ]
        atomic_jsonl(rows, directory / "final.jsonl")
        atomic_jsonl(middles, directory / "intermediate.jsonl")
        written_finals += len(rows)
        written_intermediate += len(middles)
    return written_finals, written_intermediate


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------
def default_sampling_params(*, temperature: float, top_p: float, max_tokens: int, seed: int,
                            logprobs: int | None = None):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature, top_p=top_p, top_k=-1,
        max_tokens=max_tokens, seed=seed, logprobs=logprobs,
    )


def public_answer(answer: dict) -> dict:
    """An answer without its in-memory fields (token ids, log-probs)."""
    return {key: value for key, value in answer.items() if not key.startswith("_")}


class AsyncRollout:
    """Run every sample's step loop concurrently against one async engine."""

    def __init__(
        self,
        *,
        engine,
        tokenizer,
        config: dict,
        condition: str,
        model: str,
        steps: int,
        problems: list[dict],
        output: Path,
        state: dict,
        executor,
        grade=default_grade,
        sampling_factory=default_sampling_params,
        template_kwargs: dict | None = None,
        max_inflight: int | None = None,
        max_model_len: int | None = None,
        response_max_tokens: int | None = None,
        lora_request=None,
    ):
        self.engine = engine
        # Self-correction after the update: the trained LoRA, applied to solving / carry
        # generations only (_generate);
        # the SciKnowEval judge (_judge_generate) stays on the base weights
        self.lora_request = lora_request
        self.tokenizer = tokenizer
        self.config = config
        self.condition = condition
        self.model = model
        self.spec = SPECS[condition]
        self.steps = steps
        self.problems = problems
        self.output = Path(output)
        self.state = state
        self.executor = executor
        self.grade = grade
        self.sampling_factory = sampling_factory
        self.template_kwargs = (
            thinking_template_kwargs(tokenizer)[0]
            if template_kwargs is None
            else template_kwargs
        )
        self.max_model_len = int(
            max_model_len
            if max_model_len is not None
            else config["max_model_len"]
        )
        self.response_max_tokens = int(
            response_max_tokens
            if response_max_tokens is not None
            else config["response_max_tokens"]
        )
        self.temperature = float(config["training_temperature"])
        self.top_p = float(config["training_top_p"])
        self.max_inflight = int(
            max_inflight
            if max_inflight is not None
            else config.get("rollout_max_inflight", 4 * int(config["max_num_seqs"]))
        )
        self.semaphore = asyncio.Semaphore(self.max_inflight)
        self.phase_seconds: collections.Counter[str] = collections.Counter()
        self.sample_seconds: dict[str, float] = {}
        self.context_tokens_max = 0
        self.completed = 0
        self.progress_interval = int(config.get("rollout_progress_interval", 100))
        self.state_flush_interval = int(config.get("state_flush_interval", 16))
        self.state_flush_seconds = float(config.get("state_flush_seconds", 60.0))
        self._since_flush = 0
        self._last_flush = time.monotonic()

    # -- generation ---------------------------------------------------------
    def _carried(self, problem: dict, response: str) -> str:
        """The previous turn's response as a later turn is shown it.

        Config ``leaf_carry_valid_code`` (LiveCodeBench): only the code
        the verifier extracts for grading, as a fenced block, or nothing when it
        finds none; elsewhere the whole response, as before.
        """
        if problem["source"] not in (self.config.get("leaf_carry_valid_code") or ()):
            return response
        from leaf.verifiers.answer_grader import lcb_valid_solution

        code = lcb_valid_solution(response)
        return f"```python\n{code}\n```" if code.strip() else ""

    def _generation_limit(self, prompt_tokens: int, max_tokens: int) -> int:
        """Completion budget: the response cap, the context window, and (config
        ``leaf_total_tokens``) a cap on prompt + completion together."""
        if prompt_tokens >= self.max_model_len:
            raise RuntimeError(
                f"Conversation exceeds max_model_len={self.max_model_len}; "
                f"prompt={prompt_tokens} tokens")
        caps = self.config.get("leaf_total_tokens") or {}
        total = int(caps.get(self.problems[0]["source"], self.max_model_len))
        return max(1, min(max_tokens, self.max_model_len - prompt_tokens, total - prompt_tokens))

    async def _generate(
        self, *, messages: list[dict[str, str]], step: int, problem_index: int,
        phase: int, max_tokens: int,
    ) -> tuple[dict, float]:
        want_logprobs = (self.spec.get("carry_transform") == "logprob"
                         and phase < CARRY_SUMMARY_PHASE)
        prompt_ids = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                **self.template_kwargs,
            )
        )
        self.context_tokens_max = max(self.context_tokens_max, len(prompt_ids))
        available = self._generation_limit(len(prompt_ids), max_tokens)
        seed = request_seed(self.config, step, problem_index, phase)
        params = self.sampling_factory(
            temperature=self.temperature, top_p=self.top_p,
            max_tokens=min(max_tokens, available), seed=seed,
            # logprobs=0: only the sampled token's log-prob, read off the
            # sampling pass itself (compress_lowest_logprob); no extra forward.
            **({"logprobs": 0} if want_logprobs else {}),
        )
        request_id = request_identifier(self.condition, step, problem_index, phase)
        started = time.monotonic()
        lora = {"lora_request": self.lora_request} if self.lora_request is not None else {}
        async with self.semaphore:
            output = None
            async for chunk in self.engine.generate(
                {"prompt_token_ids": prompt_ids}, params, request_id, **lora
            ):
                output = chunk
        elapsed = time.monotonic() - started
        if output is None:
            raise RuntimeError(f"Engine produced no output for {request_id}")
        completion = output.outputs[0]
        answer = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion.token_ids),
            "response": completion.text,
            "finish_reason": completion.finish_reason,
            "seed": seed,
            "request_id": request_id,
            "_token_ids": list(completion.token_ids),
        }
        if want_logprobs:
            answer["_logprobs"] = [
                float(position[token].logprob)
                for token, position in zip(completion.token_ids, completion.logprobs)
            ]
        return answer, elapsed

    # -- history-compression carry transforms --------------------------------
    def _ask(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """summary_block: every solving turn also asks for a closing <summary> block."""
        if self.spec.get("carry_transform") != "summary_block":
            return messages
        last = dict(messages[-1])
        last["content"] = f"{last['content']}\n\n{SUMMARY_BLOCK_INSTRUCTION}"
        return [*messages[:-1], last]

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    async def _carry_text(
        self, *, problem: dict, answer: dict, messages: list[dict[str, str]],
        step: int, problem_index: int, phase_seconds, intermediate: list[dict],
    ) -> str:
        """What a later turn is shown in place of ``answer``'s response.

        ``messages`` is the conversation that produced ``answer`` (llm_summary asks
        for the summary as the next turn of it, so the prefix is cached).
        """
        transform = self.spec.get("carry_transform")
        response = answer["response"]
        if not transform:
            return self._carried(problem, response)
        if transform == "summary_block":
            summary = summary_block_text(response)
            ids = self.tokenizer.encode(summary, add_special_tokens=False)
            if len(ids) > SUMMARY_BLOCK_TOKENS:
                summary = self._decode(ids[:SUMMARY_BLOCK_TOKENS])
            body = strip_summary_block(response)
            if problem["task_type"] == "code_lcb":
                from leaf.verifiers.answer_grader import lcb_valid_solution

                code = lcb_valid_solution(body)
                tail = f"```python\n{code}\n```" if code.strip() else ""
            else:
                tail = final_answer_text(body)
            head = f"<summary>\n{summary}\n</summary>" if summary else ""
            return "\n\n".join(part for part in (head, tail) if part)
        ids = answer["_token_ids"]
        if len(ids) <= CARRY_LONG_TOKENS:
            return response
        if transform == "tail":
            return self._decode(ids[-CARRY_LONG_TOKENS:])
        if transform == "head_tail":
            return (f"{self._decode(ids[:CARRY_HEAD_TAIL_TOKENS])}\n...\n"
                    f"{self._decode(ids[-CARRY_HEAD_TAIL_TOKENS:])}")
        if transform == "logprob":
            keep = logprob_keep_indices(answer["_logprobs"], CARRY_LOGPROB_KEEP)
            return self._decode([ids[i] for i in keep])
        if transform == "llm_summary":
            phase = CARRY_SUMMARY_PHASE + int(answer["solution_turn"])
            summary, elapsed = await self._generate(
                messages=[
                    *messages,
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": CARRY_SUMMARY_INSTRUCTION},
                ],
                step=step, problem_index=problem_index, phase=phase,
                max_tokens=CARRY_SUMMARY_TOKENS,
            )
            phase_seconds["carry_summary"] += elapsed
            summary["solution_turn"] = phase
            intermediate.append({
                "phase": "carry_summary", "step": step,
                "problem_uid": problem["uid"], **public_answer(summary),
            })
            return summary["response"].strip()
        raise ValueError(f"Unknown carry_transform {transform!r}")

    # -- LLM judge (open-answer SciKnowEval) ---------------------------------
    async def _judge_generate(self, prompt_ids: list[int], params: dict,
                              request_id: str) -> tuple[str, int, str | None]:
        """One greedy judge request on the base weights of this engine."""
        sampling = self.sampling_factory(
            temperature=params["temperature"], top_p=params["top_p"],
            max_tokens=params["max_tokens"], seed=params["seed"],
        )
        async with self.semaphore:
            output = None
            async for chunk in self.engine.generate(
                {"prompt_token_ids": prompt_ids}, sampling, request_id
            ):
                output = chunk
        if output is None:
            raise RuntimeError(f"Engine produced no output for {request_id}")
        completion = output.outputs[0]
        return completion.text, len(completion.token_ids), completion.finish_reason

    async def _judge(self, final: dict, step: int, problem_index: int) -> tuple[dict, float]:
        """Judge the step's scored solution; the verdict is stored on the record."""
        from leaf.verifiers import llm_judge

        ids, content, info = llm_judge.prompt_ids(
            self.tokenizer, final["verifier"], final["response"], self.template_kwargs,
        )
        params = llm_judge.sampling_params(
            self.config, request_seed(self.config, step, problem_index, llm_judge.JUDGE_PHASE),
        )
        params["max_tokens"] = max(1, min(params["max_tokens"], self.max_model_len - len(ids)))
        request_id = request_identifier(
            self.condition, step, problem_index, llm_judge.JUDGE_PHASE)
        started = time.monotonic()
        text, completion_tokens, finish_reason = await self._judge_generate(
            ids, params, request_id)
        elapsed = time.monotonic() - started
        return llm_judge.verdict_record(
            prompt=content, output=text, info=info, prompt_tokens=len(ids),
            completion_tokens=completion_tokens, finish_reason=finish_reason,
            seconds=elapsed,
        ), elapsed

    async def _verify(self, record: dict) -> tuple[dict, float]:
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        grades = await loop.run_in_executor(
            self.executor, self.grade, [record], self.config
        )
        elapsed = time.monotonic() - started
        if len(grades) != 1 or grades[0]["key"] != record["key"]:
            raise RuntimeError(f"Verifier returned a mismatched grade for {record['key']}")
        return grades[0], elapsed

    # -- one step of one sample --------------------------------------------
    async def _run_step(self, problem_index: int, problem: dict, step: int) -> dict:
        spec = self.spec
        entry = self.state["samples"][problem["uid"]]
        intermediate: list[dict] = []
        phase_seconds: collections.Counter[str] = collections.Counter()
        restart = is_restart(step, spec)
        answer: dict | None = None

        if restart:
            messages = self._ask(opening_messages(problem, self.model, entry.get("notes", "")))
            answer, elapsed = await self._generate(
                messages=messages, step=step, problem_index=problem_index,
                phase=1, max_tokens=self.response_max_tokens,
            )
            answer["solution_turn"] = 1
            phase_seconds["initial_solution"] += elapsed
            intermediate.append({
                "phase": "initial_solution", "step": step,
                "problem_uid": problem["uid"], **public_answer(answer),
                **({"messages": messages} if spec.get("record_messages") else {}),
            })
            rounds = int(spec["restart_corrections"])
        else:
            messages = self._ask(seeded_messages(problem, entry["carry"], spec, self.model))
            rounds = int(spec["carry_corrections"])

        for round_index in range(1, rounds + 1):
            if answer is not None:
                shown = await self._carry_text(
                    problem=problem, answer=answer, messages=messages, step=step,
                    problem_index=problem_index, phase_seconds=phase_seconds,
                    intermediate=intermediate,
                )
                messages = self._ask([
                    *messages,
                    {"role": "assistant", "content": shown},
                    {"role": "user", "content": correction_instruction(problem)},
                ])
            phase = round_index + 1
            answer, elapsed = await self._generate(
                messages=messages, step=step, problem_index=problem_index,
                phase=phase, max_tokens=self.response_max_tokens,
            )
            answer["solution_turn"] = phase
            phase_seconds[f"correction_{round_index}"] += elapsed
            intermediate.append({
                "phase": f"correction_{round_index}", "step": step,
                "problem_uid": problem["uid"], **public_answer(answer),
                **({"messages": messages} if spec.get("record_messages") else {}),
            })

        assert answer is not None
        # summary_block: the <summary> block is carried, not graded.
        graded = (
            {**answer, "response": strip_summary_block(answer["response"])}
            if spec.get("carry_transform") == "summary_block" else answer
        )
        final = final_record(
            condition=self.condition, step=step, problem_index=problem_index,
            problem=problem, answer=graded, problem_count=len(self.problems),
        )
        if graded is not answer:
            final["response_with_summary"] = answer["response"]

        # Open SciKnowEval: the base model judges every scored solution, in
        # every condition, before anything reads a verdict; the verifier and
        # the offline grader then only read final["judge"].
        if final["verifier"].get("kind") == "sciqa_judge":
            final["judge"], elapsed = await self._judge(final, step, problem_index)
            phase_seconds["judge"] += elapsed

        # The public verifier runs on every step final, in every condition
        # (wo_feedback too: grading, pass rate and interleaved selection read it).
        grade, elapsed = await self._verify(final)
        phase_seconds["public_verifier"] += elapsed
        # Only a step the next one continues from pays for a transform.
        if step < self.steps and not is_restart(step + 1, spec):
            solution = await self._carry_text(
                problem=problem, answer=answer, messages=messages, step=step,
                problem_index=problem_index, phase_seconds=phase_seconds,
                intermediate=intermediate,
            )
        else:
            solution = self._carried(problem, answer["response"])
        notes = entry.get("notes", "")
        carry = {"solution": solution, "notes": notes}
        carry["grade"] = {
            "correct": bool(grade["correct"]),
            "native_feedback": str(grade.get("native_feedback", "")),
            "extracted_answer": grade.get("extracted_answer", ""),
        }
        if spec.get("math_hide_answer") and problem["task_type"] == "math":
            carry["grade"]["hide_answer"] = True
        cap = spec.get("math_answer_max_tokens")
        if cap and problem["task_type"] == "math":
            extracted = str(carry["grade"]["extracted_answer"] or "").strip()
            length = len(self.tokenizer.encode(extracted, add_special_tokens=False))
            carry["grade"]["extracted_answer_tokens"] = length
            if length > int(cap):
                # feedback_section reports the length, not the text.
                carry["grade"]["extracted_answer"] = ""
                carry["grade"]["extracted_answer_too_long"] = True
        window = history_window(spec)
        if window > 1:
            # last_3_rounds: the rounds persist in the carry (state.json and the
            # sample log), so a resumed sample sees the same history.
            carry["history"] = carry_history(
                entry.get("carry"), {"solution": solution, "grade": dict(carry["grade"])},
                restart=restart, window=window)

        final["public_correct"] = bool(grade["correct"])
        return {
            "step": step,
            "problem_uid": problem["uid"],
            "problem_index": problem_index,
            "restart": restart,
            "correction_rounds": rounds,
            "final": final,
            "intermediate": intermediate,
            "phase_seconds": dict(phase_seconds),
            "carry": carry,
            "notes": notes,
        }

    # -- one sample's whole step loop --------------------------------------
    async def _run_sample(self, problem_index: int, problem: dict) -> None:
        uid = problem["uid"]
        path = sample_record_path(self.output, problem_index, uid)
        entry = self.state["samples"][uid]
        for step in range(int(entry["next_step"]), self.steps + 1):
            started = time.monotonic()
            record = await self._run_step(problem_index, problem, step)
            record["sample_step_seconds"] = time.monotonic() - started
            record["ended_at"] = time.time()
            append_sample_record(path, record)
            self.phase_seconds.update(record["phase_seconds"])
            self.sample_seconds[uid] = (
                self.sample_seconds.get(uid, 0.0) + record["sample_step_seconds"]
            )
            self.state["samples"][uid] = {
                "next_step": step + 1,
                "carry": record["carry"],
                "notes": record["notes"],
            }
            entry = self.state["samples"][uid]
            self._note_completion()

    def _note_completion(self) -> None:
        self.completed += 1
        self._since_flush += 1
        now = time.monotonic()
        if (
            self._since_flush >= self.state_flush_interval
            and now - self._last_flush >= self.state_flush_seconds
        ):
            self.flush_state()
        if self.progress_interval and self.completed % self.progress_interval == 0:
            steps = [entry["next_step"] for entry in self.state["samples"].values()]
            print(json.dumps({
                "event": "progress", "condition": self.condition,
                "completed_sample_steps": self.completed,
                "min_next_step": min(steps), "max_next_step": max(steps),
            }), flush=True)

    def flush_state(self) -> None:
        atomic_json(
            {
                "revision": self.config["revision"],
                "condition": self.condition,
                "steps": self.steps,
                "samples": self.state["samples"],
            },
            self.output / "state.json",
        )
        self._since_flush = 0
        self._last_flush = time.monotonic()

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(
                self._run_sample(index, problem), name=f"sample-{index}"
            )
            for index, problem in enumerate(self.problems)
            if int(self.state["samples"][problem["uid"]]["next_step"]) <= self.steps
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.flush_state()
        for result in results:
            if isinstance(result, BaseException):
                for task in tasks:
                    task.cancel()
                raise result


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_manifest(
    *, config: dict, condition: str, model: str, steps: int, problems: list[dict],
    dataset: Path, template_mode: str, template_kwargs: dict, max_inflight: int,
    verifier_workers: int,
) -> dict:
    sources: collections.Counter[str] = collections.Counter(
        problem["source"] for problem in problems
    )
    return {
        "revision": config["revision"],
        "condition": condition,
        "model": model,
        "model_path": str(model_path(config, model)),
        "dataset": str(dataset),
        "dataset_name": Path(dataset).name,
        "steps": steps,
        "problems_per_step": len(problems),
        "candidates": steps * len(problems),
        "sources": dict(sources),
        "temperature": config["training_temperature"],
        "top_p": config["training_top_p"],
        "response_max_tokens": model_response_max_tokens(
            config, model, benchmark_of(problems)
        ),
        "max_model_len": model_max_model_len(config, model),
        "max_num_seqs": model_max_num_seqs(config, model, benchmark_of(problems),
                                           condition=condition),
        "max_inflight_requests": max_inflight,
        "rollout_verifier_workers": config["rollout_verifier_workers"],
        "verifier_workers_used": verifier_workers,
        "verify_granularity": "per_sample_step",
        "chat_template_thinking_mode": template_mode,
        "chat_template_kwargs": dict(template_kwargs),
        "engine": "vllm.v1.engine.async_llm.AsyncLLM",
        **SPECS[condition],
    }


REVIEW_PHASES = ("carry_summary",)


def write_summary(
    *, output: Path, steps: int, problems: list[dict], phase_seconds: dict,
    generation_seconds: float, sample_seconds: dict, peak_memory_mib: int,
) -> dict:
    finals = [
        row
        for step in range(1, steps + 1)
        for row in read_jsonl(output / "steps" / f"step_{step:04d}" / "final.jsonl")
    ]
    expected = steps * len(problems)
    if len(finals) != expected:
        gaps = []
        for index, problem in enumerate(problems):
            steps_present = {
                int(row["step"])
                for row in read_sample_records(
                    sample_record_path(output, index, problem["uid"])
                )
            }
            missing = sorted(set(range(1, steps + 1)) - steps_present)
            if missing:
                gaps.append(f"{problem['uid']}:{missing}")
        raise RuntimeError(
            f"Expected {expected} final candidates, got {len(finals)}; "
            f"samples missing steps: {', '.join(gaps[:8]) or 'none'}"
        )
    intermediate = [
        row
        for step in range(1, steps + 1)
        for row in read_jsonl(output / "steps" / f"step_{step:04d}" / "intermediate.jsonl")
    ]
    atomic_jsonl(finals, output / "generations.jsonl")
    prior = (
        json.loads((output / "generation_summary.json").read_text())
        if (output / "generation_summary.json").exists()
        else {}
    )
    accumulated: collections.Counter[str] = collections.Counter(prior.get("phase_seconds", {}))
    accumulated.update(phase_seconds)
    summary = {
        "steps": steps,
        "candidates": len(finals),
        "optimizer_steps": 0,
        "final_solution_tokens": sum(row["completion_tokens"] for row in finals),
        "all_solution_tokens": sum(
            row["completion_tokens"] for row in intermediate
            if row["phase"] not in REVIEW_PHASES
        ),
        "review_tokens": sum(
            row["completion_tokens"] for row in intermediate if row["phase"] in REVIEW_PHASES
        ),
        "all_prompt_tokens": sum(row["prompt_tokens"] for row in intermediate),
        "all_completion_tokens": sum(row["completion_tokens"] for row in intermediate),
        "judge_tokens": sum(
            int(row["judge"].get("prompt_tokens", 0)) + int(row["judge"].get("completion_tokens", 0))
            for row in finals if isinstance(row.get("judge"), dict)
        ),
        "phase_seconds": dict(accumulated),
        "generation_seconds": float(prior.get("generation_seconds", 0.0)) + generation_seconds,
        "sample_seconds_sum": sum(sample_seconds.values()),
        "sample_seconds_max": max(sample_seconds.values()) if sample_seconds else 0.0,
        "peak_nvml_memory_mib": peak_memory_mib or int(prior.get("peak_nvml_memory_mib", 0)),
        "context_tokens_max": max(row["prompt_tokens"] for row in intermediate),
    }
    atomic_json(summary, output / "generation_summary.json")
    atomic_json(sample_seconds, output / "sample_seconds.json")
    return summary


def engine_kwargs(config: dict, model: str, benchmark: str | None = None,
                  condition: str | None = None, adapter: Path | None = None) -> dict:
    """AsyncEngineArgs of a rollout; ``adapter`` adds LoRA serving (rank ``lora_rank``)."""
    kwargs = dict(
        model=str(model_path(config, model)),
        dtype="bfloat16",
        max_model_len=model_max_model_len(config, model),
        max_num_seqs=model_max_num_seqs(config, model, benchmark, condition=condition),
        max_num_batched_tokens=int(config["max_num_batched_tokens"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=int(config["decode_seed"]),
        # vLLM's scheduler stats are how preemption and prefix-cache reuse
        # become visible; without them a KV-oversubscribed run looks merely slow.
        disable_log_stats=False,
    )
    if adapter is not None:
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=int(config["lora_rank"]))
    return kwargs


def build_engine(config: dict, model: str, benchmark: str | None = None,
                 condition: str | None = None, adapter: Path | None = None):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    return AsyncLLM.from_engine_args(AsyncEngineArgs(
        **engine_kwargs(config, model, benchmark, condition, adapter)))


async def _log_engine_stats(engine, interval: float = 10.0) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await engine.do_log_stats()
        except Exception:  # noqa: BLE001 -- stats are best effort
            pass


async def run_rollout(args, config: dict, problems: list[dict]) -> tuple[dict, float, dict, int]:
    adapter = getattr(args, "adapter", None)
    engine = build_engine(config, args.model, benchmark_of(problems), args.condition, adapter)
    lora_request = None
    if adapter is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("rollout_adapter", 1, str(adapter))
    tokenizer = engine.get_tokenizer()  # a coroutine on AsyncLLM
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
    state = load_state(args.output, problems)
    max_inflight = int(
        args.max_inflight
        or config.get("rollout_max_inflight", 4 * int(config["max_num_seqs"]))
    )
    workers = verifier_worker_count(config)
    atomic_json(
        build_manifest(
            config=config, condition=args.condition, model=args.model,
            steps=args.steps, problems=problems, dataset=args.dataset,
            template_mode=template_mode, template_kwargs=template_kwargs,
            max_inflight=max_inflight, verifier_workers=workers,
        ) | ({"adapter": str(adapter)} if adapter is not None else {}),
        args.output / "manifest.json",
    )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="rollout-verifier",
    )
    started = time.monotonic()
    with GpuMemoryMonitor() as monitor:
        rollout = AsyncRollout(
            engine=engine, tokenizer=tokenizer, config=config,
            condition=args.condition, model=args.model, steps=args.steps,
            problems=problems,
            output=args.output, state=state, executor=executor,
            template_kwargs=template_kwargs, max_inflight=max_inflight,
            max_model_len=model_max_model_len(config, args.model),
            response_max_tokens=model_response_max_tokens(
                config, args.model, benchmark_of(problems)
            ),
            lora_request=lora_request,
        )
        # Logging only: AsyncLLM prints its Running / Waiting / KV cache usage
        # line only when do_log_stats() is called (the API server does it every
        # 10 s); without it disable_log_stats=False leaves the log silent.
        stats_task = asyncio.create_task(_log_engine_stats(engine))
        try:
            await rollout.run()
        finally:
            stats_task.cancel()
            executor.shutdown(wait=True)
            engine.shutdown()
    elapsed = time.monotonic() - started
    return dict(rollout.phase_seconds), elapsed, rollout.sample_seconds, monitor.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-inflight", type=int, default=0,
        help="Upper bound on simultaneously in-flight engine requests "
             "(default: config rollout_max_inflight, else 4 * max_num_seqs).",
    )
    lora = parser.add_mutually_exclusive_group()
    lora.add_argument(
        "--adapter", type=Path, default=None,
        help="LoRA adapter directory served for the solving turns (judge stays on base).",
    )
    lora.add_argument(
        "--run-dir", type=Path, default=None,
        help="A verl run directory; serves the LoRA of its newest checkpoint, resolved at start.",
    )
    args = parser.parse_args()
    if args.run_dir is not None:
        from leaf.grading.test_eval import resolve_adapter

        args.adapter = resolve_adapter(None, args.run_dir)
    if args.adapter is not None:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (args.adapter / name).is_file():
                raise RuntimeError(f"Incomplete LoRA adapter: {args.adapter}")
        args.adapter = args.adapter.resolve()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    config = load_config()
    problems = read_jsonl(args.dataset)
    if not problems:
        raise RuntimeError(f"No problems in {args.dataset}")
    if len({problem["uid"] for problem in problems}) != len(problems):
        raise RuntimeError("Dataset UIDs are not unique")
    args.output.mkdir(parents=True, exist_ok=True)

    state = load_state(args.output, problems)
    pending = any(
        int(entry["next_step"]) <= args.steps for entry in state["samples"].values()
    )
    phase_seconds: dict = {}
    generation_seconds = 0.0
    sample_seconds: dict = {}
    peak_memory_mib = 0
    if pending:
        phase_seconds, generation_seconds, sample_seconds, peak_memory_mib = asyncio.run(
            run_rollout(args, config, problems)
        )
    written_finals, written_intermediate = assemble_steps(args.output, problems, args.steps)
    print(json.dumps({
        "event": "assembled", "condition": args.condition,
        "finals": written_finals, "intermediate": written_intermediate,
    }), flush=True)
    summary = write_summary(
        output=args.output, steps=args.steps, problems=problems,
        phase_seconds=phase_seconds, generation_seconds=generation_seconds,
        sample_seconds=sample_seconds, peak_memory_mib=peak_memory_mib,
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
