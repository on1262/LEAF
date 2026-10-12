"""SFT-on-y* data: rewrite every (c+, y+) pair of the LEAF update data into y*.

Input ``<data>/ystar_pairs.jsonl`` (:mod:`leaf.update.build_trace_variants`): per successful context
c+ of ``successful_contexts.jsonl`` its correct solution y+ and the plain prompt x. The base model (no LoRA)
sees x, the session of c+ (earlier attempt(s) + feedback) and y+, and writes a complete,
self-contained solution y* with the same final answer, without the wrong attempts, wrong
intermediate results or any reference to attempts / feedback / checking (no length target).

Every rewrite is verified as the rollout verified its finals: MATH by math_verify, LCB on every
test of the training problem (``leaf.paths.LCB_TRAIN_GROUND_TRUTH``), SciKnowEval by the base
model's judge on the same engine. A pair whose rewrite is wrong, truncated or empty is rewritten
again with a new seed, for up to ``sft_and_traces.rewrite_rounds`` rounds in total; pairs never verified
are dropped.

Outputs: ``<data>/sft_ystar.jsonl`` (one row per verified pair, the training set of
``leaf.update.leaf_update`` method ``sft_ystar``), ``<data>/ystar/rewrites.jsonl`` (every attempt)
and ``<data>/ystar/rewrite_summary.json``.

    LEAF_CONFIG=configs/general/sft_qwen3_8b_seed1.json python -m leaf.update.rewrite_ystar --model qwen3_8b \\
        --benchmark math500 --data $RESULTS_ROOT/general/seed1/sft/qwen3_8b/qwen3_8b/formal/update_data/math500 [--limit 16]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import time
from pathlib import Path

TASK_TYPES = {"math500": "math", "lcb": "code_lcb", "sciknoweval": "sciqa"}
INSTRUCTION = (
    "Rewrite the correct solution into a complete, self-contained solution to the problem, "
    "written as if you were solving it for the first time. Remove the wrong attempts, the invalid "
    "derivations, the wrong intermediate results and every reference to earlier attempts, feedback, "
    "checking or corrections. Present the correct reasoning in a logical order, keeping its "
    "intermediate results. {ending} Output only the rewritten solution."
)


# --------------------------------------------------------------------------- #
# pure logic
# --------------------------------------------------------------------------- #


def has_session(context: list[dict]) -> bool:
    """True when c+ holds more than the plain prompt (an earlier attempt and its feedback)."""
    return any(m["role"] == "assistant" for m in context)


def rewrite_messages(pair: dict, task_type: str, answer: str | None = None) -> list[dict]:
    """The rewrite request: x's system turn (if any) + one user turn. ``answer``: the MATH
    reference answer of y+ (``leaf.update.rewrite_utils.math_reference``)."""
    from leaf.update.trace_selection import ENDINGS, transcript

    x = pair["prompt_messages"]
    plain_user = x[-1]["content"]
    ending = ENDINGS[task_type]
    if task_type == "math":
        ending = ending.format(answer=answer or "")
    parts = [plain_user]
    if has_session(pair["context_messages"]):
        session = transcript(pair["context_messages"], plain_user)
        parts.append("Below is an earlier solving session for this problem: earlier attempts, the "
                     "feedback on them, and a final solution that was verified to be correct.")
        parts.append(f"<session>\n{session}\n</session>")
    else:
        parts.append("Below is a solution to this problem that was verified to be correct.")
    parts.append(f"<correct_solution>\n{pair['solution']}\n</correct_solution>")
    parts.append(INSTRUCTION.format(ending=ending))
    return [*(dict(m) for m in x[:-1]), {"role": "user", "content": "\n\n".join(parts)}]


def pair_key(pair: dict) -> tuple[str, int]:
    return pair["uid"], int(pair["j"])


def request_seed(base: int, problem_index: int, j: int, round_: int) -> int:
    return int(base) + 100_000 * int(round_) + 100 * int(problem_index) + int(j)


def trial_index(problem_index: int, j: int, round_: int) -> int:
    """Unique per attempt (it seeds the SciKnowEval judge)."""
    return (int(problem_index) * 100 + int(j)) * 10 + int(round_)


def grading_record(pair: dict, problem: dict, model: str, text: str, finish_reason: str | None,
                   round_: int) -> dict:
    """A canonical grading record (as ``leaf.analysis.internalization_eval.sample_records``)."""
    return {
        "key": f"ystar:{pair['uid']}:{pair['j']}:{round_}", "problem_uid": pair["uid"],
        "source": problem["source"], "task_type": problem["task_type"], "problem": problem["problem"],
        "verifier": problem["verifier"], "trial_index": trial_index(pair["problem_index"], pair["j"], round_),
        "sample_index": 0, "model": model, "response": text, "finish_reason": finish_reason,
    }


def accepted(text: str, finish_reason: str | None, grade: dict) -> bool:
    return bool(grade.get("correct")) and finish_reason != "length" and bool(text.strip())


def select_verified(attempts: list[dict]) -> dict[tuple[str, int], dict]:
    """The first accepted attempt of every pair (lowest round)."""
    out: dict[tuple[str, int], dict] = {}
    for attempt in sorted(attempts, key=lambda a: int(a["round"])):
        key = (attempt["uid"], int(attempt["j"]))
        if attempt["correct"] and key not in out:
            out[key] = attempt
    return out


def sft_rows(pairs: list[dict], verified: dict[tuple[str, int], dict]) -> list[dict]:
    """``sft_ystar.jsonl``: the verified rewrites, in pair order."""
    rows = []
    for pair in pairs:
        attempt = verified.get(pair_key(pair))
        if attempt is None:
            continue
        rows.append({"uid": pair["uid"], "problem_index": int(pair["problem_index"]), "j": int(pair["j"]),
                     "trace_step": int(pair["trace_step"]), "solution": attempt["text"],
                     "tokens": int(attempt["completion_tokens"]), "round": int(attempt["round"])})
    return rows


def grading_config(config: dict, benchmark: str) -> dict:
    """The rollout's grading of the training set (LCB: every training test; pass/fail only)."""
    from leaf import paths

    config = dict(config)
    if benchmark == "lcb":
        config["lcb_ground_truth"] = str(paths.LCB_TRAIN_GROUND_TRUTH)
        config["lcb_stop_at_first_failure"] = True
        os.environ.pop("LEAF_LCB_PUBLIC_GROUND_TRUTH", None)
    return config


def grade_workers(config: dict) -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(int(config["verifier_workers"]), cpus - 4)


# --------------------------------------------------------------------------- #
# GPU job
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=sorted(TASK_TYPES))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="first N pairs only (smokes)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from leaf.core.project_config import atomic_json, atomic_jsonl, load_config, read_jsonl

    config = load_config()
    settings = config["sft_and_traces"]
    output = args.data / "ystar"
    summary_path = output / "rewrite_summary.json"
    if summary_path.exists() and not args.force:
        done = json.loads(summary_path.read_text())
        if done.get("limit") == args.limit:
            print(json.dumps({"event": "already_rewritten", "output": str(output)}))
            return

    from vllm import LLM, SamplingParams

    from leaf.verifiers import llm_judge
    from leaf.core.request_seeds import GpuMemoryMonitor
    from leaf.core.project_config import detect_gpu_type, model_path
    from leaf.explore.leaf_rollout import model_max_model_len, model_response_max_tokens, thinking_template_kwargs
    from leaf.verifiers.answer_grader import grade_records
    from leaf.update.rewrite_utils import cache_metrics, clean_candidate, math_reference

    task_type = TASK_TYPES[args.benchmark]
    pairs = read_jsonl(args.data / "ystar_pairs.jsonl")
    pairs = pairs[: args.limit] if args.limit else pairs
    manifest = json.loads((args.data / "manifest.json").read_text())
    dataset = {row["uid"]: row for row in read_jsonl(Path(manifest["dataset"]))}
    answers = {pair_key(p): math_reference(p["solution"]) for p in pairs} if task_type == "math" else {}
    grade_config = grading_config(config, args.benchmark)
    workers = grade_workers(config)
    rounds = int(settings["rewrite_rounds"])
    max_tokens = model_response_max_tokens(config, args.model, args.benchmark)
    max_model_len = model_max_model_len(config, args.model)
    max_num_seqs = int(settings["rewrite_max_num_seqs"])
    output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    attempts: list[dict] = []
    seconds = collections.Counter()
    with GpuMemoryMonitor() as monitor:
        llm = LLM(
            model=str(model_path(config, args.model)), dtype="bfloat16", max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=int(config.get("random_sampling_max_num_batched_tokens", 32768)),
            gpu_memory_utilization=float(settings["rewrite_gpu_memory_utilization"]),
            enable_prefix_caching=True, seed=int(settings["rewrite_seed"]), disable_log_stats=False,
        )
        seconds["engine"] = time.monotonic() - started
        tokenizer = llm.get_tokenizer()
        template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
        prompt_ids = {}
        for pair in pairs:
            messages = rewrite_messages(pair, task_type, answers.get(pair_key(pair)))
            prompt_ids[pair_key(pair)] = list(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **template_kwargs))
        pending = [p for p in pairs if max_model_len - len(prompt_ids[pair_key(p)]) >= 256]
        too_long = len(pairs) - len(pending)
        for round_ in range(1, rounds + 1):
            if not pending:
                break
            began = time.monotonic()
            params = []
            for pair in pending:
                ids = prompt_ids[pair_key(pair)]
                params.append(SamplingParams(
                    n=1, temperature=float(settings["rewrite_temperature"]),
                    top_p=float(settings["rewrite_top_p"]), top_k=-1,
                    max_tokens=min(max_tokens, max_model_len - len(ids)),
                    seed=request_seed(settings["rewrite_seed"], pair["problem_index"], pair["j"], round_)))
            results = llm.generate([{"prompt_token_ids": prompt_ids[pair_key(p)]} for p in pending], params,
                                   use_tqdm=False)
            seconds["generation"] += time.monotonic() - began
            records, raw = [], []
            for pair, result in zip(pending, results, strict=True):
                candidate = result.outputs[0]
                text = clean_candidate(candidate.text)
                raw.append((pair, text, candidate))
                records.append(grading_record(pair, dataset[pair["uid"]], args.model, text,
                                              candidate.finish_reason, round_))
            began = time.monotonic()
            seconds["judge"] += llm_judge.judge_records_offline(llm, tokenizer, records, config)
            began = time.monotonic()
            grades = grade_records(records, grade_config, workers=workers)
            seconds["grading"] += time.monotonic() - began
            failed = []
            for (pair, text, candidate), record, grade in zip(raw, records, grades, strict=True):
                ok = accepted(text, candidate.finish_reason, grade)
                attempts.append({
                    "uid": pair["uid"], "problem_index": int(pair["problem_index"]), "j": int(pair["j"]),
                    "trace_step": int(pair["trace_step"]), "round": round_, "text": text,
                    "finish_reason": candidate.finish_reason, "completion_tokens": len(candidate.token_ids),
                    "prompt_tokens": len(prompt_ids[pair_key(pair)]), "correct": ok,
                    "grade": {"correct": bool(grade.get("correct")), "failure_type": grade.get("failure_type"),
                              "extracted_answer": str(grade.get("extracted_answer") or "")[:500]},
                    **({"judge": record["judge"]} if "judge" in record else {}),
                })
                if not ok:
                    failed.append(pair)
            print(json.dumps({"event": "round", "round": round_, "attempts": len(pending),
                              "verified": len(pending) - len(failed), "seconds": dict(seconds)}), flush=True)
            pending = failed
        metrics = cache_metrics(llm)

    verified = select_verified(attempts)
    rows = sft_rows(pairs, verified)
    atomic_jsonl(attempts, output / "rewrites.jsonl")
    atomic_jsonl(rows, args.data / "sft_ystar.jsonl")
    by_round = collections.Counter(int(a["round"]) for a in verified.values())
    summary = {
        "model": args.model, "benchmark": args.benchmark, "data": str(args.data), "limit": args.limit,
        "config": os.environ.get("LEAF_CONFIG"), "pairs": len(pairs), "verified": len(rows),
        "dropped": len(pairs) - len(rows), "prompt_too_long": too_long,
        "problems": len({p["uid"] for p in pairs}), "problems_with_ystar": len({r["uid"] for r in rows}),
        "verified_by_round": {str(k): v for k, v in sorted(by_round.items())},
        "attempts": len(attempts), "rounds": rounds,
        "truncated_fraction": (sum(a["finish_reason"] == "length" for a in attempts) / len(attempts)
                               if attempts else None),
        "ystar_tokens_mean": statistics.mean(r["tokens"] for r in rows) if rows else None,
        "solution_tokens_mean_attempts": (statistics.mean(a["completion_tokens"] for a in attempts)
                                          if attempts else None),
        "prompt_tokens_mean": statistics.mean(len(v) for v in prompt_ids.values()) if prompt_ids else None,
        "completion_tokens": sum(a["completion_tokens"] for a in attempts),
        "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
        "chat_template_thinking_mode": template_mode, "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs, "gpu_type": detect_gpu_type(), "grading_workers": workers,
        "seconds": {**dict(seconds), "total": time.monotonic() - started},
        "peak_gpu_memory_mib": monitor.peak_mib, **metrics,
    }
    atomic_json(summary, summary_path)
    print(json.dumps({"event": "rewritten", **{k: summary[k] for k in (
        "pairs", "verified", "dropped", "verified_by_round", "truncated_fraction")}}), flush=True)


if __name__ == "__main__":
    main()
