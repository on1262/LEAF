"""Teacher accuracy: the base model answering under a given privileged context.

The LEAF update's teacher is the policy reading a context c (successful / failure / random traces,
:mod:`leaf.update.build_trace_variants`). Its accuracy is measured by re-rolling it out: per problem of
the condition's rows file and per k in 1..``sft_and_traces.teacher_samples``, the context the trainer
would draw at training step k (:func:`draw_context`, the RNG of
:func:`leaf.update.leaf_update.teacher_messages`) is rendered with the thinking template kwargs and
answered once by the base model (no LoRA), sampled like the test evaluation.

``generate`` (GPU): vLLM; SciKnowEval responses are judged on the same engine (base weights).
``grade`` (CPU): the training grading (:func:`leaf.analysis.internalization_eval.grading_config`: LCB on
every test of the training problem; MATH math_verify; SciKnowEval the stored verdicts).

Writes ``generations.jsonl``, ``grades.jsonl``, ``generation_summary.json`` and ``summary.json``
into ``--output``.

    LEAF_CONFIG=configs/general/sft_qwen3_8b_seed1.json python -m leaf.analysis.teacher_accuracy generate \\
        --model qwen3_8b --benchmark math500 \\
        --data $RESULTS_ROOT/general/seed1/sft/qwen3_8b/qwen3_8b/formal/update_data/math500 \\
        --condition failure --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import zlib

import numpy as np

BENCHMARKS = ("lcb", "math500", "sciknoweval")
ROWS = {"successful": "successful_contexts.jsonl", "failure": "failure_contexts.jsonl",
        "random": "random_contexts.jsonl"}
# leaf.update.leaf_update.FEEDBACK_RANDOM_SEED (that module imports verl at the top)
FEEDBACK_RANDOM_SEED = 200045
SEED_BASE = 230047


def draw_index(uid: str, step: int, choices: int) -> int:
    """The context index the LEAF update trainer draws for (step, problem)."""
    rng = np.random.default_rng([FEEDBACK_RANDOM_SEED, int(step), zlib.crc32(uid.encode())])
    return int(rng.integers(choices))


def draw_context(row: dict, step: int) -> tuple[int, list[dict]]:
    choices = row["teacher_messages_list"]
    index = draw_index(row["uid"], step, len(choices))
    messages = choices[index]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{row['uid']}: teacher messages must end with a user turn")
    return index, [{"role": m["role"], "content": m["content"]} for m in messages]


def read_rows(data: Path, condition: str, limit: int | None = None) -> list[dict]:
    from leaf.core.project_config import read_jsonl

    rows = sorted(read_jsonl(Path(data) / ROWS[condition]), key=lambda r: (int(r["problem_index"]), r["uid"]))
    return rows[:limit] if limit else rows


def requests(config: dict, data: Path, condition: str, limit: int | None = None) -> list[dict]:
    from leaf.core.project_config import read_jsonl

    manifest = json.loads((Path(data) / "manifest.json").read_text())
    dataset = {row["uid"]: row for row in read_jsonl(Path(manifest["dataset"]))}
    samples = int(config["sft_and_traces"]["teacher_samples"])
    out = []
    for position, row in enumerate(read_rows(data, condition, limit)):
        problem = dataset[row["uid"]]
        for k in range(1, samples + 1):
            index, messages = draw_context(row, k)
            out.append({
                "key": f"teacher:{condition}:{row['uid']}:{k}",
                "problem_uid": row["uid"], "source": problem["source"], "task_type": problem["task_type"],
                "problem": problem.get("problem"), "verifier": problem["verifier"],
                "condition": condition, "k": k, "context_index": index,
                "contexts": len(row["teacher_messages_list"]), "fallback": bool(row.get("fallback")),
                "sample_index": k - 1, "trial_index": position * samples + k - 1,
                "seed": SEED_BASE + position * 1000 + k, "messages": messages,
            })
    return out


def generate(args, config: dict) -> None:
    from vllm import LLM

    from leaf.verifiers import llm_judge
    from leaf.core.project_config import atomic_json, atomic_jsonl, model_path, read_jsonl
    from leaf.explore.leaf_rollout import (
        default_sampling_params, model_max_model_len, model_response_max_tokens, thinking_template_kwargs,
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / "generations.jsonl"
    pending = requests(config, args.data, args.condition, args.limit)
    if target.exists() and len(read_jsonl(target)) == len(pending):
        print(json.dumps({"event": "already_generated", "records": len(pending)}))
        return
    settings = config["generalization"]
    max_len = model_max_model_len(config, args.model)
    cap = model_response_max_tokens(config, args.model, args.benchmark)
    max_num_seqs = int(args.max_num_seqs or config["sft_and_traces"]["teacher_max_num_seqs"][args.benchmark])
    started = time.monotonic()
    llm = LLM(model=str(model_path(config, args.model)), dtype="bfloat16", max_model_len=max_len,
              max_num_seqs=max_num_seqs,
              max_num_batched_tokens=int(config.get("random_sampling_max_num_batched_tokens", 32768)),
              gpu_memory_utilization=float(settings["test_gpu_memory_utilization"]),
              enable_prefix_caching=True, seed=SEED_BASE, disable_log_stats=True)
    engine_seconds = time.monotonic() - started
    tokenizer = llm.get_tokenizer()
    template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
    prompts = [list(tokenizer.apply_chat_template(item["messages"], tokenize=True, add_generation_prompt=True,
                                                  **template_kwargs)) for item in pending]
    params = [default_sampling_params(temperature=float(config["training_temperature"]),
                                      top_p=float(config["training_top_p"]),
                                      max_tokens=max(1, min(cap, max_len - len(ids))), seed=item["seed"])
              for item, ids in zip(pending, prompts, strict=True)]
    started = time.monotonic()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)
    generation_seconds = time.monotonic() - started
    records = []
    for item, ids, result in zip(pending, prompts, outputs, strict=True):
        answer = result.outputs[0]
        record = {key: value for key, value in item.items() if key != "messages"}
        record.update({"model": args.model, "response": answer.text, "finish_reason": answer.finish_reason,
                       "prompt_tokens": len(ids), "completion_tokens": len(answer.token_ids)})
        records.append(record)
    judge_seconds = llm_judge.judge_records_offline(llm, tokenizer, records, config)
    atomic_jsonl(records, target)
    atomic_json({
        "model": args.model, "benchmark": args.benchmark, "condition": args.condition,
        "data": str(args.data), "limit": args.limit, "records": len(records),
        "samples_per_problem": int(config["sft_and_traces"]["teacher_samples"]),
        "temperature": float(config["training_temperature"]), "top_p": float(config["training_top_p"]),
        "max_tokens": cap, "max_num_seqs": max_num_seqs, "chat_template_thinking_mode": template_mode,
        "engine_seconds": engine_seconds, "generation_seconds": generation_seconds, "judge_seconds": judge_seconds,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in records),
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in records),
        "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records),
    }, output / "generation_summary.json")
    print(json.dumps({"event": "generated", "records": len(records), "generation_seconds": round(generation_seconds, 1),
                      "judge_seconds": round(judge_seconds, 1)}), flush=True)


def summarize(records: list[dict], grades: list[dict]) -> dict:
    """Teacher accuracy over every sample, the mean of per-problem accuracies, and (failure) over the
    problems with a real failure context."""
    by_problem: dict[str, list[bool]] = {}
    fallback: dict[str, bool] = {}
    for record, grade in zip(records, grades, strict=True):
        by_problem.setdefault(record["problem_uid"], []).append(bool(grade["correct"]))
        fallback[record["problem_uid"]] = bool(record.get("fallback"))
    correct = [c for values in by_problem.values() for c in values]
    real = [c for uid, values in by_problem.items() if not fallback[uid] for c in values]
    mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
    return {
        "problems": len(by_problem), "samples": len(correct), "accuracy": mean(correct),
        "problem_mean_accuracy": mean([mean(v) for v in by_problem.values()]),
        "fallback_problems": sum(fallback.values()),
        "accuracy_without_fallback": mean(real),
        "truncated_fraction": mean([r.get("finish_reason") == "length" for r in records]),
        "mean_prompt_tokens": mean([r["prompt_tokens"] for r in records]),
        "mean_completion_tokens": mean([r["completion_tokens"] for r in records]),
    }


def grade(args, config: dict) -> None:
    from leaf.core.project_config import atomic_json, atomic_jsonl, read_jsonl
    from leaf.verifiers.answer_grader import grade_records
    from leaf.analysis.internalization_eval import grading_config

    records = read_jsonl(args.output / "generations.jsonl")
    expected = len(requests(config, args.data, args.condition, args.limit))
    if len(records) != expected:
        raise RuntimeError(f"{args.output}: {len(records)} generations, expected {expected}")
    started = time.monotonic()
    grades = grade_records(records, grading_config(config, args.benchmark),
                           workers=int(args.workers or config["verifier_workers"]))
    summary = {"model": args.model, "benchmark": args.benchmark, "condition": args.condition,
               **summarize(records, grades), "grading_seconds": time.monotonic() - started,
               "generation": json.loads((args.output / "generation_summary.json").read_text())}
    atomic_jsonl([{k: v for k, v in g.items() if k not in ("tests", "native_feedback")} for g in grades],
                 args.output / "grades.jsonl")
    atomic_json(summary, args.output / "summary.json")
    print(json.dumps({k: summary[k] for k in ("model", "benchmark", "condition", "accuracy",
                                              "accuracy_without_fallback")}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "grade"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--data", type=Path, required=True, help="update_data/<benchmark> directory of leaf.update.build_trace_variants")
    parser.add_argument("--condition", required=True, choices=tuple(ROWS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N problems only (smokes)")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    from leaf.core.project_config import load_config

    config = load_config()
    if args.model not in config["models"]:
        raise SystemExit(f"{args.model} is not in this config")
    (generate if args.stage == "generate" else grade)(args, config)


if __name__ == "__main__":
    main()
