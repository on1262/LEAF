"""Generalization test-set evaluation: one checkpoint (or the base model), avg@k.

A test run sees only the checkpoint -- no skill prompt, no feedback turn: every
problem of the test set gets ``generalization.test_samples`` single-turn attempts
from its plain opening prompt (:func:`leaf.prompts.task_prompts.opening_messages`
without notes), sampled like the training rollouts (training temperature /
top-p / response cap of the model and benchmark).

Two stages, so the GPU never idles through CPU-bound grading:

``generate`` (GPU)
    vLLM with the LoRA adapter (``--adapter``, or the newest verl checkpoint of
    ``--run-dir``) or the base model. SciKnowEval responses are judged on the
    same engine with base weights (no LoRA), as in training.
``grade`` (CPU)
    * lcb: every test of the v6 problem (public + private), the SDPO executor
      used for training feedback, on ``generalization.lcb_test_ground_truth``;
    * mbpp: MBPP+ public asserts and EvalPlus private inputs, correct = both;
    * math500 / sciknoweval: the rollout verifiers (math-verify / the LLM judge).

Writes ``generations.jsonl``, ``grades.jsonl`` and ``summary.json`` (avg@k =
mean correctness over every attempt, pass@k = share of problems with a correct
attempt) into ``--output``.

``--test-set`` evaluates another problem file the same way (the internalization check:
the training sets), ``--ground-truth`` then names its LCB test-case parquet, ``--limit`` keeps the
first N problems (smokes) and ``--max-num-seqs`` replaces the config's vLLM
concurrency; both stages must get the same ``--test-set`` / ``--limit``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from leaf.core.project_config import atomic_json, atomic_jsonl, load_config, model_path, read_jsonl
from leaf.explore.leaf_rollout import (
    default_sampling_params, model_max_model_len, model_response_max_tokens,
    thinking_template_kwargs,
)
from leaf.prompts.task_prompts import opening_messages

BENCHMARKS = ("lcb", "mbpp", "math500", "sciknoweval")


def test_rows(config: dict, benchmark: str, test_set: Path | None = None,
              limit: int | None = None) -> list[dict]:
    rows = read_jsonl(test_set or config["generalization"]["test_sets"][benchmark])
    return rows[:limit] if limit else rows


def resolve_adapter(adapter: Path | None, run_dir: Path | None) -> Path | None:
    """``--adapter`` as given, or the adapter of the newest checkpoint of a verl run."""
    if run_dir is not None:
        checkpoints = run_dir / "checkpoints"
        step = (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip()
        adapter = checkpoints / f"global_step_{step}" / "actor" / "lora_adapter"
    if adapter is None:
        return None
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise RuntimeError(f"Incomplete LoRA adapter: {adapter}")
    return adapter.resolve()


def requests(config: dict, benchmark: str, test_set: Path | None = None,
             limit: int | None = None) -> list[dict]:
    settings = config["generalization"]
    samples = int(settings["test_samples"])
    base = int(settings["test_seed"])
    out = []
    for position, row in enumerate(test_rows(config, benchmark, test_set, limit)):
        for sample in range(samples):
            out.append({
                "key": f"test:{benchmark}:{row['uid']}:{sample}",
                "problem_uid": row["uid"],
                "source": row["source"],
                "task_type": row["task_type"],
                "sample_index": sample,
                "trial_index": position * samples + sample,
                "seed": base + position * 1000 + sample,
                "verifier": row["verifier"],
                "row": row,
            })
    return out


# --------------------------------------------------------------------------- #
# generate (GPU)
# --------------------------------------------------------------------------- #


def generate(args, config: dict) -> None:
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from leaf.verifiers import llm_judge

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / "generations.jsonl"
    pending = requests(config, args.benchmark, args.test_set, args.limit)
    if target.exists() and len(read_jsonl(target)) == len(pending):
        print(json.dumps({"event": "already_generated", "records": len(pending)}))
        return
    adapter = resolve_adapter(args.adapter, args.run_dir)
    settings = config["generalization"]
    max_tokens = model_response_max_tokens(config, args.model, args.benchmark)
    max_num_seqs = int(args.max_num_seqs or settings["test_max_num_seqs"][args.benchmark])
    started = time.monotonic()
    llm = LLM(
        model=str(model_path(config, args.model)),
        dtype="bfloat16",
        max_model_len=model_max_model_len(config, args.model),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=int(config.get("random_sampling_max_num_batched_tokens", 32768)),
        gpu_memory_utilization=float(settings["test_gpu_memory_utilization"]),
        enable_prefix_caching=True,
        enable_lora=adapter is not None,
        max_loras=1,
        max_lora_rank=int(config["lora_rank"]),
        seed=int(settings["test_seed"]),
        disable_log_stats=True,
    )
    engine_seconds = time.monotonic() - started
    tokenizer = llm.get_tokenizer()
    template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
    prompts = [
        list(tokenizer.apply_chat_template(
            opening_messages(item["row"], args.model, ""), tokenize=True,
            add_generation_prompt=True, **template_kwargs,
        ))
        for item in pending
    ]
    params = [
        default_sampling_params(
            temperature=float(config["training_temperature"]),
            top_p=float(config["training_top_p"]),
            max_tokens=max_tokens, seed=item["seed"],
        )
        for item in pending
    ]
    lora = LoRARequest("test_adapter", 1, str(adapter)) if adapter is not None else None
    started = time.monotonic()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params,
                           lora_request=lora, use_tqdm=False)
    generation_seconds = time.monotonic() - started
    records = []
    for item, ids, result in zip(pending, prompts, outputs, strict=True):
        answer = result.outputs[0]
        record = {key: value for key, value in item.items() if key != "row"}
        record.update({
            "model": args.model,
            "response": answer.text,
            "finish_reason": answer.finish_reason,
            "prompt_tokens": len(ids),
            "completion_tokens": len(answer.token_ids),
        })
        records.append(record)
    # The judge is the base model, never the adapter under test.
    judge_seconds = llm_judge.judge_records_offline(llm, tokenizer, records, config)
    atomic_jsonl(records, target)
    completion = [r["completion_tokens"] for r in records]
    atomic_json({
        "model": args.model,
        "benchmark": args.benchmark,
        "adapter": str(adapter) if adapter else None,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "test_set": str(args.test_set or settings["test_sets"][args.benchmark]),
        "limit": args.limit,
        "records": len(records),
        "samples_per_problem": int(settings["test_samples"]),
        "temperature": float(config["training_temperature"]),
        "top_p": float(config["training_top_p"]),
        "max_tokens": max_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": float(settings["test_gpu_memory_utilization"]),
        "chat_template_thinking_mode": template_mode,
        "engine_seconds": engine_seconds,
        "generation_seconds": generation_seconds,
        "judge_seconds": judge_seconds,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in records),
        "mean_completion_tokens": statistics.mean(completion),
        "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records),
    }, output / "generation_summary.json")
    print(json.dumps({"event": "generated", "records": len(records),
                      "generation_seconds": round(generation_seconds, 1),
                      "judge_seconds": round(judge_seconds, 1)}), flush=True)


# --------------------------------------------------------------------------- #
# grade (CPU)
# --------------------------------------------------------------------------- #


def mbpp_private(records: list[dict], grades: list[dict], config: dict, workers: int) -> dict:
    """``{key: outcome}`` of the EvalPlus private check for every MBPP+ record."""
    from leaf.verifiers import mbppplus as plus
    from leaf.verifiers.answer_grader import default_grader

    grader = default_grader(config, workers)
    codes = {grade["key"]: grade.get("extracted_answer") or "" for grade in grades}
    public = {grade["key"]: bool(grade["correct"]) for grade in grades}
    jobs, retry, key_of = {}, set(), {}
    for record in records:
        code = codes[record["key"]]
        if not code.strip():
            continue
        job = (record["problem_uid"], plus.sha(code))
        jobs.setdefault(job, (record["problem_uid"], code))
        key_of[record["key"]] = job
        if public[record["key"]]:
            retry.add(job)
    scratch = Path(tempfile.mkdtemp(prefix="test_mbppplus_gt_", dir=os.environ.get("TMPDIR")))
    try:
        truth = plus.build_ground_truth(
            scratch, sorted({plus.plus_task_id(r["problem_uid"]) for r in records}))
        results = plus.run_private(jobs, truth, grader, retry)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {key: results[job] for key, job in key_of.items()}


def summarize(records: list[dict], grades: list[dict], key: str = "correct") -> dict:
    by_problem = collections.defaultdict(list)
    by_sample = collections.defaultdict(list)
    for record, grade in zip(records, grades, strict=True):
        by_problem[record["problem_uid"]].append(bool(grade[key]))
        by_sample[record["sample_index"]].append(bool(grade[key]))
    return {
        "problems": len(by_problem),
        "attempts": len(records),
        "avg_at_k": sum(map(sum, by_problem.values())) / len(records),
        "pass_at_k": sum(any(v) for v in by_problem.values()) / len(by_problem),
        "accuracy_by_sample_index": {str(k): sum(v) / len(v) for k, v in sorted(by_sample.items())},
    }


def grade(args, config: dict) -> None:
    from leaf.verifiers.answer_grader import grade_records

    records = read_jsonl(args.output / "generations.jsonl")
    expected = len(requests(config, args.benchmark, args.test_set, args.limit))
    if len(records) != expected:
        raise RuntimeError(f"{args.output}: {len(records)} generations, expected {expected}")
    config = dict(config)
    if args.benchmark == "lcb":
        # Every test of each v6 test problem, on the executor that gave training feedback.
        config["lcb_ground_truth"] = str(args.ground_truth or config["generalization"]["lcb_test_ground_truth"])
        config["lcb_stop_at_first_failure"] = False  # the early stop is a training setting
        os.environ.pop("LEAF_LCB_PUBLIC_GROUND_TRUTH", None)
    workers = int(args.workers or config["verifier_workers"])
    started = time.monotonic()
    grades = grade_records(records, config, workers=workers)
    extra = {}
    if args.benchmark == "mbpp":
        private = mbpp_private(records, grades, config, workers)
        for grade_row in grades:
            outcome = private.get(grade_row["key"])
            grade_row["public_correct"] = bool(grade_row["correct"])
            grade_row["private_correct"] = bool(outcome and outcome["passed"])
            grade_row["private_status"] = outcome["status"] if outcome else "no_code"
            grade_row["correct"] = grade_row["public_correct"] and grade_row["private_correct"]
        extra["public_only"] = summarize(records, grades, "public_correct")
        extra["private_only"] = summarize(records, grades, "private_correct")
    summary = {
        "model": args.model,
        "benchmark": args.benchmark,
        "grading": {
            "lcb": "all tests (public + private) of each v6 problem, SDPO executor "
                   "(generalization.lcb_test_ground_truth)",
            "mbpp": "MBPP+ public asserts AND EvalPlus private inputs",
            "math500": "math_verify (leaf.verifiers.answer_grader)",
            "sciknoweval": "LLM judge = base model without LoRA (leaf.verifiers.llm_judge)",
        }[args.benchmark],
        **summarize(records, grades),
        **extra,
        "failure_types": dict(collections.Counter(g.get("failure_type") for g in grades)),
        "grading_seconds": time.monotonic() - started,
        "generation": json.loads((args.output / "generation_summary.json").read_text()),
    }
    atomic_jsonl([{key: value for key, value in g.items()
                   if key not in ("tests", "native_feedback")} for g in grades],
                 args.output / "grades.jsonl")
    atomic_json(summary, args.output / "summary.json")
    print(json.dumps({k: summary[k] for k in ("model", "benchmark", "avg_at_k", "pass_at_k")}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "grade"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="a verl run directory; evaluates its newest checkpoint")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--test-set", type=Path, default=None,
                        help="problem file to evaluate instead of generalization.test_sets[benchmark]")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="lcb: test-case parquet of --test-set (default generalization.lcb_test_ground_truth)")
    parser.add_argument("--limit", type=int, default=None, help="first N problems only (smokes)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="vLLM concurrency (default generalization.test_max_num_seqs[benchmark])")
    args = parser.parse_args()
    config = load_config()
    if args.model not in config["models"]:
        raise SystemExit(f"{args.model} is not in this config")
    (generate if args.stage == "generate" else grade)(args, config)


if __name__ == "__main__":
    main()
