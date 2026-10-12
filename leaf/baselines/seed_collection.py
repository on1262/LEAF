"""Collect, verify, and hindsight-analyze SEED SFT trajectories.

One collection run per benchmark.  SEED spends the first
``round(seed_collection_fraction x U)`` whole updates of that benchmark's budget
on collection - a 4-of-25 = 16% share taken proportionally - so that
collection trials plus RL trials equal what the other rollout-N=8 methods spend
on the same benchmark::

    math500  8 updates -> 2,048 trajectories
    lcb      2 updates ->   512 trajectories
    mbpp     8 updates -> 2,048 trajectories
    sciknow  3 updates ->   768 trajectories

Stage order: sample the solutions, grade them with the public verifier, ask the
frozen base model for one hindsight Skill per trajectory, then write the SFT
parquet.

Resume
------
Pure vLLM, so verl's checkpointing does not apply and resume is per record:
``solutions.jsonl`` and ``analyses.jsonl`` are append-only and keyed by ``key``.
A restarted chunk re-reads both, regenerates only the missing keys, and writes
each chunk of ``random_sampling_chunk_size`` responses before starting the next, so a
chunk killed by the wall clock loses at most one chunk of work.  Once both
files are complete and ``sft.parquet`` exists the run exits immediately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from leaf.baselines import seed_prompts
from leaf.prompts import task_prompts
from leaf.verifiers import llm_judge
from leaf.baselines.random_sampling import kv_cache_tokens
from leaf.core.project_config import (
    BENCHMARKS, MODELS, MODES, apply_chat_messages, apply_chat_template, atomic_json,
    benchmark_dataset_path, benchmark_rl_data_dir, detect_gpu_type,
    launch_manifest, load_config, mark_complete, model_path, parquet_mode,
    read_jsonl, verifier_record, write_launch_manifest,
)
from leaf.baselines.seed_analyzer import validate_seed_skill
from leaf.explore.leaf_rollout import model_max_num_seqs, model_response_max_tokens


def collection_data(config: dict, benchmark: str, mode: str, model: str) -> Path:
    path = (
        benchmark_rl_data_dir(config, benchmark, model)
        / f"seed_collection_{parquet_mode(mode)}.parquet"
    )
    if not path.exists():
        raise RuntimeError(f"{path} is missing; run leaf.data.prepare_verl_data first")
    return path


def append_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def generate_in_chunks(llm, sampling, requests, chunk_size, on_chunk):
    """Generate in chunks so a killed chunk loses at most ``chunk_size`` work."""
    seconds = 0.0
    for start in range(0, len(requests), chunk_size):
        batch = requests[start:start + chunk_size]
        started = time.monotonic()
        outputs = llm.generate(
            [{"prompt_token_ids": item["prompt_ids"]} for item in batch],
            [sampling(item) for item in batch],
            use_tqdm=True,
        )
        seconds += time.monotonic() - started
        on_chunk(batch, outputs)
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-type", choices=["a100", "h200"])
    args = parser.parse_args()

    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sft_path = output / "sft.parquet"

    data = collection_data(config, args.benchmark, args.mode, args.model)
    questions = pq.read_table(data).to_pylist()
    problems = {
        row["uid"]: row
        for row in read_jsonl(benchmark_dataset_path(config, args.benchmark, args.mode))
    }
    group = int(config["rollout_group_size"])
    expected = len(questions) * group

    requests = []
    for question_index, question in enumerate(questions):
        extra = question["extra_info"]
        row = problems[extra["problem_uid"]]
        for rollout_index in range(group):
            trial_index = question_index * group + rollout_index
            requests.append({
                "key": f"seed:collection:{args.benchmark}:{trial_index}",
                "row": row,
                "trial_index": trial_index,
                "attempt_index": rollout_index,
                "seed": int(question["rollout_seed"]) + rollout_index,
            })

    manifest = launch_manifest(
        config=config, method="seed_collection", model=args.model,
        benchmark=args.benchmark, mode=args.mode, gpu_type=gpu_type,
        updates=len(questions) // int(config["question_batch_size"]) or None,
        overrides=[],
        extra={
            "stage": "hindsight_collection",
            "data": str(data),
            "question_rows": len(questions),
            "rollout_group_size": group,
            "verifier_trials": expected,
            "skill_max_tokens": int(config["skill_max_tokens"]),
            "analyzer": f"frozen_base_{args.model}",
        },
    )
    write_launch_manifest(output, manifest)

    solutions_path = output / "solutions.jsonl"
    analyses_path = output / "analyses.jsonl"
    done_solutions = {row["key"]: row for row in read_jsonl(solutions_path)}
    done_analyses = {row["key"] for row in read_jsonl(analyses_path)}
    if sft_path.exists() and len(done_solutions) == expected and len(done_analyses) == expected:
        print(json.dumps({"event": "already_complete", "trajectories": expected}), flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    base = model_path(config, args.model)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    llm = LLM(
        model=str(base), dtype="bfloat16", max_model_len=int(config["max_model_len"]),
        # Single-turn sampling like Random Sampling, so the same per-cell concurrency.
        max_num_seqs=model_max_num_seqs(
            config, args.model, args.benchmark, key="random_sampling_max_num_seqs",
            condition="seed_collection",
        ),
        max_num_batched_tokens=int(config["max_num_batched_tokens"]),
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        enable_prefix_caching=True, enable_chunked_prefill=True,
        # Stats only: vLLM logs Running / Waiting / KV cache usage every 10 s.
        seed=int(config["decode_seed"]), disable_log_stats=False,
    )
    chunk_size = int(config["random_sampling_chunk_size"])

    # --- stage 1: solutions --------------------------------------------------
    pending = [item for item in requests if item["key"] not in done_solutions]
    for item in pending:
        item["prompt_ids"] = apply_chat_messages(
            tokenizer, task_prompts.opening_messages(item["row"], args.model)
        )

    def solution_params(item):
        return SamplingParams(
            temperature=float(config["training_temperature"]),
            top_p=float(config["training_top_p"]), top_k=-1,
            # The cell's own cap, the one every other condition generates under.
            max_tokens=model_response_max_tokens(config, args.model, args.benchmark),
            seed=item["seed"],
        )

    def write_solutions(batch, outputs):
        rows = []
        for item, output_value in zip(batch, outputs, strict=True):
            answer = output_value.outputs[0]
            rows.append(verifier_record(
                item["row"], answer.text, item["key"],
                trial_index=item["trial_index"],
                attempt_index=item["attempt_index"],
                seed=item["seed"],
                prompt_tokens=len(item["prompt_ids"]),
                completion_tokens=len(answer.token_ids),
                finish_reason=answer.finish_reason,
            ))
        # Open SciKnowEval: the same base engine judges the chunk before it is
        # written, so the public verifier below only reads the verdicts.
        judge_seconds[0] += llm_judge.judge_records_offline(llm, tokenizer, rows, config)
        append_jsonl(rows, solutions_path)
        done_solutions.update({row["key"]: row for row in rows})

    from leaf.baselines.verl_reward import compute_score_batch

    # Config ``seed_collection_overlap_chunk``: each generated chunk
    # is graded on a CPU thread while the next one decodes, so the public
    # verifier -- LCB's every-test reward in particular -- does not idle the GPU.
    # The grades are the same calls on the same records; only their timing moves.
    overlap_chunk = int((config.get("seed_collection_overlap_chunk") or {}).get(args.benchmark, 0))
    grading = concurrent.futures.ThreadPoolExecutor(max_workers=1) if overlap_chunk else None
    grade_futures = []
    on_chunk = write_solutions
    solution_chunk = overlap_chunk or chunk_size
    if grading is not None:

        def on_chunk(batch, outputs):
            write_solutions(batch, outputs)
            rows = [done_solutions[item["key"]] for item in batch]
            grade_futures.append(grading.submit(compute_score_batch, rows))

    judge_seconds = [0.0]
    solution_seconds = generate_in_chunks(
        llm, solution_params, pending, solution_chunk, on_chunk
    )
    solutions = [done_solutions[item["key"]] for item in requests]

    # --- stage 2: public verifier -------------------------------------------
    grades_path = output / "grades_public.json"
    tail_started = time.monotonic()
    if grading is not None:
        by_key = {grade["key"]: grade for future in grade_futures for grade in future.result()}
        grading.shutdown()
        rest = [record for record in solutions if record["key"] not in by_key]
        by_key.update({grade["key"]: grade for grade in compute_score_batch(rest)})
        graded = [by_key[record["key"]] for record in solutions]
    else:
        graded = compute_score_batch(solutions)
    grading_tail_seconds = time.monotonic() - tail_started
    atomic_json(
        [{k: v for k, v in grade.items() if k != "native_feedback"} for grade in graded],
        grades_path,
    )
    feedback_by_key = {grade["key"]: grade for grade in graded}

    # --- stage 3: hindsight analyses ----------------------------------------
    analysis_requests = []
    for record in solutions:
        if record["key"] in done_analyses:
            continue
        grade = feedback_by_key[record["key"]]
        content = seed_prompts.analyzer_content_from_prompt(
            task_prompts.opening_user_text(problems[record["problem_uid"]], args.model),
            record["response"],
            grade["feedback"], bool(grade["correct"]),
        )
        prompt_ids = apply_chat_template(tokenizer, content)
        if len(prompt_ids) + int(config["skill_max_tokens"]) > int(config["max_model_len"]):
            raise RuntimeError(
                f"Analyzer prompt for {record['key']} exceeds the context window"
            )
        analysis_requests.append({
            "key": record["key"], "record": record, "content": content,
            "prompt_ids": prompt_ids,
            "seed": int(config["seed_analyzer_seed"]) + record["trial_index"],
        })

    def analysis_params(item):
        return SamplingParams(
            temperature=float(config["seed_analyzer_temperature"]), top_p=1.0, top_k=-1,
            max_tokens=int(config["skill_max_tokens"]), seed=item["seed"],
        )

    def write_analyses(batch, outputs):
        rows = []
        for item, output_value in zip(batch, outputs, strict=True):
            answer = output_value.outputs[0]
            valid, reason, skill = validate_seed_skill(
                answer.text, item["record"]["response"], tokenizer,
                int(config["skill_max_tokens"]),
            )
            rows.append({
                "key": item["key"],
                "problem_uid": item["record"]["problem_uid"],
                "source": item["record"]["source"],
                "task_type": item["record"]["task_type"],
                "prompt": item["content"],
                "prompt_tokens": len(item["prompt_ids"]),
                "completion_tokens": len(answer.token_ids),
                "skill": skill,
                "valid": valid,
                "rejection_reason": reason,
            })
        append_jsonl(rows, analyses_path)

    analyzer_seconds = generate_in_chunks(
        llm, analysis_params, analysis_requests, chunk_size, write_analyses
    )

    # --- stage 4: SFT parquet ------------------------------------------------
    analyses = {row["key"]: row for row in read_jsonl(analyses_path)}
    sft_rows = [
        {
            "prompt": analyses[record["key"]]["prompt"],
            "response": analyses[record["key"]]["skill"],
            "trajectory_key": record["key"],
            "problem_uid": record["problem_uid"],
            "source": record["source"],
        }
        for record in solutions
        if analyses.get(record["key"], {}).get("valid")
    ]
    if not sft_rows:
        raise RuntimeError("No valid SEED hindsight Skills were produced")
    import pandas as pd

    temporary = sft_path.with_suffix(".parquet.tmp")
    pd.DataFrame(sft_rows).to_parquet(temporary, index=False)
    temporary.replace(sft_path)

    summary = {
        "method": "seed_collection",
        "steps": len(questions),
        "generation_seconds": solution_seconds + judge_seconds[0] + analyzer_seconds,
        "solution_seconds": solution_seconds,
        "judge_seconds": judge_seconds[0],
        "analyzer_seconds": analyzer_seconds,
        "grading_tail_seconds": grading_tail_seconds,
        "verifier_trials": len(solutions),
        "valid_skills": len(sft_rows),
        "valid_skill_fraction": len(sft_rows) / len(solutions),
        "public_correct": sum(bool(grade["correct"]) for grade in graded),
        "solution_completion_tokens": sum(
            int(row["completion_tokens"]) for row in solutions
        ),
        "solution_prompt_tokens": sum(int(row["prompt_tokens"]) for row in solutions),
        "analyzer_prompt_tokens": sum(int(row["prompt_tokens"]) for row in analyses.values()),
        "analyzer_completion_tokens": sum(
            int(row["completion_tokens"]) for row in analyses.values()),
        "kv_cache_tokens": kv_cache_tokens(llm),
        "sft_parquet": str(sft_path),
    }
    mark_complete(output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
