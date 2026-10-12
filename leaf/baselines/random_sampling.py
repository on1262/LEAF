"""Generate the Random Sampling baseline with plain vLLM (no verl).

One run per (model, benchmark).  The frozen ``trial_schedule.jsonl`` decides
every generation: ``n x 24`` trials balanced to within one per problem, each
with its own decode seed, and the trial's ``attempt_index`` is its aligned step,
so the 24 aligned steps fall out of the schedule rather than being imposed
afterwards.

Records are written in the canonical record shape (see
:func:`leaf.core.project_config.verifier_record`), so the offline grader can feed
``generations.jsonl`` straight to ``leaf.verifiers.answer_grader.grade_records``
without knowing which benchmark produced it.

Chat templates: ``enable_thinking=False`` is passed unconditionally.  Qwen3
honours it and renders an empty ``<think></think>`` block; the Olmo-3 template
ignores unknown kwargs and renders identically either way, so one code path
covers both families.

Prompt and decoding
-------------------
By default the opening prompt is the cell's own (``problem_prompt``) and the
decoding is the config's training temperature / top-p, which is what every
condition in a cell shares. ``--prompt-style lcb_think`` instead builds
LiveCodeBench's own chat prompt with a concise-``<think>`` line
(:func:`leaf.prompts.task_prompts.lcb_think_messages`), and ``--temperature`` /
``--top-p`` override the decoding. Both are recorded in the launch manifest.
They exist for prompt probes written to their own output directory: using them
on a cell's real Random Sampling run would leave it on a different prompt than the
cell's other conditions.

Resume
------
There is no trainer here, so resume is per trial.  ``generations.jsonl`` is
append-only and keyed by ``key``; a restarted chunk reads it, drops the keys it
already holds, and generates the rest in chunks of ``random_sampling_chunk_size``,
flushing after each chunk.  A chunk killed by the wall clock therefore loses at
most ``random_sampling_chunk_size`` generations, and a chunk that starts after the
schedule is exhausted exits immediately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from leaf.prompts import task_prompts
from leaf.verifiers import llm_judge
from leaf.explore.leaf_rollout import (
    model_max_model_len, model_max_num_seqs, model_response_max_tokens,
)
from leaf.core.project_config import (
    MODES,
    BENCHMARKS, GPU_TYPES, MODELS, apply_chat_messages, apply_chat_template,
    atomic_json, lcb_raw_rows,
    benchmark_budget, benchmark_dataset_path, benchmark_rl_data_dir,
    detect_gpu_type, launch_manifest, load_config, mark_complete, model_path,
    read_jsonl, verifier_record, write_launch_manifest,
)


def schedule_for(config: dict, benchmark: str, mode: str) -> list[dict]:
    """The frozen trial schedule, or a one-per-problem sweep in smoke mode."""
    problems = read_jsonl(benchmark_dataset_path(config, benchmark, mode))
    if not problems:
        raise RuntimeError(f"{benchmark}: no {mode} dataset file")
    if mode == "smoke":
        steps = int(config["smoke_steps"])
        base = int(config["decode_seed"]) + 800_000_000
        return [
            {
                "trial_index": step * len(problems) + index,
                "problem_uid": row["uid"],
                "attempt_index": step,
                "aligned_step": step,
                "seed": base + step * len(problems) + index,
            }
            for step in range(steps)
            for index, row in enumerate(problems)
        ]
    path = benchmark_rl_data_dir(config, benchmark) / "trial_schedule.jsonl"
    schedule = read_jsonl(path)
    if not schedule:
        raise RuntimeError(f"{path} is missing; run leaf.data.prepare_verl_data first")
    expected = int(benchmark_budget(config, benchmark)["random_sampling_generations"])
    if len(schedule) != expected:
        raise RuntimeError(f"{benchmark}: trial schedule has {len(schedule)}, want {expected}")
    if mode == "smoke_full":
        # The frozen schedule's own first steps, not a re-derived sweep: the
        # seeds and the problem order are the ones the formal run would spend.
        steps = int(config["smoke_steps"])
        return [row for row in schedule if int(row["aligned_step"]) < steps]
    return schedule


def kv_cache_tokens(llm) -> int | None:
    """KV cache capacity (tokens) of an offline engine, for the tuning analysis."""
    try:
        cache = llm.llm_engine.vllm_config.cache_config
        return int(cache.num_gpu_blocks) * int(cache.block_size)
    except Exception:  # noqa: BLE001 -- reporting only
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-type", choices=list(GPU_TYPES))
    parser.add_argument("--prompt-style", choices=("cell", "lcb_think"), default="cell",
                        help="cell: the cell's own opening prompt; lcb_think: LCB's "
                             "chat prompt with a concise-<think> line")
    parser.add_argument("--temperature", type=float, default=None,
                        help="default: the config's training_temperature")
    parser.add_argument("--top-p", type=float, default=None,
                        help="default: the config's training_top_p")
    args = parser.parse_args()

    config = load_config()
    temperature = (float(config["training_temperature"])
                   if args.temperature is None else args.temperature)
    top_p = float(config["training_top_p"]) if args.top_p is None else args.top_p
    if args.prompt_style == "lcb_think" and args.benchmark != "lcb":
        raise SystemExit("--prompt-style lcb_think is LiveCodeBench's prompt; use it on lcb")
    gpu_type = args.gpu_type or detect_gpu_type()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    problems = {
        row["uid"]: row
        for row in read_jsonl(benchmark_dataset_path(config, args.benchmark, args.mode))
    }
    schedule = schedule_for(config, args.benchmark, args.mode)
    requests = [
        {
            "key": f"random_sampling:{args.benchmark}:{item['trial_index']}",
            "row": problems[item["problem_uid"]],
            "trial_index": int(item["trial_index"]),
            "attempt_index": int(item["attempt_index"]),
            "aligned_step": int(item.get("aligned_step", item["attempt_index"])),
            "seed": int(item["seed"]),
        }
        for item in schedule
    ]

    manifest = launch_manifest(
        config=config, method="random_sampling", model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=None, overrides=[],
        extra={
            "generations": len(requests),
            "aligned_steps": max(item["aligned_step"] for item in requests) + 1,
            "temperature": temperature,
            "top_p": top_p,
            "prompt_style": args.prompt_style,
            "max_tokens": model_response_max_tokens(config, args.model, args.benchmark),
            "resume": "per-trial append to generations.jsonl",
        },
    )
    write_launch_manifest(output, manifest)

    target = output / "generations.jsonl"
    done = {row["key"] for row in read_jsonl(target)}
    pending = [item for item in requests if item["key"] not in done]
    if not pending:
        summary = {
            "method": "random_sampling",
            "steps": manifest["aligned_steps"],
            "generations": len(requests),
            "generation_seconds": 0.0,
            "event": "already_complete",
        }
        mark_complete(output, summary)
        print(json.dumps(summary), flush=True)
        return

    from vllm import LLM, SamplingParams

    base = model_path(config, args.model)
    llm = LLM(
        model=str(base), dtype="bfloat16",
        max_model_len=model_max_model_len(config, args.model),
        max_num_seqs=model_max_num_seqs(
            config, args.model, args.benchmark, key="random_sampling_max_num_seqs",
            condition="random_sampling",
        ),
        max_num_batched_tokens=int(config["random_sampling_max_num_batched_tokens"]),
        gpu_memory_utilization=float(config["random_sampling_gpu_memory_utilization"]),
        enforce_eager=False, enable_prefix_caching=True, enable_chunked_prefill=True,
        # Stats only: vLLM logs Running / Waiting / KV cache usage every 10 s.
        seed=int(config["decode_seed"]), disable_log_stats=False,
    )
    tokenizer = llm.get_tokenizer()
    if args.prompt_style == "lcb_think":
        raw = lcb_raw_rows(config)
        for item in pending:
            lcb = raw[item["row"]["meta"]["question_id"]]
            item["prompt_ids"] = apply_chat_messages(
                tokenizer,
                task_prompts.lcb_think_messages(
                    lcb["question_content"], lcb.get("starter_code") or ""),
            )
    else:
        for item in pending:
            item["prompt_ids"] = apply_chat_messages(
                tokenizer, task_prompts.opening_messages(item["row"], args.model)
            )

    chunk_size = int(config["random_sampling_chunk_size"])
    completion_tokens = 0
    seconds = 0.0
    judge_seconds = 0.0
    with target.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), chunk_size):
            batch = pending[start:start + chunk_size]
            params = [
                SamplingParams(
                    temperature=temperature, top_p=top_p, top_k=-1,
                    max_tokens=model_response_max_tokens(
                        config, args.model, args.benchmark),
                    seed=item["seed"],
                )
                for item in batch
            ]
            started = time.monotonic()
            started_at = time.time()
            results = llm.generate(
                [{"prompt_token_ids": item["prompt_ids"]} for item in batch],
                params, use_tqdm=True,
            )
            chunk_seconds = time.monotonic() - started
            seconds += chunk_seconds
            records = []
            for item, result in zip(batch, results, strict=True):
                answer = result.outputs[0]
                completion_tokens += len(answer.token_ids)
                records.append(verifier_record(
                    item["row"], answer.text, item["key"],
                    condition="random_sampling",
                    model=args.model,
                    mode=args.mode,
                    trial_index=item["trial_index"],
                    attempt_index=item["attempt_index"],
                    aligned_step=item["aligned_step"],
                    seed=item["seed"],
                    prompt_tokens=len(item["prompt_ids"]),
                    completion_tokens=len(answer.token_ids),
                    finish_reason=answer.finish_reason,
                ))
            # Open SciKnowEval: the same base engine judges the chunk, greedy,
            # and the verdict is written with the record it grades.
            chunk_judge = llm_judge.judge_records_offline(
                llm, tokenizer, records, config)
            judge_seconds += chunk_judge
            # Per-chunk timing (logging only): the schedule is sorted by
            # aligned step, so each chunk covers a contiguous step range.
            per_step: dict = {}
            for item, record in zip(batch, records, strict=True):
                entry = per_step.setdefault(str(item["aligned_step"]), {
                    "requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
                entry["requests"] += 1
                entry["prompt_tokens"] += int(record["prompt_tokens"])
                entry["completion_tokens"] += int(record["completion_tokens"])
            with (output / "chunk_timings.jsonl").open("a", encoding="utf-8") as timings:
                timings.write(json.dumps({
                    "slurm_job_id": manifest.get("slurm_job_id"),
                    "chunk_start": start, "requests": len(batch),
                    "started_at": started_at, "ended_at": time.time(),
                    "generate_seconds": chunk_seconds, "judge_seconds": chunk_judge,
                    "per_step": per_step,
                }) + "\n")
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    total = len({row["key"] for row in read_jsonl(target)})
    summary = {
        "method": "random_sampling",
        "steps": manifest["aligned_steps"],
        "generations": total,
        "generated_this_chunk": len(pending),
        "generation_seconds": seconds + judge_seconds,
        "judge_seconds": judge_seconds,
        "completion_tokens": completion_tokens,
        "prompt_tokens": sum(len(item["prompt_ids"]) for item in pending),
        "kv_cache_tokens": kv_cache_tokens(llm),
    }
    if total >= len(requests):
        mark_complete(output, summary)
    else:
        atomic_json(summary, output / "generation_summary.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
