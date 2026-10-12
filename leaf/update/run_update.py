"""Launch a LEAF policy update (:mod:`leaf.update.leaf_update`) through the SDPO/verl fork.

The verl side (FSDP PEFT LoRA actor, colocated vLLM server with the LoRA synced
on wake-up, checkpoint every step, ``resume_mode=auto``) is the verl baselines'
stack and the override list starts from the same base
(:func:`leaf.baselines.sdpo.hydra_overrides`). ``trainer.trainer_class``
swaps in :class:`leaf.update.leaf_update.LeafUpdateTrainer`.

One training step = one optimizer step: the step's rows (one student sample per
problem) always form one mini batch (``ppo_mini_batch_size`` = the step's
problems), and the loss modes normalise by the step's own response tokens.

Step counts come from :func:`leaf.update.leaf_update.plan_steps`; ``smoke_full``
runs ``--steps`` (default ``smoke_steps`` = 2) of the same batches.

SFT methods (``sft_ystar``) never generate: their step is 128 SFT rows, the vLLM
engine gets a short context (``sft_vllm_max_model_len``) and a small memory share
(``sft_vllm_gpu_memory_utilization``), and micro batches are packed by the student's tokens
(``sft_ppo_max_token_len``).

The ``leaf_update`` config block holds the H200 values; ``by_gpu.<gpu>``
overrides them for the card the job got (:func:`gpu_tuning`). The student sampling
concurrency is the config's ``update_student_max_num_seqs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leaf.core.project_config import (
    benchmark_dataset_path, benchmark_rl_data_dir, completion_marker, detect_gpu_type,
    drop_prefix, launch_manifest, load_config, mark_complete, replace_prefix,
    write_launch_manifest,
)
from leaf.explore.leaf_rollout import model_max_model_len
from leaf.update.leaf_update import LOSS_MODES, UPDATE_METHODS, is_sft, method_spec, plan_steps
from leaf.baselines.sdpo import (
    completed_updates, exit_if_done, hydra_overrides, run_trainer,
)

MODES = ("smoke_full", "formal")
BENCHMARKS = ("lcb", "math500", "sciknoweval")


def gpu_tuning(config: dict, gpu_type: str) -> dict:
    """The ``leaf_update`` tunables for one GPU type (``by_gpu`` over the H200 values)."""
    tuning = {key: value for key, value in config["leaf_update"].items() if key != "by_gpu"}
    tuning.update((config["leaf_update"].get("by_gpu") or {}).get(gpu_type, {}))
    return tuning


def training_dataset(config: dict, benchmark: str, data: Path) -> Path:
    """The problem rows x is built from: the rollout's dataset, as the data manifest records it."""
    manifest = data / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text()).get("dataset")
        if recorded:
            return Path(recorded)
    return benchmark_dataset_path(config, benchmark, "formal")


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def update_overrides(*, output: Path, config: dict, model: str, benchmark: str,
                     method: str, mode: str, steps: int, plan: dict, gpu_type: str,
                     dataset: Path, data: Path) -> list[str]:
    tuning = gpu_tuning(config, gpu_type)
    loss = method_spec(method, benchmark)["loss"]
    # verl builds its (unused) train dataloader from a parquet; the generalization
    # training parquet of the benchmark serves.
    parquet = benchmark_rl_data_dir(config, benchmark, model) / "train_formal.parquet"
    if not parquet.exists():
        raise RuntimeError(f"{parquet} is missing")
    values = hydra_overrides(
        output, steps, config, model, benchmark, "formal", gpu_type, parquet,
        condition="grpo",
    )
    values = drop_prefix(values, "actor_rollout_ref.actor.self_distillation.")
    values = drop_prefix(values, "algorithm.rollout_correction.")
    for prefix in ("reward_model.use_reward_loop=", "reward_model.launch_reward_fn_async=",
                   "reward_model.reward_loop_", "actor_rollout_ref.rollout.agent.num_workers="):
        if any(value.startswith(prefix) for value in values):
            values = drop_prefix(values, prefix)
    values += ["reward_model.use_reward_loop=False", "reward_model.launch_reward_fn_async=True"]
    # one student sample per problem: one mini batch covers every step
    rows_cap = plan["batch_problems"]
    max_num_seqs = int(config["update_student_max_num_seqs"][benchmark])
    max_len = model_max_model_len(config, model)
    if is_sft(method):
        ppo_tokens = int(tuning["sft_ppo_max_token_len"][benchmark])
        utilization = float(tuning["sft_vllm_gpu_memory_utilization"])
        # the vLLM server takes its context from max_position_embeddings; the FSDP actor's
        # RoPE frequencies do not depend on it
        max_len = min(max_len, int(tuning["sft_vllm_max_model_len"]))
    elif tuning["ppo_max_token_len"][benchmark] is None:
        raise SystemExit(f"ppo_max_token_len[{benchmark}] is null for this GPU: the cell does not run on it")
    else:
        ppo_tokens = int(tuning["ppo_max_token_len"][benchmark])
        utilization = float(tuning["vllm_gpu_memory_utilization"][benchmark])
    log_prob_tokens = int(tuning["log_prob_max_token_len"][benchmark])
    for prefix, value in [
        ("data.train_batch_size=",
         f"data.train_batch_size={min(rows_cap, parquet_rows(parquet))}"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=",
         f"actor_rollout_ref.actor.ppo_mini_batch_size={rows_cap}"),
        ("actor_rollout_ref.rollout.n=", "actor_rollout_ref.rollout.n=1"),
        ("actor_rollout_ref.rollout.gpu_memory_utilization=",
         f"actor_rollout_ref.rollout.gpu_memory_utilization={utilization}"),
        ("actor_rollout_ref.rollout.max_num_seqs=",
         f"actor_rollout_ref.rollout.max_num_seqs={max_num_seqs}"),
        ("actor_rollout_ref.rollout.max_model_len=",
         f"actor_rollout_ref.rollout.max_model_len={max_len}"),
        ("+actor_rollout_ref.model.override_config.max_position_embeddings=",
         f"+actor_rollout_ref.model.override_config.max_position_embeddings={max_len}"),
        # distillation micro batches are packed by the teacher's tokens (the longer forward), SFT by the student's
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
         f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_tokens}"),
        ("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=",
         f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={log_prob_tokens}"),
        ("trainer.group_name=", f"trainer.group_name=leaf_update-{method}-{benchmark}"),
        ("trainer.experiment_name=",
         f"trainer.experiment_name={method}_{model}_{benchmark}_{mode}"),
        ("trainer.project_name=", "trainer.project_name=leaf_update"),
    ]:
        replace_prefix(values, prefix, value)
    values.extend([
        f"max_model_len={max_len}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "algorithm.use_kl_in_reward=False",
        "trainer.balance_batch=False",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "+trainer.trainer_class=leaf.update.leaf_update.LeafUpdateTrainer",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={LOSS_MODES[loss]}",
        "actor_rollout_ref.actor.self_distillation.alpha=1.0",
        f"actor_rollout_ref.actor.self_distillation.vocab_chunk_size={tuning['kl_row_chunk_tokens'][benchmark]}",
        f"+leaf_update.method={method}",
        f"+leaf_update.model={model}",
        f"+leaf_update.benchmark={benchmark}",
        f"+leaf_update.dataset={dataset}",
        f"+leaf_update.data={data}",
        f"+leaf_update.output={output}",
        f"+leaf_update.steps={steps}",
    ])
    if mode == "smoke_full":
        # Smokes size vLLM from its logged KV capacity and usage.
        key = "actor_rollout_ref.rollout.disable_log_stats="
        values = [value for value in values if not value.startswith(key)]
        values.append(f"{key}False")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--method", required=True, choices=sorted(UPDATE_METHODS))
    parser.add_argument("--data", type=Path, required=True,
                        help="update data directory (leaf.update.build_contexts: successful_contexts.jsonl)")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="smoke_full only (default: config smoke_steps)")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = load_config()
    gpu_type = detect_gpu_type()
    data = args.data.resolve()
    plan = plan_steps(config, data, args.method)
    if args.mode == "formal":
        if args.steps:
            raise ValueError("--steps is for smoke_full runs only")
        steps = plan["steps"]
    else:
        steps = args.steps or int(config["smoke_steps"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method = args.method
    if exit_if_done(output, steps, method):
        # A chunk killed after the last checkpoint leaves no COMPLETE.json
        if not completion_marker(output).exists():
            mark_complete(output, {"method": method, "steps": completed_updates(output),
                                   "target_steps": steps, "generation_seconds": 0.0})
        return
    dataset = training_dataset(config, args.benchmark, data)
    overrides = update_overrides(
        output=output, config=config, model=args.model, benchmark=args.benchmark,
        method=args.method, mode=args.mode, steps=steps, plan=plan, gpu_type=gpu_type,
        dataset=dataset, data=data,
    )
    manifest = launch_manifest(
        config=config, method=method, model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=steps, overrides=overrides,
        extra={
            "dataset": str(dataset), "data": str(data), "plan": plan,
            "loss": method_spec(args.method, args.benchmark),
            "verl_config_name": "baseline_grpo", "resume_mode": "auto",
            "completed_updates_at_launch": completed_updates(output),
        },
    )
    write_launch_manifest(output, manifest)
    print(json.dumps({"event": "update_launch", "plan": plan, "steps": steps,
                      "overrides": overrides}), flush=True)
    run_trainer(
        config=config, output=output, overrides=overrides,
        experiment=f"{method}_{args.model}_{args.benchmark}_{args.mode}",
        config_name="baseline_grpo", updates=steps, method=method,
        print_config=args.print_config,
    )


if __name__ == "__main__":
    main()
