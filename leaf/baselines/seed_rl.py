"""Run confidence-gated SEED RL on the budget left after collection.

One SEED policy is trained per benchmark.  The first
``round(seed_collection_fraction x U)`` whole updates of that benchmark's budget
went to hindsight collection - a 4-of-25 = 16% share taken
proportionally - so RL runs for the rest and the two stages together spend
exactly the same trials as SDPO, GRPO and SCoRe on that benchmark::

    math500  U=47  collection 8  RL 39
    lcb      U=13  collection 2  RL 11
    mbpp     U=47  collection 8  RL 39
    sciknow  U=19  collection 3  RL 16

The share is proportional so that small benchmarks such as ``lcb`` do
not spend a large fraction of their run on the reservation.

The initial LoRA is the hindsight-analyzer adapter from
:mod:`leaf.baselines.seed_sft`, found by convention at
``<baselines>/seed_sft/<benchmark>/final_adapter_path.txt``; ``--adapter`` overrides it.

The fork's SEED path imports its analyzer from ``leaf.baselines.seed_analyzer``, whose
prompt is written for code.  ``LEAF_SEED_ANALYZER=1`` is exported here so
that module delegates to the task-agnostic wording (:mod:`leaf.baselines.seed_prompts`);
the programming wording is unchanged when the variable is unset.

Resume: ``resume_mode=auto`` with a full checkpoint after every update, as in
:mod:`leaf.baselines.sdpo`. ``model.lora_adapter_path`` (the SFT adapter)
only initializes a fresh run: on resume the fork's worker overwrites the LoRA
with ``<ckpt>/actor/lora_adapter`` and verifies it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from leaf.core.project_config import (
    BENCHMARKS, MODELS, MODES, benchmark_budget, benchmark_rl_data_dir,
    detect_gpu_type, launch_manifest, load_config, parquet_mode,
    replace_prefix, write_launch_manifest,
)
from leaf.baselines.sdpo import (
    completed_updates, exit_if_done, hydra_overrides, run_trainer,
    smoke_question_batch,
)


def rl_data(config: dict, benchmark: str, mode: str, model: str) -> Path:
    path = (
        benchmark_rl_data_dir(config, benchmark, model)
        / f"seed_rl_{parquet_mode(mode)}.parquet"
    )
    if not path.exists():
        raise RuntimeError(f"{path} is missing; run leaf.data.prepare_verl_data first")
    return path


def default_adapter(output: Path) -> Path | None:
    # Same layout as above: the SFT stage is a sibling directory for this
    # benchmark, not a child of the RL stage's directory.
    marker = output.parent.parent / "seed_sft" / output.name / "final_adapter_path.txt"
    return Path(marker.read_text().strip()) if marker.exists() else None


def seed_rl_updates(config: dict, benchmark: str, mode: str) -> tuple[int, int, int]:
    """(total updates, updates reserved for collection, RL updates)."""
    if mode == "smoke":
        return int(config["smoke_updates"]), 0, 1
    if mode == "smoke_full":
        # The formal seed_rl parquet is the schedule; the run just stops early.
        return int(config["smoke_updates"]), 0, int(config["smoke_updates"])
    budget = benchmark_budget(config, benchmark)
    return (
        int(budget["updates"]),
        int(budget["seed_collection_updates"]),
        int(budget["seed_rl_updates"]),
    )


def seed_overrides(
    output: Path, updates: int, config: dict, model: str, benchmark: str,
    mode: str, gpu_type: str, data: Path, adapter: Path,
) -> list[str]:
    values = hydra_overrides(
        output, updates, config, model, benchmark, mode, gpu_type, data,
        condition="seed_rl",
    )
    replace_prefix(
        values, "actor_rollout_ref.actor.self_distillation.teacher_update_rate=",
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0",
    )
    replace_prefix(
        values, "trainer.group_name=", f"trainer.group_name=baselines-seed-{benchmark}"
    )
    replace_prefix(
        values, "trainer.experiment_name=",
        f"trainer.experiment_name=seed_{model}_{benchmark}_lora{config['lora_rank']}_{mode}",
    )
    values.extend([
        f"actor_rollout_ref.model.lora_adapter_path={adapter}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=seed",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config['seed_reference_kl']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.clip_ratio={config['seed_ppo_clip']}",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"+algorithm.seed.skill_max_tokens={config['skill_max_tokens']}",
        f"+algorithm.seed.analyzer_temperature={config['seed_analyzer_temperature']}",
        f"+algorithm.seed.analyzer_seed={config['seed_analyzer_seed']}",
        f"actor_rollout_ref.actor.seed.gate_scale={config['seed_gate_scale']}",
        f"actor_rollout_ref.actor.seed.opd_coefficient={config['seed_opd_coefficient']}",
    ])
    if mode == "smoke":
        batch = smoke_question_batch(config, "seed_smoke_question_batch_size", model)
        replace_prefix(values, "data.train_batch_size=", f"data.train_batch_size={batch}")
        replace_prefix(
            values, "actor_rollout_ref.actor.ppo_mini_batch_size=",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--gpu-type", choices=["a100", "h200"])
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    total_updates, reserved, updates = seed_rl_updates(
        config, args.benchmark, args.mode
    )
    if exit_if_done(output, updates, "seed_rl"):
        return

    adapter = (args.adapter or default_adapter(output))
    if adapter is None or not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            "SEED RL needs the hindsight-SFT adapter; run "
            "leaf.baselines.seed_sft first"
        )
    adapter = adapter.resolve()
    data = rl_data(config, args.benchmark, args.mode, args.model)
    overrides = seed_overrides(
        output, updates, config, args.model, args.benchmark, args.mode, gpu_type,
        data, adapter,
    )
    batch = int(
        smoke_question_batch(config, "seed_smoke_question_batch_size", args.model)
        if args.mode == "smoke"
        else config["question_batch_size"]
    )
    manifest = launch_manifest(
        config=config, method="seed_rl", model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=updates, overrides=overrides,
        extra={
            "stage": "rl",
            "train_files": str(data),
            "initial_adapter": str(adapter),
            "common_budget_updates": total_updates,
            "updates_reserved_for_collection": reserved,
            "rl_updates": updates,
            "verifier_trials": updates * batch * int(config["rollout_group_size"]),
            "resume_mode": "auto",
            "completed_updates_at_launch": completed_updates(output),
        },
    )
    write_launch_manifest(output, manifest)
    os.environ["LEAF_SEED_ANALYZER"] = "1"
    run_trainer(
        config=config, output=output, overrides=overrides,
        experiment=f"seed_{args.model}_{args.benchmark}_{args.mode}", config_name="sdpo",
        updates=updates, method="seed_rl", print_config=args.print_config,
    )


if __name__ == "__main__":
    main()
