"""Launch the "enhanced GRPO" baseline through the SDPO/verl fork.

One GRPO policy is trained per benchmark.  It runs from the fork's own GRPO entry point (``--config-name baseline_grpo``, the config behind
``third_party/SDPO/run_local_grpo.sh``) with the same LoRA, optimizer, sampling
and reward settings as the other verl baselines, so the only difference
between this arm and SDPO is the loss.

What "enhanced" means, concretely
---------------------------------
It is not a new algorithm; it is textbook GRPO plus the four fork-specific
options listed below, all of which are on here:

``algorithm.rollout_correction.rollout_is=token``
``algorithm.rollout_correction.rollout_is_threshold=2.0``
    The fork's rollout-correction module (``docs/algo/rollout_corr.md``).  The
    vLLM rollout policy and the FSDP training policy do not produce identical
    log probabilities, so every token's gradient is reweighted by a truncated
    per-token importance ratio between them.  ``bypass_mode=false`` (the fork
    default) keeps the decoupled three-policy formulation.  This is what
    ``run_local_grpo.sh`` passes and what makes the fork's GRPO stronger than a
    plain implementation.

``actor_rollout_ref.rollout.calculate_log_probs=True``
    Required by the above: the rollout engine must return its own log
    probabilities for the correction to be computable.

``algorithm.norm_adv_by_std_in_grpo=True``
    Group advantages are divided by the group standard deviation.  The fork's
    ``baseline_grpo.yaml`` ships this as ``False``; the baseline turns it on.

``actor_rollout_ref.actor.clip_ratio_high=0.28``
    Asymmetric ("clip-higher") PPO clipping, the fork's ``user.yaml`` default,
    kept explicit here so the report can quote it.

Everything else is deliberately the same as the other baselines: rank-64 LoRA
on ``all-linear``, ``rollout.n=8``, ``data.train_batch_size=32``,
``ppo_mini_batch_size=32``, temperature 1.0 / top-p 1.0, and
``leaf/baselines/verl_reward.py`` as the reward, i.e. the binary public-verifier score.

Resume: identical to :mod:`leaf.baselines.sdpo` - ``resume_mode=auto``,
a checkpoint (adapter + optimizer + extra) after every update with only the
newest kept, all three restored on resume, and an immediate exit once U updates
are done.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leaf.core.project_config import (
    detect_gpu_type, drop_prefix, launch_manifest, load_config, replace_prefix,
    write_launch_manifest,
)
from leaf.baselines.sdpo import (
    add_common_arguments, completed_updates, exit_if_done, hydra_overrides,
    resolve_updates, run_trainer, train_files,
)


def grpo_overrides(
    output: Path, updates: int, config: dict, model: str, benchmark: str,
    mode: str, gpu_type: str, data: Path,
) -> list[str]:
    values = hydra_overrides(
        output, updates, config, model, benchmark, mode, gpu_type, data,
        condition="grpo",
    )
    # SDPO's self-distillation knobs have no meaning under a GRPO loss.
    values = drop_prefix(values, "actor_rollout_ref.actor.self_distillation.")
    replace_prefix(
        values, "trainer.group_name=", f"trainer.group_name=baselines-grpo-{benchmark}"
    )
    replace_prefix(
        values, "trainer.experiment_name=",
        f"trainer.experiment_name=grpo_{model}_{benchmark}_lora{config['lora_rank']}_{mode}",
    )
    values.extend([
        # baseline_grpo.yaml pins max_model_len at 10240; user.yaml interpolates
        # it into several fields, so it has to move with the rest of the settings.
        f"max_model_len={config['max_model_len']}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.clip_ratio_high={config['grpo_clip_ratio_high']}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={config['grpo_norm_adv_by_std_in_grpo']}",
    ])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    updates = resolve_updates(config, args.benchmark, args.mode, args.updates)
    args.output.mkdir(parents=True, exist_ok=True)
    if exit_if_done(args.output, updates, "grpo"):
        return
    data = args.train_files or train_files(
        config, args.benchmark, args.mode, "grpo", args.model
    )
    overrides = grpo_overrides(
        args.output.resolve(), updates, config, args.model, args.benchmark,
        args.mode, gpu_type, data,
    )
    manifest = launch_manifest(
        config=config, method="grpo", model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=updates, overrides=overrides,
        extra={
            "train_files": str(data),
            "verl_config_name": "baseline_grpo",
            "enhanced_grpo_options": {
                "algorithm.rollout_correction.rollout_is": "token",
                "algorithm.rollout_correction.rollout_is_threshold":
                    config["rollout_is_clip"],
                "algorithm.rollout_correction.bypass_mode": False,
                "algorithm.norm_adv_by_std_in_grpo":
                    config["grpo_norm_adv_by_std_in_grpo"],
                "actor_rollout_ref.actor.clip_ratio_high":
                    config["grpo_clip_ratio_high"],
                "actor_rollout_ref.rollout.calculate_log_probs": True,
            },
            "verifier_trials": updates * int(config["question_batch_size"])
            * int(config["rollout_group_size"]),
            "resume_mode": "auto",
            "completed_updates_at_launch": completed_updates(args.output),
        },
    )
    write_launch_manifest(args.output, manifest)
    run_trainer(
        config=config, output=args.output.resolve(), overrides=overrides,
        experiment=f"grpo_{args.model}_{args.benchmark}_{args.mode}", config_name="baseline_grpo",
        updates=updates, method="grpo", print_config=args.print_config,
    )


if __name__ == "__main__":
    main()
