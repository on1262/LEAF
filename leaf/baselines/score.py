"""Launch the SCoRe baseline through the SDPO/verl fork.

The two-turn rollout loop (``leaf/baselines/score_agent.py``) is task-agnostic -
it just re-prompts the model to revise its own answer with no correctness
signal - so it is reused unchanged, together with
``configs/baselines/score_agent_loop.yaml``.  Only the reward is specific to these benchmarks
(:mod:`leaf.baselines.score_reward`), because it has to grade four task types.

One SCoRe policy is trained per benchmark.  ``rollout.n=4`` trajectories x 2
turns = 8 sequences per question occurrence, the same eight the rollout-N=8
methods spend, so SCoRe runs to that benchmark's frozen U (math500 47, lcb 13,
mbpp 47, sciknoweval 19).

Resume: ``resume_mode=auto`` with a full checkpoint after every update, exactly
as in :mod:`leaf.baselines.sdpo`; the parquet is pre-materialised with a
fixed stage-1/stage-2 split per update index, so restarting mid-chain keeps
every occurrence in its original stage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from leaf.core.project_config import (
    PROJECT, benchmark_rl_data_dir, detect_gpu_type, launch_manifest,
    parquet_mode,
    load_config, replace_prefix, write_launch_manifest,
)
from leaf.baselines.sdpo import (
    add_common_arguments, completed_updates, exit_if_done, hydra_overrides,
    resolve_updates, response_max_tokens, run_trainer, score_total_tokens,
    smoke_question_batch,
)

SCORE_REWARD = PROJECT / "leaf" / "baselines" / "score_reward.py"
AGENT_CONFIG = PROJECT / "configs" / "baselines" / "score_agent_loop.yaml"


def score_data(config: dict, benchmark: str, mode: str, model: str) -> Path:
    path = (
        benchmark_rl_data_dir(config, benchmark, model)
        / f"score_{parquet_mode(mode)}.parquet"
    )
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing; run leaf.data.prepare_verl_data first"
        )
    return path


def score_overrides(
    output: Path, updates: int, config: dict, model: str, benchmark: str,
    mode: str, gpu_type: str, data: Path,
) -> list[str]:
    values = hydra_overrides(
        output, updates, config, model, benchmark, mode, gpu_type, data,
        condition="score",
    )
    replace_prefix(
        values, "data.max_response_length=",
        f"data.max_response_length={score_total_tokens(config, model, benchmark)}",
    )
    replace_prefix(
        values, "actor_rollout_ref.rollout.n=",
        f"actor_rollout_ref.rollout.n={config['score_rollout_group_size']}",
    )
    replace_prefix(
        values, "custom_reward_function.path=",
        f"custom_reward_function.path={SCORE_REWARD}",
    )
    replace_prefix(
        values, "trainer.group_name=", f"trainer.group_name=baselines-score-{benchmark}"
    )
    replace_prefix(
        values, "trainer.experiment_name=",
        f"trainer.experiment_name=score_{model}_{benchmark}_lora{config['lora_rank']}_{mode}",
    )
    values.extend([
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=1.0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.clip_ratio={config['seed_ppo_clip']}",
        "actor_rollout_ref.actor.score.enabled=True",
        f"actor_rollout_ref.actor.score.beta1={config['score_beta1']}",
        f"actor_rollout_ref.actor.score.beta2={config['score_beta2']}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=score_agent",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={AGENT_CONFIG}",
        (
            "+actor_rollout_ref.rollout.custom="
            f"{{score_response_length:{response_max_tokens(config, model, benchmark)}}}"
        ),
    ])
    if mode == "smoke":
        batch = smoke_question_batch(config, "score_smoke_question_batch_size", model)
        replace_prefix(values, "data.train_batch_size=", f"data.train_batch_size={batch}")
        replace_prefix(
            values, "actor_rollout_ref.actor.ppo_mini_batch_size=",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    updates = resolve_updates(config, args.benchmark, args.mode, args.updates)
    args.output.mkdir(parents=True, exist_ok=True)
    if exit_if_done(args.output, updates, "score"):
        return
    data = score_data(config, args.benchmark, args.mode, args.model)
    overrides = score_overrides(
        args.output.resolve(), updates, config, args.model, args.benchmark,
        args.mode, gpu_type, data,
    )
    batch = int(
        smoke_question_batch(config, "score_smoke_question_batch_size", args.model)
        if args.mode == "smoke"
        else config["question_batch_size"]
    )
    manifest = launch_manifest(
        config=config, method="score", model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=updates, overrides=overrides,
        extra={
            "train_files": str(data),
            "rollout_group_size": int(config["score_rollout_group_size"]),
            "turns_per_trajectory": 2,
            "trajectories_per_update": batch * int(config["score_rollout_group_size"]),
            "verifier_trials": updates * batch
            * int(config["score_rollout_group_size"]) * 2,
            "stage1_updates": max(
                1, math.floor(float(config["score_stage1_fraction"]) * updates)
            ),
            "resume_mode": "auto",
            "completed_updates_at_launch": completed_updates(args.output),
        },
    )
    write_launch_manifest(args.output, manifest)
    run_trainer(
        config=config, output=args.output.resolve(), overrides=overrides,
        experiment=f"score_{args.model}_{args.benchmark}_{args.mode}", config_name="sdpo",
        updates=updates, method="score", print_config=args.print_config,
    )


if __name__ == "__main__":
    main()
