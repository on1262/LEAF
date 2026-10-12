"""Launch the SDPO baseline through the vendored SDPO/verl fork.

One SDPO policy is trained per benchmark, since every experiment is
independent single-GPU jobs, one per benchmark; U comes from that benchmark's frozen budget
(math500 47, lcb 13, mbpp 47, sciknoweval 19 updates).

This module also owns the override list every other verl-based runner
starts from (:func:`hydra_overrides`).

Resume
------
Formal runs are executed as a sequence of time-capped chunks (5 h each),
each started after the previous one ends whatever its exit state, so a chunk that
hits the wall clock must be picked up by the next one (re-running the run script).  Three things make that work and they are set here for every
verl runner:

* ``trainer.resume_mode=auto`` - verl reads
  ``<output>/checkpoints/latest_checkpointed_iteration.txt`` and restores the
  step count, the dataloader position, the adapter and the optimizer;
* a checkpoint after every update (``trainer.save_freq=1``) holding the LoRA
  adapter, the optimizer and the extra state (RNG, lr scheduler, ``data.pt``);
  the fork keeps only the newest ``global_step_*`` directory, pruning older
  ones only after the tracker points at the new one.  ``load_contents`` is
  ``[optimizer,extra]`` and the worker loads ``<ckpt>/actor/lora_adapter`` (and
  SDPO's ``ema_lora_adapter``) back into the FSDP module, verifies it tensor by
  tensor and fails loudly if anything is missing
  (``third_party/SDPO/verl/utils/checkpoint/lora_resume.py``).  Loading only
  ``[extra]`` would restart the LoRA from B = 0 on every mid-run resume;
* ``data.shuffle=False`` - the frozen question schedule *is* the order, so a
  resumed run continues the same sequence of questions.

A chunk that starts after the run already reached U exits immediately instead of
burning its allocation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from leaf.core.project_config import (
    MODES,
    BENCHMARKS, GPU_TYPES, MODELS, PROJECT, benchmark_budget,
    benchmark_rl_data_dir, detect_gpu_type, execution_profile, launch_manifest,
    load_config, mark_complete, model_path, write_launch_manifest,
)

REWARD = PROJECT / "leaf" / "baselines" / "verl_reward.py"


# --------------------------------------------------------------------------- #
# resume helpers
# --------------------------------------------------------------------------- #


def checkpoint_dir(output: Path) -> Path:
    return Path(output) / "checkpoints"


def completed_updates(output: Path) -> int:
    marker = checkpoint_dir(output) / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return 0
    try:
        return int(marker.read_text().strip())
    except ValueError:
        return 0


def exit_if_done(output: Path, updates: int, method: str) -> bool:
    """True when a chained chunk has nothing left to do."""
    done = completed_updates(output)
    if done < updates:
        return False
    print(json.dumps({
        "event": "already_complete", "method": method,
        "completed_updates": done, "target_updates": updates,
    }), flush=True)
    return True


# --------------------------------------------------------------------------- #
# overrides
# --------------------------------------------------------------------------- #


def train_files(
    config: dict, benchmark: str, mode: str, method: str, model: str
) -> Path:
    data = benchmark_rl_data_dir(config, benchmark, model)
    path = data / ("smoke.parquet" if mode == "smoke" else "train_formal.parquet")
    if mode != "smoke" and method not in {"sdpo", "grpo"}:
        raise ValueError(f"{method} builds its own formal train file")
    if not path.exists():
        raise RuntimeError(f"{path} is missing; run leaf.data.prepare_verl_data first")
    return path


def checkpoint_freqs(config: dict, benchmark: str, mode: str) -> tuple[int, int]:
    """Checkpoint every update, optimizer included, in every mode.

    The per-benchmark ``checkpoint_save_freq`` / ``optimizer_save_freq`` frozen
    in the budget manifest are no longer used: a resume needs the optimizer of
    the exact step it restarts from, and a chunk killed at the wall clock should
    lose at most the update in flight. Only the newest checkpoint is kept.
    """
    return 1, 1


def ppo_max_token_len(config: dict, model: str, benchmark: str, condition: str) -> int:
    """Tokens per update micro batch: one whole longest-possible sequence.

    With ``use_remove_padding`` the budget counts real tokens, so a value of
    prompt cap + response cap guarantees the longest sequence a run can produce
    still fits in one micro batch while letting shorter ones pack together.
    """
    response = response_max_tokens(config, model, benchmark)
    if condition == "score":
        response = max(response, score_total_tokens(config, model, benchmark))
    floor = int(config["ordinary_prompt_max_tokens"]) + response
    # A cell may pack more than one sequence per micro batch where the card has
    # room for it: the update peak measured 1.42 MiB per token, so the floor
    # leaves most of an H200 idle on the short benchmarks.
    override = _cell_override(config, "ppo_max_token_len_overrides", model, benchmark, condition)
    return max(floor, int(override)) if override else floor


def _cell_override(config: dict, key: str, model: str, benchmark: str, condition: str):
    """``config[key][model][benchmark]``: an int, or ``{"default": int,
    "by_condition": {condition: int}}`` for a per-condition value."""
    value = ((config.get(key) or {}).get(model, {}) or {}).get(benchmark)
    if isinstance(value, dict):
        value = (value.get("by_condition") or {}).get(condition, value.get("default"))
    return value


def log_prob_max_token_len(config: dict, model: str, benchmark: str, condition: str) -> int:
    """Old / ref log-prob micro batch tokens: the update budget unless the config
    sets ``log_prob_max_token_len_overrides`` for the cell (never below it)."""
    update = ppo_max_token_len(config, model, benchmark, condition)
    override = _cell_override(config, "log_prob_max_token_len_overrides", model, benchmark, condition)
    return max(update, int(override)) if override else update


def smoke_question_batch(config: dict, key: str, model: str) -> int:
    """Questions per smoke update for the conditions that shrink their batch.

    SCoRe and SEED-RL smoke on a two-question batch because Qwen and Olmo write
    long two-turn responses; at that size the step is mostly fixed overhead, so
    the measured seconds per update do not scale to the formal batch. A model
    whose sequences are short enough smokes at the formal batch instead, and
    then the projection is a measurement.
    """
    by_model = (config.get(f"{key}_by_model") or {})
    return int(by_model.get(model, config[key]))


def score_total_tokens(config: dict, model: str, benchmark: str) -> int:
    """SCoRe writes two turns plus a bridge into one response field.

    Both turns get the cell's own generation cap, so the total follows the cap
    rather than a constant; at the shared 8,192 cap the total is 16,896.
    """
    per_turn = response_max_tokens(config, model, benchmark)
    return 2 * per_turn + int(config.get("score_bridge_tokens", 512))


def response_max_tokens(config: dict, model: str, benchmark: str) -> int:
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
    return int(override.get(benchmark, base))


def ray_init_overrides(config: dict) -> list[str]:
    """Size the local Ray cluster to this job's allocation, not to the node.

    Left at ``num_cpus: null`` Ray counts every core on the host and prestarts
    one Python worker per core. On a 192-core host that is 192 workers booting
    inside a 16-core, 120 GB job; with two such jobs on one host none of the 384
    registered before Ray's timeout and the driver never got past its own
    start-up. verl's own config says to pin this on a shared machine. The
    object store is sized the same way: Ray otherwise defaults it to a share of
    /dev/shm that can exceed the job's memory limit.
    """
    try:
        cpus = len(os.sched_getaffinity(0))  # the cgroup's cpuset, if any
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    # NUM_CPUS (run/common.sh) wins, then a scheduler allocation, then the affinity mask
    cpus = int(os.environ.get("NUM_CPUS") or os.environ.get("SLURM_CPUS_PER_TASK") or cpus)
    store = int(float(config.get("ray_object_store_gib", 16)) * 2**30)
    return [
        f"ray_kwargs.ray_init.num_cpus={cpus}",
        f"+ray_kwargs.ray_init.object_store_memory={store}",
    ]


def batch_sizes(config: dict, mode: str, data: Path) -> tuple[int, int]:
    """Question batch and PPO mini-batch, clamped to the rows actually present.

    The smoke parquets hold only that benchmark's smoke subset (10 rows for
    lcb), so asking verl for the formal batch of 32 would either abort or
    silently wrap the dataloader and invalidate the timing the formal chain is
    sized from. ``smoke_full`` reads the formal parquet, so it keeps the formal
    batch and is clamped by nothing.
    """
    batch = int(config["question_batch_size"])
    mini = int(config["ppo_mini_batch_size"])
    if mode == "smoke":
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(data).metadata.num_rows
        batch = max(1, min(batch, rows))
        mini = max(1, min(mini, batch))
    return batch, mini


def hydra_overrides(
    output: Path, updates: int, config: dict, model: str, benchmark: str,
    mode: str, gpu_type: str, data: Path, condition: str = "default",
) -> list[str]:
    """The shared verl override list (SDPO settings as the base)."""
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    profile = execution_profile(config, gpu_type, model, benchmark, condition)
    base_model = model_path(config, model)
    save_freq, optimizer_save_freq = checkpoint_freqs(config, benchmark, mode)
    train_batch, mini_batch = batch_sizes(config, mode, data)
    values = [
        f"data.train_files={data}",
        f"data.val_files={data}",
        f"data.train_batch_size={train_batch}",
        f"data.max_prompt_length={config['ordinary_prompt_max_tokens']}",
        f"data.max_response_length={response_max_tokens(config, model, benchmark)}",
        f"data.seed={config['schedule_seed']}",
        # The frozen schedule is the order; never reshuffle, so a resumed chunk
        # continues the same question sequence.
        "data.shuffle=False",
        "data.validation_shuffle=False",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "data.dataloader_num_workers=2",
        "data.apply_chat_template_kwargs.enable_thinking=False",
        f"actor_rollout_ref.model.path={base_model}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # vLLM silently clamps max_tokens to what the context leaves, so the
        # context has to cover the longest response the condition can write.
        "+actor_rollout_ref.model.override_config.max_position_embeddings="
        f"{max(int(config['max_model_len']), ppo_max_token_len(config, model, benchmark, condition))}",
        f"actor_rollout_ref.model.lora_rank={config['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={config['lora_alpha']}",
        f"actor_rollout_ref.model.lora_dropout={config['lora_dropout']}",
        f"actor_rollout_ref.model.target_modules={config['lora_target_modules']}",
        f"actor_rollout_ref.actor.optim.lr={config['learning_rate']}",
        f"actor_rollout_ref.actor.optim.betas=[{config['adam_beta1']},{config['adam_beta2']}]",
        f"actor_rollout_ref.actor.optim.weight_decay={config['weight_decay']}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch}",
        # Token-based batching packs short sequences into one micro batch.
        f"actor_rollout_ref.actor.use_dynamic_bsz={config['use_dynamic_bsz']}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu="
        f"{ppo_max_token_len(config, model, benchmark, condition)}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="
        f"{log_prob_max_token_len(config, model, benchmark, condition)}",
        # Pack without padding so the token budget above counts real tokens.
        f"actor_rollout_ref.model.use_remove_padding={config['use_remove_padding']}",
        # sleep level 1: keep vLLM's base weights resident so only the rank-64
        # LoRA is re-synced each step. layered_summon refuses to run unless
        # vLLM loaded the base weights itself, so the default `dummy` init --
        # which leaves the base for the first full sync to supply -- has to
        # give way to a real load from disk.
        f"actor_rollout_ref.rollout.layered_summon={profile['layered_summon']}",
        *(["actor_rollout_ref.rollout.load_format=safetensors"]
          if profile["layered_summon"] else []),
        (
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
            f"{profile['ppo_micro_batch_size_per_gpu']}"
        ),
        f"actor_rollout_ref.actor.grad_clip={config['gradient_clip']}",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={profile['actor_param_offload']}",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        "actor_rollout_ref.ref.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.checkpoint.save_contents=[optimizer,extra]",
        # All three parts on resume; the fork asserts optimizer and extra are
        # listed and loads/verifies the adapter itself.
        "actor_rollout_ref.actor.checkpoint.load_contents=[optimizer,extra]",
        (
            "actor_rollout_ref.actor.checkpoint.optimizer_save_freq="
            f"{optimizer_save_freq}"
        ),
        f"actor_rollout_ref.actor.self_distillation.distillation_topk={config['distillation_topk']}",
        f"actor_rollout_ref.actor.self_distillation.vocab_chunk_size={config['sdpo_vocab_chunk_tokens']}",
        "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True",
        "actor_rollout_ref.actor.self_distillation.alpha=1.0",
        f"actor_rollout_ref.actor.self_distillation.is_clip={config['rollout_is_clip']}",
        f"actor_rollout_ref.actor.self_distillation.teacher_update_rate={config['teacher_update_rate']}",
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        f"actor_rollout_ref.actor.self_distillation.max_reprompt_len={config['augmented_prompt_max_tokens']}",
        f"actor_rollout_ref.rollout.n={config['rollout_group_size']}",
        f"actor_rollout_ref.rollout.temperature={config['training_temperature']}",
        f"actor_rollout_ref.rollout.top_p={config['training_top_p']}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.data_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={profile['vllm_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_model_len={config['max_model_len']}",
        f"actor_rollout_ref.rollout.max_num_seqs={profile['vllm_max_num_seqs']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={profile['vllm_max_num_batched_tokens']}",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "algorithm.rollout_correction.rollout_is=token",
        f"algorithm.rollout_correction.rollout_is_threshold={config['rollout_is_clip']}",
        f"custom_reward_function.path={REWARD}",
        "custom_reward_function.name=compute_score",
        "reward_model.use_reward_loop=False",
        # The public verifier runs on CPU; overlapping it with old_log_prob and
        # ref leaves the GPU busy instead of idle while code executes.
        "reward_model.launch_reward_fn_async=True",
        "trainer.project_name=baselines",
        f"trainer.group_name=baselines-sdpo-{benchmark}",
        f"trainer.experiment_name=sdpo_{model}_{benchmark}_lora{config['lora_rank']}_{mode}",
        "trainer.logger=[console,file]",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"trainer.total_training_steps={updates}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        f"trainer.save_freq={save_freq}",
        "trainer.max_actor_ckpt_to_keep=null",
        # Chained chunks: pick up the latest checkpoint instead of restarting.
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={checkpoint_dir(output)}",
        f"trainer.rollout_data_dir={output / 'rollouts'}",
        f"trainer.validation_data_dir={output / 'validation'}",
        *ray_init_overrides(config),
    ]
    # Hydra rejects a key given twice: replace in place, append what is new.
    extra = list(reward_loop_overrides(config, benchmark))
    if config.get("vllm_log_stats"):
        # vLLM's server prints Running / Waiting / KV cache usage every 10 s.
        extra.append("actor_rollout_ref.rollout.disable_log_stats=False")
    for override in extra:
        key = override.split("=", 1)[0] + "="
        hits = [i for i, value in enumerate(values) if value.startswith(key)]
        if hits:
            values[hits[0]] = override
        else:
            values.append(override)
    return values


def reward_loop_overrides(config: dict, benchmark: str) -> list[str]:
    """Score each trajectory inside the agent loop, overlapping reward with decoding.

    Only for benchmarks listed in the config's ``reward_loop_benchmarks`` (the
    generalization runs: LiveCodeBench, whose all-tests reward otherwise idles the GPU for minutes
    per update); never for a benchmark with an LLM-judge phase, which must run
    before the reward. Later overrides win in Hydra, so these replace the two
    reward lines above. :mod:`leaf.baselines.reward_loop` keeps the legacy
    manager's reward inputs.
    """
    if benchmark not in (config.get("reward_loop_benchmarks") or ()):
        return []
    if benchmark == "sciknoweval":
        raise RuntimeError("the SciKnowEval judge must run before reward; no reward loop")
    return [
        "reward_model.use_reward_loop=True",
        "reward_model.launch_reward_fn_async=False",
        "reward_model.reward_loop_source=importlib",
        f"reward_model.reward_loop_module_path={PROJECT / 'leaf' / 'baselines' / 'reward_loop.py'}",
        "reward_model.reward_loop_class_name=LegacyInputsRewardManager",
        f"actor_rollout_ref.rollout.agent.num_workers={int(config.get('reward_loop_agent_workers', 4))}",
    ]


def resolve_updates(
    config: dict, benchmark: str, mode: str, requested: int | None
) -> int:
    """U comes from the benchmark's frozen budget, never from the CLI.

    Only the ``formal`` mode spends the whole budget; ``smoke`` and
    ``smoke_full`` both stop after ``smoke_updates``.
    """
    if mode == "formal":
        updates = int(benchmark_budget(config, benchmark)["updates"])
        if requested is not None and requested != updates:
            raise RuntimeError(
                f"Requested updates differ from {benchmark}'s frozen U={updates}"
            )
        return updates
    return requested or int(config["smoke_updates"])


# --------------------------------------------------------------------------- #
# launching
# --------------------------------------------------------------------------- #


def training_environment(config: dict, output: Path, experiment: str) -> dict:
    environment = os.environ.copy()
    environment.setdefault("TASK", "leaf")
    environment.setdefault("EXPERIMENT", experiment)
    environment.setdefault("VERL_FILE_LOGGER_PATH", str(output / "metrics.jsonl"))
    # The fork's naive reward manager overlaps candidate grading across this
    # many threads; one thread per candidate would serialise the verifier.
    environment.setdefault(
        "LEAF_VERIFIER_CANDIDATE_WORKERS", str(config["rollout_verifier_workers"])
    )
    # leaf.baselines.reward_loop: candidate threads over all agent-loop reward workers.
    environment.setdefault(
        "LEAF_REWARD_CANDIDATE_WORKERS",
        str(config.get("lcb_verifier_workers", config["rollout_verifier_workers"])),
    )
    return environment


def run_trainer(
    *, config: dict, output: Path, overrides: list[str], experiment: str,
    config_name: str, updates: int, method: str, print_config: bool,
) -> None:
    if config.get("verl_kv_poller") and not any(
            value.startswith(("trainer.trainer_class=", "+trainer.trainer_class="))
            for value in overrides):
        # Logging only: the plain trainer plus a vLLM KV-cache stats poller.
        overrides = [*overrides, "+trainer.trainer_class=leaf.baselines.kv_logging.KvLoggingTrainer"]
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-name", config_name, *overrides,
    ]
    if print_config:
        command += ["--cfg", "job", "--resolve"]
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    if print_config:
        raise SystemExit(subprocess.call(
            command, cwd=PROJECT / "third_party/SDPO",
            env=training_environment(config, output, experiment),
        ))
    started = time.monotonic()
    code = subprocess.call(
        command, cwd=PROJECT / "third_party/SDPO",
        env=training_environment(config, output, experiment),
    )
    seconds = time.monotonic() - started
    done = completed_updates(output)
    if code == 0 and done >= updates:
        mark_complete(output, {
            "method": method, "steps": done, "target_steps": updates,
            "generation_seconds": seconds,
        })
    raise SystemExit(code)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    # Tuning runs need the formal batch on formal-length problems without
    # claiming the formal budget: point a smoke run at its own parquet.
    parser.add_argument("--train-files", type=Path)
    parser.add_argument("--gpu-type", choices=list(GPU_TYPES))
    parser.add_argument("--print-config", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    updates = resolve_updates(config, args.benchmark, args.mode, args.updates)
    args.output.mkdir(parents=True, exist_ok=True)
    if exit_if_done(args.output, updates, "sdpo"):
        return
    data = args.train_files or train_files(
        config, args.benchmark, args.mode, "sdpo", args.model
    )
    overrides = hydra_overrides(
        args.output.resolve(), updates, config, args.model, args.benchmark,
        args.mode, gpu_type, data, condition="sdpo",
    )
    manifest = launch_manifest(
        config=config, method="sdpo", model=args.model, benchmark=args.benchmark,
        mode=args.mode, gpu_type=gpu_type, updates=updates, overrides=overrides,
        extra={
            "train_files": str(data),
            "verifier_trials": updates * int(config["question_batch_size"])
            * int(config["rollout_group_size"]),
            "resume_mode": "auto",
            "completed_updates_at_launch": completed_updates(args.output),
        },
    )
    write_launch_manifest(args.output, manifest)
    run_trainer(
        config=config, output=args.output.resolve(), overrides=overrides,
        experiment=f"sdpo_{args.model}_{args.benchmark}_{args.mode}", config_name="sdpo",
        updates=updates, method="sdpo", print_config=args.print_config,
    )


if __name__ == "__main__":
    main()
