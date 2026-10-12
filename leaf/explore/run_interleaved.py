"""Launch LEAF with interleaved policy updates (:mod:`leaf.explore.interleaved_trainer`)
through the SDPO/verl fork.

The verl side is the stack of the LEAF policy update (FSDP LoRA actor, colocated vLLM
server with the LoRA synced on wake-up, checkpoint every step, ``resume_mode=auto``);
the override list starts from :func:`leaf.baselines.sdpo.hydra_overrides`
and ``trainer.trainer_class`` swaps in the interleaved trainer. The config comes
from ``LEAF_CONFIG``:

* ``configs/explore/<model>_seed1.json`` (update every step) /
  ``configs/ablation/explore_<model>.json`` (update every 2 / 5 steps) -- the
  exploration benchmarks MATH500 / LCB / MBPP+ / SciKnowEval. Those configs carry no
  ``leaf_update`` block, so the update tunables are read from the same model's
  ``configs/general/leaf_rollout_math500_<model>_seed1.json`` (a benchmark it lacks,
  MBPP+, takes the largest per-benchmark value);
* a config with its own ``leaf_update`` block (a training set) uses that block.

Formal steps = config ``interleaved_steps`` (else 20), learning rate = config
``interleaved_lr`` (else 1e-5); ``--steps`` for the smoke modes. The update
interval is ``--update-interval``, else config ``interleaved_update_interval``, else 1
(an update every step); ``ppo_mini_batch_size`` = the number of problems, so an
update's solved set (at most every problem) is one mini batch = one optimizer step.
``--mode smoke`` is ``smoke_full`` on a ``--dataset`` subset.

vLLM (:func:`tuning`): utilization = ``leaf_update.vllm_gpu_memory_utilization``
(H200 0.78, A100 0.5). The rollout concurrency is the condition's standalone
``max_num_seqs_by_condition`` value (measured at ``gpu_memory_utilization`` 0.9)
scaled by the KV share left at the colocated utilization,
``(u - w) / (u0 - w)`` with ``w`` = bf16 weights / card memory; the student
sampling concurrency is the rollout value; the server gets the larger of the two.
An ``interleaved`` config block (``{key: {benchmark: value}}``, optional ``by_gpu``)
overrides any of these after a smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leaf.core.project_config import (
    PROJECT, benchmark_dataset_path, benchmark_rl_data_dir, detect_gpu_type, drop_prefix,
    launch_manifest, load_config, replace_prefix, write_launch_manifest,
)
from leaf import paths
from leaf.explore.leaf_rollout import model_max_model_len, model_max_num_seqs
from leaf.explore.interleaved_trainer import CONDITION, LEARNING_RATE, METHOD, STEPS, update_loss
from leaf.baselines.sdpo import completed_updates, exit_if_done, hydra_overrides, run_trainer

MODES = ("smoke", "smoke_full", "formal")
BENCHMARKS = ("math500", "lcb", "mbpp", "sciknoweval")
TASK_TYPES = {"math500": "math", "lcb": "code_lcb", "mbpp": "code_mbpp", "sciknoweval": "sciqa"}
SIBLING_CONFIGS = {model: f"general/leaf_rollout_math500_{model}_seed1.json"
                   for model in ("qwen3_8b", "olmo3_7b")}
# bf16 weights (GiB) and card memory (GiB, as torch reports it) for the KV-share scaling
WEIGHTS_GIB = {"qwen3_8b": 15.3, "olmo3_7b": 13.6}
CARD_GIB = {"h200": 139.8, "a100": 79.3}
UPDATE_KEYS = ("vllm_gpu_memory_utilization", "ppo_max_token_len", "log_prob_max_token_len",
               "kl_row_chunk_tokens")


def dataset_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _per_benchmark(value, benchmark: str):
    if isinstance(value, dict):
        if benchmark in value:
            return value[benchmark]
        return max(v for v in value.values() if v is not None)
    return value


def update_block(config: dict, model: str) -> tuple[dict, str]:
    """``leaf_update`` of the config, else of the model's seed-1 LEAF rollout config."""
    if "leaf_update" in config:
        return config["leaf_update"], "config"
    sibling = PROJECT / "configs" / SIBLING_CONFIGS[model]
    return paths.load_json(sibling)["leaf_update"], str(sibling)


def tuning(config: dict, model: str, benchmark: str, gpu_type: str, problems: int) -> dict:
    block, source = update_block(config, model)
    merged = {k: v for k, v in block.items() if k != "by_gpu"}
    merged.update((block.get("by_gpu") or {}).get(gpu_type, {}))
    out: dict = {"sources": {}}
    for key in UPDATE_KEYS:
        out[key] = _per_benchmark(merged[key], benchmark)
        out["sources"][key] = f"{source}:leaf_update.{key}"
    standalone_u = float(config["gpu_memory_utilization"])
    tuned = model_max_num_seqs(config, model, benchmark, condition=CONDITION)
    share = WEIGHTS_GIB[model] / CARD_GIB[gpu_type]
    scale = min(1.0, (float(out["vllm_gpu_memory_utilization"]) - share) / (standalone_u - share))
    out["rollout_max_num_seqs"] = max(1, min(problems, int(tuned * scale)))
    out["sources"]["rollout_max_num_seqs"] = (
        f"max_num_seqs_by_condition[{model}][{CONDITION}][{benchmark}]={tuned} (u0={standalone_u}) "
        f"x KV share {scale:.3f}")
    overrides = dict(config.get("interleaved") or {})
    by_gpu = (overrides.pop("by_gpu", None) or {}).get(gpu_type, {})
    for layer, name in ((overrides, "interleaved"), (by_gpu, f"interleaved.by_gpu.{gpu_type}")):
        for key, value in layer.items():
            if key.startswith("_"):
                continue
            picked = value.get(benchmark) if isinstance(value, dict) else value
            if picked is not None:
                out[key] = picked
                out["sources"][key] = name
    if "student_max_num_seqs" not in out:
        out["student_max_num_seqs"] = out["rollout_max_num_seqs"]
        out["sources"]["student_max_num_seqs"] = "rollout_max_num_seqs"
    out["server_max_num_seqs"] = max(int(out["rollout_max_num_seqs"]), int(out["student_max_num_seqs"]))
    return out


def dataloader_parquet(config: dict, benchmark: str, model: str) -> Path:
    """verl builds an (unused) train dataloader: the benchmark's parquet of the config's root."""
    path = benchmark_rl_data_dir(config, benchmark, model) / "train_formal.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} does not exist (run/data_prepare.sh roots builds it)")
    return path


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def set_override(values: list[str], key: str, value) -> None:
    prefix = f"{key}="
    hits = [i for i, v in enumerate(values) if v.startswith(prefix)]
    if len(hits) > 1:
        raise RuntimeError(f"{key} given twice")
    if hits:
        values[hits[0]] = f"{key}={value}"
    else:
        values.append(f"{key}={value}")


def interleaved_steps(config: dict) -> int:
    """Formal steps: ``interleaved_steps`` of the config (exploration: 24), else 20."""
    return int(config.get("interleaved_steps", STEPS))


def interleaved_lr(config: dict) -> float:
    """``interleaved_lr`` of the config (exploration: 1e-6), else 1e-5."""
    return float(config.get("interleaved_lr", LEARNING_RATE))


def update_interval(config: dict, override: int | None = None) -> int:
    """Policy update every ``N`` steps: ``--update-interval``, else config
    ``interleaved_update_interval``, else 1 (every step)."""
    value = int(override if override is not None else config.get("interleaved_update_interval", 1))
    if value < 1:
        raise ValueError(f"update interval must be >= 1, got {value}")
    return value


def method_label(interval: int) -> str:
    """``interleaved_every_<N>``: the condition name of update interval ``N``."""
    return f"{METHOD}_every_{interval}"


def update_overrides(*, output: Path, config: dict, model: str, benchmark: str, mode: str,
                     steps: int, gpu_type: str, dataset: Path, interval: int = 1) -> tuple[list[str], dict]:
    problems = dataset_rows(dataset)
    tune = tuning(config, model, benchmark, gpu_type, problems)
    parquet = dataloader_parquet(config, benchmark, model)
    values = hydra_overrides(output, steps, config, model, benchmark, "formal", gpu_type, parquet,
                             condition="grpo")
    values = drop_prefix(values, "actor_rollout_ref.actor.self_distillation.")
    values = drop_prefix(values, "algorithm.rollout_correction.")
    # the trainer grades in its own verifier pool, never through verl's reward
    for prefix in ("reward_model.use_reward_loop=", "reward_model.launch_reward_fn_async=",
                   "reward_model.reward_loop_", "actor_rollout_ref.rollout.agent.num_workers="):
        if any(value.startswith(prefix) for value in values):
            values = drop_prefix(values, prefix)
    values += ["reward_model.use_reward_loop=False", "reward_model.launch_reward_fn_async=True"]
    loss_mode, tau = update_loss(TASK_TYPES[benchmark])
    max_len = model_max_model_len(config, model)
    for prefix, value in [
        ("data.train_batch_size=", f"data.train_batch_size={min(problems, parquet_rows(parquet))}"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=", f"actor_rollout_ref.actor.ppo_mini_batch_size={problems}"),
        ("actor_rollout_ref.rollout.n=", "actor_rollout_ref.rollout.n=1"),
        ("actor_rollout_ref.actor.optim.lr=", f"actor_rollout_ref.actor.optim.lr={interleaved_lr(config)}"),
        ("actor_rollout_ref.rollout.gpu_memory_utilization=",
         f"actor_rollout_ref.rollout.gpu_memory_utilization={tune['vllm_gpu_memory_utilization']}"),
        ("actor_rollout_ref.rollout.max_num_seqs=",
         f"actor_rollout_ref.rollout.max_num_seqs={tune['server_max_num_seqs']}"),
        ("actor_rollout_ref.rollout.max_model_len=", f"actor_rollout_ref.rollout.max_model_len={max_len}"),
        ("+actor_rollout_ref.model.override_config.max_position_embeddings=",
         f"+actor_rollout_ref.model.override_config.max_position_embeddings={max_len}"),
        # micro batches packed by the privileged context + response (the longer forward)
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=",
         f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={tune['ppo_max_token_len']}"),
        ("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=",
         f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={tune['log_prob_max_token_len']}"),
        ("trainer.group_name=", f"trainer.group_name={method_label(interval)}-{benchmark}"),
        ("trainer.experiment_name=",
         f"trainer.experiment_name={method_label(interval)}_{model}_{benchmark}_{mode}"),
        ("trainer.project_name=", "trainer.project_name=interleaved"),
    ]:
        replace_prefix(values, prefix, value)
    # KV usage per step (kv_stats.jsonl) in every mode
    set_override(values, "actor_rollout_ref.rollout.disable_log_stats", "False")
    sources = json.dumps(tune["sources"], separators=(",", ":"))
    values.extend([
        f"max_model_len={max_len}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "algorithm.use_kl_in_reward=False",
        "trainer.balance_batch=False",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "+trainer.trainer_class=leaf.explore.interleaved_trainer.InterleavedTrainer",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        "actor_rollout_ref.actor.self_distillation.alpha=1.0",
        f"actor_rollout_ref.actor.self_distillation.vocab_chunk_size={tune['kl_row_chunk_tokens']}",
        f"+interleaved.setting={'training' if 'leaf_update' in config else 'exploration'}",
        f"+interleaved.model={model}",
        f"+interleaved.benchmark={benchmark}",
        f"+interleaved.dataset={dataset}",
        f"+interleaved.output={output}",
        f"+interleaved.steps={steps}",
        f"+interleaved.rollout_max_num_seqs={tune['rollout_max_num_seqs']}",
        f"+interleaved.student_max_num_seqs={tune['student_max_num_seqs']}",
        f"+interleaved.learning_rate_config={config['learning_rate']}",
        # quoted: a JSON string, not a Hydra dict
        f"+interleaved.tuning_sources='{sources}'",
    ])
    if interval > 1:
        values.append(f"+interleaved.update_interval={interval}")
    tune = {**tune, "problems": problems, "loss_mode": loss_mode, "clip_tau": tau,
            "dataloader_parquet": str(parquet)}
    return values, tune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="smoke modes only (default: config smoke_steps)")
    parser.add_argument("--dataset", type=Path, help="smoke only: the problem subset (required there)")
    parser.add_argument("--update-interval", type=int,
                        help="policy update every N steps (default: config interleaved_update_interval, else 1)")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = load_config()
    gpu_type = detect_gpu_type()
    if args.mode == "formal":
        if args.steps:
            raise ValueError("--steps is for the smoke modes only")
        steps = interleaved_steps(config)
    else:
        steps = args.steps or int(config["smoke_steps"])
    if (args.dataset is not None) != (args.mode == "smoke"):
        raise ValueError("--dataset is required for --mode smoke and only allowed there")
    interval = update_interval(config, args.update_interval)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method = method_label(interval)
    if exit_if_done(output, steps, method):
        return
    # smoke_full = the formal dataset stopped after `steps` steps; smoke = a subset of it
    dataset = (args.dataset.resolve() if args.mode == "smoke"
               else benchmark_dataset_path(config, args.benchmark, "formal"))
    overrides, tune = update_overrides(output=output, config=config, model=args.model,
                                       benchmark=args.benchmark, mode=args.mode, steps=steps,
                                       gpu_type=gpu_type, dataset=dataset, interval=interval)
    manifest = launch_manifest(
        config=config, method=method, model=args.model, benchmark=args.benchmark, mode=args.mode,
        gpu_type=gpu_type, updates=steps, overrides=overrides,
        extra={"dataset": str(dataset), "condition": CONDITION, "tuning": tune,
               **({"update_interval": interval} if interval > 1 else {}),
               "learning_rate": interleaved_lr(config), "learning_rate_config": config["learning_rate"],
               "verl_config_name": "baseline_grpo", "resume_mode": "auto",
               "completed_updates_at_launch": completed_updates(output)})
    write_launch_manifest(output, manifest)
    print(json.dumps({"event": "interleaved_launch", "steps": steps, "tuning": tune, "overrides": overrides}),
          flush=True)
    run_trainer(config=config, output=output, overrides=overrides,
                experiment=f"{method}_{args.model}_{args.benchmark}_{args.mode}",
                config_name="baseline_grpo", updates=steps, method=method,
                print_config=args.print_config)


if __name__ == "__main__":
    main()
