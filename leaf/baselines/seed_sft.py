"""Train the rank-64 hindsight-analyzer LoRA that initializes SEED RL.

One analyzer LoRA per benchmark.  Input is ``sft.parquet`` from
:mod:`leaf.baselines.seed_collection`, which
by convention sits next to this run's output directory
(``<baselines>/seed_collection/<benchmark>/sft.parquet``); ``--data`` overrides that.

Resume: verl's SFT trainer has its own ``trainer.resume_mode=auto``. It saves
the adapter, the optimizer and the extra state after every SFT step, keeps only
the newest checkpoint, and on resume loads all three (the fork asserts it and
verifies the adapter tensor by tensor), so a chained chunk continues from the
last step (loading only ``[extra]`` would restart the LoRA of a resumed SFT
from B = 0).  A chunk that starts after the adapter has
already been exported exits immediately.
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
    BENCHMARKS, MODELS, PROJECT, detect_gpu_type, launch_manifest,
    load_config, mark_complete, model_path, write_launch_manifest,
)


def default_data(output: Path) -> Path:
    # The run scripts lay stages out as <baselines>/<stage>/<benchmark>, so the
    # collection output is a sibling stage directory for the same benchmark,
    # not a child of this stage's directory.
    return output.parent.parent / "seed_collection" / output.name / "sft.parquet"


def sft_overrides(
    config: dict, model: str, benchmark: str, data: Path, output: Path
) -> list[str]:
    base = model_path(config, model)
    return [
        f"data.train_files={data}",
        f"data.val_files={data}",
        f"data.train_batch_size={config['seed_sft_batch_size']}",
        f"data.micro_batch_size_per_gpu={config['seed_sft_micro_batch_size_per_gpu']}",
        "data.train_max_samples=-1",
        "data.val_max_samples=8",
        f"data.max_length={config['max_model_len']}",
        "data.truncation=error",
        "data.drop_last=False",
        "data.prompt_key=prompt",
        "data.response_key=response",
        "+data.shuffle=False",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"model.partial_pretrain={base}",
        "model.trust_remote_code=True",
        "model.enable_gradient_checkpointing=True",
        "model.fsdp_config.model_dtype=bfloat16",
        "model.fsdp_config.cpu_offload=False",
        "model.fsdp_config.offload_params=False",
        "model.strategy=fsdp",
        f"model.lora_rank={config['lora_rank']}",
        f"model.lora_alpha={config['lora_alpha']}",
        f"model.lora_dropout={config['lora_dropout']}",
        f"model.target_modules={config['lora_target_modules']}",
        f"optim.lr={config['learning_rate']}",
        f"optim.betas=[{config['adam_beta1']},{config['adam_beta2']}]",
        f"optim.weight_decay={config['weight_decay']}",
        "optim.lr_warmup_steps_ratio=0.0",
        "optim.lr_scheduler=constant",
        f"optim.clip_grad={config['gradient_clip']}",
        "ulysses_sequence_parallel_size=1",
        "use_remove_padding=True",
        "trainer.project_name=baselines",
        f"trainer.experiment_name=seed_sft_{model}_{benchmark}_lora{config['lora_rank']}",
        f"trainer.total_epochs={config['seed_sft_epochs']}",
        # Every SFT step by default (chained chunks resume from it); a time-capped
        # single-job run (config ``seed_sft_save_freq``) saves less often.
        f"trainer.save_freq={int(config.get('seed_sft_save_freq', 1))}",
        "trainer.test_freq=-1",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        "trainer.max_ckpt_to_keep=null",
        # Chained chunks continue from the latest SFT checkpoint.
        "trainer.resume_mode=auto",
        "trainer.logger=[console]",
        "trainer.checkpoint.save_contents=[optimizer,extra]",
        "trainer.checkpoint.load_contents=[optimizer,extra]",
        "+trainer.checkpoint.optimizer_save_freq=1",
        f"trainer.default_local_dir={output / 'checkpoints'}",
    ]


def trainer_command(values: list[str]) -> list[str]:
    """Launch even one-GPU FSDP through the distributed env rendezvous."""
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc-per-node=1",
        # remove-padding at SP 1, see leaf/baselines/verl_sft_rmpad.py
        "-m", "leaf.baselines.verl_sft_rmpad", *values,
    ]


def export_adapter(output: Path) -> Path:
    checkpoints = sorted(
        (output / "checkpoints").glob("global_step_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise RuntimeError("SEED SFT completed without a checkpoint")
    adapter = checkpoints[-1] / "lora_adapter"
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError("SEED SFT checkpoint has no exported LoRA adapter")
    (output / "final_adapter_path.txt").write_text(str(adapter.resolve()) + "\n")
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--gpu-type", choices=["a100", "h200"])
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    config = load_config()
    gpu_type = args.gpu_type or detect_gpu_type()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "final_adapter_path.txt"
    if marker.exists() and Path(marker.read_text().strip()).exists():
        print(json.dumps({
            "event": "already_complete", "adapter": marker.read_text().strip(),
        }), flush=True)
        return

    data = (args.data or default_data(output)).resolve()
    if not data.exists():
        raise FileNotFoundError(
            f"{data} is missing; run leaf.baselines.seed_collection first"
        )
    values = sft_overrides(config, args.model, args.benchmark, data, output)
    manifest = launch_manifest(
        config=config, method="seed_sft", model=args.model,
        benchmark=args.benchmark, mode=args.mode, gpu_type=gpu_type,
        updates=None, overrides=values,
        extra={
            "stage": "hindsight_sft",
            "data": str(data),
            "resume_mode": "auto",
        },
    )
    write_launch_manifest(output, manifest)

    command = trainer_command(values)
    if args.print_config:
        command += ["--cfg", "job", "--resolve"]
    print(json.dumps({"command": command}), flush=True)
    started = time.monotonic()
    code = subprocess.call(command, cwd=PROJECT / "third_party/SDPO", env=os.environ.copy())
    if code != 0 or args.print_config:
        raise SystemExit(code)
    adapter = export_adapter(output)
    mark_complete(output, {
        "method": "seed_sft",
        "steps": int(config["seed_sft_epochs"]),
        "generation_seconds": time.monotonic() - started,
        "adapter": str(adapter),
    })


if __name__ == "__main__":
    main()
