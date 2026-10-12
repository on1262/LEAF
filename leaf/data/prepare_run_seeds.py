"""Configs and data of run seeds 2 / 3 of the exploration (Fig. 3) and generalization
(Table 2) experiments; seed 1 runs the seed-1 configs themselves. CPU only.

Every seed-``s`` config is the seed-1 config with a new ``output_root`` / ``revision``
and every *run* seed shifted by ``(s - 1) * SEED_STRIDE``:
``decode_seed`` (vLLM request seeds, verl ``rollout_seed`` column, random-sampling trials),
``schedule_seed`` (question order of the verl baselines / SEED, policy-update and SFT problem
stream), ``seed_analyzer_seed``, ``feedback_random_seed`` (teacher-context draw of the LEAF
update), ``generalization.test_seed`` (test samples), ``sft_and_traces.rewrite_seed`` (y*
rewrite). The dataset seeds (``mix_seed``, ``sciknoweval_sample_seed``,
``generalization.sample_seed`` / ``mix_seed``) and the frozen dataset files stay, so every seed
runs the same problems.

Configs and roots (``leaf.paths.explore_root`` / ``general_root``), per kind:

``explore``        exploration, from ``configs/explore/<m>_seed1.json``
                   -> ``configs/explore/<m>_seed<s>.json``, root ``explore/seed<s>/<m>``
``baselines``      GRPO / SDPO / SCoRe + Base test, from ``configs/general/baselines_<m>_seed1.json``
                   -> ``configs/general/baselines_<m>_seed<s>.json``, root ``general/seed<s>/baselines/<m>``
``seed_baseline``  SEED, from ``configs/general/seed_baseline_<m>_seed1.json``
                   -> ``configs/general/seed_baseline_<m>_seed<s>.json``, root ``general/seed<s>/seed_baseline/<m>``
``leaf``           LEAF rollout -> policy update and the y* SFT, from
                   ``configs/general/sft_<m>_seed1.json`` (the update settings + the
                   ``sft_and_traces`` block) -> ``configs/general/leaf_<m>_seed<s>.json``, root
                   ``general/seed<s>/leaf/<m>``; its ``data_root`` stays the seed-1 one (read only:
                   the unused verl dataloader parquet)

``configs``  writes the configs. ``data`` links the frozen datasets into each root and builds
``rl_data`` (``leaf.data.prepare_verl_data`` under the seed's config) for every kind but ``leaf``.

    python -m leaf.data.prepare_run_seeds configs
    python -m leaf.data.prepare_run_seeds data [--seed 2] [--model qwen3_8b]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from leaf import paths

PROJECT = Path(__file__).resolve().parents[2]
SEEDS = (2, 3)
MODELS = ("qwen3_8b", "olmo3_7b")
# above every formal per-request offset of seed 1 (decode_seed + < 40M: four benchmark windows of
# leaf.data.prepare_verl_data.SEED_STRIDE), and every derived seed stays below 2**31
SEED_STRIDE = 100_000_000
FEEDBACK_RANDOM_SEED = 200045  # leaf.update.leaf_update.FEEDBACK_RANDOM_SEED (seed 1)
TOP_SEEDS = ("decode_seed", "schedule_seed", "seed_analyzer_seed")
NESTED_SEEDS = (("generalization", "test_seed"), ("sft_and_traces", "rewrite_seed"))
KINDS = ("explore", "baselines", "seed_baseline", "leaf")


def config_name(kind: str, model: str, seed: int) -> str:
    """Config of a kind, relative to configs/ and without ``.json``."""
    if kind == "explore":
        return f"explore/{model}_seed{seed}"
    if kind == "leaf" and seed == 1:
        return f"general/sft_{model}_seed1"
    return f"general/{kind}_{model}_seed{seed}"


def source_config(kind: str, model: str) -> Path:
    return PROJECT / "configs" / f"{config_name(kind, model, 1)}.json"


def config_path(kind: str, model: str, seed: int) -> Path:
    return PROJECT / "configs" / f"{config_name(kind, model, seed)}.json"


def root(kind: str, model: str, seed: int) -> Path:
    if kind == "explore":
        return paths.explore_root(seed) / model
    return paths.general_root(seed, kind) / model


def dataset_source(kind: str, model: str) -> Path:
    return (paths.EXPLORE_SETS if kind == "explore" else paths.TRAIN_SETS) / model


def build_config(kind: str, model: str, seed: int) -> dict:
    config = copy.deepcopy(json.loads(source_config(kind, model).read_text()))
    shift = (seed - 1) * SEED_STRIDE
    for key in TOP_SEEDS:
        config[key] = int(config[key]) + shift
    config["feedback_random_seed"] = FEEDBACK_RANDOM_SEED + shift
    for block, key in NESTED_SEEDS:
        if key in (config.get(block) or {}):
            config[block][key] = int(config[block][key]) + shift
    # written back to configs/, so the root stays a ${RESULTS_ROOT} placeholder
    out = paths.placeholder(root(kind, model, seed))
    config["output_root"] = out
    if kind != "leaf":
        config["data_root"] = out
    config.pop("report", None)
    config["revision"] = config_name(kind, model, seed).replace("/", "_")
    return config


def write_configs(seeds, models) -> None:
    for seed in seeds:
        for model in models:
            for kind in KINDS:
                path = config_path(kind, model, seed)
                path.write_text(json.dumps(build_config(kind, model, seed), indent=2,
                                           ensure_ascii=False) + "\n")
                print(path.relative_to(PROJECT))


def link_data(seeds, models) -> None:
    python = sys.executable
    for seed in seeds:
        for model in models:
            for kind in KINDS:
                target = root(kind, model, seed)
                target.mkdir(parents=True, exist_ok=True)
                source = dataset_source(kind, model)
                for dataset in sorted(source.glob("dataset_*.jsonl")):
                    link = target / dataset.name
                    if not link.exists():
                        os.symlink(dataset, link)
                if kind == "leaf":
                    continue
                if (target / "rl_data" / "budget_manifest.json").exists():
                    print(f"{target}/rl_data exists", flush=True)
                    continue
                env = {**os.environ, "LEAF_CONFIG": str(config_path(kind, model, seed))}
                benchmarks = paths.load_json(config_path(kind, model, seed))["source_sizes"]
                subprocess.run([python, "-m", "leaf.data.prepare_verl_data",
                                *[arg for name in sorted(benchmarks) for arg in ("--benchmark", name)]],
                               cwd=PROJECT, env=env, check=True, stdout=subprocess.DEVNULL)
                print(f"{target}/rl_data", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("configs", "data"))
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS)
    parser.add_argument("--model", action="append", choices=MODELS)
    args = parser.parse_args()
    seeds, models = args.seed or SEEDS, args.model or MODELS
    if args.command == "configs":
        write_configs(seeds, models)
    else:
        link_data(seeds, models)


if __name__ == "__main__":
    main()
