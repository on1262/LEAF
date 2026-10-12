"""Freeze the open-answer SciKnowEval dataset under a config's data root.

Writes ``dataset_sciknoweval.jsonl`` (400 rows built by
:func:`leaf.data.sciknoweval_pool.build_rows`, shuffled with the config's
``mix_seed``) and ``smoke_dataset_sciknoweval.jsonl`` (its first
``smoke_source_sizes.sciknoweval`` rows), and records them in the root's
``data_manifest.json`` beside the benchmarks it already lists. Then build the
verl parquets::

    LEAF_CONFIG=configs/data/explore_pools_qwen3_8b.json python -m leaf.data.prepare_sciknoweval
    LEAF_CONFIG=configs/data/explore_pools_qwen3_8b.json python -m leaf.data.prepare_verl_data --benchmark sciknoweval
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from leaf.data import sciknoweval_pool
from leaf.core.project_config import (
    CONFIG_PATH, atomic_json, load_config, read_jsonl, root,
)
from leaf.data.prepare_test_sets import frozen_write


def main() -> None:
    config = load_config()
    if not sciknoweval_pool.enabled(config):
        raise SystemExit(f"{CONFIG_PATH} does not set {sciknoweval_pool.CONFIG_FLAG}")
    style = config["sciknoweval_prompt_style"]
    source = Path(config["sources"]["sciknoweval"])
    seed = int(config["sciknoweval_sample_seed"])
    rows, audit = sciknoweval_pool.build_rows(source, seed, style)
    expected = int(config["source_sizes"]["sciknoweval"])
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} SciKnowEval rows, built {len(rows)}")
    if len({row["uid"] for row in rows}) != len(rows):
        raise SystemExit("duplicate SciKnowEval uid")
    # The verl grader maps a rollout back to its problem by prompt or payload.
    keys = {(row.get("prompt_user") or row["problem"]) for row in rows}
    payloads = {json.dumps(row["verifier"], sort_keys=True) for row in rows}
    if len(keys) != len(rows) or len(payloads) != len(rows):
        raise SystemExit("two SciKnowEval rows share a prompt or a verifier payload")
    random.Random(int(config["mix_seed"])).shuffle(rows)
    for index, row in enumerate(rows):
        row["index"] = index
    target = root(config)
    smoke_size = int(config["smoke_source_sizes"]["sciknoweval"])
    frozen_write(rows, target / "dataset_sciknoweval.jsonl")
    frozen_write(sorted(rows[:smoke_size], key=lambda r: r["index"]),
                 target / "smoke_dataset_sciknoweval.jsonl")

    path = target / "data_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["revision"] = config["revision"]
    manifest.setdefault("per_benchmark_files", {})["sciknoweval"] = str(
        target / "dataset_sciknoweval.jsonl")
    manifest.setdefault("benchmarks", {})["sciknoweval"] = {
        "rows": len(rows),
        "smoke_rows": smoke_size,
        "source": str(source),
        "builder": "leaf.data.sciknoweval_pool.build_rows",
        "prompt_style": style,
        "verifier": "sciqa_judge (leaf.verifiers.llm_judge): base model without LoRA, greedy",
        "sample_seed": seed,
        "mix_seed": config["mix_seed"],
        "audit": audit,
    }
    atomic_json(manifest, path)
    print(json.dumps({"root": str(target), "sciknoweval": manifest["benchmarks"]["sciknoweval"]},
                     indent=2))
    assert read_jsonl(target / "dataset_sciknoweval.jsonl") == rows


if __name__ == "__main__":
    main()
