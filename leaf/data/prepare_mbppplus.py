"""Freeze the MBPP+ dataset under a config's data root.

Writes ``dataset_mbpp.jsonl`` (all 378 MbppPlus v0.2.0 tasks, rows built by
:func:`leaf.verifiers.mbppplus.build_rows`, shuffled with the config's
``mix_seed``) and ``smoke_dataset_mbpp.jsonl`` (its first
``smoke_source_sizes.mbpp`` rows), and records them in the root's
``data_manifest.json`` next to whatever other benchmarks it already lists.
Only MBPP is touched, so other benchmarks can be added to the same root later.

The prompt follows the config's ``mbpp_prompt_style``: ``chat`` (Qwen3-8B,
``MBPP_INSTRUCTION`` + description + 3 public asserts) or ``olmes``
(Olmo-3-7B, ``mbppplus::olmo3:adapt`` + the 3 public asserts). Then build the
verl parquets::

    LEAF_CONFIG=configs/data/explore_pools_qwen3_8b.json python -m leaf.data.prepare_mbppplus
    LEAF_CONFIG=configs/data/explore_pools_qwen3_8b.json python -m leaf.data.prepare_verl_data --benchmark mbpp
"""

from __future__ import annotations

import json
import random

from leaf.core.project_config import (
    CONFIG_PATH, atomic_json, load_config, read_jsonl, root,
)
from leaf.verifiers import mbppplus
from leaf.data.prepare_test_sets import frozen_write


def main() -> None:
    config = load_config()
    if not mbppplus.enabled(config):
        raise SystemExit(f"{CONFIG_PATH} does not set {mbppplus.CONFIG_FLAG}")
    style = config["mbpp_prompt_style"]
    rows = mbppplus.build_rows(style)
    expected = int(config["source_sizes"]["mbpp"])
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} MBPP+ rows, built {len(rows)}")
    # Two rows the verl grader could not tell apart would break its prompt map.
    keys = {(r.get("prompt_user") or r["problem"],
             json.dumps(r["verifier"], sort_keys=True)) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("two MBPP+ rows share both prompt and verifier payload")
    random.Random(int(config["mix_seed"])).shuffle(rows)
    for index, row in enumerate(rows):
        row["index"] = index
    target = root(config)
    smoke_size = int(config["smoke_source_sizes"]["mbpp"])
    frozen_write(rows, target / "dataset_mbpp.jsonl")
    frozen_write(sorted(rows[:smoke_size], key=lambda r: r["index"]),
                 target / "smoke_dataset_mbpp.jsonl")

    path = target / "data_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["revision"] = config["revision"]
    manifest.setdefault("per_benchmark_files", {})["mbpp"] = str(target / "dataset_mbpp.jsonl")
    manifest.setdefault("benchmarks", {})["mbpp"] = {
        "rows": len(rows),
        "smoke_rows": smoke_size,
        "source": str(mbppplus.MBPP_PLUS_PATH),
        "builder": "leaf.verifiers.mbppplus.build_rows",
        "prompt_style": style,
        "public_tests": "first 3 asserts of MBPP+ 'assertion'",
        "private_tests": "MBPP+ plus_input (base_input when empty), graded offline",
        "private_split_tasks": {
            split: sum(1 for r in rows if r["meta"]["private_split"] == split)
            for split in ("plus_input", "base_input")
        },
        "mix_seed": config["mix_seed"],
    }
    atomic_json(manifest, path)
    print(json.dumps({"root": str(target), "mbpp": manifest["benchmarks"]["mbpp"]}, indent=2))
    # Every row round-trips.
    assert read_jsonl(target / "dataset_mbpp.jsonl") == rows


if __name__ == "__main__":
    main()
