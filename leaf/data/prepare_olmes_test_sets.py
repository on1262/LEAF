"""Freeze the Olmo-3-7B test sets: the base test-set rows with OLMES prompts attached.

The build keeps the base rows' problems, order, uids, indices and verifier payloads
(``base_test_sets_root``, the build of configs/data/test_sets_qwen3_8b.json), so every
schedule and seed derived from them is unchanged. Each row gains

* ``prompt_system`` / ``prompt_user`` -- its OLMES olmo3:adapt opening prompt,
  built by :func:`leaf.prompts.olmes_prompts.build_prompt`;
* ``verifier.answer_format = "olmes"`` -- which tells the verifier to read the
  answer the way that template asks for it.

Run with ``LEAF_CONFIG=configs/data/test_sets_olmo3_7b.json``; afterwards build the verl
parquets with ``leaf.data.prepare_verl_data`` under the same variable.
"""

from __future__ import annotations

import json
from pathlib import Path

from leaf.core.project_config import (
    BENCHMARKS, CONFIG_PATH, atomic_json, atomic_jsonl, lcb_raw_rows,
    load_config, read_jsonl, root,
)
from leaf.prompts.olmes_prompts import ANSWER_FORMAT, build_prompt


def augment(row: dict, lcb_raw: dict, mbpp_raw: dict, mbpp_asserts: bool = False) -> dict:
    out = dict(row)
    out.update(build_prompt(
        row, lcb_raw=lcb_raw, mbpp_raw=mbpp_raw, mbpp_asserts=mbpp_asserts,
    ))
    out["verifier"] = {**row["verifier"], "answer_format": ANSWER_FORMAT}
    return out


def main() -> None:
    config = load_config()
    # ``mbpp_prompt_asserts`` freezes the same rows with the asserts added to the
    # MBPP prompt.
    if config.get("experiment") not in ("data_test_sets_olmes", "data_explore_pools"):
        raise SystemExit(f"{CONFIG_PATH} is not an Olmo test-set / exploration-pool build config")
    mbpp_asserts = bool(config.get("mbpp_prompt_asserts", False))
    source = Path(config["base_test_sets_root"])
    target = root(config)
    if source.resolve() == target.resolve():
        raise SystemExit("the Olmo build must not write into the base test-set root")
    lcb_raw = lcb_raw_rows(config)
    mbpp_raw = {int(row["task_id"]): row for row in read_jsonl(config["sources"]["mbpp"])}

    names = ["dataset.jsonl", "smoke_dataset.jsonl"] + [
        f"{prefix}_{benchmark}.jsonl"
        for prefix in ("dataset", "smoke_dataset") for benchmark in BENCHMARKS
    ]
    files = {}
    for name in names:
        rows = [
            augment(row, lcb_raw, mbpp_raw, mbpp_asserts)
            for row in read_jsonl(source / name)
        ]
        if not rows:
            raise SystemExit(f"{source / name} is empty or missing")
        atomic_jsonl(rows, target / name)
        files[name] = {
            "rows": len(rows),
        }

    # A row keeps the same prompt wherever it appears.
    by_uid: dict[str, tuple] = {}
    for name in names:
        for row in read_jsonl(target / name):
            key = (row["prompt_system"], row["prompt_user"])
            if by_uid.setdefault(row["uid"], key) != key:
                raise SystemExit(f"{row['uid']} got two prompts")

    base_test_manifest = json.loads((source / "data_manifest.json").read_text())
    manifest = {
        **base_test_manifest,
        "revision": config["revision"],
        "base_test_sets_revision": base_test_manifest.get("revision"),
        "per_benchmark_files": {
            benchmark: str(target / f"dataset_{benchmark}.jsonl") for benchmark in BENCHMARKS
        },
        "files": files,
        "prompts": {
            "builder": "leaf.prompts.olmes_prompts.build_prompt",
            "templates": "OLMES olmo3:adapt (allenai/olmes 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e)",
            "answer_format": ANSWER_FORMAT,
            "mbpp_prompt_asserts": mbpp_asserts,
            "lcb_raw": config["lcb_hf_snapshot"],
            "mbpp_raw": config["sources"]["mbpp"],
        },
    }
    atomic_json(manifest, target / "data_manifest.json")
    print(json.dumps({"root": str(target), "files": files}, indent=2))


if __name__ == "__main__":
    main()
