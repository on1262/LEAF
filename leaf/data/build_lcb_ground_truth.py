"""Build the LiveCodeBench ground-truth parquets from the shipped splits and the raw release.

    python -m leaf.data.build_lcb_ground_truth split131 [--out DIR]
    python -m leaf.data.build_lcb_ground_truth train478 [--out DIR] [--limit N]

``split131`` reads ``$DATA_ROOT/lcb/lcb131_split.jsonl`` (the 131-problem split) and
writes, in the schema of ``leaf.data.export_lcb_split``,

* ``train.parquet``: the public tests of every problem (rollout reward during
  exploration; ``leaf.paths.LCB_PUBLIC_GROUND_TRUTH``),
* ``test.parquet``: all tests of every problem (generalization test;
  config ``generalization.lcb_test_ground_truth``).

``train478`` rebuilds the 478 training problems with ``leaf.data.prepare_train_sets.lcb_rows``
(scan of release files ``test2..test5.jsonl``, deduplicated against the 131 split) and
writes all their tests to ``lcb_ground_truth_full.parquet`` (~2.2 GB; config
``lcb_ground_truth``) with ``leaf.data.prepare_train_sets.write_lcb_ground_truth``. It
also checks that the rebuilt problems are the ones of the shipped
``$DATA_ROOT/train/*/dataset_lcb.jsonl``. ``--limit N`` writes only the first N rows
(to a ``.partial`` file name; for spot checks).

Both need the raw release in ``$DATA_ROOT/raw/code_generation_lite`` (``run/data_fetch.sh``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leaf import paths
from leaf.data.export_lcb_split import read_jsonl, verl_row, write_parquet

TRAIN_CONFIG = paths.REPO / "configs" / "general" / "baselines_qwen3_8b_seed1.json"


def split131(out: Path) -> None:
    rows = read_jsonl(paths.LCB131_SPLIT)
    files = {
        "train": write_parquet((verl_row(row, "public", "train") for row in rows),
                               out / "train.parquet", len(rows)),
        "test": write_parquet((verl_row(row, "full", "test") for row in rows),
                              out / "test.parquet", len(rows)),
    }
    print(json.dumps(files, indent=2))


def train478(out: Path, limit: int | None) -> None:
    from leaf.data.prepare_train_sets import lcb_rows, write_lcb_ground_truth

    config = paths.load_json(TRAIN_CONFIG)
    lcb_split_rows, lcb_train_rows, audit = lcb_rows(config)
    rebuilt = {row["uid"]: (Path(row["verifier"]["evaluation_ref"]["path"]).name,
                            row["verifier"]["evaluation_ref"]["offset"]) for row in lcb_train_rows}
    for model_dir in sorted(paths.TRAIN_SETS.iterdir()):
        shipped = {row["uid"]: (Path(row["verifier"]["evaluation_ref"]["path"]).name,
                                row["verifier"]["evaluation_ref"]["offset"])
                   for row in read_jsonl(model_dir / "dataset_lcb.jsonl")}
        if shipped != rebuilt:
            raise SystemExit(f"rebuilt LCB training problems differ from {model_dir}/dataset_lcb.jsonl")
    print(json.dumps({key: value for key, value in audit.items() if key != "excluded_rows"},
                     indent=2, default=dict))
    if limit:
        rows = lcb_split_rows[:limit]
        result = write_parquet((verl_row(row, "full", "train") for row in rows),
                               out / "lcb_ground_truth_full.partial.parquet", len(rows))
    else:
        result = write_lcb_ground_truth(lcb_split_rows, out / "lcb_ground_truth_full.parquet")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=("split131", "train478"))
    parser.add_argument("--out", type=Path, default=paths.LCB_DATA,
                        help="output directory (default: $DATA_ROOT/lcb)")
    parser.add_argument("--limit", type=int, default=None, help="train478 only: first N rows")
    args = parser.parse_args()
    if args.target == "split131":
        split131(args.out)
    else:
        train478(args.out, args.limit)


if __name__ == "__main__":
    main()
