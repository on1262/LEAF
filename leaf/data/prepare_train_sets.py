"""Freeze the generalization training sets under a config's data root.

The generalization experiment trains every method on a training set and tests the
final checkpoint on the test set of the same task. The training sets keep the
benchmark keys of the test sets, so every runner reads them unchanged:

``dataset_lcb.jsonl``
    LiveCodeBench release_v5 minus release_v1, i.e. the rows of ``test2`` ..
    ``test5.jsonl`` that pass the LCB split's static checks, with duplicates and
    anything colliding with the 131 v6 test problems dropped
    (:func:`leaf.data.prepare_lcb_split.deduplicate_candidates`). Every test case is
    feedback: ``public_test_indices`` covers the whole suite and the SDPO
    ground-truth export ``lcb_ground_truth_full.parquet`` (config
    ``lcb_ground_truth``) holds all of them.
``dataset_math500.jsonl``
    500 ``Level 5`` problems of the MATH train split (EleutherAI/hendrycks_math,
    all seven subjects), answer = the last ``\\boxed{}`` of the reference
    solution; problems whose text matches a MATH500 problem are dropped.
``dataset_sciknoweval.jsonl``
    100 L5 open problems per domain from the SciKnowEval v2 pool of
    :func:`leaf.data.sciknoweval_pool.open_pool`, none of them in the
    400-problem test subset.

Each file is shuffled with ``generalization.mix_seed``; ``smoke_dataset_<bm>.jsonl``
is its first ``smoke_source_sizes`` rows. Olmo-3-7B rows get their OLMES
prompt frozen in (:func:`leaf.data.prepare_olmes_test_sets.augment`). Then::

    LEAF_CONFIG=configs/general/baselines_qwen3_8b_seed1.json python -m leaf.data.prepare_train_sets
    LEAF_CONFIG=configs/general/baselines_qwen3_8b_seed1.json python -m leaf.data.prepare_verl_data
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from leaf.data import sciknoweval_pool
from leaf.verifiers.math_grader import _boxed_values
from leaf.core.project_config import (
    atomic_json, lcb_raw_rows, load_config, read_jsonl, root,
)
from leaf.data.prepare_test_sets import frozen_write

BENCHMARK_FILES = ("lcb", "math500", "sciknoweval")


def olmes(config: dict) -> bool:
    return config.get("sciknoweval_prompt_style") == "olmes"


# --------------------------------------------------------------------------- #
# LiveCodeBench
# --------------------------------------------------------------------------- #


def lcb_rows(config: dict) -> tuple[list[dict], list[dict], dict]:
    """LCB-split-shaped rows (all tests public), benchmark rows, and the audit."""
    from leaf.data.prepare_lcb_split import deduplicate_candidates, static_row

    settings = config["generalization"]
    source = Path(config["sources"]["lcb"])
    scanned = []
    order = 0  # position in the release_v6 concatenation, as the LCB split numbers it
    for name in settings["lcb_all_files"]:
        path = source / name
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if name in settings["lcb_train_files"]:
                    raw = json.loads(line)
                    row = static_row(raw, path, offset, order)
                    row["release_file"] = name
                    scanned.append(row)
                order += 1
    test = read_jsonl(settings["lcb_test_problems"])
    accepted, excluded = deduplicate_candidates(scanned, test)
    lcb_split_rows, lcb_train_rows = [], []
    for row in accepted:
        full = list(range(row["full_test_count"]))
        row = {**row, "split": "train", "index": len(lcb_split_rows),
               "public_test_indices": full, "heldout_test_indices": [],
               "public_test_count": len(full), "heldout_test_count": 0}
        lcb_split_rows.append(row)
        lcb_train_rows.append({
            "uid": f"lcb/{row['uid']}",
            "source": "lcb",
            "task_type": "code_lcb",
            "problem": row["problem"],
            "verifier": {
                "kind": "lcb",
                # lcb_ground_truth is keyed by this uid, like the LCB split's export.
                "problem_uid": row["uid"],
                "evaluation_ref": row["evaluation_ref"],
                "public_test_indices": full,
                "heldout_test_indices": [],
                "full_test_count": row["full_test_count"],
                "heldout_test_count": 0,
            },
            "meta": {
                "platform": row["platform"],
                "question_id": row["question_id"],
                "difficulty": row["difficulty"],
                "interface": row["interface"],
                "contest_date": row["contest_date"],
                "release_file": row["release_file"],
                "question_hash": row["question_hash"],
            },
        })
    audit = {
        "scanned": len(scanned),
        "accepted": len(accepted),
        "excluded": collections.Counter(item["reason"] for item in excluded),
        "excluded_rows": excluded,
        "by_release_file": collections.Counter(row["release_file"] for row in accepted),
        "by_difficulty_interface": collections.Counter(
            f"{row['difficulty']}|{row['interface']}" for row in accepted),
        "full_test_count": sum(row["full_test_count"] for row in accepted),
        "contest_dates": [min(r["contest_date"] for r in accepted)[:10],
                          max(r["contest_date"] for r in accepted)[:10]],
        "test_problems_checked_against": len(test),
    }
    return lcb_split_rows, lcb_train_rows, audit


def write_lcb_ground_truth(lcb_split_rows: list[dict], path: Path) -> dict:
    """The all-tests export (~2.2 GB); shared by both models, so reused if present."""
    from leaf.data.export_lcb_split import verl_row, write_parquet

    if path.exists():
        uids = [row["problem_uid"] for row in
                pq.read_table(path, columns=["extra_info"]).column("extra_info").to_pylist()]
        if sorted(uids) != sorted(row["uid"] for row in lcb_split_rows):
            raise RuntimeError(f"{path} holds another LCB problem set")
        return {"path": str(path), "rows": len(uids), "size": path.stat().st_size,
                "reused": True}
    return write_parquet((verl_row(row, "full", "train") for row in lcb_split_rows),
                         path, len(lcb_split_rows))


# --------------------------------------------------------------------------- #
# MATH
# --------------------------------------------------------------------------- #


def normalise_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def math_rows(config: dict) -> tuple[list[dict], dict]:
    settings = config["generalization"]
    snapshot = Path(config["sources"]["math500"])
    level = settings["math_train_level"]
    test = {normalise_problem(row["problem"])
            for row in read_jsonl(settings["test_sets"]["math500"])}
    pool, no_answer, overlap = [], 0, 0
    for subject_dir in sorted(p for p in snapshot.iterdir() if p.is_dir()):
        table = pq.read_table(subject_dir / "train-00000-of-00001.parquet").to_pylist()
        for position, row in enumerate(table):
            if row["level"] != level:
                continue
            boxed = _boxed_values(row["solution"])
            answer = boxed[-1].strip() if boxed else ""
            if not answer:
                no_answer += 1
                continue
            if normalise_problem(row["problem"]) in test:
                overlap += 1
                continue
            pool.append({"subject": subject_dir.name, "position": position,
                         "problem": row["problem"], "answer": answer,
                         "type": row["type"], "level": row["level"]})
    size = int(config["source_sizes"]["math500"])
    rng = random.Random(f"{settings['sample_seed']}:math")
    chosen = sorted(rng.sample(pool, size), key=lambda r: (r["subject"], r["position"]))
    rows = [{
        "uid": f"math500/t_{row['subject']}_{row['position']:04d}",
        "source": "math500",
        "task_type": "math",
        "problem": row["problem"],
        "verifier": {"kind": "math", "answer": row["answer"]},
        "meta": {"subject": row["type"], "level": row["level"], "split": "train",
                 "source_subject": row["subject"], "source_position": row["position"]},
    } for row in chosen]
    audit = {
        "pool": len(pool), "dropped_no_boxed_answer": no_answer,
        "dropped_math500_overlap": overlap,
        "pool_by_subject": collections.Counter(row["subject"] for row in pool),
        "sampled_by_subject": collections.Counter(row["subject"] for row in chosen),
    }
    return rows, audit


# --------------------------------------------------------------------------- #
# SciKnowEval
# --------------------------------------------------------------------------- #


def sciknoweval_rows(config: dict) -> tuple[list[dict], dict]:
    settings = config["generalization"]
    style = config["sciknoweval_prompt_style"]
    pool, audit = sciknoweval_pool.open_pool(
        sciknoweval_pool.read_source(Path(config["sources"]["sciknoweval"])))
    test_positions = {row["meta"]["source_position"]
                      for row in read_jsonl(settings["test_sets"]["sciknoweval"])}
    per_domain = int(config["source_sizes"]["sciknoweval"]) // len(sciknoweval_pool.DOMAINS)
    chosen, available = [], {}
    for domain in sciknoweval_pool.DOMAINS:
        stratum = [row for row in pool if row["domain"] == domain
                   and row["details"]["level"] == "L5"
                   and row["_position"] not in test_positions]
        available[domain] = len(stratum)
        rng = random.Random(f"{settings['sample_seed']}:{domain}:L5")
        chosen.extend(sorted(rng.sample(stratum, per_domain), key=lambda r: r["_position"]))
    rows = [sciknoweval_pool.build_row(row, style) for row in chosen]
    if {row["meta"]["source_position"] for row in rows} & test_positions:
        raise RuntimeError("SciKnowEval training rows overlap the test subset")
    keys = {(row.get("prompt_user") or row["problem"]) for row in rows}
    payloads = {json.dumps(row["verifier"], sort_keys=True) for row in rows}
    if len(keys) != len(rows) or len(payloads) != len(rows):
        raise RuntimeError("two SciKnowEval rows share a prompt or a verifier payload")
    audit = {
        "l5_available_outside_test": available,
        "sampled": collections.Counter(
            f"{row['meta']['domain']}/{row['meta']['subtask']}" for row in rows),
        "criteria": collections.Counter(row["verifier"]["criterion"] for row in rows),
        "pool_audit": audit,
    }
    return rows, audit


# --------------------------------------------------------------------------- #


def with_olmes_prompts(rows: list[dict], lcb_raw: dict) -> list[dict]:
    from leaf.data.prepare_olmes_test_sets import augment

    return [row if "prompt_user" in row else augment(row, lcb_raw, {}) for row in rows]


def main() -> None:
    config = load_config()
    if config.get("experiment") != "generalization":
        raise SystemExit("run under a generalization config "
                         "(LEAF_CONFIG=configs/general/baselines_<model>_seed1.json)")
    target = root(config)
    target.mkdir(parents=True, exist_ok=True)
    settings = config["generalization"]

    lcb_split, lcb, lcb_audit = lcb_rows(config)
    math, math_audit = math_rows(config)
    science, science_audit = sciknoweval_rows(config)
    built = {"lcb": lcb, "math500": math, "sciknoweval": science}
    audits = {"lcb": lcb_audit, "math500": math_audit, "sciknoweval": science_audit}
    if olmes(config):
        raw = lcb_raw_rows(config)
        built = {name: with_olmes_prompts(rows, raw) for name, rows in built.items()}

    manifest_path = target / "data_manifest.json"
    manifest = {"revision": config["revision"], "benchmarks": {}}
    for name in BENCHMARK_FILES:
        rows = built[name]
        expected = int(config["source_sizes"][name])
        if expected and len(rows) != expected:
            raise SystemExit(f"{name}: config expects {expected} rows, built {len(rows)}")
        if len({row["uid"] for row in rows}) != len(rows):
            raise SystemExit(f"{name}: duplicate uid")
        random.Random(f"{settings['mix_seed']}:{name}").shuffle(rows)
        for index, row in enumerate(rows):
            row["index"] = index
        smoke = int(config["smoke_source_sizes"][name])
        frozen_write(rows, target / f"dataset_{name}.jsonl")
        frozen_write(rows[:smoke], target / f"smoke_dataset_{name}.jsonl")
        manifest["benchmarks"][name] = {
            "rows": len(rows),
            "smoke_rows": smoke,
            "source": config["sources"][name],
            "test_set": settings["test_sets"][name],
            "audit": audits[name],
        }
    manifest["benchmarks"]["lcb"]["ground_truth"] = write_lcb_ground_truth(
        lcb_split, Path(config["lcb_ground_truth"]))
    atomic_json(json.loads(json.dumps(manifest, default=dict)), manifest_path)
    print(json.dumps({name: {k: v for k, v in value.items() if k != "audit"}
                      | {"audit": {k: v for k, v in value["audit"].items()
                                   if k != "excluded_rows"}}
                      for name, value in manifest["benchmarks"].items()},
                     indent=2, default=dict))


if __name__ == "__main__":
    main()
