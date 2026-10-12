"""Freeze the 131-question SDPO split and stratified 400-question IID test."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np

from leaf import paths
from leaf.core.io_utils import (
    CONFIG_PATH, atomic_json, atomic_jsonl, decode_private_tests,
    function_name, normalize_problem, official_problem_text,
    lcb_model_manifest_path, require_root, test_signature,
)


def git_output(path, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True,
        stderr=subprocess.DEVNULL).strip()


# Pinned upstream commits of the sources (recorded when a tree is not a git checkout).
PINNED_COMMITS = {
    "SDPO": "7c457fc1b1f636ae794eb0362ba37d4743b06fbc",
    "LiveCodeBench": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
    "code_generation_lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
}
PROJECT = Path(__file__).resolve().parents[2]


def repository_state(path):
    try:
        if not (Path(path) / ".git").exists():  # never read an enclosing repository's HEAD
            raise OSError(f"{path} is not a git checkout")
        return _git_repository_state(path)
    except (OSError, subprocess.CalledProcessError):
        name = Path(path).name
        return {
            "commit": PINNED_COMMITS.get(name),
        }


def _git_repository_state(path):
    value = {
        "path": str(Path(path).resolve()),
        "commit": git_output(path, "rev-parse", "HEAD"),
        "modified_paths": git_output(
            path, "status", "--short", "--untracked-files=no").splitlines(),
    }
    return value


def model_fingerprint(path):
    path = Path(path).resolve()
    if not path.is_dir():
        return {"path": str(path), "files": None}
    names = [
        item for item in sorted(path.iterdir())
        if item.is_file() and (
            item.name.endswith(".safetensors")
            or item.name in {
                "config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "model.safetensors.index.json",
            }
        )
    ]
    return {
        "path": str(path),
        "files": [
            {"name": item.name, "size": item.stat().st_size}
            for item in names
        ],
    }


def static_row(raw, path, offset, source_order):
    uid = f"{raw['platform']}/{raw['question_id']}"
    tests = decode_private_tests(raw["private_test_cases"])
    test_types = sorted({test["testtype"] for test in tests})
    interface = "call_based" if raw.get("starter_code", "").strip() else "standard_input"
    reason = None
    try:
        fn_name = function_name(raw)
        official_problem_text(raw)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fn_name = None
        reason = f"invalid_problem_metadata:{type(exc).__name__}"
    if reason is None and len(test_types) != 1:
        reason = "mixed_test_types"
    elif reason is None and interface == "call_based" and not fn_name:
        reason = "functional_test_missing_fn_name"
    question_normalized = normalize_problem(raw["question_content"])
    starter_normalized = normalize_problem(raw.get("starter_code", ""))
    return {
        "uid": uid,
        "platform": raw["platform"],
        "question_id": str(raw["question_id"]),
        "contest_date": raw["contest_date"],
        "difficulty": raw["difficulty"],
        "interface": interface,
        "question_hash": hashlib.sha256(question_normalized.encode()).hexdigest(),
        "starter_hash": (
            hashlib.sha256(starter_normalized.encode()).hexdigest()
            if starter_normalized else None
        ),
        "test_signature_hash": test_signature(tests),
        "full_test_count": len(tests),
        "test_types": test_types,
        "function_name": fn_name,
        "problem": official_problem_text(raw) if reason is None else raw.get("question_content", ""),
        "evaluation_ref": {"path": str(path.resolve()), "offset": offset},
        "source_order": source_order,
        "static_exclusion_reason": reason,
    }


def scan(cfg):
    root = Path(cfg["data_source"])
    rows = []
    source_order = 0
    for filename in cfg["data_files"]:
        path = root / filename
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                raw = json.loads(line)
                rows.append(static_row(raw, path, offset, source_order))
                source_order += 1
    return rows


def add_public_split(rows, fraction, seed):
    rng = np.random.RandomState(seed)
    result = []
    for index, source in enumerate(rows):
        row = dict(source)
        keep = max(1, int(row["full_test_count"] * fraction))
        public = sorted(int(value) for value in rng.choice(
            row["full_test_count"], size=keep, replace=False))
        public_set = set(public)
        heldout = [value for value in range(row["full_test_count"]) if value not in public_set]
        row.update(
            split="train", index=index, public_test_indices=public,
            heldout_test_indices=heldout, public_test_count=len(public),
            heldout_test_count=len(heldout),
        )
        result.append(row)
    return result


def collision_keys(row):
    keys = {
        ("uid", row["uid"]),
        ("question", row["question_hash"]),
        ("tests", row["test_signature_hash"]),
    }
    if row["starter_hash"] is not None:
        keys.add(("starter", row["starter_hash"]))
    return keys


def deduplicate_candidates(rows, training):
    occupied = set().union(*(collision_keys(row) for row in training))
    accepted = []
    excluded = []
    for row in rows:
        if row["static_exclusion_reason"]:
            excluded.append({"uid": row["uid"], "reason": row["static_exclusion_reason"]})
            continue
        overlap = sorted(collision_keys(row) & occupied)
        if overlap:
            excluded.append({"uid": row["uid"], "reason": "duplicate_or_training_overlap", "keys": overlap})
            continue
        occupied.update(collision_keys(row))
        accepted.append(row)
    return accepted, excluded


def largest_remainder_targets(training, size):
    counts = collections.Counter((row["difficulty"], row["interface"]) for row in training)
    ideals = {key: size * value / len(training) for key, value in counts.items()}
    targets = {key: math.floor(value) for key, value in ideals.items()}
    remaining = size - sum(targets.values())
    order = sorted(ideals, key=lambda key: (-(ideals[key] - targets[key]), key))
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def stratified_sample(candidates, training, size, seed):
    targets = largest_remainder_targets(training, size)
    grouped = collections.defaultdict(list)
    for row in candidates:
        grouped[(row["difficulty"], row["interface"])].append(row)
    shortages = {
        key: targets[key] - len(grouped[key])
        for key in targets if len(grouped[key]) < targets[key]
    }
    if shortages:
        raise RuntimeError(f"Insufficient IID candidates for strata: {shortages}")
    rng = random.Random(seed)
    selected = []
    for key in sorted(targets):
        values = sorted(grouped[key], key=lambda row: row["uid"])
        rng.shuffle(values)
        selected.extend(values[:targets[key]])
    random.Random(seed + 1).shuffle(selected)
    output = []
    for index, source in enumerate(selected):
        row = dict(source)
        row.update(split="iid_test", index=index)
        output.append(row)
    return output, targets


def frozen_write(rows, path):
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text() != serialized:
        raise RuntimeError(f"Frozen LCB split data changed: {path}")
    atomic_jsonl(rows, path)
    return hashlib.sha256(serialized.encode()).hexdigest()


def main():
    cfg = paths.load_json(CONFIG_PATH)
    ROOT = require_root()
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = scan(cfg)
    train_source = [
        row for row in rows
        if cfg["train_start_date"] <= row["contest_date"][:10] < cfg["train_end_date_exclusive"]
    ]
    if len(train_source) != cfg["train_size"]:
        raise RuntimeError(f"Expected {cfg['train_size']} training questions, found {len(train_source)}")
    if any(row["static_exclusion_reason"] for row in train_source):
        bad = [(row["uid"], row["static_exclusion_reason"]) for row in train_source if row["static_exclusion_reason"]]
        raise RuntimeError(f"Training questions fail static verifier checks: {bad}")
    train = add_public_split(
        train_source, cfg["public_test_fraction"], cfg["public_split_seed"])
    train_uids = {row["uid"] for row in train}
    candidates, excluded = deduplicate_candidates(
        [row for row in rows if row["uid"] not in train_uids], train)
    iid_test, targets = stratified_sample(
        candidates, train, cfg["iid_test_size"], cfg["iid_test_seed"])

    train_path = ROOT / "train.jsonl"
    iid_path = ROOT / "iid_test.jsonl"
    frozen_write(train, train_path)
    frozen_write(iid_test, iid_path)
    source_root = Path(cfg["data_source"])
    repositories = cfg["source_repositories"]
    manifest = {
        "revision": cfg["revision"],
        "train": {
            "count": len(train),
            "date_range": [cfg["train_start_date"], cfg["train_end_date_exclusive"]],
            "distribution": dict(collections.Counter(
                f"{row['difficulty']}|{row['interface']}" for row in train)),
            "full_test_count": sum(row["full_test_count"] for row in train),
            "public_test_count": sum(row["public_test_count"] for row in train),
            "heldout_test_count": sum(row["heldout_test_count"] for row in train),
            "public_split": {
                "algorithm": "numpy RandomState.choice without replacement in source order",
                "seed": cfg["public_split_seed"],
                "fraction": cfg["public_test_fraction"],
            },
        },
        "iid_test": {
            "count": len(iid_test),
            "selection_seed": cfg["iid_test_seed"],
            "target_distribution": {f"{key[0]}|{key[1]}": value for key, value in targets.items()},
            "actual_distribution": dict(collections.Counter(
                f"{row['difficulty']}|{row['interface']}" for row in iid_test)),
            "eligible_candidates": len(candidates),
            "excluded": excluded,
        },
        "source_files": [
            {
                "path": str((source_root / name).resolve()),
                "size": (source_root / name).stat().st_size,
            }
            for name in cfg["data_files"]
        ],
        "repositories": {
            "sdpo": repository_state(repositories["sdpo"]),
            "seed": repository_state(repositories["seed"]),
            "livecodebench": repository_state(repositories["livecodebench"]),
            "livecodebench_data": repository_state(repositories["livecodebench_data"]),
            "sdpo_data_revision": "refs/pr/6@eef204666b9879915189d71010db8aab4e766ed5",
        },
        "model": model_fingerprint(cfg["model"]),
    }
    target = lcb_model_manifest_path(cfg)
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise RuntimeError(f"Frozen LCB split manifest changed: {target}")
    atomic_json(manifest, target)
    print(json.dumps({
        "train": manifest["train"],
        "iid_test": {key: value for key, value in manifest["iid_test"].items() if key != "excluded"},
        "excluded_candidates": len(excluded),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
