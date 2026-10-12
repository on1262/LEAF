"""Freeze the base mixed-benchmark test-set build (configs/data/test_sets_qwen3_8b.json).

Four benchmarks are normalised onto one record schema and shuffled into a
single dataset of 1,331 problems:

    MATH500        500   all of the official test split
    LiveCodeBench  131   the frozen 131-problem LCB split, verbatim
    MBPP           500   the original `full` test split, task_id 11-510
    SciKnowEval    200   stratified over the L3 (reasoning) pool

Every row carries a `verifier` payload holding exactly what that problem's
public verifier needs, so no downstream code has to know where a problem came
from beyond its `task_type`.

The frozen file never changes silently: re-running this script rewrites it only
if the bytes are identical, otherwise it raises.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import re

from leaf import paths

PROJECT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT / "configs" / "data" / "test_sets_qwen3_8b.json"

SCIKNOWEVAL_LEVEL = "L3"
ANSWER_TYPES = {
    "mcq-4-choices": "mcq",
    "mcq-2-choices": "mcq",
    "true_or_false": "yes_no",
    "filling": "filling",
}


def load_config() -> dict:
    return paths.load_json(CONFIG_PATH)


def read_jsonl(path: Path) -> list[dict]:
    # Split on "\n" only: some source records contain U+2028/U+2029, which
    # str.splitlines would treat as line breaks and corrupt.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_write(rows: list[dict], path: Path) -> None:
    """Write once; afterwards only ever confirm the bytes are unchanged."""
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    encoded = payload.encode()
    if path.exists():
        existing = path.read_bytes()
        if existing == encoded:
            print(f"unchanged {path} ({len(rows)} rows)")
            return
        raise RuntimeError(f"{path} already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    print(f"wrote {path} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# per-source normalisation
# --------------------------------------------------------------------------- #


def build_math500(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    built = []
    for index, row in enumerate(rows):
        built.append({
            "uid": f"math500/{index:04d}",
            "source": "math500",
            "task_type": "math",
            "problem": row["problem"],
            "verifier": {"kind": "math", "answer": str(row["answer"])},
            "meta": {
                "unique_id": row["unique_id"],
                "subject": row.get("subject"),
                "level": row.get("level"),
            },
        })
    return built


def build_lcb(path: Path) -> list[dict]:
    """Carry the 131-problem LCB split over untouched."""
    rows = read_jsonl(path)
    built = []
    for row in rows:
        built.append({
            "uid": f"lcb/{row['uid']}",
            "source": "lcb",
            "task_type": "code_lcb",
            "problem": row["problem"],
            "verifier": {
                "kind": "lcb",
                # The LCB ground-truth parquet is keyed by this uid.
                "problem_uid": row["uid"],
                "evaluation_ref": row["evaluation_ref"],
                "public_test_indices": row["public_test_indices"],
                "heldout_test_indices": row["heldout_test_indices"],
                "full_test_count": row["full_test_count"],
                "heldout_test_count": row["heldout_test_count"],
            },
            "meta": {
                "platform": row["platform"],
                "question_id": row["question_id"],
                "difficulty": row["difficulty"],
                "interface": row["interface"],
                "question_hash": row["question_hash"],
            },
        })
    return built


def mbpp_problem_text(row: dict) -> str:
    """The task plus the asserts, so the required entry point is unambiguous."""
    tests = "\n".join(row["test_list"])
    setup = str(row.get("test_setup_code") or "").strip()
    body = f"{row['text'].strip()}\n\nYour code should pass these tests:\n\n{tests}"
    if setup:
        body += f"\n\nThe tests run after this setup code:\n\n{setup}"
    return body


def mbpp_entry_point(row: dict) -> str | None:
    for assertion in row["test_list"]:
        match = re.search(r"assert\s+\(?\s*([A-Za-z_]\w*)\s*\(", assertion)
        if match:
            return match.group(1)
    return None


def build_mbpp(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    built = []
    for row in rows:
        built.append({
            "uid": f"mbpp/{row['task_id']}",
            "source": "mbpp",
            "task_type": "code_mbpp",
            "problem": mbpp_problem_text(row),
            "verifier": {
                "kind": "mbpp",
                # All three official asserts are public; there is no held-out
                # suite here.
                "test_list": list(row["test_list"]),
                "test_setup_code": str(row.get("test_setup_code") or ""),
                "entry_point": mbpp_entry_point(row),
            },
            "meta": {"task_id": row["task_id"]},
        })
    return built


def sciknoweval_problem_text(row: dict) -> str:
    question = row["question"].strip()
    labels = row.get("choices", {}).get("label") or []
    texts = row.get("choices", {}).get("text") or []
    if labels:
        options = "\n".join(
            f"{label}. {text.strip()}" for label, text in zip(labels, texts, strict=True)
        )
        return f"{question}\n\nOptions:\n{options}"
    return question


def sciknoweval_gold(row: dict) -> str:
    """MCQ gold lives in answerKey; filling and true/false gold lives in answer."""
    key = str(row.get("answerKey") or "").strip()
    return key if key else str(row.get("answer") or "").strip()


def stratified_allocation(counts: dict, total: int) -> dict:
    """Proportional allocation with largest remainders, at least 1 per stratum."""
    population = sum(counts.values())
    if total < len(counts):
        raise ValueError("total is smaller than the number of strata")
    exact = {key: value * total / population for key, value in counts.items()}
    allocation = {key: max(1, int(value)) for key, value in exact.items()}
    while sum(allocation.values()) > total:
        key = max(
            (k for k in allocation if allocation[k] > 1),
            key=lambda k: allocation[k] - exact[k],
        )
        allocation[key] -= 1
    while sum(allocation.values()) < total:
        key = min(allocation, key=lambda k: allocation[k] - exact[k])
        allocation[key] += 1
    for key, value in allocation.items():
        if value > counts[key]:
            raise ValueError(f"stratum {key} wants {value} of {counts[key]}")
    return allocation


def build_sciknoweval(path: Path, size: int, seed: int) -> tuple[list[dict], dict]:
    rows = read_jsonl(path)
    pool = [row for row in rows if row["details"]["level"] == SCIKNOWEVAL_LEVEL]
    # Stratify on (domain, task, type): task is the finer axis and type decides
    # how the answer is graded, so both belong in the stratum key.
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for position, row in enumerate(pool):
        strata[(row["domain"], row["details"]["task"], row["type"])].append(
            {**row, "_position": position}
        )
    counts = {key: len(value) for key, value in strata.items()}
    allocation = stratified_allocation(counts, size)

    built = []
    for key in sorted(strata, key=str):
        # A per-stratum seed keeps one stratum's draw independent of the others.
        rng = random.Random(f"{seed}:{key}")
        chosen = rng.sample(strata[key], allocation[key])
        for row in sorted(chosen, key=lambda value: value["_position"]):
            gold = sciknoweval_gold(row)
            if not gold:
                raise RuntimeError(f"SciKnowEval row {row['_position']} has no gold answer")
            built.append({
                "uid": f"sciknoweval/{row['_position']:05d}",
                "source": "sciknoweval",
                "task_type": "sciqa",
                "problem": sciknoweval_problem_text(row),
                "verifier": {
                    "kind": "sciqa",
                    "answer_type": ANSWER_TYPES[row["type"]],
                    "gold": gold,
                    "choice_labels": list(row.get("choices", {}).get("label") or []),
                },
                "meta": {
                    "domain": row["domain"],
                    "task": row["details"]["task"],
                    "subtask": row["details"]["subtask"],
                    "type": row["type"],
                    "source_position": row["_position"],
                },
            })
    audit = {
        "level": SCIKNOWEVAL_LEVEL,
        "pool_size": len(pool),
        "strata": [
            {"domain": k[0], "task": k[1], "type": k[2], "available": counts[k],
             "sampled": allocation[k]}
            for k in sorted(strata, key=str)
        ],
    }
    return built, audit


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="also write the stratified smoke subset")
    args = parser.parse_args()
    config = load_config()
    root = Path(config["data_root"])
    sources = config["sources"]
    expected = config["source_sizes"]

    parts = {
        "math500": build_math500(Path(sources["math500"])),
        "lcb": build_lcb(Path(sources["lcb"])),
        "mbpp": build_mbpp(Path(sources["mbpp"])),
    }
    parts["sciknoweval"], sci_audit = build_sciknoweval(
        Path(sources["sciknoweval"]),
        expected["sciknoweval"],
        int(config["sciknoweval_sample_seed"]),
    )
    for name, rows in parts.items():
        if len(rows) != expected[name]:
            raise RuntimeError(f"{name}: expected {expected[name]} rows, got {len(rows)}")

    merged = [row for name in sorted(parts) for row in parts[name]]
    if len({row["uid"] for row in merged}) != len(merged):
        raise RuntimeError("duplicate uid in the merged dataset")
    random.Random(int(config["mix_seed"])).shuffle(merged)
    for index, row in enumerate(merged):
        row["index"] = index
    if len(merged) != int(config["dataset_size"]):
        raise RuntimeError(f"expected {config['dataset_size']} rows, got {len(merged)}")

    frozen_write(merged, root / "dataset.jsonl")

    # Every experiment runs one job per benchmark, so every source also gets its
    # own frozen file. The rows are the same objects in the same mixed order,
    # so a per-benchmark run and a slice of the mixed run are identical.
    for name in sorted(parts):
        frozen_write(
            [row for row in merged if row["source"] == name],
            root / f"dataset_{name}.jsonl",
        )

    if args.smoke:
        # A stratified slice that keeps every source and every sciqa answer
        # type, so a smoke run exercises all four verifier routes.
        size = int(config["smoke_dataset_size"])
        by_source = collections.defaultdict(list)
        for row in merged:
            by_source[row["source"]].append(row)
        allocation = stratified_allocation(
            {name: len(rows) for name, rows in by_source.items()}, size
        )
        smoke = []
        for name in sorted(by_source):
            smoke.extend(by_source[name][: allocation[name]])
        smoke.sort(key=lambda row: row["index"])
        frozen_write(smoke, root / "smoke_dataset.jsonl")
        for name in sorted(by_source):
            frozen_write(
                [row for row in smoke if row["source"] == name],
                root / f"smoke_dataset_{name}.jsonl",
            )

    manifest = {
        "revision": config["revision"],
        "dataset_size": len(merged),
        "source_sizes": {
            name: sum(1 for row in merged if row["source"] == name)
            for name in sorted(expected)
        },
        "per_benchmark_files": {
            name: str(root / f"dataset_{name}.jsonl") for name in sorted(expected)
        },
        "task_type_sizes": {
            task: sum(1 for row in merged if row["task_type"] == task)
            for task in sorted({row["task_type"] for row in merged})
        },
        "sources": sources,
        "mix_seed": config["mix_seed"],
        "sciknoweval": sci_audit,
    }
    path = root / "data_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sciknoweval"}, indent=2))


if __name__ == "__main__":
    main()
