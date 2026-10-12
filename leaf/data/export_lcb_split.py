"""Export the frozen LCB split records to the parquet schema consumed by SDPO/verl."""
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from leaf.core.io_utils import (
    BASE_INSTRUCTION, atomic_json, decode_private_tests, function_name,
    load_lcb_split_config, load_raw, require_root,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def description(problem):
    match = re.search(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:input|example|examples)\b|^.*```|^.*-----",
        problem,
    )
    return problem[: match.start() if match else len(problem)].strip()


def tests_json(row, suite):
    raw = load_raw(row["evaluation_ref"])
    tests = decode_private_tests(raw["private_test_cases"])
    if suite == "public":
        indices = row["public_test_indices"]
    elif suite == "heldout":
        indices = row["heldout_test_indices"]
    elif suite == "full":
        indices = range(len(tests))
    else:
        raise ValueError(f"Unknown suite: {suite}")
    selected = [tests[index] for index in indices]
    if not selected:
        raise ValueError(f"{row['uid']} has no tests in suite {suite}")
    return json.dumps({
        "inputs": [test["input"] for test in selected],
        "outputs": [test["output"] for test in selected],
        "testtype": selected[0]["testtype"],
        "fn_name": function_name(raw) or "",
        "time_limit": 6,
    }, ensure_ascii=False)


def verl_row(row, suite, split):
    prompt = BASE_INSTRUCTION + "\n\n" + row["problem"]
    return {
        "data_source": "livecodebench",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "code",
        "reward_model": {"style": "code", "ground_truth": tests_json(row, suite)},
        "extra_info": {
            "split": split,
            "index": str(row["index"]),
            "problem_uid": row["uid"],
            "description": description(row["problem"]),
            "problem": prompt,
            "test_suite": suite,
        },
    }


def write_parquet(rows, path, expected_rows, batch_size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = None
    pending = []
    count = 0
    try:
        for row in rows:
            pending.append(row)
            if len(pending) < batch_size:
                continue
            table = pa.Table.from_pylist(pending)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            count += len(pending)
            pending = []
        if pending:
            table = pa.Table.from_pylist(pending)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            count += len(pending)
    finally:
        if writer is not None:
            writer.close()
    if count != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} parquet rows, wrote {count}")
    temporary.replace(path)
    return {
        "path": str(path), "rows": count, "size": path.stat().st_size,
    }


def main():
    ROOT = require_root()
    train = read_jsonl(ROOT / "train.jsonl")
    iid_test = read_jsonl(ROOT / "iid_test.jsonl")
    out = ROOT / "sdpo_data"
    files = {
        "train_public": write_parquet(
            (verl_row(row, "public", "train") for row in train),
            out / "train.parquet", len(train)),
        "validation_full": write_parquet(
            (verl_row(row, "full", "test") for row in train),
            out / "test.parquet", len(train)),
        "validation_heldout": write_parquet(
            (
                verl_row(row, "heldout", "test") for row in train
                if row["heldout_test_count"] > 0
            ),
            out / "heldout.parquet",
            sum(row["heldout_test_count"] > 0 for row in train)),
        "iid_test_full": write_parquet(
            (verl_row(row, "full", "test") for row in iid_test),
            out / "iid_test.parquet", len(iid_test)),
    }
    manifest = {
        "revision": load_lcb_split_config()["revision"],
        "schema": "SDPO/verl RLHFDataset parquet",
        "files": files,
    }
    target = out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise RuntimeError(f"Frozen SDPO export changed: {target}")
    atomic_json(manifest, target)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
