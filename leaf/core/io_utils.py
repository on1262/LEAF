"""LiveCodeBench split semantics (configs/data/lcb_split.json), the coding prompt, schedules
and JSON(L) writers shared by every component."""
import base64
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import re
import zlib

from leaf import paths


PROJECT = Path(__file__).resolve().parents[2]


def lcb_split_config_path(project: Path | None = None) -> Path:
    """The LiveCodeBench split profile (split dates, seeds, verifier settings)."""
    return (project or PROJECT) / "configs" / "data" / "lcb_split.json"


CONFIG_PATH = lcb_split_config_path()
_CONFIG = paths.load_json(CONFIG_PATH)
# LCB split build root (train.jsonl, iid_test.jsonl, sdpo_data/) written by
# leaf.data.prepare_lcb_split / leaf.data.export_lcb_split. The shipped data holds its
# export (paths.LCB_PUBLIC_GROUND_TRUTH), so the root is only set when
# LEAF_LCB_SPLIT_ROOT or the profile's "root" names one.
_ROOT = os.environ.get("LEAF_LCB_SPLIT_ROOT") or _CONFIG.get("root")
ROOT = Path(_ROOT) if _ROOT else None


def require_root() -> Path:
    if ROOT is None:
        raise SystemExit("LCB split build root is not set: export LEAF_LCB_SPLIT_ROOT")
    return ROOT


def load_lcb_split_config() -> dict:
    return dict(_CONFIG)


def lcb_model_manifest_path(config: dict | None = None) -> Path:
    config = config or load_lcb_split_config()
    value = Path(config.get("model_manifest", "data_manifest.json"))
    return value if value.is_absolute() else require_root() / value


BASE_INSTRUCTION = (
    "You are a coding expert. You will be given a coding problem, and you need "
    "to write a correct Python program that matches the specification and "
    "passes all tests. The time limit is 1 second. You may start by outlining "
    "your thought process. In the end, please provide the complete code in a "
    "code block enclosed with ``` ```."
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_problem(text):
    return re.sub(r"\s+", " ", text).strip()


def decode_private_tests(value):
    """Decode the LCB lite hidden tests using the same path as SDPO/LCB."""
    if not isinstance(value, str):
        tests = value
    else:
        try:
            tests = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            tests = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))
    if not isinstance(tests, list) or not tests:
        raise ValueError("LiveCodeBench hidden test list is empty or malformed")
    required = {"input", "output", "testtype"}
    if any(not isinstance(test, dict) or not required <= test.keys() for test in tests):
        raise ValueError("LiveCodeBench hidden test is missing a required field")
    return tests


def test_signature(tests):
    """Hash normalized test triples without materializing another large JSON value."""
    digest = hashlib.sha256()
    for test in tests:
        for field in ("testtype", "input", "output"):
            value = str(test[field]).encode()
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def function_name(raw):
    metadata = raw.get("metadata") or "{}"
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata.get("func_name") or None


def official_problem_text(raw):
    """Reproduce SDPO's LCB problem formatting before CODE_PROMPT is applied."""
    problem = raw["question_content"]
    starter = raw.get("starter_code", "").strip()
    if starter:
        if "def " not in starter:
            raise ValueError("Functional problem starter code has no function signature")
        signature = "def " + starter.split("def ", 1)[1].split("Input\n", 1)[0].strip()
        problem += (
            "\n\nYour solution should have the following signature: ```python\n"
            + signature + "\n```"
        )
    return problem


def ordinary_prompt(row):
    """Build the ordinary prompt from either a raw LCB row or a frozen row."""
    problem = official_problem_text(row) if "question_content" in row else row["problem"]
    return BASE_INSTRUCTION + "\n\n" + problem


def load_raw(evaluation_ref):
    # The stored path is resolved by its file name (test.jsonl .. test6.jsonl) under the
    # local LiveCodeBench release (paths.LCB_RAW_DIR); the byte offset needs identical bytes.
    with (paths.LCB_RAW_DIR / Path(evaluation_ref["path"]).name).open("rb") as handle:
        handle.seek(evaluation_ref["offset"])
        return json.loads(handle.readline())


def evaluation_sample(row, suite):
    """Build an official-runner sample for public, heldout, or full hidden tests."""
    if suite not in {"public", "heldout", "full"}:
        raise ValueError(f"Unknown test suite: {suite}")
    raw = load_raw(row["evaluation_ref"])
    tests = decode_private_tests(raw["private_test_cases"])
    if suite == "public":
        indices = row["public_test_indices"]
    elif suite == "heldout":
        indices = row["heldout_test_indices"]
    else:
        indices = range(len(tests))
    selected = [tests[index] for index in indices]
    return {"input_output": json.dumps({
        "inputs": [test["input"] for test in selected],
        "outputs": [test["output"] for test in selected],
        "fn_name": function_name(raw),
    }, ensure_ascii=False)}


def balanced_schedule(problem_uids, trials, seed):
    """Return a deterministic schedule whose prefix counts differ by at most one."""
    if not problem_uids or trials < 0:
        raise ValueError("Schedule requires problems and a nonnegative trial count")
    if len(set(problem_uids)) != len(problem_uids):
        raise ValueError("Schedule problem UIDs must be unique")
    schedule = []
    round_index = 0
    while len(schedule) < trials:
        current = list(problem_uids)
        random.Random(seed + round_index).shuffle(current)
        schedule.extend(current[: trials - len(schedule)])
        round_index += 1
    return schedule


# JSON does not require escaping U+0085, U+2028 or U+2029, so a record holding
# one -- model output does, occasionally, in degenerate text -- is written as a
# literal. `str.splitlines()` then treats it as a line break and shreds the
# record on read-back. Escaping them on the way out keeps every record on one
# physical line while leaving the rest of the text readable.
LINE_BREAK_ESCAPES = {0x85: "\\u0085", 0x2028: "\\u2028", 0x2029: "\\u2029"}


def json_line(value) -> str:
    return json.dumps(value, ensure_ascii=False).translate(LINE_BREAK_ESCAPES)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries the pid: two processes writing the same output
    # (a manually started grade job racing the scheduled one, say) would
    # otherwise share one temp file and one would rename it out from under
    # the other.
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2).translate(LINE_BREAK_ESCAPES)
        + "\n"
    )
    temporary.replace(path)


def atomic_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries the pid: two processes writing the same output
    # (a manually started grade job racing the scheduled one, say) would
    # otherwise share one temp file and one would rename it out from under
    # the other.
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text("".join(json_line(row) + "\n" for row in rows))
    temporary.replace(path)
