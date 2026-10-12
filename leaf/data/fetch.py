"""Download the upstream datasets and base models at the revisions the paper used.

    python -m leaf.data.fetch [data] [models] [--dry-run]

Datasets go to ``$DATA_ROOT/raw`` (see ``leaf.paths``), models to ``$MODEL_ROOT``.
Files that are already present are not downloaded again.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from leaf import paths

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"  # release_v6
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")

MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "ff5b20257d8185524591543f8ff5993951537bb8"

MATH_TRAIN_REPO = "EleutherAI/hendrycks_math"
MATH_TRAIN_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_TRAIN_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                       "number_theory", "prealgebra", "precalculus")  # <subject>/train-00000-of-00001.parquet

SCIKNOWEVAL_REPO = "hicai-zju/SciKnowEval"
SCIKNOWEVAL_REVISION = "92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2"
SCIKNOWEVAL_FILE = "data/v2/sciknoweval_test_v2.jsonl"

MBPPPLUS_URL = "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"

# Original MBPP full/test, only needed to re-run the 1,331-problem mixed-set shuffle
# (leaf.data.prepare_test_sets) that fixed the order of the shipped exploration files.
MBPP_REPO = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_FILE = "full/test-00000-of-00001.parquet"

MODELS = {
    "Qwen3-8B": ("Qwen/Qwen3-8B", "2069b3fae1114555f3c020c81410e51fa0f656f2"),
    "Olmo-3-7B-Instruct": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
}


def present(path: Path) -> bool:
    if path.is_file():
        print(f"ok       {path}", flush=True)
        return True
    return False


def fetched(path: Path) -> None:
    print(f"fetched  {path}", flush=True)


def hub_file(repo: str, revision: str, filename: str, target: Path, dry_run: bool) -> None:
    """One file of a Hub dataset repo, placed at ``target`` (no hub cache dirs left behind)."""
    if present(target):
        return
    print(f"get      {repo}@{revision[:8]}/{filename} -> {target}", flush=True)
    if dry_run:
        return
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".download_") as scratch:
        local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                local_dir=scratch)
        os.replace(local, target)
    fetched(target)


def fetch_lcb(dry_run: bool) -> None:
    for name in LCB_FILES:
        hub_file(LCB_REPO, LCB_REVISION, name, paths.LCB_RAW_DIR / name, dry_run)


def fetch_math500(dry_run: bool) -> None:
    hub_file(MATH500_REPO, MATH500_REVISION, "test.jsonl", paths.MATH500_RAW / "test.jsonl",
             dry_run)


def fetch_math_train(dry_run: bool) -> None:
    for subject in MATH_TRAIN_SUBJECTS:
        name = f"{subject}/train-00000-of-00001.parquet"
        hub_file(MATH_TRAIN_REPO, MATH_TRAIN_REVISION, name, paths.MATH_TRAIN_RAW / name, dry_run)


def fetch_sciknoweval(dry_run: bool) -> None:
    hub_file(SCIKNOWEVAL_REPO, SCIKNOWEVAL_REVISION, SCIKNOWEVAL_FILE,
             paths.SCIKNOWEVAL_RAW / SCIKNOWEVAL_FILE, dry_run)


def fetch_mbppplus(dry_run: bool) -> None:
    target = paths.MBPPPLUS_RAW
    if present(target):
        return
    print(f"get      {MBPPPLUS_URL} -> {target}", flush=True)
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with urllib.request.urlopen(MBPPPLUS_URL) as response, \
            gzip.GzipFile(fileobj=response) as source, temporary.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    os.replace(temporary, target)
    fetched(target)


def fetch_mbpp(dry_run: bool) -> None:
    target = paths.MBPP_RAW
    if present(target):
        return
    print(f"get      {MBPP_REPO}@{MBPP_REVISION[:8]}/{MBPP_FILE} -> {target} (as jsonl)", flush=True)
    if dry_run:
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".download_") as scratch:
        local = hf_hub_download(MBPP_REPO, MBPP_FILE, repo_type="dataset",
                                revision=MBPP_REVISION, local_dir=scratch)
        rows = pq.read_table(local).to_pylist()
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    os.replace(temporary, target)
    fetched(target)


def fetch_models(dry_run: bool) -> None:
    for name, (repo, revision) in MODELS.items():
        target = paths.MODEL_ROOT / name
        print(f"get      {repo}@{revision[:8]} -> {target}", flush=True)
        if dry_run:
            continue
        from huggingface_hub import snapshot_download

        snapshot_download(repo, revision=revision, local_dir=target)
        print(f"fetched  {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", nargs="*", metavar="{data,models}",
                        help="what to fetch (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be downloaded")
    args = parser.parse_args()
    parts = args.parts or ["data", "models"]
    unknown = sorted(set(parts) - {"data", "models"})
    if unknown:
        parser.error(f"invalid part(s) {unknown} (choose from 'data', 'models')")
    if "data" in parts:
        fetch_lcb(args.dry_run)
        fetch_math500(args.dry_run)
        fetch_math_train(args.dry_run)
        fetch_sciknoweval(args.dry_run)
        fetch_mbppplus(args.dry_run)
        fetch_mbpp(args.dry_run)
    if "models" in parts:
        fetch_models(args.dry_run)


if __name__ == "__main__":
    main()
