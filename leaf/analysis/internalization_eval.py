"""Internalization check: last-step training accuracy of the LEAF update.

Per model x training set (LCB / MATH / SciKnowEval) it collects, into one json:

(a) ``rollout_step20``: the LEAF rollout that produced the update data (the run manifest ->
    its data manifest's ``rollout``), ``final.public_correct`` of step 20 -- the accuracy
    *with* the privileged context; over every problem and over the update problems (the uids
    of ``successful_contexts.jsonl``);
(b) ``update_last_step``: the update's own last-step student samples (``samples/step_0020.jsonl``,
    no context), graded offline like the test evaluation: LCB on every test of the training
    problem (``leaf.paths.LCB_TRAIN_GROUND_TRUTH``), MATH by math_verify, SciKnowEval by the
    base-model judge, which needs a GPU: ``judge`` writes ``update_samples_sciknoweval_judged.jsonl``
    and ``summarize`` (CPU) grades those records;
(d) ``update_last_rollout``: per update problem (uid of ``successful_contexts.jsonl``) its last
    student rollout of the update run -- the sample of the highest ``samples/step_*.jsonl``
    containing the uid (the last such line of that step) -- graded as in (b); SciKnowEval from
    ``judge --samples last_rollout`` (``update_last_rollout_sciknoweval_judged.jsonl``);
(c) ``trainset``: avg@4 / pass@4 of the train-set evaluations under ``--trainset-root``
    (``<label>/<bm>/{generations,grades}.jsonl`` of ``leaf.grading.test_eval`` with
    ``--test-set``, the ``trainset`` runs of run/general_ablations.sh), over every problem and
    restricted to the update uids (:func:`restricted_summary`).

    LEAF_CONFIG=configs/ablation/update_qwen3_8b.json python -m leaf.analysis.internalization_eval judge \\
        --model qwen3_8b --output DIR
    LEAF_CONFIG=configs/ablation/update_qwen3_8b.json python -m leaf.analysis.internalization_eval judge \\
        --model qwen3_8b --output DIR --samples last_rollout
    LEAF_CONFIG=configs/ablation/update_qwen3_8b.json python -m leaf.analysis.internalization_eval summarize \\
        --model qwen3_8b --output DIR --trainset-root $RESULTS_ROOT/ablation/update/qwen3_8b/qwen3_8b/trainset
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from leaf import paths

BENCHMARKS = ("lcb", "math500", "sciknoweval")
# the update run of each training set (leaf.update.leaf_update.UPDATE_METHODS)
SOURCE_METHOD = "leaf_update"
ROLLOUT_STEP = 20
JUDGED = "update_samples_sciknoweval_judged.jsonl"
JUDGED_LAST = "update_last_rollout_sciknoweval_judged.jsonl"
JUDGED_FILES = {"last_step": JUDGED, "last_rollout": JUDGED_LAST}


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines also breaks at U+2028 inside JSON strings
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def leaf_update_run(model: str, benchmark: str, method: str | None = None, root: Path | None = None) -> Path:
    """The LEAF update run of one model x training set."""
    method = method or SOURCE_METHOD
    return Path(root or paths.GENERAL_LEAF_SEED1) / model / model / "formal" / "update" / method / benchmark


def run_sources(run: Path) -> dict:
    """``{data, dataset, rollout, steps}`` of an update run, from its manifests."""
    manifest = json.loads((run / "manifest.json").read_text())
    data = Path(manifest["data"])
    data_manifest = json.loads((data / "manifest.json").read_text())
    return {"data": data, "dataset": Path(manifest["dataset"]), "rollout": Path(data_manifest["rollout"]),
            "steps": int(manifest["steps"])}


def update_uids(data: Path) -> set[str]:
    return {row["uid"] for row in read_jsonl(data / "successful_contexts.jsonl")}


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def rollout_accuracy(rollout: Path, uids: set[str], step: int = ROLLOUT_STEP) -> dict:
    """(a): ``final.public_correct`` of one rollout step (with the privileged context)."""
    path = rollout / "steps" / f"step_{step:04d}" / "final.jsonl"
    rows = read_jsonl(path)
    restricted = [r for r in rows if r["problem_uid"] in uids]
    return {"path": str(path), "step": step, "problems": len(rows),
            "accuracy": mean(bool(r["public_correct"]) for r in rows),
            "update_problems": len(restricted),
            "update_accuracy": mean(bool(r["public_correct"]) for r in restricted)}


def sample_records(samples: list[dict], dataset: dict[str, dict], model: str, benchmark: str) -> list[dict]:
    """Canonical grading records (leaf.core.project_config) of update student samples."""
    out = []
    for sample in samples:
        row = dataset[sample["uid"]]
        out.append({
            "key": f"internalization_update:{benchmark}:{sample['uid']}:{sample['slot']}",
            "problem_uid": sample["uid"], "source": row["source"], "task_type": row["task_type"],
            "problem": row["problem"], "verifier": row["verifier"], "trial_index": int(sample["slot"]),
            "sample_index": 0, "model": model, "response": sample["response"],
            "finish_reason": sample.get("finish_reason"),
        })
    return out


def last_step_records(model: str, benchmark: str, limit: int | None = None,
                      run: Path | None = None) -> tuple[list[dict], dict]:
    run = run or leaf_update_run(model, benchmark)
    sources = run_sources(run)
    path = run / "samples" / f"step_{sources['steps']:04d}.jsonl"
    samples = read_jsonl(path)[:limit] if limit else read_jsonl(path)
    dataset = {row["uid"]: row for row in read_jsonl(sources["dataset"])}
    return sample_records(samples, dataset, model, benchmark), {"run": str(run), "samples": str(path), **{
        key: str(value) for key, value in sources.items() if key != "steps"}, "step": sources["steps"]}


def last_samples(run: Path, uids: set[str] | None = None) -> list[dict]:
    """The last student rollout of each uid: from the highest ``samples/step_*.jsonl`` containing it,
    the last line of that step; with the step recorded as ``step``. Sorted by uid."""
    last: dict[str, dict] = {}
    for path in sorted((run / "samples").glob("step_*.jsonl")):
        step = int(path.stem.split("_")[1])
        for sample in read_jsonl(path):
            if uids is None or sample["uid"] in uids:
                last[sample["uid"]] = {**sample, "step": step}  # later steps / lines overwrite
    return [last[uid] for uid in sorted(last)]


def last_rollout_records(model: str, benchmark: str, limit: int | None = None,
                         run: Path | None = None, subset: set[str] | None = None) -> tuple[list[dict], dict]:
    """(d): grading records of the last student rollout of every update problem; with ``subset`` only
    the update problems in it (smokes: the problems of the base train-set evaluation), else with
    ``limit`` the first N uids."""
    run = run or leaf_update_run(model, benchmark)
    sources = run_sources(run)
    uids = update_uids(sources["data"])
    samples = last_samples(run, uids & subset if subset is not None else uids)
    samples = samples[:limit] if limit and subset is None else samples
    dataset = {row["uid"]: row for row in read_jsonl(sources["dataset"])}
    records = sample_records(samples, dataset, model, benchmark)
    for record, sample in zip(records, samples, strict=True):
        record["key"] = f"internalization_last:{benchmark}:{sample['uid']}:{sample['step']}:{sample['slot']}"
        record["update_step"] = sample["step"]
    steps = [s["step"] for s in samples]
    return records, {"run": str(run), "uids": len(samples),
                     "step_min": min(steps) if steps else None, "step_max": max(steps) if steps else None,
                     "step_mean": mean(steps)}


def grading_config(config: dict, benchmark: str) -> dict:
    """The test evaluation's grading setup on a training set (LCB: every training test)."""
    config = dict(config)
    if benchmark == "lcb":
        config["lcb_ground_truth"] = str(paths.LCB_TRAIN_GROUND_TRUTH)
        config["lcb_stop_at_first_failure"] = False
        os.environ.pop("LEAF_LCB_PUBLIC_GROUND_TRUTH", None)
    return config


def restricted_summary(records: list[dict], grades: list[dict], uids: set[str] | None) -> dict:
    """avg@k / pass@k (leaf.grading.test_eval.summarize) over the records of ``uids``."""
    from leaf.grading.test_eval import summarize

    pairs = [(r, g) for r, g in zip(records, grades, strict=True) if uids is None or r["problem_uid"] in uids]
    if not pairs:
        return {"problems": 0, "attempts": 0}
    return summarize([p[0] for p in pairs], [p[1] for p in pairs])


def trainset_results(root: Path, benchmark: str, uids: set[str]) -> dict:
    """(c): every ``<label>/<benchmark>`` train-set evaluation under ``root``."""
    out = {}
    for directory in sorted(Path(root).glob(f"*/{benchmark}")):
        generations, grades = directory / "generations.jsonl", directory / "grades.jsonl"
        if not grades.exists():
            out[directory.parent.name] = {"missing": str(grades)}
            continue
        records, rows = read_jsonl(generations), read_jsonl(grades)
        out[directory.parent.name] = {"path": str(directory), "all": restricted_summary(records, rows, None),
                                      "update_uids": restricted_summary(records, rows, uids)}
    return out


def smoke_subset(args, benchmark: str) -> set[str] | None:
    """Under ``--limit`` with ``--trainset-root``: the problem uids of the base train-set evaluation,
    so every column of a smoke uses the same problems. ``None`` otherwise (formal: all update uids)."""
    if not args.limit or not args.trainset_root:
        return None
    generations = Path(args.trainset_root) / "base" / benchmark / "generations.jsonl"
    return {row["problem_uid"] for row in read_jsonl(generations)} if generations.exists() else None


# --------------------------------------------------------------------------- #
# judge (GPU) / summarize (CPU)
# --------------------------------------------------------------------------- #

def judge(args, config: dict) -> None:
    """Judge the update's last-step SciKnowEval samples with the base model (leaf.verifiers.llm_judge)."""
    from vllm import LLM

    from leaf.verifiers import llm_judge
    from leaf.core.project_config import atomic_jsonl, model_path
    from leaf.explore.leaf_rollout import model_max_model_len

    if args.samples == "last_rollout":
        records, info = last_rollout_records(args.model, "sciknoweval", args.limit,
                                             subset=smoke_subset(args, "sciknoweval"))
    else:
        records, info = last_step_records(args.model, "sciknoweval", args.limit)
    settings = config["generalization"]
    started = time.monotonic()
    llm = LLM(model=str(model_path(config, args.model)), dtype="bfloat16",
              max_model_len=model_max_model_len(config, args.model),
              max_num_seqs=int(settings["test_max_num_seqs"]["sciknoweval"]),
              max_num_batched_tokens=int(config.get("random_sampling_max_num_batched_tokens", 32768)),
              gpu_memory_utilization=float(settings["test_gpu_memory_utilization"]),
              enable_prefix_caching=True, seed=int(settings["test_seed"]), disable_log_stats=True)
    engine_seconds = time.monotonic() - started
    judge_seconds = llm_judge.judge_records_offline(llm, llm.get_tokenizer(), records, config)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(records, args.output / JUDGED_FILES[args.samples])
    print(json.dumps({"event": "judged", "samples": args.samples, "records": len(records), **info, "engine_seconds": engine_seconds,
                      "judge_seconds": judge_seconds}), flush=True)


def last_rollout_summary(args, config: dict, benchmark: str) -> dict:
    """(d): accuracy of the last student rollout per update problem (SciKnowEval: judged records)."""
    from leaf.verifiers.answer_grader import grade_records

    records, info = last_rollout_records(args.model, benchmark, args.limit, subset=smoke_subset(args, benchmark))
    if benchmark == "sciknoweval":
        judged = args.output / JUDGED_LAST
        if not judged.exists():
            return {"missing": str(judged), **info}
        records = read_jsonl(judged)
    started = time.monotonic()
    grades = grade_records(records, grading_config(config, benchmark),
                           workers=int(args.workers or config["verifier_workers"]))
    correct = [bool(g["correct"]) for g in grades]
    return {**info, "problems": len(records), "accuracy": mean(correct),
            "truncated_fraction": mean(r.get("finish_reason") == "length" for r in records),
            "grading_seconds": time.monotonic() - started}


def summarize(args, config: dict) -> None:
    from leaf.core.project_config import atomic_json
    from leaf.verifiers.answer_grader import grade_records

    out = {"model": args.model, "config": os.environ.get("LEAF_CONFIG"), "limit": args.limit,
           "trainset_root": str(args.trainset_root) if args.trainset_root else None, "benchmarks": {}}
    for benchmark in args.benchmark or BENCHMARKS:
        records, info = last_step_records(args.model, benchmark, args.limit)
        sources = run_sources(Path(info["run"]))
        uids = update_uids(sources["data"])
        subset = smoke_subset(args, benchmark)
        uids_eval = uids & subset if subset is not None else uids  # the fixed subset of every column
        cell = {"sources": info, "update_problems": len(uids), "subset_problems": len(uids_eval),
                "rollout_step20": rollout_accuracy(sources["rollout"], uids_eval)}
        if benchmark == "sciknoweval":
            judged = args.output / JUDGED
            records = read_jsonl(judged) if judged.exists() else None
        if records is None:
            cell["update_last_step"] = {"missing": str(args.output / JUDGED)}
        else:
            started = time.monotonic()
            grades = grade_records(records, grading_config(config, benchmark),
                                   workers=int(args.workers or config["verifier_workers"]))
            correct = [bool(g["correct"]) for g in grades]
            cell["update_last_step"] = {
                "samples": len(records), "accuracy": mean(correct),
                "truncated_fraction": mean(r.get("finish_reason") == "length" for r in records),
                "grading_seconds": time.monotonic() - started,
            }
        cell["update_last_rollout"] = last_rollout_summary(args, config, benchmark)
        if args.trainset_root:
            cell["trainset"] = trainset_results(args.trainset_root, benchmark, uids_eval)
        out["benchmarks"][benchmark] = cell
        print(json.dumps({"benchmark": benchmark, **{k: v for k, v in cell.items() if k != "trainset"}}),
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(out, args.output / "summary.json")
    print(f"wrote {args.output / 'summary.json'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("judge", "summarize"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory of summary.json / judged records")
    parser.add_argument("--benchmark", action="append", default=[], choices=BENCHMARKS)
    parser.add_argument("--trainset-root", type=Path, default=None,
                        help="<root>/<label>/<bm> train-set evaluations to tabulate")
    parser.add_argument("--limit", type=int, default=None, help="first N last-step samples only (smokes)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--samples", choices=tuple(JUDGED_FILES), default="last_step",
                        help="judge: the update's last-step samples (b) or the last rollout per update problem (d)")
    args = parser.parse_args(argv)
    from leaf.core.project_config import load_config

    config = load_config()
    if args.model not in config["models"]:
        raise SystemExit(f"{args.model} is not in {os.environ.get('LEAF_CONFIG')}")
    (judge if args.stage == "judge" else summarize)(args, config)


if __name__ == "__main__":
    main()
