"""Ablation tables (exploration ablations, history compression, divergence ablation,
internalization, repeated errors) as JSON under ``--tables`` (default ``paths.REPORT_ABLATION``)
and markdown on stdout.

1. exploration ablations (``exploration.COMPONENT_ABLATIONS`` / ``INTERLEAVED_ABLATIONS``): the LEAF
   and update-every-step rows read the seed-1 exploration cells (formal), the ablation rows the
   ``--stage`` cells under ``paths.ABLATION_EXPLORE``;
2. history-compression ablation (``exploration.COMPRESSION_ABLATIONS``, formal);
3. divergence ablation (``update.LOSS_ABLATION``: test avg@4 / pass@4, training GPU h / FLOPs);
4. internalization (``leaf.analysis.internalization_eval`` ``summary.json`` per model, under
   ``paths.ABLATION_UPDATE/<m>/<m>/trainset[_smoke]/trainacc``), on the update problems only: Base
   avg@4 (``trainset.base.update_uids``), last student rollout of the update per problem
   (``update_last_rollout``), step-20 rollout accuracy (``rollout_step20.update_accuracy``),
   relative gain;
5. feedback and repeated errors (``repeat.analyse``): LEAF against w/o feedback / w/o last response
   on memory steps after a wrong answer, and the Random Sampling base rate of the same wrong answer.

Outputs: ``curves.json`` (``{"ablation": ..., "compression": ...}``), ``update.json`` (``loss``,
``trainacc``), ``repeat.json``; ``_smoke`` is inserted before ``.json`` with ``--stage smoke``.
Missing cells print "running" (a job of the cell's condition is still queued) or "—".
``--stage smoke`` reads the ablation smoke outputs (exploration ``smoke``, loss ``smoke_full``,
``trainset_smoke``).

    python -m leaf.report.ablation_tables [--tables DIR] [--stage formal|smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

from leaf import paths
from leaf.report import exploration, repeat, update
from leaf.report.common import MODELS, Slurm, pct, read_json, read_jsonl

TRAIN_NAMES = (("lcb", "LCB"), ("math500", "MATH"), ("sciknoweval", "SciKnowEval"))
TRAINSET_LABELS = ("base", "leaf_update")
UPDATE_STAGE = {"formal": "formal", "smoke": "smoke_full"}
TRAINSET_KIND = {"formal": "trainset", "smoke": "trainset_smoke"}


# --------------------------------------------------------------------------- exploration
def ablation_methods(methods) -> list:
    return [m for m in methods if m.root == paths.ABLATION_EXPLORE]


def explore_curves(stage: str, methods=exploration.EXPLORE_ABLATIONS) -> tuple[dict, list]:
    """Curves of the exploration ablation sets: seed-1 rows formal, ablation rows ``stage``."""
    new = ablation_methods(methods)
    data, errors = exploration.curves([m for m in methods if m not in new], "formal")
    extra, extra_errors = exploration.curves(new, stage)
    for model, block in extra.items():
        data.setdefault(model, {}).update(block)
    return data, errors + extra_errors


def queued_jobs() -> set[str]:
    """Job ids of this account still in a Slurm queue (pending / running); empty without Slurm."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%i"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _resolved(path) -> str:
    """Run directory as a symlink-resolved string (jobs.jsonl may record it through a symlinked root)."""
    return str(Path(path).resolve())


def queued_outputs(jobs: Path, queued: set[str]) -> set[str]:
    """``output`` directories of the ``jobs.jsonl`` rows (submitters' job records) whose job is
    still queued: rollout / training / grading / test jobs all record the directory they fill."""
    return {_resolved(r["output"]) for r in read_jsonl(jobs) if r.get("output") and str(r.get("job")) in queued}


def running(model: str, methods, stage: str, queued: set[str]) -> frozenset:
    """(method key, benchmark) of the ablation cells with a queued job (rollout, grading or LCB
    private regrade) for this model / stage; a missing cell prints "running", one with data is
    marked as still running."""
    out = set()
    for m in ablation_methods(methods):
        outputs = queued_outputs(m.run_root(model) / "jobs.jsonl", queued)
        out |= {(m.key, b) for b in exploration.BENCH_KEYS if _resolved(m.cell(model, b, stage)) in outputs}
    return frozenset(out)


def explore_tables(data: dict, stage: str, queued: set[str]) -> tuple[str, str]:
    perf, cost = [], []
    for model, name in MODELS:
        run = running(model, exploration.EXPLORE_ABLATIONS, stage, queued)
        for title, methods in (("LEAF ablations", exploration.COMPONENT_ABLATIONS),
                               ("interleaved update interval", exploration.INTERLEAVED_ABLATIONS)):
            perf.append(f"#### {name} · {title}: mean accuracy over 24 steps / cumulative pass rate at step 24 "
                        "(or at the 5 GPU h cutoff)\n\n"
                        + exploration.perf_table(data, model, methods, run, strict=True, annotate=True) + "\n")
            cost.append(f"#### {name} · {title}: GPU h / PFLOPs / generated tokens (M), cumulative at each "
                        "benchmark's cutoff\n\n"
                        + exploration.cost_table(data, model, methods, run, annotate=True) + "\n")
    return "\n".join(perf), "\n".join(cost)


def compression_tables(data: dict) -> str:
    out = []
    for model, name in MODELS:
        methods = [m for m in exploration.COMPRESSION_ABLATIONS if m.key in data.get(model, {})]
        if not methods:
            continue
        out.append(f"#### {name}: mean accuracy / cumulative pass rate\n\n" + exploration.perf_table(data, model, methods))
        out.append(f"#### {name}: GPU h / PFLOPs / generated tokens (M)\n\n"
                   + exploration.cost_table(data, model, methods, mark=""))
    return "\n\n".join(out) + "\n" if out else "(no compression-ablation results)\n"


# --------------------------------------------------------------------------- policy update
def loss_methods(stage: str) -> tuple:
    """``update.LOSS_ABLATION`` with the ablation runs read from ``stage``'s run directories."""
    return tuple(dataclasses.replace(m, stage=UPDATE_STAGE[stage]) if m.root == paths.ABLATION_UPDATE else m
                 for m in update.LOSS_ABLATION)


def update_jobs(model: str, root: Path | None = None) -> Path:
    return (root or paths.ABLATION_UPDATE) / model / "jobs.jsonl"


def loss_running(model: str, methods, queued: set[str], root: Path | None = None) -> tuple[frozenset, frozenset]:
    """((label, test) with a queued test job, (label, training set) with a queued training job)
    of the divergence-ablation rows."""
    outputs = queued_outputs(update_jobs(model, root), queued)
    tests, trains = set(), set()
    for m in methods:
        if m.root != (root or paths.ABLATION_UPDATE):
            continue
        tests |= {(m.label, t) for t, _, train in update.TESTS if _resolved(m.test_dir(model, t, train)) in outputs}
        trains |= {(m.label, r.name) for r in m.run_dirs(model) if _resolved(r) in outputs}
    return frozenset(tests), frozenset(trains)


def loss_tables(methods, slurm: Slurm, queued: set[str] = frozenset()) -> tuple[str, str, dict]:
    run = {model: loss_running(model, methods, queued) for model, _ in MODELS}
    perf = "\n\n".join(f"#### {name}: test avg@4 / pass@4\n\n" + update.perf_table(model, slurm, methods, run[model][0])
                       for model, name in MODELS)
    cost = "\n\n".join(f"#### {name}: training GPU time (h) and FLOPs\n\n"
                       + update.train_cost_table(model, slurm, methods, run[model][1]) for model, name in MODELS)
    rows = {model: [{"label": m.label, "tests": res, "overall": update.overall(res)}
                    for m, res in update.rows(model, methods)] for model, _ in MODELS}
    return perf, cost, rows


# --------------------------------------------------------------------------- internalization
def trainacc_dir(model: str, stage: str, root: Path | None = None) -> Path:
    return (root or paths.ABLATION_UPDATE) / model / model / TRAINSET_KIND[stage] / "trainacc"


def trainacc(model: str, stage: str, root: Path | None = None) -> dict:
    """``summary.json`` of ``leaf.analysis.internalization_eval`` (``{"benchmarks": {}}`` while missing), with
    each missing train-set "all problems" entry filled from the evaluation's own summary.json."""
    directory = trainacc_dir(model, stage, root)
    summary = read_json(directory / "summary.json") or {"benchmarks": {}}
    for bench, _ in TRAIN_NAMES:
        sets = summary["benchmarks"].setdefault(bench, {}).setdefault("trainset", {})
        for label in TRAINSET_LABELS:
            if "all" in (sets.get(label) or {}):
                continue
            own = read_json(directory.parent / label / bench / "summary.json")
            if own:
                sets[label] = {"all": {"problems": own.get("problems"), "avg_at_k": own["avg_at_k"],
                                       "pass_at_k": own["pass_at_k"]}}
    return summary


def trainacc_running(model: str, stage: str, queued: set[str], root: Path | None = None) -> frozenset:
    """(label, benchmark) of the train-set evaluations with a queued generate / grade job, plus
    "trainacc" while the judge / summary job is queued."""
    outputs = queued_outputs(update_jobs(model, root), queued)
    directory = trainacc_dir(model, stage, root)
    out = {(label, b) for label in TRAINSET_LABELS for b, _ in TRAIN_NAMES
           if _resolved(directory.parent / label / b) in outputs}
    return frozenset(out | ({"trainacc"} if _resolved(directory) in outputs else set()))


def relative_gain(base: float | None, update: float | None, train: float | None) -> str:
    """(policy update acc - base acc) / (training acc - base acc) as a percentage ("—" if undefined)."""
    if None in (base, update, train) or abs(train - base) < 1e-9:
        return "—"
    return pct((update - base) / (train - base))


def trainacc_table(summary: dict, running=frozenset()) -> str:
    """Per training set, on the update problems (uids of ``successful_contexts.jsonl``): Base avg@4,
    the last student rollout of the update per problem, the step-20 rollout accuracy and the
    relative gain."""
    out = ["| Training set | Problems | Base acc | Policy update acc | Training acc | Relative gain |",
           "|---|---:|---:|---:|---:|---:|"]
    for bench, name in TRAIN_NAMES:
        cell = summary["benchmarks"].get(bench) or {}
        summ = "running" if "trainacc" in running else "—"
        base_entry = ((cell.get("trainset") or {}).get("base") or {}).get("update_uids") or {}
        base = base_entry.get("avg_at_k")
        update_acc = (cell.get("update_last_rollout") or {}).get("accuracy")
        train = (cell.get("rollout_step20") or {}).get("update_accuracy")
        base_missing = "running" if ("base", bench) in running or "trainacc" in running else "—"
        count = cell.get("subset_problems", cell.get("update_problems", "—"))
        cols = [str(count), pct(base) if base is not None else base_missing,
                pct(update_acc) if update_acc is not None else summ,
                pct(train) if train is not None else summ, relative_gain(base, update_acc, train)]
        out.append(f"| {name} | " + " | ".join(cols) + " |")
    return "\n".join(out)


def trainacc_tables(stage: str, root: Path | None = None, queued: set[str] = frozenset()) -> tuple[str, dict]:
    data = {model: trainacc(model, stage, root) for model, _ in MODELS}
    text = "\n\n".join(f"#### {name}\n\n"
                       + trainacc_table(data[model], trainacc_running(model, stage, queued, root))
                       for model, name in MODELS)
    return text, data


# --------------------------------------------------------------------------- main
def step(name: str, func):
    start = time.time()
    value = func()
    print(f"[{time.strftime('%T')}] {name}: {time.time() - start:.0f} s", file=sys.stderr, flush=True)
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablation tables as JSON (and markdown on stdout).")
    parser.add_argument("--tables", type=Path, default=paths.REPORT_ABLATION,
                        help="json table directory (default: paths.REPORT_ABLATION)")
    parser.add_argument("--stage", choices=("formal", "smoke"), default="formal",
                        help="which ablation outputs to read (the seed-1 and compression rows are always formal)")
    args = parser.parse_args(argv)
    args.tables.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.stage == "formal" else "_smoke"

    data, errors = step("exploration curves", lambda: explore_curves(args.stage))
    compression, compression_errors = step("compression curves",
                                           lambda: exploration.curves(exploration.COMPRESSION_ABLATIONS))
    errors += compression_errors
    (args.tables / f"curves{suffix}.json").write_text(
        json.dumps({"ablation": data, "compression": compression}, indent=1) + "\n")
    queued = queued_jobs()
    explore_perf, explore_cost = explore_tables(data, args.stage, queued)

    methods = loss_methods(args.stage)
    slurm = step("job accounting", lambda: Slurm(update.slurm_roots(methods),
                                                 stages=("formal", UPDATE_STAGE[args.stage])))
    loss_perf, loss_cost, loss_rows = step("loss tables", lambda: loss_tables(methods, slurm, queued))
    internal, internal_data = trainacc_tables(args.stage, queued=queued)
    repeats = step("repeat errors", lambda: repeat.analyse(args.stage))
    (args.tables / f"repeat{suffix}.json").write_text(json.dumps(repeats, indent=1) + "\n")
    (args.tables / f"update{suffix}.json").write_text(
        json.dumps({"loss": loss_rows, "trainacc": internal_data}, indent=1, default=str) + "\n")

    for title, text in (("Exploration ablations: performance", explore_perf),
                        ("Exploration ablations: cost", explore_cost),
                        ("History compression ablations", compression_tables(compression)),
                        ("Loss ablation: performance", loss_perf), ("Loss ablation: training cost", loss_cost),
                        ("Internalization (update training problems)", internal),
                        ("Feedback and repeated errors", repeat.tables(repeats))):
        print(f"\n### {title}\n\n{text}")
    print(f"wrote {args.tables}; {len(errors)} exploration cells failed", flush=True)
    if errors:
        print("\n".join(json.dumps(e) for e in errors), file=sys.stderr)


if __name__ == "__main__":
    main()
