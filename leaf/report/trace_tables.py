"""Trace tables (SFT on y*, teacher accuracy under successful / failure / random traces, and
train-set self-correction after the policy update): json tables + markdown on stdout.

1. SFT on y*: y* rewrite statistics (``<sft root>/<m>/<m>/formal/update_data/<bm>/ystar/rewrite_summary.json``,
   sft root = ``paths.GENERAL_SFT_SEED1``), test avg@4 / pass@4 of ``update.SFT_YSTAR`` next to Base and
   the LEAF update, training cost;
2. traces: context statistics (``contexts_summary.json`` of the same update data), teacher accuracy
   (``<traces root>/<m>/<m>/teacher/<condition>/<bm>/summary.json``, traces root =
   ``paths.ABLATION_TRACES``, :mod:`leaf.analysis.teacher_accuracy`);
3. train-set self-correction (:func:`leaf.analysis.self_correction_eval.summarize_run`, update problems
   only): the base model = steps 1..10 of the update data's source rollout, trained models =
   ``<traces root>/<m>/<m>/selfcorr/<label>/<bm>``.

Missing cells print "running" (a job recorded for that output directory is still queued) or "—".

    python -m leaf.report.trace_tables [--tables DIR] [--stage formal|smoke]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from leaf import paths
from leaf.analysis import self_correction_eval
from leaf.report import update
from leaf.report.common import MODELS, Slurm, pct, read_json
from leaf.report.ablation_tables import queued_jobs, queued_outputs

TRAIN_NAMES = (("lcb", "LCB"), ("math500", "MATH"), ("sciknoweval", "SciKnowEval"))
CONDITIONS = (("successful", "Successful traces"), ("failure", "Failure traces"), ("random", "Random traces"))
SELFCORR_ROWS = (("base", "Base"), ("leaf_update", "Ours"), ("sft_ystar", "SFT"))
STAGE = {"formal": "formal", "smoke": "smoke_full"}
KIND = {"formal": "", "smoke": "_smoke"}


def sft_root(model: str) -> Path:
    """SFT on y* of seed 1: its update data (contexts, y* pairs and rewrites), runs and tests."""
    return paths.GENERAL_SFT_SEED1 / model / model


def traces_root(model: str) -> Path:
    """Teacher accuracy and self-correction runs."""
    return paths.ABLATION_TRACES / model / model


def data_dir(model: str, benchmark: str, stage: str) -> Path:
    return sft_root(model) / STAGE[stage] / "update_data" / benchmark


def cell(value, running: bool = False) -> str:
    return value if value is not None else ("running" if running else "—")


class Queue:
    """Output directories with a queued job, per model (from ``<root>/<m>/jobs.jsonl`` of the SFT and
    traces roots)."""

    def __init__(self):
        jobs = queued_jobs()
        self.outputs = {m: queued_outputs(paths.GENERAL_SFT_SEED1 / m / "jobs.jsonl", jobs)
                        | queued_outputs(paths.ABLATION_TRACES / m / "jobs.jsonl", jobs) for m, _ in MODELS}

    def __call__(self, model: str, directory: Path) -> bool:
        return str(Path(directory).resolve()) in self.outputs.get(model, set())


# --------------------------------------------------------------------------- SFT data
def rewrite_table(stage: str, queue: Queue) -> tuple[str, dict]:
    out = ["| Model | Training set | (c+, y+) pairs | verified y* | dropped | verified per round (1 / 2 / 3 / 4 / 5) | y* mean tokens | truncated | generation + verification GPU h |",
           "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    tables = {}
    slurm = Slurm([paths.GENERAL_SFT_SEED1], stages=(f"{stage}_rewrite",))
    for model, name in MODELS:
        for bench, bname in TRAIN_NAMES:
            directory = data_dir(model, bench, stage) / "ystar"
            s = read_json(directory / "rewrite_summary.json")
            tables[f"{model}/{bench}"] = s
            if not s:
                out.append(f"| {name} | {bname} | " + " | ".join([cell(None, queue(model, directory))] * 7) + " |")
                continue
            rounds = s.get("verified_by_round") or {}
            per = " / ".join(str(rounds.get(str(r), rounds.get(r, 0))) for r in range(1, 6))
            hours = slurm.hours(directory)
            out.append(f"| {name} | {bname} | {s['pairs']:,} | {s['verified']:,} | {s['dropped']:,} | {per} | "
                       f"{s['ystar_tokens_mean']:.0f} | {pct(s['truncated_fraction'])} | "
                       f"{'—' if hours is None else f'{hours:.2f}'} |")
    return "\n".join(out), tables


# --------------------------------------------------------------------------- contexts
def context_table(stage: str) -> tuple[str, dict]:
    out = ["| Model | Training set | Problems | successful contexts / problem | failure contexts / problem | problems without failure context (plain prompt) | random-trace contexts | of which feedback from other problems | restart contexts (unchanged) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    tables = {}
    for model, name in MODELS:
        for bench, bname in TRAIN_NAMES:
            s = read_json(data_dir(model, bench, "formal") / "contexts_summary.json")
            tables[f"{model}/{bench}"] = s
            if not s:
                out.append(f"| {name} | {bname} | " + " | ".join(["—"] * 7) + " |")
                continue
            out.append(f"| {name} | {bname} | {s['problems']} | {s['success_contexts_mean']:.2f} | "
                       f"{s['fail_contexts_mean']:.2f} | {s['fallback']} | {s['shuffled']:,} | {s['other_problem']:,} | "
                       f"{s['kept_restart']:,} |")
    return "\n".join(out), tables


def teacher_table(stage: str, queue: Queue) -> tuple[str, dict]:
    kind = "teacher" + KIND[stage]
    heads = [f"{bname}" for _, bname in TRAIN_NAMES]
    out = []
    tables = {}
    for model, name in MODELS:
        rows = [f"#### {name}: teacher accuracy (base model rollout under the given context)\n",
                "| Context | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
        for condition, cname in CONDITIONS:
            cells = []
            for bench, _ in TRAIN_NAMES:
                directory = traces_root(model) / kind / condition / bench
                s = read_json(directory / "summary.json")
                tables[f"{model}/{condition}/{bench}"] = s
                if not s:
                    cells.append(cell(None, queue(model, directory)))
                    continue
                text = pct(s["accuracy"])
                if condition == "failure" and s.get("accuracy_without_fallback") is not None:
                    text += f" (without plain-prompt problems: {pct(s['accuracy_without_fallback'])})"
                cells.append(text)
            rows.append(f"| {cname} | " + " | ".join(cells) + " |")
        out.append("\n".join(rows))
    return "\n\n".join(out), tables


# --------------------------------------------------------------------------- tests / cost
def methods_for(stage: str, methods) -> tuple:
    return tuple(dataclasses.replace(m, stage=STAGE[stage]) if m.root == paths.GENERAL_SFT_SEED1 else m
                 for m in methods)


def running(model: str, methods, queue: Queue) -> tuple[frozenset, frozenset]:
    tests, trains = set(), set()
    for m in methods:
        if m.root != paths.GENERAL_SFT_SEED1:
            continue
        tests |= {(m.label, t) for t, _, train in update.TESTS if queue(model, m.test_dir(model, t, train))}
        trains |= {(m.label, r.name) for r in m.run_dirs(model) if queue(model, r)}
    return frozenset(tests), frozenset(trains)


def update_tables(methods, slurm: Slurm, queue: Queue) -> tuple[str, str]:
    perf, cost = [], []
    for model, name in MODELS:
        tests, trains = running(model, methods, queue)
        perf.append(f"#### {name}: test avg@4 / pass@4\n\n" + update.perf_table(model, slurm, methods, tests))
        cost.append(f"#### {name}: training GPU time (h) and FLOPs\n\n" + update.train_cost_table(model, slurm, methods, trains))
    return "\n\n".join(perf), "\n\n".join(cost)


# --------------------------------------------------------------------------- self-correction
def selfcorr_run(model: str, bench: str, label: str, stage: str) -> Path:
    if label == "base":
        manifest = read_json(paths.GENERAL_LEAF_SEED1 / model / model / "formal" / "update_data" / bench / "manifest.json")
        return Path(manifest["rollout"])
    return traces_root(model) / ("selfcorr" + KIND[stage]) / label / bench


def selfcorr_tables(stage: str, queue: Queue, steps: int) -> tuple[str, dict]:
    out, tables = [], {}
    for model, name in MODELS:
        for bench, bname in TRAIN_NAMES:
            uids = data_dir(model, bench, "formal") / "successful_contexts.jsonl"
            rows = [f"#### {name} · {bname}\n",
                    "| Model | Transitions | wrong to right | right to wrong | repeated error | wrong to wrong | restart step accuracy | mean accuracy |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for label, lname in SELFCORR_ROWS:
                run = selfcorr_run(model, bench, label, stage)
                done = (run / "generation_summary.json").exists() if label != "base" else run.exists()
                if not done:
                    rows.append(f"| {lname} | " + " | ".join([cell(None, label != "base" and queue(model, run))] * 7) + " |")
                    continue
                s = self_correction_eval.summarize_run(run, steps if label == "base" or stage == "formal" else 2, uids)
                tables[f"{model}/{bench}/{label}"] = s
                m = s["update_uids"]
                rows.append(f"| {lname} | {m['transitions']:,} | {pct(m['wrong_to_right_rate'])} | "
                            f"{pct(m['right_to_wrong_rate'])} | {pct(m['repeat_error_rate'])} | "
                            f"{pct(m['wrong_to_wrong_rate'])} | {pct(m['turn1_accuracy'])} | {pct(m['accuracy'])} |")
            out.append("\n".join(rows))
    return "\n\n".join(out), tables


# --------------------------------------------------------------------------- main
SECTIONS = (("rewrite", "SFT ablation: y* data"), ("sft_perf", "SFT ablation: test avg@4 / pass@4"),
            ("sft_cost", "SFT ablation: training cost"), ("contexts", "Context statistics"),
            ("teacher", "Teacher accuracy"), ("selfcorr", "Train-set self-correction"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace tables as JSON (and markdown on stdout).")
    parser.add_argument("--tables", type=Path, default=paths.REPORT_TRACES)
    parser.add_argument("--stage", default="formal", choices=tuple(STAGE))
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    queue = Queue()
    base = dataclasses.replace(update.GENERAL_METHODS[0], group="Base")
    sft = methods_for(args.stage, (base, update.LEAF_UPDATE_TRAIN_COST, update.SFT_YSTAR))
    stages = ("formal",) if args.stage == "formal" else ("formal", "smoke_full")
    slurm = Slurm(update.slurm_roots(sft), stages=stages)
    sections, tables = {}, {}
    sections["rewrite"], tables["rewrite"] = rewrite_table(args.stage, queue)
    sections["sft_perf"], sections["sft_cost"] = update_tables(sft, slurm, queue)
    sections["contexts"], tables["contexts"] = context_table(args.stage)
    sections["teacher"], tables["teacher"] = teacher_table(args.stage, queue)
    sections["selfcorr"], tables["selfcorr"] = selfcorr_tables(args.stage, queue, args.steps)
    for key, title in SECTIONS:
        print(f"\n### {title}\n\n{sections[key]}")
    args.tables.mkdir(parents=True, exist_ok=True)
    out = args.tables / f"tables_{args.stage}.json"
    out.write_text(json.dumps(tables, indent=1, default=str))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
