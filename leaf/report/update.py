"""Policy-update tables: test avg@4 / pass@4 and training cost per method.

A method is one ``UpdateMethod`` entry: its output root, the test directory label per training
set and the training runs per training set. Layout: ``<root>/<m>/<m>/test/<label>/<test>``
(``leaf.grading.test_eval`` ``summary.json`` + ``generations.jsonl``) and the runs
``<root>/<m>/<m>/formal/<run>``; GPU hours come from the job rows of ``<root>/<m>/jobs.jsonl``.
Overall = mean avg@4 of LCB v6 / MATH-500 / SciKnowEval; MBPP+ uses the LCB-trained checkpoint.
"""

from __future__ import annotations

import dataclasses
import functools
import statistics
from pathlib import Path

from leaf import paths
from leaf.report import compute
from leaf.report.common import PARAMS, Slurm, pct, read_json, read_jsonl

TRAINS = ("lcb", "math500", "sciknoweval")
# (test key, column, training set)
TESTS = (("lcb", "LCB v6", "lcb"), ("math500", "MATH-500", "math500"),
         ("sciknoweval", "SciKnowEval", "sciknoweval"), ("mbpp", "MBPP+", "lcb"))
OVERALL = ("lcb", "math500", "sciknoweval")
H200 = "*"
SEED_STAGES = ("seed_collection", "seed_sft", "seed_rl")


@dataclasses.dataclass(frozen=True)
class UpdateMethod:
    """One row of the policy-update tables.

    tests   training set -> test directory label (``<root>/<m>/<m>/test/<label>/<test>``);
    runs    run directories under ``<root>/<m>/<m>/formal``, each ending in its training set;
            GPU h sums every run of a training set, FLOPs come from its last run;
    flops   ``leaf_update`` / ``sft`` / ``seed`` / a verl method key (``sdpo``, ``grpo``,
            ``score``) / None;
    rollout True: the runs train on LEAF update data; add the LEAF rollout that produced it
            (``<rollout root>/<m>/<m>/formal/leaf/leaf/<bm>``);
    rollout_root  root of that rollout when it is not ``root`` (SFT on y* of seed 1, whose data
            comes from the seed-1 LEAF root);
    models  the models the row is reported for (None: all);
    stage   the runs' stage directory (``formal``; ``smoke_full`` for a smoke view);
    short / group / color   figure labels and legend group.
    """
    label: str
    root: Path
    tests: dict
    runs: tuple = ()
    flops: str | None = None
    rollout: bool = False
    rollout_root: Path | None = None
    models: tuple | None = None
    stage: str = "formal"
    short: str = ""
    group: str = "Base / baselines"
    color: str = "#898781"

    def test_dir(self, model: str, test: str, train: str) -> Path:
        return self.root / model / model / "test" / self.tests[train] / test

    def run_dirs(self, model: str) -> list[Path]:
        return [self.root / model / model / self.stage / r for r in self.runs]

    def rollout_dir(self, model: str, bench: str) -> Path | None:
        """The LEAF rollout of the training set whose correct solutions the run trained on."""
        if not self.rollout:
            return None
        return (self.rollout_root or self.root) / model / model / "formal" / "leaf" / "leaf" / bench


def verl_baseline(key: str, name: str) -> UpdateMethod:
    return UpdateMethod(name, paths.GENERAL_BASELINES_SEED1, {t: f"{key}_{t}" for t in TRAINS},
                        tuple(f"baselines/{key}/{t}" for t in TRAINS), flops=key, short=name)


def seed(label: str, root: Path, short: str, models=None) -> UpdateMethod:
    return UpdateMethod(label, root, {t: f"seed_{t}" for t in TRAINS},
                        tuple(f"baselines/{k}/{t}" for k in SEED_STAGES for t in TRAINS),
                        flops="seed", models=models, short=short)


def leaf_update(label: str, root: Path, method: str, short: str, rollout: bool = True,
                **style) -> UpdateMethod:
    """LEAF update rows: ``method`` (leaf.update.leaf_update.UPDATE_METHODS; the divergence per
    training set) on every training set; test labels ``<method>_<training set>``."""
    runs = tuple(f"update/{method}/{t}" for t in ("math500", "lcb", "sciknoweval"))
    tests = {t: f"{method}_{t}" for t in TRAINS}
    return UpdateMethod(label, root, tests, runs, flops="leaf_update", rollout=rollout, short=short,
                        **{"group": "Ours", "color": "#1baf7a", **style})


LEAF_UPDATE = leaf_update("Ours", paths.GENERAL_LEAF_SEED1, "leaf_update", "Ours")
GENERAL_METHODS = (
    UpdateMethod("Base", paths.GENERAL_BASELINES_SEED1, {t: "base" for t in TRAINS}, short="Base"),
    verl_baseline("sdpo", "SDPO"),
    verl_baseline("grpo", "GRPO"),
    verl_baseline("score", "SCoRe"),
    seed("SEED (20 verifier calls)", paths.GENERAL_BASELINES_SEED1, "SEED 20 calls"),
    seed("SEED", paths.GENERAL_SEED_BASELINE_SEED1, "SEED"),
    LEAF_UPDATE,
)


# Divergence ablation of the LEAF update: reverse KL = the seed-1 LEAF update runs; forward KL /
# JSD = run/general_ablations.sh ``loss`` (MATH keeps the per-entry high clip tau = 1e-3), test
# labels leaf_update_{forward_kl,jsd}_<training set>. The data rollout is shared by all three, so
# the rows carry training cost only.
LOSS_ABLATION = (
    dataclasses.replace(GENERAL_METHODS[0], group="Base"),
    dataclasses.replace(LEAF_UPDATE, label="Ours (reverse KL)", short="reverse KL", rollout=False,
                        group="reverse KL"),
    *(leaf_update(f"Ours w/ {name}", paths.ABLATION_UPDATE, f"leaf_update_{loss}", name, rollout=False,
                  group=name, color=color)
      for loss, name, color in (("forward_kl", "forward KL", "#2a78d6"), ("jsd", "JSD", "#eda100"))),
)


# SFT on the rewritten solutions y* (test labels sft_ystar_<training set>), on the LEAF update's
# data / problems; training cost only (the rewrite job is listed separately).
SFT_YSTAR = UpdateMethod("SFT", paths.GENERAL_SFT_SEED1, {t: f"sft_ystar_{t}" for t in TRAINS},
                         tuple(f"update/sft_ystar/{t}" for t in TRAINS), flops="sft", rollout_root=paths.GENERAL_LEAF_SEED1,
                         short="SFT y*", group="SFT y*", color="#eda100")
LEAF_UPDATE_TRAIN_COST = dataclasses.replace(LEAF_UPDATE, short="Ours", rollout=False, group="Ours")
# row labels are the keys of the tables and of the figures' lookups: unique within a table
for _table in (GENERAL_METHODS, LOSS_ABLATION):
    assert len({m.label for m in _table}) == len(_table), [m.label for m in _table]


def slurm_roots(methods) -> list[Path]:
    """Output roots whose jobs.jsonl the tables need (runs + LEAF data rollouts)."""
    roots = {m.root for m in methods if m.runs or m.rollout}
    roots |= {m.rollout_root for m in methods if m.rollout and m.rollout_root is not None}
    return sorted(roots)


@functools.lru_cache(maxsize=None)
def result(directory: Path) -> dict | None:
    """{"avg", "pass", "problems", "tokens", "truncated"} of one test directory."""
    s = read_json(directory / "summary.json")
    if not s:
        return None
    found = {"avg": s["avg_at_k"], "pass": s["pass_at_k"], "problems": s.get("problems")}
    rows = read_jsonl(directory / "generations.jsonl")
    if rows:
        found["tokens"] = statistics.mean(r["completion_tokens"] for r in rows)
        found["truncated"] = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
    return found


def rows(model: str, methods=GENERAL_METHODS) -> list[tuple[UpdateMethod, dict]]:
    """[(method, {test: result})] of the methods reported for ``model``."""
    return [(m, {t: result(m.test_dir(model, t, train)) for t, _, train in TESTS})
            for m in methods if m.models is None or model in m.models]


def overall(results: dict) -> float | None:
    values = [results[t]["avg"] if results.get(t) else None for t in OVERALL]
    return None if None in values else sum(values) / len(values)


def h200_by_train(slurm: Slurm, runs: list[Path]) -> dict:
    """training set -> True if every training run of it ran on H200 only (None: no runs)."""
    out = {}
    for b in TRAINS:
        types = set().union(*[slurm.gpus(r) for r in runs if r.name == b]) if any(r.name == b for r in runs) else set()
        out[b] = (types == {"h200"}) if types else None
    return out


def starred(label: str, flags: dict) -> str:
    return label + H200 if flags and all(flags.values()) else label


def perf_table(model: str, slurm: Slurm, methods=GENERAL_METHODS, running=frozenset()) -> str:
    """LCB v6 / MATH-500 / SciKnowEval / Overall / MBPP+ (``*``: trained on H200 only).
    ``running``: (label, test) pairs whose missing cell prints "running" (Overall too while any of
    its three tests does)."""
    heads = [c for _, c, _ in TESTS[:3]] + ["Overall", "MBPP+"]
    out = ["| Method | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for m, results in rows(model, methods):
        flags = h200_by_train(slurm, m.run_dirs(model))
        cells = [f"{pct(results[t]['avg'])} / {pct(results[t]['pass'])}" + (H200 if flags.get(train) else "")
                 if results[t] else ("running" if (m.label, t) in running else "—") for t, _, train in TESTS]
        total = overall(results)
        total = pct(total) if total is not None else (
            "running" if any(not results[t] and (m.label, t) in running for t in OVERALL) else "—")
        out.append(f"| {starred(m.label, flags)} | " + " | ".join(cells[:3] + [total, cells[3]]) + " |")
    return "\n".join(out)


def run_flops(model: str, method: UpdateMethod, run: Path) -> float | None:
    """Training FLOPs of one run (None: missing / unfinished data)."""
    try:
        if method.flops == "leaf_update":
            return compute.leaf_update_run_flops(run, model)
        if method.flops == "sft":
            return compute.sft_run_flops(run, model)
        if method.flops == "seed":
            return compute.seed_flops(run.parent.parent, model, run.name)
        return compute.verl_run_flops(run, model, method.flops)
    except (OSError, KeyError, ValueError):
        return None


def cost_rows(model: str, slurm: Slurm, methods=GENERAL_METHODS) -> list[tuple[UpdateMethod, dict]]:
    n = PARAMS[model]
    out = []
    for m, _ in rows(model, methods):
        runs = m.run_dirs(model)
        if not runs:
            continue
        cell = {"train_h": [], "roll_h": [], "train_f": [], "roll_f": [], "h200": h200_by_train(slurm, runs)}
        for bench in TRAINS:
            bench_runs = [r for r in runs if r.name == bench]
            hours = [slurm.hours(r) for r in bench_runs]  # a method without this training set: "—"
            cell["train_h"].append(sum(hours) if bench_runs and None not in hours else None)
            cell["train_f"].append(run_flops(model, m, bench_runs[-1]) if bench_runs else None)
            roll = m.rollout_dir(model, bench)
            cell["roll_h"].append(slurm.hours(roll) if roll else 0.0)
            cell["roll_f"].append(compute.rollout_flops(roll, n) if roll else 0.0)
        out.append((m, cell))
    return out


def cost_table(model: str, slurm: Slurm, methods=GENERAL_METHODS) -> str:
    """Training / data-rollout / total GPU h and FLOPs, summed over the three training sets."""
    def total(values):
        return None if None in values else sum(values)

    def h(v):
        return "—" if v is None else f"{v:.2f}"

    def f(v):
        return "—" if v is None else f"{v:.2e}"

    def plus(a, b):
        return None if a is None or b is None else a + b

    out = ["| Method | Training GPU h | Data rollout GPU h | Total GPU h | Training FLOPs | Data rollout FLOPs | Total FLOPs |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m, c in cost_rows(model, slurm, methods):
        flags = c["h200"]
        th, rh, tf, rf = (total(c[k]) for k in ("train_h", "roll_h", "train_f", "roll_f"))
        per = " / ".join(h(v) + (H200 if flags.get(b) else "") for v, b in zip(c["train_h"], TRAINS))
        out.append(f"| {starred(m.label, flags)} | {h(th)} ({per}) | {h(rh)} | {h(plus(th, rh))} | "
                   f"{f(tf)} | {f(rf)} | {f(plus(tf, rf))} |")
    return "\n".join(out)


def train_cost_table(model: str, slurm: Slurm, methods, running=frozenset()) -> str:
    """Training GPU h and FLOPs only (per training set and summed), for rows sharing their data.
    ``running``: (label, training set) pairs whose missing values print "running"."""
    def h(v, run=False):
        return ("running" if run else "—") if v is None else f"{v:.2f}" + (" (running)" if run else "")

    def f(v, run=False):
        return ("running" if run else "—") if v is None else f"{v:.2e}" + (" (running)" if run else "")

    def total(values):
        return None if None in values else sum(values)

    out = ["| Method | Training GPU h (total) | LCB / MATH / Sci GPU h | Training FLOPs (total) | LCB / MATH / Sci FLOPs |",
           "|---|---:|---:|---:|---:|"]
    for m, c in cost_rows(model, slurm, methods):
        flags = c["h200"]
        run = {b: (m.label, b) in running for b in TRAINS}
        per_h = " / ".join(h(v, run[b]) + (H200 if flags.get(b) else "") for v, b in zip(c["train_h"], TRAINS))
        per_f = " / ".join(f(v, run[b]) for v, b in zip(c["train_f"], TRAINS))
        out.append(f"| {starred(m.label, flags)} | {h(total(c['train_h']), any(run.values()))} | {per_h} | "
                   f"{f(total(c['train_f']), any(run.values()))} | {per_f} |")
    return "\n".join(out)
