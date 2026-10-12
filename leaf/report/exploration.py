"""Exploration curves and tables: per aligned step k = 1..24, cost and accuracy of every cell.

A method is one ``Method`` entry (output root, kind, run directory name); a report lists the
methods it compares (``EXPLORE_METHODS``, ``COMPRESSION_ABLATIONS``), so adding a condition is adding an
entry. For each (model, method, benchmark) cell ``<root>/<m>/<m>/<stage>/{leaf,baselines}/<name>/<b>``
and pooled over the four benchmarks, per aligned step k:

* ``hours``   cumulative GPU wall time (measured), start-up included;
* ``flops``   cumulative FLOPs, 2N per forward token + 6N per trained token;
* ``tokens``  cumulative generated tokens (prompt + completion of every generation call,
              judge calls included);
* ``accuracy`` (step k's mean), ``pass_rate`` (problems solved within the first k answers),
  and on LCB / MBPP+ also ``accuracy_private`` / ``pass_rate_private`` (public AND private).

Alignment
---------
Step k holds every problem's k-th answer. LEAF / Random Sampling answer each problem once per step.
The batch methods (SDPO, GRPO, SCoRe, SEED) answer a problem 8 times per question occurrence,
so a problem's k-th answer is its k-th call: each call carries an equal share of its update's
measured seconds, FLOPs and tokens, and step k's cost is the sum over calls with per-problem
index <= k. Accuracy / pass rate are the graders' aligned-step summaries (``rollout_summary*.json``),
falling back to the raw public verdicts (LEAF ``public_correct``, verl ``acc``) when a cell is
not graded.

Wall time per step
------------------
* verl: start-up (newest job's wall - its summed ``timing_s/step``) + per-call shares of
  ``timing_s/step``; SEED adds the collection and SFT jobs' wall times, collection spread over
  its calls, SFT over the RL calls.
* LEAF (async, one task per problem): the measured rollout wall time (first sample start to
  last ``ended_at``) split over the steps in proportion to each step's summed
  ``sample_step_seconds``, plus the job's start-up before the rollout. Records without
  ``ended_at`` (interleaved driver, compression-ablation cells): the rollout time is the longest
  per-problem running sum of ``sample_step_seconds`` and the start-up is the summed wall time of
  the jobs that did work (chunks >= 2 min; a compression-ablation cell runs as two chunks) minus it.
  Jobs come from ``launches/``, else from the cell-stage rows of ``jobs.jsonl`` (LEAF rollout
  cells; grading jobs excluded).
* LEAF with interleaved updates: ``step_timings.jsonl`` end-of-step times minus the job start.
* Random Sampling: ``chunk_timings.jsonl`` (chunk wall split over its aligned steps by tokens),
  plus start-up.

Truncation: steps up to min(24, the last step whose cumulative wall time <= 5 h). The pooled
row needs all four benchmarks and stops at the shortest; its accuracy / pass rate are the
macro average (unweighted mean of the four benchmarks) and its hours, FLOPs and tokens are
sums.

    python -m leaf.report.exploration --set explore|compression|ablation [--json OUT.json] [--stage formal]

(``run/report.sh`` writes the ``explore`` set (seed 1, with the interleaved-update runs) to
``paths.REPORT_EXPLORE/curves_seed1_with_interleaved.json``.)
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

from leaf import paths
from leaf.report.common import (MODELS, PARAMS, START_LINE, config, flops, job_ids,
                                   job_logs, pct, read_json, read_jsonl)

STEPS = 24
CAP_HOURS = 5.0
BENCHMARKS = (("math500", "MATH-500"), ("lcb", "LCB"), ("mbpp", "MBPP+"), ("sciknoweval", "SciKnowEval"))
BENCH_KEYS = tuple(b for b, _ in BENCHMARKS)
PRIVATE = {"lcb": "rollout_summary_lcbprivate.json", "mbpp": "rollout_summary_mbppplus.json"}
KINDS = ("leaf", "random_sampling", "verl", "seed")


@dataclasses.dataclass(frozen=True)
class Method:
    """One exploration method / condition.

    key    id in the curves JSON and the tables;
    kind   ``leaf`` (LEAF rollout, with or without interleaved updates), ``random_sampling``,
           ``verl`` (SDPO / SCoRe / enhanced GRPO), ``seed`` (collection + SFT + RL);
    root   output root holding ``<model>/`` (jobs.jsonl, logs/, dataset_*.jsonl, runs);
    name   run directory name (default ``key``; SEED reads seed_collection / seed_sft / seed_rl);
    algo   verl algorithm of the FLOPs estimate (``sdpo`` adds the teacher forward; default
           ``key``, SEED ``seed_rl``), independent of the directory name;
    config the model's run config under configs/ (``{model}`` = the model key), used by verl /
           SEED for the rl_data parquets that map rollout prompts to problem uids;
    data   root holding ``<model>/dataset_<b>.jsonl`` when it is not ``root`` (the exploration
           ablations read the seed-1 exploration datasets); a smoke stage reads
           ``<root>/<model>/smoke_dataset_<b>.jsonl``;
    color / marker / style  figure styling.
    """
    key: str
    label: str
    kind: str
    root: Path
    name: str = ""
    algo: str = ""
    config: str = "explore/{model}_seed1.json"
    data: Path | None = None
    color: str = "#898781"
    marker: str = "o"
    style: str = "-"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")

    def run_root(self, model: str) -> Path:
        return self.root / model

    def dataset(self, model: str, benchmark: str, stage: str = "formal") -> Path:
        """The cell's problem list (the smoke subset when the stage has one)."""
        if stage == "smoke":
            smoke = self.run_root(model) / f"smoke_dataset_{benchmark}.jsonl"
            if smoke.is_file():
                return smoke
        return (self.data or self.root) / model / f"dataset_{benchmark}.jsonl"

    def cell(self, model: str, benchmark: str, stage: str = "formal") -> Path:
        sub = "leaf" if self.kind == "leaf" else "baselines"
        name = "seed_rl" if self.kind == "seed" else (self.name or self.key)
        return self.root / model / model / stage / sub / name / benchmark

    @property
    def flops_algo(self) -> str:
        return self.algo or ("seed_rl" if self.kind == "seed" else self.key)


LEAF = "leaf"
INTERLEAVED = "interleaved_every_1"

EXPLORE_METHODS = (
    Method("random_sampling", "Random Sampling", "random_sampling", paths.EXPLORE_SEED1, color="#2a78d6", marker="s",
           style="--"),
    Method("sdpo", "SDPO", "verl", paths.EXPLORE_SEED1, color="#eb6834", marker="s", style="--"),
    Method("score", "SCoRe", "verl", paths.EXPLORE_SEED1, color="#1baf7a", marker="s", style="--"),
    Method("seed", "SEED", "seed", paths.EXPLORE_SEED1, color="#eda100", marker="s", style="--"),
    Method("grpo", "GRPO", "verl", paths.EXPLORE_SEED1, color="#e87ba4", marker="s", style="--"),
    Method(LEAF, "Ours", "leaf", paths.EXPLORE_SEED1, color="#4a3aa7"),
    Method(INTERLEAVED, "Ours + update every step", "leaf", paths.EXPLORE_SEED1, color="#d4241c"),
)

# History-compression ablation (cells run with other hardware / settings than the H200-tuned
# exploration runs, so their wall time is not comparable). Qwen: compress_none / tail / head_tail
# on all four benchmarks; Olmo: tail / head_tail on four, llm_summary / summary_block /
# lowest_logprob on LCB + SciKnowEval.
COMPRESSION_ABLATIONS = tuple(
    Method(key, label, "leaf", paths.ABLATION_COMPRESSION)
    for key, label in (("compress_none", "No compression"), ("compress_llm_summary", "LLM summary"),
                       ("compress_tail", "Tail"), ("compress_head_tail", "Head and tail"),
                       ("compress_summary_block", "Summary block"),
                       ("compress_lowest_logprob", "Lowest log-prob")))

# Exploration ablations of LEAF and of the interleaved update interval (data and tuned engine
# settings of the seed-1 exploration, cells under paths.ABLATION_EXPLORE). The LEAF and
# update-every-step rows are the seed-1 exploration runs.
COMPONENT_ABLATIONS = (
    next(m for m in EXPLORE_METHODS if m.key == LEAF),
    *(Method(key, label, "leaf", paths.ABLATION_EXPLORE, data=paths.EXPLORE_SEED1, color=color, marker=marker,
             style="--")
      for key, label, color, marker in (("wo_last_response", "w/o last response", "#2a78d6", "s"),
                                        ("wo_feedback", "w/o feedback", "#eb6834", "^"),
                                        ("restart_every_2", "Restart every 2", "#1baf7a", "v"),
                                        ("restart_every_10", "Restart every 10", "#eda100", "D"),
                                        ("no_restart", "No restart", "#e87ba4", "P"),
                                        ("last_3_rounds", "Last 3 rounds", "#898781", "X"))),
)
INTERLEAVED_ABLATIONS = (
    next(m for m in EXPLORE_METHODS if m.key == INTERLEAVED),
    *(Method(f"interleaved_every_{n}", f"Ours + update every {n} steps", "leaf", paths.ABLATION_EXPLORE,
             data=paths.EXPLORE_SEED1, color=color, marker=marker, style=":")
      for n, color, marker in ((2, "#0f8b8d", "s"), (5, "#8c564b", "D"))),
)
EXPLORE_ABLATIONS = COMPONENT_ABLATIONS + INTERLEAVED_ABLATIONS

METHOD_SETS = {"explore": EXPLORE_METHODS, "compression": COMPRESSION_ABLATIONS, "ablation": EXPLORE_ABLATIONS}
# keys and labels identify the rows of the curves JSON and the tables: unique within a set
for _methods in METHOD_SETS.values():
    assert len({m.key for m in _methods}) == len({m.label for m in _methods}) == len(_methods)


# --------------------------------------------------------------------------- job timing
# a rollout chunk shorter than this did no work (a resume chunk that found the run finished);
# the GPU-hour accounting drops it as well
SPARE_CHUNK_SECONDS = 120


def job_windows(cell: Path, run_root: Path) -> list[tuple[float, float]]:
    """(start, end) epoch seconds of every job that ran this cell, oldest first (job log start
    line, newest log mtime); jobs without a start line are skipped."""
    out = []
    for job in job_ids(cell, run_root):
        start = end = None
        for path in job_logs(job, run_root):
            match = START_LINE.search(path.read_text(errors="ignore"))
            if match and start is None:
                start = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            end = max(end or 0.0, path.stat().st_mtime)
        if start:
            out.append((start, end))
    return out


def job_window(cell: Path, run_root: Path) -> tuple[float | None, float | None]:
    """(start, end) of the newest job that ran this cell."""
    windows = job_windows(cell, run_root)
    return windows[-1] if windows else (None, None)


def job_hours(cell: Path, run_root: Path) -> float | None:
    start, end = job_window(cell, run_root)
    return (end - start) / 3600 if start and end else None


def work_hours(cell: Path, run_root: Path) -> float | None:
    """Summed wall time of the cell's jobs that did work (chunks >= SPARE_CHUNK_SECONDS)."""
    walls = [end - start for start, end in job_windows(cell, run_root) if end - start >= SPARE_CHUNK_SECONDS]
    return sum(walls) / 3600 if walls else None


# --------------------------------------------------------------------------- accuracy
def graded_steps(cell: Path) -> dict:
    """{public: [(acc, pass)], private: [...]} from the graders' summaries."""
    out = {}
    summary = read_json(cell / "rollout_summary.json")
    if summary and summary.get("steps"):
        out["public"] = [(r["accuracy"], r["pass_rate_cumulative"]) for r in summary["steps"]]
    for name in PRIVATE.values():
        private = read_json(cell / name)
        if private and private.get("steps"):
            out["private"] = [(r["accuracy"], r["pass_rate_cumulative"]) for r in private["steps"]]
    return out


def attempt_accuracy(verdicts: dict[str, list[bool]], problems: int) -> list[tuple[float, float]]:
    """Raw public verdicts in per-problem answer order -> aligned (acc, pass) while every
    problem has a k-th answer."""
    if len(verdicts) < problems:
        return []
    out, solved = [], set()
    for k in range(STEPS):
        if any(len(v) <= k for v in verdicts.values()):
            break
        correct = [uid for uid, v in verdicts.items() if v[k]]
        solved.update(correct)
        out.append((len(correct) / problems, len(solved) / problems))
    return out


# --------------------------------------------------------------------------- per-kind costs
def leaf_cell(cell: Path, run_root: Path, problems: list[dict], n: float) -> dict:
    """Per-step increments (seconds, flops, tokens) + raw verdicts of a LEAF rollout."""
    from leaf.explore.leaf_rollout import read_sample_records, sample_record_path

    ends_abs = collections.defaultdict(float)
    ends_rel = collections.defaultdict(float)
    step_seconds = collections.defaultdict(float)  # summed sample_step_seconds of step k
    starts = []
    toks = collections.defaultdict(float)
    verdicts: dict[str, list[bool]] = {}
    for index, problem in enumerate(problems):
        seen = {int(r["step"]): r for r in sorted(read_sample_records(sample_record_path(cell, index, problem["uid"])),
                                                   key=lambda r: int(r["step"]))}
        if not seen:
            continue
        contiguous = []
        while len(contiguous) + 1 in seen:
            contiguous.append(bool(seen[len(contiguous) + 1]["final"].get("public_correct")))
        verdicts[problem["uid"]] = contiguous
        running = 0.0
        for k in sorted(seen):
            r = seen[k]
            running += float(r.get("sample_step_seconds") or 0.0)
            step_seconds[k] += float(r.get("sample_step_seconds") or 0.0)
            ends_rel[k] = max(ends_rel[k], running)
            if r.get("ended_at"):
                ends_abs[k] = max(ends_abs[k], float(r["ended_at"]))
                if k == 1:
                    starts.append(float(r["ended_at"]) - float(r.get("sample_step_seconds") or 0.0))
            gen = sum(int(m.get("prompt_tokens") or 0) + int(m.get("completion_tokens") or 0)
                      for m in r.get("intermediate") or [])
            judge = r["final"].get("judge") if isinstance(r["final"].get("judge"), dict) else None
            toks[k] += gen + ((int(judge.get("prompt_tokens") or 0) + int(judge.get("completion_tokens") or 0))
                              if judge else 0)
    steps = sorted(k for k in ends_rel if k <= STEPS)
    if ends_abs and starts and len(ends_abs) == len(ends_rel):
        job_start, _ = job_window(cell, run_root)
        rollout_start = min(starts)
        offset = max(0.0, rollout_start - job_start) if job_start else 0.0
        rollout = max(ends_abs[k] for k in steps) - rollout_start if steps else 0.0
    else:  # no ended_at (interleaved driver, compression-ablation cells): every rollout chunk that did work
        total = work_hours(cell, run_root)
        rollout = ends_rel[steps[-1]] if steps else 0.0
        offset = max(0.0, total * 3600 - rollout) if total else 0.0
    # the measured rollout wall time is split over the steps in proportion to each step's summed
    # per-problem seconds, so step k's cumulative time = start-up + the rollout time attributed to
    # every problem's first k answers (the share definition of the batch methods)
    share_total = sum(step_seconds[k] for k in steps) or 1.0
    end_of, acc = {}, 0.0
    for k in steps:
        acc += step_seconds[k]
        end_of[k] = offset + rollout * acc / share_total
    extra = interleaved_extras(cell, run_root) if (cell / "step_timings.jsonl").is_file() else None
    seconds, f, t, prev = [], [], [], 0.0
    for k in steps:
        cum = extra["end"].get(k, end_of[k]) if extra else end_of[k]
        seconds.append(cum - prev)
        prev = cum
        up_fwd, up_train, up_tok = extra["update"].get(k, (0.0, 0.0, 0.0)) if extra else (0.0, 0.0, 0.0)
        f.append(flops(toks[k] + up_fwd, up_train, n))
        t.append(toks[k] + up_tok)
    return {"seconds": seconds, "flops": f, "tokens": t, "raw": attempt_accuracy(verdicts, len(problems))}


def interleaved_extras(cell: Path, run_root: Path) -> dict:
    """End-of-step wall times (``time`` stamps) and the update's tokens per step."""
    job_start, _ = job_window(cell, run_root)
    rows = {int(r["step"]): r for r in read_jsonl(cell / "step_timings.jsonl")}  # last row of a step wins
    end, update = {}, {}
    for k, r in rows.items():
        stamp = r.get("time")
        if stamp and job_start:
            end[k] = dt.datetime.strptime(" ".join(stamp.split()[:2]), "%Y-%m-%d %H:%M:%S").timestamp() - job_start
        b = r.get("batch") or {}
        if r.get("update_skipped") or not b.get("student_tokens"):
            update[k] = (0.0, 0.0, 0.0)
            continue
        gen = float(b.get("generation_prompt_tokens", 0)) + float(b.get("generation_completion_tokens", 0))
        # student sampling + behaviour log-prob (x+y) + teacher forward (c+y); LoRA update on x+y
        update[k] = (gen + float(b["student_tokens"]) + float(b.get("teacher_tokens", 0)),
                     float(b["student_tokens"]), gen)
    return {"end": end, "update": update}


def random_sampling_cell(cell: Path, run_root: Path, problem_rows: list[dict], n: float) -> dict:
    rows = read_jsonl(cell / "generations.jsonl")
    problems = len(problem_rows)
    toks = collections.defaultdict(float)
    for r in rows:
        judge = r.get("judge") if isinstance(r.get("judge"), dict) else None
        toks[int(r["aligned_step"]) + 1] += int(r["prompt_tokens"]) + int(r["completion_tokens"]) + (
            int(judge.get("prompt_tokens") or 0) + int(judge.get("completion_tokens") or 0) if judge else 0)
    seconds = collections.defaultdict(float)
    chunks = read_jsonl(cell / "chunk_timings.jsonl")
    if not chunks:
        raise FileNotFoundError(cell / "chunk_timings.jsonl")
    job_start, _ = job_window(cell, run_root)
    first = min(c["started_at"] for c in chunks)
    startup = max(0.0, first - job_start) if job_start else 0.0
    for c in chunks:
        wall = c["ended_at"] - c["started_at"]
        weights = {int(s) + 1: v["prompt_tokens"] + v["completion_tokens"] for s, v in c["per_step"].items()}
        total = sum(weights.values()) or 1
        for s, w in weights.items():
            seconds[s] += wall * w / total
    seconds[min(seconds)] += startup
    grades = {g["key"]: bool(g["correct"]) for g in read_jsonl(cell / "grades_public.jsonl")}
    verdicts: dict[str, list] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: int(r["aligned_step"])) if grades else ():
        verdicts[r["problem_uid"]].append(grades.get(r["key"], False))
    per_step = collections.Counter(int(r["aligned_step"]) + 1 for r in rows)
    steps = sorted(s for s in toks if s <= STEPS and per_step[s] == problems)
    return {"seconds": [seconds[s] for s in steps], "flops": [flops(toks[s], 0.0, n) for s in steps],
            "tokens": [toks[s] for s in steps], "raw": attempt_accuracy(verdicts, problems) if grades else []}


def verl_calls(cell: Path, run_config: dict, benchmark: str, method: str, model: str):
    """[(uid, ok, update)] in rollout order, and per-update (seconds, flops, tokens)."""
    import leaf.grading.grade_baseline as gb
    from leaf.core.project_config import benchmark_rl_data_dir
    from leaf.report.compute import verl_components

    parquet = benchmark_rl_data_dir(run_config, benchmark, model) / gb.MAPPING_PARQUET["formal"]
    resolver = gb.PromptResolver(gb.prompt_uid_map(parquet), gb.ground_truth_uid_map(parquet))
    metrics = {int(r["step"]): r["data"] for r in read_jsonl(cell / "metrics.jsonl")}
    calls, per_update = [], {}
    for path in sorted((cell / "rollouts").glob("*.jsonl"), key=lambda p: int(p.stem)):
        u = int(path.stem)
        for r in read_jsonl(path):
            uid = resolver.resolve_row(r)
            if "score_r1" in r:
                calls += [(uid, float(r["score_r1"]) > 0.5, u), (uid, float(r["score_r2"]) > 0.5, u)]
            else:
                calls.append((uid, float(r["acc"]) > 0.5, u))
        d = metrics.get(u)
        if d:
            fwd, train = verl_components(d, model, method)
            judge = float(d.get("judge/requests") or 0) * float(d.get("judge/completion_tokens_mean") or 0)
            per_update[u] = (float(d["timing_s/step"]), flops(fwd, train, PARAMS[model]),
                             float(d["perf/total_num_tokens"]) + judge)
    return calls, per_update


def batch_cell(cell: Path, run_root: Path, benchmark: str, method: Method, model: str,
               problem_rows: list[dict]) -> dict:
    """verl methods and SEED (collection + SFT + RL), attempt-aligned per-call shares."""
    from leaf.report.compute import SFT_EPOCHS, sft_tokens

    n = PARAMS[model]
    problems = len(problem_rows)
    attempts: dict[str, int] = collections.defaultdict(int)
    step_cost = [[0.0, 0.0, 0.0] for _ in range(STEPS + 1)]
    verdicts: dict[str, list[bool]] = collections.defaultdict(list)

    def call(uid, ok, share):
        attempts[uid] += 1
        verdicts[uid].append(ok)
        if attempts[uid] <= STEPS:
            for i in range(3):
                step_cost[attempts[uid]][i] += share[i]

    lump_seconds = lump_flops = 0.0
    if method.kind == "seed":
        collection = cell.parent.parent / "seed_collection" / cell.name
        sft = cell.parent.parent / "seed_sft" / cell.name
        sols = sorted(read_jsonl(collection / "solutions.jsonl"), key=lambda r: int(r["trial_index"]))
        grades = {g["key"]: bool(g["correct"]) for g in (read_json(collection / "grades_public.json") or [])}
        gen = sum(int(r["prompt_tokens"]) + int(r["completion_tokens"]) for r in sols)
        gen += sum(int(a["prompt_tokens"]) + int(a["completion_tokens"]) for a in read_jsonl(collection / "analyses.jsonl"))
        collect_s = (job_hours(collection, run_root) or 0.0) * 3600
        for r in sols:
            call(r["problem_uid"], grades.get(r["key"], False),
                 (collect_s / len(sols), flops(gen, 0.0, n) / len(sols), gen / len(sols)))
        lump_seconds = (job_hours(sft, run_root) or 0.0) * 3600
        try:
            lump_flops = flops(0.0, sft_tokens(model, collection) * SFT_EPOCHS, n)
        except Exception:  # noqa: BLE001 -- tokenizer / parquet missing on a partial chain
            lump_flops = 0.0
    run_config = config(method.config.format(model=model))
    calls, per_update = verl_calls(cell, run_config, benchmark, method.flops_algo, model)
    hours = job_hours(cell, run_root)
    startup = max(0.0, hours * 3600 - sum(v[0] for v in per_update.values())) if hours else 0.0
    by_update = collections.Counter(u for _, _, u in calls)
    for uid, ok, u in calls:
        s, f, t = per_update.get(u, (0.0, 0.0, 0.0))
        m = by_update[u]
        call(uid, ok, (s / m + (lump_seconds + startup) / len(calls), f / m + lump_flops / len(calls), t / m))
    steps = [k for k in range(1, STEPS + 1) if sum(1 for a in attempts.values() if a >= k) == problems]
    return {"seconds": [step_cost[k][0] for k in steps], "flops": [step_cost[k][1] for k in steps],
            "tokens": [step_cost[k][2] for k in steps], "raw": attempt_accuracy(dict(verdicts), problems)}


# --------------------------------------------------------------------------- assembly
def cumulative(values):
    out, total = [], 0.0
    for v in values:
        total += v
        out.append(total)
    return out


def cell_curve(method: Method, model: str, benchmark: str, stage: str = "formal") -> dict | None:
    cell = method.cell(model, benchmark, stage)
    if not cell.is_dir():
        return None
    run_root, n = method.run_root(model), PARAMS[model]
    problems = read_jsonl(method.dataset(model, benchmark, stage))
    if not problems:
        raise FileNotFoundError(method.dataset(model, benchmark, stage))
    if method.kind == "leaf":
        raw = leaf_cell(cell, run_root, problems, n)
    elif method.kind == "random_sampling":
        raw = random_sampling_cell(cell, run_root, problems, n)
    else:
        raw = batch_cell(cell, run_root, benchmark, method, model, problems)
    graded = graded_steps(cell)
    public = graded.get("public") or raw["raw"]
    steps = min(len(raw["seconds"]), len(public))
    hours = [s / 3600 for s in cumulative(raw["seconds"][:steps])]
    keep = sum(1 for h in hours if h <= CAP_HOURS + 1e-9)
    out = {"model": model, "method": method.key, "benchmark": benchmark, "cell": str(cell),
           "accuracy_source": "graded" if graded.get("public") else "raw public verdicts",
           "problems": len(problems),
           "steps": list(range(1, keep + 1)), "hours": hours[:keep],
           "flops": cumulative(raw["flops"][:keep]), "tokens": cumulative(raw["tokens"][:keep]),
           "accuracy": [a for a, _ in public[:keep]], "pass_rate": [p for _, p in public[:keep]],
           "steps_measured": steps}
    if graded.get("private"):
        private = graded["private"][:keep]
        out["accuracy_private"] = [a for a, _ in private]
        out["pass_rate_private"] = [p for _, p in private]
    return out


def pooled(cells: list[dict]) -> dict | None:
    if len(cells) != len(BENCHMARKS) or any(not c["steps"] for c in cells):
        return None
    steps = min(len(c["steps"]) for c in cells)
    out = {"steps": list(range(1, steps + 1)), "problems": sum(c["problems"] for c in cells)}
    for key in ("hours", "flops", "tokens"):
        out[key] = [sum(c[key][i] for c in cells) for i in range(steps)]
    # accuracy / pass rate: macro average, every benchmark weighted equally
    for key in ("accuracy", "pass_rate"):
        out[key] = [sum(c[key][i] for c in cells) / len(cells) for i in range(steps)]
    # LCB / MBPP+ on public AND private, the other two benchmarks on their (only) verifier
    if all(("pass_rate_private" in c) or c["benchmark"] not in PRIVATE for c in cells):
        steps_p = min(steps, *(len(c.get("pass_rate_private", c["pass_rate"])) for c in cells))
        for key, src in (("accuracy_private", "accuracy"), ("pass_rate_private", "pass_rate")):
            out[key] = [sum(c.get(key, c[src])[i] for c in cells) / len(cells) for i in range(steps_p)]
    return out


def curves(methods, stage: str = "formal") -> tuple[dict, list]:
    """({model: {method key: {"cells": [...], "pooled": ...}}}, [error rows]). A model appears
    when one of the methods' roots has ``<root>/<m>/<m>/<stage>``; a cell that raises is
    reported in the error rows (and on stderr) and left out."""
    result, errors = {}, []
    for model, _ in MODELS:
        if not any((m.root / model / model / stage).is_dir() for m in methods):
            continue
        block = result.setdefault(model, {})
        for method in methods:
            cells = []
            for benchmark in BENCH_KEYS:
                try:
                    curve = cell_curve(method, model, benchmark, stage)
                except Exception as error:  # noqa: BLE001 -- report and keep going
                    errors.append({"cell": [model, method.key, benchmark], "error": repr(error)})
                    print(json.dumps(errors[-1]), file=sys.stderr, flush=True)
                    curve = None
                if curve:
                    cells.append(curve)
            if cells:
                block[method.key] = {"cells": cells, "pooled": pooled(cells)}
    return result, errors


# --------------------------------------------------------------------------- tables
def _last(c: dict | None) -> dict | None:
    """Per-step lists -> values at the last counted step, plus ``nsteps``."""
    if not c:
        return None
    out = dict(c)
    for k in ("hours", "flops", "tokens"):
        out[k] = c[k][-1] if c[k] else None
    out["nsteps"] = len(c["steps"])
    return out


def _acc_pass(x: dict | None) -> tuple[float, float] | None:
    """(mean accuracy, last cumulative pass rate), public AND private where graded."""
    if not x or not x["nsteps"]:
        return None
    a = x.get("accuracy_private") or x["accuracy"]
    p = x.get("pass_rate_private") or x["pass_rate"]
    return sum(a) / len(a), p[-1]


def _cells(data: dict, model: str, key: str) -> tuple[dict, dict | None]:
    entry = data.get(model, {}).get(key) or {}
    return {c["benchmark"]: _last(c) for c in entry.get("cells") or []}, _last(entry.get("pooled"))


def _is_running(running, key: str, bench: str | None, cells: dict | None = None) -> bool:
    """``running`` holds method keys (every cell of the method) and / or (key, benchmark) pairs;
    ``bench`` None = the pooled cell: running while a benchmark of the method is and every
    benchmark either has a cell (``cells``) or is running (otherwise it can never be filled)."""
    if key in running:
        return True
    if bench is None:
        return (any(_is_running(running, key, b) for b in BENCH_KEYS)
                and all(b in (cells or {}) or _is_running(running, key, b) for b in BENCH_KEYS))
    return (key, bench) in running


def _pending(c: dict, bench: str) -> bool:
    """Accuracy not from the graders yet (raw rollout verdicts, or LCB / MBPP+ without the
    private / plus-test regrade)."""
    return c.get("accuracy_source", "graded") != "graded" or (bench in PRIVATE and "pass_rate_private" not in c)


def curve_complete(data: dict, model: str, key: str, running=()) -> bool:
    """A method's pooled curve is drawable: all four benchmark cells exist, every one is graded
    (LCB / MBPP+ including the private / plus-test regrade, so the pooled curve is public AND
    private) and no rollout / grading job of the method is still queued for the model."""
    entry = data.get(model, {}).get(key) or {}
    cells = {c["benchmark"]: c for c in entry.get("cells") or []}
    pooled = entry.get("pooled")
    return (set(cells) == set(BENCH_KEYS) and bool(pooled) and bool(pooled.get("pass_rate_private"))
            and not any(_pending(c, b) for b, c in cells.items())
            and not any(_is_running(running, key, b) for b in BENCH_KEYS))


def perf_table(data: dict, model: str, methods, running=(), steps: int = STEPS, strict: bool = False,
               annotate: bool = False) -> str:
    """Rows: per benchmark and pooled, "mean accuracy / last cumulative pass rate" (`` (N steps)``
    when the 5 GPU h cap stopped the cell before ``steps``). ``running`` (method keys or
    (key, benchmark) pairs, see ``_is_running``): a missing cell prints "running"; with ``annotate``
    a cell with data prints `` (N steps, running)``. ``strict``: a cell whose grading is not finished (``_pending``) prints
    "grading", and so does Overall while any of its benchmarks does."""
    out = ["| Method | " + " | ".join(b for _, b in BENCHMARKS) + " | Overall (mean of 4 benchmarks) |",
           "|---|" + "---:|" * (len(BENCHMARKS) + 1)]
    for method in methods:
        cells, pool = _cells(data, model, method.key)
        row = []
        for bench, c in [(b, cells.get(b)) for b in BENCH_KEYS] + [(None, pool)]:
            run = _is_running(running, method.key, bench, cells)
            ap = _acc_pass(c)
            if not ap:
                row.append("running" if run else "—")
                continue
            if strict and (any(_pending(x, b) for b, x in cells.items()) if bench is None else _pending(c, bench)):
                row.append("grading")
                continue
            if run and annotate:
                mark = f" ({c['nsteps']} steps, running)"
            else:
                mark = "" if c["nsteps"] >= steps else f" ({c['nsteps']} steps)"
            row.append(f"{pct(ap[0])} / {pct(ap[1])}{mark}")
        out.append(f"| {method.label} | " + " | ".join(row) + " |")
    return "\n".join(out)


def cost_table(data: dict, model: str, methods, running=(), mark: str = "*", annotate: bool = False) -> str:
    """Rows: per benchmark "GPU h / PFLOPs / M tokens" at the cell's last counted step; Total =
    the sum of the benchmark cells (each at its own last step); ``running`` as in ``perf_table``
    (with ``annotate`` a cell with data gets `` (running)``); ``mark`` follows the label of a row with cost data."""
    out = ["| Method | " + " | ".join(b for _, b in BENCHMARKS) + " | Total |",
           "|---|" + "---:|" * (len(BENCHMARKS) + 1)]

    def fmt(hours, fl, tok):
        return f"{hours:.2f} / {fl / 1e15:,.0f} / {tok / 1e6:.1f}"

    for method in methods:
        cells, pool = _cells(data, model, method.key)
        done = {b: c for b, c in cells.items() if c["nsteps"]}
        row = []
        for b in BENCH_KEYS:
            run = _is_running(running, method.key, b)
            if b in done:
                row.append(fmt(done[b]["hours"], done[b]["flops"], done[b]["tokens"])
                           + (" (running)" if run and annotate else ""))
            else:
                row.append("running" if run else "—")
        tot = [done[b] for b in BENCH_KEYS if b in done]
        run = _is_running(running, method.key, None, cells)
        if pool:
            row.append(fmt(sum(c["hours"] for c in tot), sum(c["flops"] for c in tot), sum(c["tokens"] for c in tot))
                       + (" (running)" if run and annotate else ""))
        else:
            row.append("running" if run else "—")
        out.append(f"| {method.label}{mark if done else ''} | " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=sorted(METHOD_SETS), default="compression")
    parser.add_argument("--stage", default="formal")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    methods = METHOD_SETS[args.set]
    data, errors = curves(methods, args.stage)
    for model, name in MODELS:
        if model in data:
            print(f"\n#### {name}\n\n{perf_table(data, model, methods)}\n\n{cost_table(data, model, methods, mark='')}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=1) + "\n")
    if errors:
        sys.exit(f"{len(errors)} cells failed")


if __name__ == "__main__":
    main()
