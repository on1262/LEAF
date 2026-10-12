"""FLOPs accounting of the training runs (verl methods, SEED, LEAF update) and the LEAF rollouts.

FLOPs = 2N x forward tokens + 6N x trained tokens (``common.flops``).

* verl update (``metrics.jsonl`` row): generation ``perf/total_num_tokens``, plus the extra
  turns (SCoRe), old / ref log-prob, SDPO teacher (reprompt fraction x (seq + mean response)),
  SEED-RL analyzer + OPD teacher, judge calls; trained = ``perf/total_num_tokens``. An update
  whose metrics row was lost when a later chunk resumed takes the run's mean.
* SEED: collection (solutions + their judge + analyses) + SFT (tokenized ``sft.parquet`` x
  ``SFT_EPOCHS``) + its RL run.
* LEAF update (``step_timings.jsonl`` batch): student sampling + teacher forward + behaviour
  forward, student update.
* LEAF rollout: every generation's prompt + completion, judge included.
"""

from __future__ import annotations

import functools
import json
import statistics
from pathlib import Path

from leaf import paths
from leaf.report.common import PARAMS, config, flops, read_json, read_jsonl

SFT_EPOCHS = 3


@functools.lru_cache(maxsize=None)
def judge_delta(model: str) -> float:
    """Mean judge prompt tokens minus generation prompt + response, measured once per model on
    the seed-1 Base SciKnowEval test generations (the judge prompt wraps the response)."""
    path = paths.GENERAL_BASELINES_SEED1 / model / model / "test" / "base" / "sciknoweval" / "generations.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(path)
    return statistics.mean(r["judge"]["prompt_tokens"] - r["prompt_tokens"] - r["completion_tokens"]
                           for r in rows if r.get("judge"))


def verl_components(d: dict, model: str, method: str) -> tuple[float, float]:
    """(forward tokens, trained tokens) of one verl update's ``metrics.jsonl`` data."""
    total = float(d["perf/total_num_tokens"])
    p, r = float(d["prompt_length/mean"]), float(d["response_length/mean"])
    n_seq = float(d.get("judge/requests") or round(total / (p + r)))
    fwd = total
    turns = float(d.get("num_turns/mean", 2.0))
    if turns > 2.0 + 1e-6:
        fwd += (turns / 2.0 - 1.0) * n_seq * (p + r / 2.0)
    if "timing_s/old_log_prob" in d:
        fwd += total
    if "timing_s/ref" in d:
        fwd += total
    if method == "sdpo":
        fwd += float(d.get("self_distillation/reprompt_sample_fraction", 1.0)) * (total + n_seq * r)
    if method == "seed_rl":
        fwd += float(d.get("seed/analyzer_prompt_tokens", 0)) + float(d.get("seed/analyzer_completion_tokens", 0))
        fwd += total + n_seq * float(d.get("seed/skill_tokens_mean", 0))
    if "judge/requests" in d:
        jn = float(d["judge/requests"])
        fwd += jn * float(d.get("judge/completion_tokens_mean", 0)) + jn * (p + r + judge_delta(model))
    return fwd, total


def verl_run_flops(run: Path, model: str, method: str) -> float:
    """Total FLOPs of a verl run: one ``metrics.jsonl`` row per dumped ``rollouts/<u>.jsonl``."""
    rows = {int(r["step"]): r["data"] for r in read_jsonl(run / "metrics.jsonl")}
    comps = {u: verl_components(d, model, method) for u, d in rows.items()}
    if not comps:
        raise ValueError(f"no metrics rows in {run}")
    mean = (statistics.mean(c[0] for c in comps.values()), statistics.mean(c[1] for c in comps.values()))
    n = PARAMS[model]
    return sum(flops(*comps.get(int(p.stem), mean), n) for p in (run / "rollouts").glob("*.jsonl"))


def seed_flops(training: Path, model: str, bench: str) -> float:
    """SEED collection + SFT + RL FLOPs; ``training`` = ``<root>/<m>/<m>/formal/baselines``."""
    collection = training / "seed_collection" / bench
    n = PARAMS[model]
    records = read_jsonl(collection / "solutions.jsonl")
    if not records:
        raise FileNotFoundError(collection / "solutions.jsonl")
    fwd = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    fwd += sum(r["judge"]["prompt_tokens"] + r["judge"]["completion_tokens"]
               for r in records if isinstance(r.get("judge"), dict) and "prompt_tokens" in r["judge"])
    fwd += sum(a["prompt_tokens"] + a["completion_tokens"] for a in read_jsonl(collection / "analyses.jsonl"))
    return (flops(fwd, 0.0, n) + flops(0.0, sft_tokens(model, collection) * SFT_EPOCHS, n)
            + verl_run_flops(training / "seed_rl" / bench, model, "seed_rl"))


def leaf_update_step_flops(batch: dict, n: float) -> float:
    """One LEAF update ``step_timings`` row's batch (a missing field counts 0)."""
    get = lambda k: batch.get(k, 0)  # noqa: E731
    fwd = (get("generation_prompt_tokens") + get("generation_completion_tokens") + get("teacher_tokens")
           + get("student_tokens"))
    return flops(fwd, get("student_tokens"), n)


def sft_run_flops(run: Path, model: str) -> float | None:
    """Total FLOPs of a finished SFT run (``sft_ystar``): 6N x the trained tokens (prompt +
    response) of every step; no sampling, no separate forward."""
    rows = {int(r["step"]): r for r in read_jsonl(run / "step_timings.jsonl")}
    if not rows or not (run / "COMPLETE.json").is_file():
        return None
    return sum(flops(0, (r.get("batch") or {}).get("student_tokens", 0), PARAMS[model]) for r in rows.values())


def leaf_update_run_flops(run: Path, model: str) -> float | None:
    """Total FLOPs of a finished LEAF update run (None: no ``COMPLETE.json`` / no step log); the last
    row of a step wins after a resume, ``update_skipped`` steps count nothing."""
    rows = {int(r["step"]): r for r in read_jsonl(run / "step_timings.jsonl")}
    if not rows or not (run / "COMPLETE.json").is_file():
        return None
    return sum(leaf_update_step_flops(r.get("batch") or {}, PARAMS[model]) for _, r in sorted(rows.items())
               if not r.get("update_skipped"))


# --------------------------------------------------------------------------- SEED SFT tokens
def sft_tokens(model: str, collection: Path) -> int:
    """Prompt + response tokens of a SEED collection's ``sft.parquet``. Tokenizing takes ~30 s,
    so the count is cached in ``paths.SFT_TOKENS_CACHE`` keyed on the file's path, size, mtime."""
    parquet = (collection / "sft.parquet").resolve()
    stat = parquet.stat()
    key = f"{model}|{parquet}|{stat.st_size}|{stat.st_mtime}"
    cache = read_json(paths.SFT_TOKENS_CACHE) or {}
    if key in cache:
        return int(cache[key])
    count = _sft_tokens(model, parquet)
    cache = read_json(paths.SFT_TOKENS_CACHE) or {}
    cache[key] = count
    paths.SFT_TOKENS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    paths.SFT_TOKENS_CACHE.write_text(json.dumps(cache, indent=1))
    return count


def _sft_tokens(model: str, parquet: Path) -> int:
    import pandas as pd
    from transformers import AutoTokenizer

    name = f"general/baselines_{model}_seed1.json"
    tok = AutoTokenizer.from_pretrained(config(name)["models"][model], trust_remote_code=True)
    frame = pd.read_parquet(parquet)
    return sum(len(tok.encode(str(row["prompt"]), add_special_tokens=False))
               + len(tok.encode(str(row["response"]), add_special_tokens=False))
               for _, row in frame.iterrows())


# --------------------------------------------------------------------------- LEAF rollouts
def rollout_flops(cell: Path, n: float) -> float:
    """FLOPs of a finished LEAF rollout (``steps/step_*/{intermediate,final}.jsonl``), cached in
    ``paths.FLOPS_CACHE`` keyed on the directory and its newest mtime (itself + children)."""
    key = f"rollout|{cell}|{n}"
    stamp = max([cell.stat().st_mtime] + [c.stat().st_mtime for c in cell.iterdir()]) if cell.is_dir() else 0.0
    hit = (read_json(paths.FLOPS_CACHE) or {}).get(key)
    if hit and hit["stamp"] == stamp:
        return hit["value"]
    fwd = 0.0
    for step in sorted((cell / "steps").glob("step_*")):
        fwd += sum(r["prompt_tokens"] + r["completion_tokens"] for r in read_jsonl(step / "intermediate.jsonl"))
        fwd += sum(r["judge"]["prompt_tokens"] + r["judge"]["completion_tokens"]
                   for r in read_jsonl(step / "final.jsonl")
                   if isinstance(r.get("judge"), dict) and "prompt_tokens" in r["judge"])
    value = flops(fwd, 0.0, n)
    cache = read_json(paths.FLOPS_CACHE) or {}
    cache[key] = {"stamp": stamp, "value": value}
    paths.FLOPS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    paths.FLOPS_CACHE.write_text(json.dumps(cache))
    return value
