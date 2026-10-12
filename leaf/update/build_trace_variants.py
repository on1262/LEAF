"""Trace variants of the LEAF update data and the SFT-on-y* pairs (CPU only).

Per model x training set, from ``leaf.paths.GENERAL_LEAF_SEED1/<m>/<m>/formal/update_data/<bm>``
(``successful_contexts.jsonl`` and its source rollout, condition ``leaf``, restart every 5 steps)
into ``<output_root>/<m>/formal/update_data/<bm>``:

``successful_contexts.jsonl``  symlink to the source (successful contexts c+: the round prompt of
                      every distinct untruncated correct step-final)
``failure_contexts.jsonl``  same problems; ``teacher_messages_list`` = the distinct round prompts of the
                      problem's incorrect step-finals (failure contexts). A problem whose 20 steps
                      were all correct gets its plain prompt as its only context (``fallback``).
``random_contexts.jsonl``  same problems and contexts as ``successful_contexts.jsonl``, but every carry-step
                      context (earlier attempt + feedback) has its feedback turn replaced by a
                      feedback turn that differs from it: drawn uniformly among the distinct
                      feedback texts of the problem's carry steps, else among those of the other
                      problems. MATH / SciKnowEval feedback states only the verdict, so there this
                      flips it. Restart-step contexts (initial solution + blind correction) carry
                      no feedback and are kept.
``ystar_pairs.jsonl`` one row per (c+, y+) pair of ``successful_contexts.jsonl`` (same order): the context,
                      the correct solution y+ and the plain prompt x, for :mod:`leaf.update.rewrite_ystar`.
``contexts_summary.json``

    LEAF_CONFIG=configs/general/sft_qwen3_8b_seed1.json python -m leaf.update.build_trace_variants --model qwen3_8b
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import zlib
from pathlib import Path

import numpy as np

from leaf import paths

BENCHMARKS = ("lcb", "math500", "sciknoweval")
SHUFFLE_SEED = 230046
RESTART_INTERVAL = 5


def is_restart_step(step: int) -> bool:
    return (int(step) - 1) % RESTART_INTERVAL == 0


def source_dir(model: str, benchmark: str, source_root: Path | None = None) -> Path:
    """LEAF update data of the training set; ``source_root`` = another run's
    ``<root>/<m>/formal/update_data`` (run seeds 2 / 3)."""
    if source_root is not None:
        return source_root / benchmark
    return paths.GENERAL_LEAF_SEED1 / model / model / "formal" / "update_data" / benchmark


def plain(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def step_contexts(rows: list[dict]) -> list[dict]:
    """``{step, restart, correct, messages}`` of every step-final of one problem, in step order."""
    from leaf.update.trace_selection import final_generation

    out = []
    for row in sorted(rows, key=lambda r: int(r["step"])):
        messages = plain(final_generation(row).get("messages") or [])
        if not messages or messages[-1]["role"] != "user":
            continue
        out.append({"step": int(row["step"]), "restart": bool(row.get("restart")),
                    "correct": bool(row["final"].get("public_correct")), "messages": messages})
    return out


def failure_contexts(contexts: list[dict]) -> list[list[dict]]:
    """Distinct round prompts of the incorrect step-finals, in step order."""
    out, seen = [], set()
    for context in contexts:
        if context["correct"]:
            continue
        key = json.dumps(context["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(context["messages"])
    return out


def shuffle_feedback(uid: str, index: int, messages: list[dict], pool: list[str],
                     every: list[str]) -> tuple[list[dict], dict]:
    """``messages`` with its last (feedback) turn replaced by a uniformly drawn distinct feedback
    text that differs from it: of the problem's carry steps (``pool``), else of any problem."""
    rng = np.random.default_rng([SHUFFLE_SEED, zlib.crc32(uid.encode()), index])
    out = [dict(m) for m in messages]
    original = out[-1]["content"]
    others, source = sorted({t for t in pool if t != original}), "problem"
    if not others:
        others, source = [t for t in every if t != original], "other_problem"
    text = others[int(rng.integers(len(others)))]
    out[-1]["content"] = text
    return out, {"source": source, "identical": text == original}


def prepare(model: str, benchmark: str, output: Path, source_root: Path | None = None) -> dict:
    from leaf.core.project_config import atomic_json, atomic_jsonl, read_jsonl
    from leaf.prompts.task_prompts import opening_messages
    from leaf.update.trace_selection import correct_traces
    from leaf.update.rollout_rows import run_rows

    source = source_dir(model, benchmark, source_root)
    manifest = json.loads((source / "manifest.json").read_text())
    rollout = Path(manifest["rollout"])
    dataset = {row["uid"]: row for row in read_jsonl(Path(manifest["dataset"]))}
    successful = read_jsonl(source / "successful_contexts.jsonl")
    by_uid = run_rows(rollout)
    contexts = {uid: step_contexts(rows) for uid, rows in by_uid.items()}

    # feedback turns of every carry step, per problem (the shuffle pool) and over all problems
    pools = {uid: [(c["step"], c["messages"][-1]["content"]) for c in ctx
                   if not c["restart"] and len(c["messages"]) >= 3]
             for uid, ctx in contexts.items()}
    every = sorted({text for pool in pools.values() for _, text in pool})

    fail_rows, shuf_rows, pairs = [], [], []
    stats = {"fallback": 0, "shuffled": 0, "kept_restart": 0, "identical": 0, "other_problem": 0}
    fail_counts = []
    for row in successful:
        uid = row["uid"]
        x = plain(opening_messages(dataset[uid], model, ""))
        failures = failure_contexts(contexts[uid])
        stats["fallback"] += not failures
        fail_counts.append(len(failures))
        fail_rows.append({**{k: row[k] for k in ("uid", "problem_index", "prompt_messages")},
                          "teacher_messages_list": failures or [x], "fallback": not failures})

        # the solutions y+ of the success contexts, aligned with teacher_messages_list
        traces = [t for t in correct_traces(by_uid[uid]) if t["messages"] and t["messages"][-1]["role"] == "user"]
        if [t["step"] for t in traces] != [int(s) for s in row["teacher_steps"]]:
            raise RuntimeError(f"{uid}: correct traces do not match teacher_steps")
        shuffled = []
        for index, (messages, step) in enumerate(zip(row["teacher_messages_list"], row["teacher_steps"],
                                                     strict=True)):
            if is_restart_step(step):
                shuffled.append(plain(messages))
                stats["kept_restart"] += 1
                continue
            new, info = shuffle_feedback(uid, index, plain(messages), [t for _, t in pools[uid]], every)
            shuffled.append(new)
            stats["shuffled"] += 1
            stats["identical"] += info["identical"]
            stats["other_problem"] += info["source"] == "other_problem"
        shuf_rows.append({**{k: row[k] for k in ("uid", "problem_index", "prompt_messages", "teacher_steps")},
                          "teacher_messages_list": shuffled})
        for j, (trace, messages) in enumerate(zip(traces, row["teacher_messages_list"], strict=True)):
            pairs.append({"uid": uid, "problem_index": int(row["problem_index"]), "j": j,
                          "trace_step": int(trace["step"]), "prompt_messages": x,
                          "context_messages": plain(messages), "solution": trace["solution"]})

    output.mkdir(parents=True, exist_ok=True)
    link = output / "successful_contexts.jsonl"
    if not link.exists():
        os.symlink(os.path.realpath(source / "successful_contexts.jsonl"), link)
    atomic_jsonl(fail_rows, output / "failure_contexts.jsonl")
    atomic_jsonl(shuf_rows, output / "random_contexts.jsonl")
    atomic_jsonl(pairs, output / "ystar_pairs.jsonl")
    atomic_json({**manifest, "source": str(source)}, output / "manifest.json")
    summary = {
        "model": model, "benchmark": benchmark, "source": str(source), "rollout": str(rollout),
        "problems": len(successful), "pairs": len(pairs),
        "success_contexts_mean": statistics.mean(len(r["teacher_messages_list"]) for r in successful),
        "fail_contexts_mean": statistics.mean(fail_counts),
        "fail_contexts_mean_nonfallback": statistics.mean([n for n in fail_counts if n] or [0]),
        **stats,
    }
    atomic_json(summary, output / "contexts_summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", action="append", default=[], choices=BENCHMARKS)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="default: the config's output_root/<model>/formal/update_data")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="LEAF update data <root>/<m>/formal/update_data "
                             "(default: that of leaf.paths.GENERAL_LEAF_SEED1)")
    args = parser.parse_args(argv)
    from leaf.core.project_config import load_config

    config = load_config()
    root = args.output_root or Path(config["output_root"]) / args.model / "formal" / "update_data"
    for benchmark in args.benchmark or BENCHMARKS:
        print(json.dumps(prepare(args.model, benchmark, root / benchmark, args.source_root)), flush=True)


if __name__ == "__main__":
    main()
