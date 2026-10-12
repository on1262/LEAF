"""Response length per answer of the exploration runs, within the min(24 step, 5 GPU h) budget.

For every (model, method, benchmark) cell of the seed-1 exploration comparison and the
exploration ablations, the answers counted are exactly those the report's exploration curves
count (``leaf/report/exploration.py``): the per-cell kept steps ``len(cell["steps"])`` of
``explore/curves_seed1_with_interleaved.json`` / ``ablation/curves.json`` under REPORT_ROOT, and for the batch methods each
problem's first ``keep`` calls in rollout order (SEED: collection solutions first, then the
SEED-RL rollouts; SCoRe: both turns of a trajectory are calls).

Response tokens of one answer:
* LEAF / ablations: ``final.completion_tokens`` of the step's sample record (vLLM count of the
  graded answer). ``gen_tokens`` additionally sums every generation call of the step
  (restart steps generate an initial solution, then the correction that is graded).
* Random Sampling, SEED collection: ``completion_tokens`` of ``generations.jsonl`` / ``solutions.jsonl``.
* verl rollouts (GRPO / SDPO / SCoRe / SEED-RL) store decoded text only: the answer text is
  re-tokenized with the model's tokenizer (``add_special_tokens=False``) plus 1 for the
  end-of-turn token missing from the decoded text. SCoRe outputs hold turn 1 + the templated second-turn prompt + turn 2;
  the two answers are split at the second-turn instruction and the template text is dropped.

Summary per (model, method): ``mean_resp_tokens`` / ``mean_gen_tokens`` and ``pass_rate`` are
macro averages (each benchmark weighted 1/4: per-benchmark mean tokens per answer, and the
report's macro ``pooled`` pass rate). Cached cells are not re-tokenized; their summary is
recomputed on every run.

Run seeds 2 / 3 of the Table 1 methods are stored under ``seed<s>|<model>|<method>``
(seed 1 = ``<model>|<method>``).

Writes ``$REPORT_ROOT/figures/data/len_explore.json``.

    python -m figures.len_explore_data   (from the package root)
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaf import paths  # noqa: E402
from leaf.report import exploration as ex  # noqa: E402
from leaf.report.common import config, read_jsonl  # noqa: E402

OUT = paths.FIGURES_OUT / "data" / "len_explore.json"
CURVES_SEED1 = paths.REPORT_EXPLORE / "curves_seed1_with_interleaved.json"
CURVES_ABLATION = paths.REPORT_ABLATION / "curves.json"
# run seeds 2 / 3 (leaf.report.main_results curves; seed 1 = the curves above)
CURVES_SEEDS = str(paths.REPORT_MAIN / "curves_seed{seed}.json")
SEEDS = (2, 3)
TOKENIZERS = {model: f"general/baselines_{model}_seed1.json" for model in ("qwen3_8b", "olmo3_7b")}
SECOND_TURN = "Review your previous answer carefully"
# decoded rollout text lacks the generated end-of-turn token that vLLM counts in completion_tokens
# (re-tokenized length = verl response_length/mean - 1 on GRPO, checked on MATH-500 updates 1 / 5)
EOS = 1

_tok = {}


def tokenizer(model):
    if model not in _tok:
        from transformers import AutoTokenizer
        _tok[model] = AutoTokenizer.from_pretrained(config(TOKENIZERS[model])["models"][model])
    return _tok[model]


def n_tokens(model, texts):
    if not texts:
        return []
    enc = tokenizer(model)(texts, add_special_tokens=False)["input_ids"]
    return [len(x) + EOS for x in enc]


def read_records(path: Path) -> dict[int, dict]:
    """LEAF per-problem sample log, read-only (last record of a step wins, torn lines skipped)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[int(r["step"])] = r
    return out


def score_split(model: str, text: str) -> list[str]:
    """SCoRe trajectory output -> [turn 1 answer, turn 2 answer] (template text dropped)."""
    i = text.find(SECOND_TURN)
    if i < 0:
        return [text]
    head = "\nsystem\n" if model == "olmo3_7b" else "\nuser\n"
    cut = text.rfind(head, 0, i)
    first = text[:cut] if cut >= 0 else text[:i]
    j = text.find("\nassistant\n", i)
    second = text[j + len("\nassistant\n"):] if j >= 0 else ""
    if second.startswith("<think>\n\n</think>\n\n"):
        second = second[len("<think>\n\n</think>\n\n"):]
    return [first, second]


def leaf_cell(method, model, bench, keep):
    from leaf.explore.leaf_rollout import sample_record_path
    cell = method.cell(model, bench)
    problems = read_jsonl(method.dataset(model, bench))
    resp, gen, n = 0, 0, 0
    for index, p in enumerate(problems):
        recs = read_records(sample_record_path(cell, index, p["uid"]))
        for k in range(1, keep + 1):
            r = recs.get(k)
            if r is None:
                continue
            resp += int(r["final"]["completion_tokens"])
            gen += sum(int(m.get("completion_tokens") or 0) for m in r.get("intermediate") or [])
            n += 1
    return {"answers": n, "resp_tokens": resp, "gen_tokens": gen, "expected": keep * len(problems)}


def random_sampling_cell(method, model, bench, keep):
    cell = method.cell(model, bench)
    rows = [r for r in read_jsonl(cell / "generations.jsonl") if int(r["aligned_step"]) + 1 <= keep]
    tot = sum(int(r["completion_tokens"]) for r in rows)
    problems = len(read_jsonl(method.dataset(model, bench)))
    return {"answers": len(rows), "resp_tokens": tot, "gen_tokens": tot, "expected": keep * problems}


def batch_cell(method, model, bench, keep):
    import leaf.grading.grade_baseline as gb
    from leaf.core.project_config import benchmark_rl_data_dir

    cell = method.cell(model, bench)
    problems = len(read_jsonl(method.dataset(model, bench)))
    attempts = collections.Counter()
    lengths = []  # counted answers' token lengths
    pending = []  # (text) to tokenize, counted answers

    def call(uid, length=None, text=None):
        attempts[uid] += 1
        if attempts[uid] <= keep:
            if length is not None:
                lengths.append(length)
            else:
                pending.append(text)

    if method.kind == "seed":
        collection = cell.parent.parent / "seed_collection" / cell.name
        for r in sorted(read_jsonl(collection / "solutions.jsonl"), key=lambda r: int(r["trial_index"])):
            call(r["problem_uid"], length=int(r["completion_tokens"]))
    run_config = config(method.config.format(model=model))
    parquet = benchmark_rl_data_dir(run_config, bench, model) / gb.MAPPING_PARQUET["formal"]
    resolver = gb.PromptResolver(gb.prompt_uid_map(parquet), gb.ground_truth_uid_map(parquet))
    split_missing = 0
    for path in sorted((cell / "rollouts").glob("*.jsonl"), key=lambda p: int(p.stem)):
        for r in read_jsonl(path):
            uid = resolver.resolve_row(r)
            if "score_r1" in r:
                turns = score_split(model, r["output"])
                if len(turns) < 2:
                    split_missing += 1
                    turns = turns + [""]
                call(uid, text=turns[0])
                call(uid, text=turns[1])
            else:
                call(uid, text=r["output"])
    lengths += n_tokens(model, pending)
    return {"answers": len(lengths), "resp_tokens": sum(lengths), "gen_tokens": sum(lengths),
            "expected": keep * problems, "retokenized": len(pending), "score_unsplit": split_missing}


def cell_stats(method, model, bench, keep):
    fn = {"leaf": leaf_cell, "random_sampling": random_sampling_cell}.get(method.kind, batch_cell)
    return dict(fn(method, model, bench, keep), keep=keep)


# cells are independent: they are computed in a process pool (forked, so the task list is
# inherited and only the small result dicts are pickled). LEN_WORKERS overrides the pool size.
_TASKS: list = []


def _run_task(i):
    tag, method, model, bench, keep = _TASKS[i]
    return i, cell_stats(method, model, bench, keep)


def workers() -> int:
    if os.environ.get("LEN_WORKERS"):
        return max(1, int(os.environ["LEN_WORKERS"]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def compute_cells(tasks, on_tag_done=None) -> dict:
    """{(tag, benchmark): stats} for every (tag, method, model, benchmark, keep) task.

    ``on_tag_done(tag, {benchmark: stats})`` is called as soon as every cell of a tag is done, so
    finished tags can be saved before a later cell fails."""
    global _TASKS
    _TASKS = list(tasks)
    results = {}
    pending = collections.Counter(t[0] for t in _TASKS)

    def done(i, stats):
        results[i] = stats
        tag = _TASKS[i][0]
        pending[tag] -= 1
        if pending[tag] == 0 and on_tag_done is not None:
            on_tag_done(tag, {t[3]: results[j] for j, t in enumerate(_TASKS) if t[0] == tag})

    n = min(workers(), len(_TASKS))
    if n <= 1:
        for i in range(len(_TASKS)):
            done(*_run_task(i))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n) as pool:
            for i, stats in pool.imap_unordered(_run_task, range(len(_TASKS))):
                tag, _, model, bench, _ = _TASKS[i]
                print(tag, bench, stats, flush=True)
                done(i, stats)
    return {(t[0], t[3]): results[i] for i, t in enumerate(_TASKS)}


def summarize(model, key, method, entry, cached, curves_path, computed=None):
    """Token counts of one (model, method) curve entry (``cached`` cells are reused) + the macro summary."""
    if cached:
        cells = cached["cells"]
    else:
        cells = {c["benchmark"]: computed[c["benchmark"]] for c in entry["cells"]}
    p = entry["pooled"]
    # macro average: per-benchmark mean tokens per answer, then the mean over benchmarks
    mean = lambda f: sum(v[f] / v["answers"] for v in cells.values()) / len(cells)  # noqa: E731
    return {
        "model": model, "method": key, "cells": cells,
        "answers": sum(v["answers"] for v in cells.values()),
        "mean_resp_tokens": mean("resp_tokens"),
        "mean_gen_tokens": mean("gen_tokens"),
        "pass_rate": (p.get("pass_rate_private") or p["pass_rate"])[-1],
        "pooled_steps": len(p["steps"]),
        "curves": str(curves_path),
    }


def main():
    from leaf.report import main_results

    seed1 = json.loads(CURVES_SEED1.read_text())
    ablation = json.loads(CURVES_ABLATION.read_text())["ablation"]
    methods = {m.key: m for m in ex.EXPLORE_METHODS + ex.EXPLORE_ABLATIONS}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    # entries in output order: (tag, model, key, method, entry, curves path)
    # seed 1 and the exploration ablations: tags "<model>|<method>"
    entries = []
    for model, _ in ex.MODELS:
        for key, method in methods.items():
            src = seed1 if key in seed1.get(model, {}) else ablation
            entry = src.get(model, {}).get(key)
            if not entry or not entry.get("pooled"):
                continue
            entries.append((f"{model}|{key}", model, key, method, entry,
                            CURVES_SEED1 if src is seed1 else CURVES_ABLATION))
    # run seeds 2 / 3 of the Table 1 methods: tags "seed<s>|<model>|<method>"
    for seed in SEEDS:
        path = CURVES_SEEDS.format(seed=seed)
        curves = json.loads(Path(path).read_text())
        for method in main_results.explore_methods(seed):
            for model, _ in ex.MODELS:
                entry = curves.get(model, {}).get(method.key)
                if not entry or not entry.get("pooled"):
                    continue
                entries.append((f"seed{seed}|{model}|{method.key}", model, method.key, method, entry, path))
    tasks = [(tag, method, model, c["benchmark"], len(c["steps"]))
             for tag, model, _, method, entry, _ in entries if not out.get(tag)
             for c in entry["cells"]]
    print(f"{len(tasks)} cells to compute with {min(workers(), max(1, len(tasks)))} workers", flush=True)
    by_tag = {e[0]: e for e in entries}
    partial = dict(out)  # finished tags are saved as they complete (the cache of a re-run)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save(tag, cells):
        _, model, key, method, entry, curves_path = by_tag[tag]
        partial[tag] = summarize(model, key, method, entry, None, curves_path, cells)
        OUT.write_text(json.dumps(partial, indent=1))

    computed = compute_cells(tasks, save)
    for tag, model, key, method, entry, curves_path in entries:
        cells = {b: s for (t, b), s in computed.items() if t == tag}
        out[tag] = summarize(model, key, method, entry, out.get(tag), curves_path, cells)
    OUT.write_text(json.dumps(out, indent=1))
    for tag, v in out.items():
        print(f"{tag:70s} {v['mean_resp_tokens']:8.1f} {v['mean_gen_tokens']:8.1f} {100 * v['pass_rate']:6.2f}")


if __name__ == "__main__":
    main()
