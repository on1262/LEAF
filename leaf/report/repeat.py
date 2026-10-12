"""Feedback and repeated errors in exploration (paper appendix: Tables tab:repeat-random, tab:repeat-paired, tab:repeat-base).

Does the verifier feedback make Ours less likely to repeat the wrong answer it was shown? Three
conditions differ only in the memory: Ours (last response + feedback), ``wo_feedback`` (last
response + a blind instruction: no correctness, no test output) and ``wo_last_response``
(problem + feedback).

Transition: a step t whose prompt holds step t-1's answer (t not a restart step 1 / 6 / 11 / 16 /
21) and whose previous answer is wrong. Wrong-answer signature: MATH-500, the normalised final
answer (``extracted_answer``); LCB / MBPP+, the exact public-test feedback (``native_feedback``:
failing input, output or error). SciKnowEval is open-ended (no comparable answer) and left out.

* repeat: step t is wrong with the previous signature; fix: step t is correct (public tests on code);
* matched comparison: per-problem means over its transitions, averaged over the problems both
  conditions have; 95% CI of the paired difference by bootstrap over problems (fixed seed);
* base rate: share of Random Sampling's 24 independent samples of the problem with the same signature;
  ratio = repeats / summed base rates over the same transitions.
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from leaf import paths
from leaf.report import exploration
from leaf.report.common import MODELS

BENCHES = (("math500", "MATH-500"), ("lcb", "LCB"), ("mbpp", "MBPP+"))
STEPS = 24
RESTART = {1, 6, 11, 16, 21}
CONDS = ((exploration.LEAF, "Ours"), ("wo_feedback", "w/o feedback"),
         ("wo_last_response", "w/o last response"))
BOOTSTRAP = 2000


def norm(answer: str) -> str:
    a = answer.strip().rstrip(".")
    a = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ |\$", "", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"\s+", "", a)


def answers(cell: Path, bench: str) -> dict:
    """{problem: {step: (correct, signature)}} over the first ``STEPS`` steps (1-based)."""
    path = cell / "grades_public.jsonl"
    if not path.is_file():
        return {}
    out = collections.defaultdict(dict)
    for line in path.open():
        r = json.loads(line)
        step = int(r["step"]) if "step" in r else int(r["aligned_step"]) + 1
        if step > STEPS:
            continue
        if bench == "math500":
            ok, sig = str(r["correct"]) == "True", norm(r.get("extracted_answer") or "") or None
        else:
            ok, sig = r["failure_type"] == "pass", (r.get("native_feedback") or "").strip() or None
        out[r["problem_uid"]][step] = (ok, sig)
    return out


def transitions(rows: dict) -> dict:
    """{problem: [(previous wrong signature, repeat, fix)]} over memory steps."""
    out = {}
    for p, by in rows.items():
        ts = []
        for t in range(2, STEPS + 1):
            if t in RESTART or t - 1 not in by or t not in by:
                continue
            (ok0, a), (ok1, s1) = by[t - 1], by[t]
            if not ok0 and a is not None:
                ts.append((a, (not ok1) and s1 == a, ok1))
        if ts:
            out[p] = ts
    return out


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def paired(a: dict, b: dict, idx: int) -> dict | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    pa = {p: _mean([x[idx] for x in a[p]]) for p in common}
    pb = {p: _mean([x[idx] for x in b[p]]) for p in common}
    diffs = [pa[p] - pb[p] for p in common]
    rng = random.Random(0)
    boot = sorted(_mean([rng.choice(diffs) for _ in diffs]) for _ in range(BOOTSTRAP))
    return {"problems": len(common), "a": _mean(list(pa.values())), "b": _mean(list(pb.values())),
            "diff": _mean(diffs), "lo": boot[int(0.025 * BOOTSTRAP)], "hi": boot[int(0.975 * BOOTSTRAP)]}


def base_ratio(tr: dict, base: dict) -> dict | None:
    freq = {p: collections.Counter(sig for ok, sig in by.values() if not ok and sig) for p, by in base.items()}
    n = rep = expect = 0
    for p, ts in tr.items():
        for a, r, _ in ts:
            n += 1
            rep += r
            expect += freq.get(p, {}).get(a, 0) / STEPS
    if not n or not expect:
        return None
    return {"transitions": n, "repeat": rep / n, "base": expect / n, "ratio": rep / expect}


def analyse(stage: str = "formal") -> dict:
    """{model: {bench: {"vs": {cond: {"repeat": paired, "fix": paired}}, "base": {cond: ratio}}}}."""
    methods = {m.key: m for m in exploration.EXPLORE_METHODS + exploration.EXPLORE_ABLATIONS}
    out = {}
    for model, _ in MODELS:
        block = out.setdefault(model, {})
        for bench, _ in BENCHES:
            tr = {}
            for key, _ in CONDS:
                m = methods[key]
                tr[key] = transitions(answers(m.cell(model, bench, "formal" if m.root == paths.EXPLORE_SEED1
                                                       else stage), bench))
            base = answers(methods["random_sampling"].cell(model, bench), bench)
            ours = CONDS[0][0]
            block[bench] = {
                "vs": {key: {"repeat": paired(tr[ours], tr[key], 1), "fix": paired(tr[ours], tr[key], 2)}
                       for key, _ in CONDS[1:] if tr[key]},
                "base": {key: base_ratio(tr[key], base) for key, _ in CONDS[:2] if tr[key]},
            }
    return out


def random_repeats(stage: str = "formal") -> dict:
    """Repeated wrong answers under independent sampling (paper appendix, Table tab:repeat-random).

    For each problem, the 24 Random Sampling responses are taken in the order in which
    they were drawn (``aligned_step``); a wrong response is a repeat when its signature (``answers``)
    equals the signature of an earlier wrong response to the same problem. Wrong responses without
    a signature count as non-repeats. Correctness on LCB / MBPP+ is the public-test verdict.

    {model: {bench: {"wrong": total wrong responses, "repeats": repeats, "repeat": repeats / wrong}}}
    """
    cell = next(m for m in exploration.EXPLORE_METHODS if m.key == "random_sampling").cell
    out = {}
    for model, _ in MODELS:
        block = out.setdefault(model, {})
        for bench, _ in BENCHES:
            wrong = repeats = 0
            for by in answers(cell(model, bench, stage), bench).values():
                seen = set()
                for step in sorted(by):
                    ok, sig = by[step]
                    if ok:
                        continue
                    wrong += 1
                    if sig is not None:
                        repeats += sig in seen
                        seen.add(sig)
            block[bench] = {"wrong": wrong, "repeats": repeats, "repeat": repeats / wrong if wrong else None}
    return out


def random_table(data: dict) -> str:
    names = dict(MODELS)
    rows = ["| Model | " + " | ".join(f"{b} Wrong | {b} Repeat (%)" for _, b in BENCHES) + " |",
            "|---|" + "---:|" * (2 * len(BENCHES))]
    for model, block in data.items():
        cells = []
        for bench, _ in BENCHES:
            r = block.get(bench) or {}
            cells += [f"{r['wrong']:,}" if r else "—",
                      f"{100 * r['repeat']:.1f}" if r.get("repeat") is not None else "—"]
        rows.append(f"| {names[model]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Repeated wrong answers of Random Sampling (seed 1).")
    parser.add_argument("--out", type=Path, default=paths.REPORT_ABLATION / "repeat_random.json")
    args = parser.parse_args(argv)
    data = random_repeats()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(random_table(data))
    print(f"wrote {args.out}")


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def tables(data: dict) -> str:
    label = dict(CONDS)
    names = dict(MODELS)
    rows = ["| Model | Benchmark | Comparison | Shared problems | Repeat Ours / comparison | Diff pp (95% CI) "
            "| Fix Ours / comparison | Diff pp (95% CI) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for model, block in data.items():
        for bench, bname in BENCHES:
            for key, res in block.get(bench, {}).get("vs", {}).items():
                r, f = res["repeat"], res["fix"]
                if not r:
                    continue
                rows.append(f"| {names[model]} | {bname} | {label[key]} | {r['problems']} "
                            f"| {100 * r['a']:.1f}% / {100 * r['b']:.1f}% | {_pp(r['diff'])} ({_pp(r['lo'])}, {_pp(r['hi'])}) "
                            f"| {100 * f['a']:.1f}% / {100 * f['b']:.1f}% | {_pp(f['diff'])} ({_pp(f['lo'])}, {_pp(f['hi'])}) |")
    base = ["| Model | Benchmark | Ours: repeat / base frequency (ratio) | w/o feedback: repeat / base frequency (ratio) |",
            "|---|---|---:|---:|"]
    for model, block in data.items():
        for bench, bname in BENCHES:
            cells = []
            for key, _ in CONDS[:2]:
                b = block.get(bench, {}).get("base", {}).get(key)
                cells.append(f"{100 * b['repeat']:.1f}% / {100 * b['base']:.1f}% ({b['ratio']:.2f})" if b else "—")
            base.append(f"| {names[model]} | {bname} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\n" + "\n".join(base)


if __name__ == "__main__":
    main()
