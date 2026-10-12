"""Headline exploration numbers (printed as markdown) from the same series as
explore_curves.py (the three seeds' ``pooled.pass_rate_private``: macro average, the
unweighted mean of the four benchmarks, averaged over the seeds).

* final pass rate per method (last step within min(24 step, 5 GPU h per benchmark));
* Ours - Random Sampling and Ours - strongest baseline (pp);
* steps Ours needs to reach a baseline's final pass rate (linear interpolation between
  aligned steps), speedup = baseline steps / that step;
* the same in GPU h and PFLOPs: Ours' cumulative cost linearly interpolated at that
  fractional step vs the baseline's final cumulative cost; ratio = baseline cost / Ours cost.

"avg" = the two-model mean curve (pass rate and x averaged per aligned step), and the mean of
the per-model ratios is listed separately.

    python figures/explore_headline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_curves import METHODS, MODELS, LEAF, averaged, load, seed_mean  # noqa: E402

LABEL = {k: lab for k, lab, *_ in METHODS}
BASELINES = [k for k, *_ in METHODS if k != LEAF]


def reach(s: dict, target: float):
    """(fractional step, GPU h, PFLOPs) at which series ``s`` first reaches ``target``."""
    y = s["pass"]
    for i, v in enumerate(y):
        if v >= target - 1e-9:
            if i == 0:
                return s["steps"][0], s["hours"][0], s["pflops"][0]
            t = (target - y[i - 1]) / (v - y[i - 1])
            lerp = lambda a: a[i - 1] + t * (a[i] - a[i - 1])  # noqa: E731
            return lerp(s["steps"]), lerp(s["hours"]), lerp(s["pflops"])
    return None


def row(ours: dict, base: dict) -> dict:
    r = reach(ours, base["pass"][-1])
    out = {"base_final": base["pass"][-1], "base_steps": base["steps"][-1], "base_h": base["hours"][-1],
           "base_pf": base["pflops"][-1]}
    if r:
        out.update(step=r[0], h=r[1], pf=r[2], x_step=base["steps"][-1] / r[0], x_h=base["hours"][-1] / r[1],
                   x_pf=base["pflops"][-1] / r[2])
    return out


def main() -> None:
    data = load()
    series = {m: {k: seed_mean(data, m, k) for k in LABEL} for m, _ in MODELS}
    series["avg"] = {k: averaged(data, k) for k in LABEL}
    names = [n for _, n in MODELS] + ["2-model mean curve"]
    keys = [m for m, _ in MODELS] + ["avg"]

    print("## Final macro-average pass rate (%) / steps / GPU h / PFLOPs at the last pooled step\n")
    print("| Method | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for k in LABEL:
        cells = []
        for m in keys:
            s = series[m][k]
            cells.append(f"{s['pass'][-1]:.2f} / {s['steps'][-1]} / {s['hours'][-1]:.2f} / {s['pflops'][-1]:,.0f}")
        print(f"| {LABEL[k]} | " + " | ".join(cells) + " |")

    print("\n## Ours minus baseline final pass rate (pp)\n")
    print("| Baseline | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for b in BASELINES:
        print(f"| {LABEL[b]} | " + " | ".join(f"{series[m][LEAF]['pass'][-1] - series[m][b]['pass'][-1]:+.2f}"
                                              for m in keys) + " |")
    for m, n in zip(keys, names):
        best = max(BASELINES, key=lambda b: series[m][b]["pass"][-1])
        print(f"- strongest baseline, {n}: {LABEL[best]} ({series[m][best]['pass'][-1]:.2f}); "
              f"Ours - it = {series[m][LEAF]['pass'][-1] - series[m][best]['pass'][-1]:+.2f} pp")

    print("\n## Cost for Ours to reach a baseline's final pass rate\n")
    print("| Baseline | Curve | baseline final % (steps) | Ours step | speedup (steps) | Ours GPU h vs baseline GPU h "
          "(ratio) | Ours PFLOPs vs baseline PFLOPs (ratio) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    ratios = {}
    for b in BASELINES:
        for m, n in zip(keys, names):
            r = row(series[m][LEAF], series[m][b])
            if "step" not in r:
                print(f"| {LABEL[b]} | {n} | {r['base_final']:.2f} ({r['base_steps']}) | not reached | | | |")
                continue
            if m != "avg":
                ratios.setdefault(b, []).append(r)
            print(f"| {LABEL[b]} | {n} | {r['base_final']:.2f} ({r['base_steps']}) | {r['step']:.2f} | "
                  f"{r['x_step']:.2f}x | {r['h']:.2f} vs {r['base_h']:.2f} ({r['x_h']:.2f}x) | "
                  f"{r['pf']:,.0f} vs {r['base_pf']:,.0f} ({r['x_pf']:.2f}x) |")
        rs = ratios.get(b, [])
        if len(rs) == len(MODELS):
            mean = lambda f: sum(r[f] for r in rs) / len(rs)  # noqa: E731
            print(f"| {LABEL[b]} | mean of per-model ratios | | {mean('step'):.2f} | {mean('x_step'):.2f}x | "
                  f"({mean('x_h'):.2f}x) | ({mean('x_pf'):.2f}x) |")
    for m, n in zip(keys, names):
        print(f"- steps present, {n}: " + ", ".join(f"{LABEL[k]} {series[m][k]['steps'][-1]}" for k in LABEL))


if __name__ == "__main__":
    main()
