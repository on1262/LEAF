"""Main results: the exploration (Fig. 3 / Table 1) and generalization (Table 2) results over
run seeds 1-3 (``paths.explore_root(seed)`` / ``paths.general_root(seed, part)``; the seed 2 / 3
configs are prepared by :mod:`leaf.data.prepare_run_seeds`).

Exploration: :func:`leaf.report.exploration.curves` per seed (``curves_seed<s>.json`` under
``paths.REPORT_MAIN``); Pass = the pooled (four-benchmark macro average) cumulative pass rate at
the last aligned step (LCB / MBPP+ public AND private), GPU h / PFLOPs = the pooled sums there.
Generalization: :mod:`leaf.report.update` rows per seed (``general.json``); GPU h / EFLOPs as
Table 2 (Ours = LEAF update + data rollout, SFT = data rollout + y* rewrite + SFT, SEED incl.
collection). mean ± std = sample std over the seeds that have the value (n in the cell when < 3).
``sections.json`` holds the markdown tables printed on stdout.

    python -m leaf.report.main_results [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

from leaf import paths
from leaf.report import exploration, update
from leaf.report.common import MODELS, PARAMS, Slurm, read_json, same_path

SEEDS = (1, 2, 3)
REPORT_TABLES = paths.REPORT_MAIN
LEAF = exploration.LEAF
# key, paper label, color, linestyle, linewidth, marker (figures/explore_curves.py)
EXPLORE = (("random_sampling", "Random Sampling", "#8A8B8C", (0, (4, 2)), 1.3, None),
           ("grpo", "GRPO", "#00AEBB", "-", 1.0, "o"),
           ("sdpo", "SDPO", "#6A51A3", "-", 1.0, "s"),
           ("score", "SCoRe", "#E69F00", "-", 1.0, "^"),
           ("seed", "SEED", "#3A3A3A", "-", 1.0, "D"),
           (LEAF, "Ours", "#C00000", "-", 2.0, None))
LABEL = {k: label for k, label, *_ in EXPLORE}
GENERAL = ("Base", "GRPO", "SDPO", "SCoRe", "SEED", "SFT", "Ours")


# --------------------------------------------------------------------------- statistics
def mean_std(values) -> tuple[float | None, float | None, int]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None), len(values)


def fmt_ms(values, digits: int = 2, scale: float = 1.0) -> str:
    m, s, n = mean_std(values)
    if m is None:
        return "—"
    text = f"{m * scale:,.{digits}f}" + (f" ± {s * scale:,.{digits}f}" if s is not None else "")
    return text + (f" (n={n})" if n < len(SEEDS) else "")


def fmt(value, digits: int = 2, scale: float = 1.0) -> str:
    return "—" if value is None else f"{value * scale:,.{digits}f}"


# --------------------------------------------------------------------------- exploration
def explore_methods(seed: int) -> tuple:
    base = tuple(m for m in exploration.EXPLORE_METHODS if m.key in LABEL)
    if seed == 1:
        return base
    return tuple(dataclasses.replace(m, root=paths.explore_root(seed),
                                     config=f"explore/{{model}}_seed{seed}.json") for m in base)


def final(data: dict, model: str, key: str) -> dict | None:
    """Pooled pass rate (%), GPU h, PFLOPs and step count at the last step; None unless the four
    cells are graded (``exploration.curve_complete``)."""
    if not exploration.curve_complete(data, model, key):
        return None
    p = data[model][key]["pooled"]
    n = len(p["pass_rate_private"])
    return {"pass": 100 * p["pass_rate_private"][-1], "hours": p["hours"][n - 1],
            "pflops": p["flops"][n - 1] / 1e15, "steps": n}


def explore_tables(curves: dict) -> str:
    out = []
    for model, name in MODELS:
        rows = ["| Method | Pass seed 1 | seed 2 | seed 3 | Pass mean ± std | GPU h mean ± std | PFLOPs mean ± std |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for key, label, *_ in EXPLORE:
            f = [final(curves[s], model, key) for s in SEEDS]
            short = [f"{x['pass']:.2f}" + (f" ({x['steps']} steps)" if x["steps"] < exploration.STEPS else "")
                     if x else "—" for x in f]
            rows.append(f"| {label} | " + " | ".join(short) + " | "
                        + " | ".join(fmt_ms([x[k] if x else None for x in f], d)
                                     for k, d in (("pass", 2), ("hours", 2), ("pflops", 0))) + " |")
        out.append(f"#### {name}\n\n" + "\n".join(rows))
    return "\n\n".join(out)


def averaged(data: dict, key: str) -> dict | None:
    """Two-model mean per aligned step present for both models (explore_curves.averaged)."""
    series = []
    for model, _ in MODELS:
        if not exploration.curve_complete(data, model, key):
            return None
        p = data[model][key]["pooled"]
        y = p["pass_rate_private"]
        series.append({"steps": p["steps"][:len(y)], "pass": [100 * v for v in y], "hours": p["hours"][:len(y)],
                       "pflops": [f / 1e15 for f in p["flops"][:len(y)]]})
    n = min(len(s["steps"]) for s in series)
    return {"steps": series[0]["steps"][:n],
            **{f: [statistics.mean(s[f][i] for s in series) for i in range(n)] for f in ("pass", "hours", "pflops")}}


def seed_mean(curves: dict, key: str) -> tuple[dict | None, int]:
    """Across-seed mean (and std) of the two-model averaged curve, over steps all seeds have."""
    per = [averaged(curves[s], key) for s in SEEDS]
    per = [p for p in per if p]
    if not per:
        return None, 0
    n = min(len(p["steps"]) for p in per)
    out = {"steps": per[0]["steps"][:n]}
    for f in ("pass", "hours", "pflops"):
        out[f] = [statistics.mean(p[f][i] for p in per) for i in range(n)]
        out[f + "_std"] = [statistics.stdev(p[f][i] for p in per) if len(per) > 1 else 0.0 for i in range(n)]
    return out, len(per)


def reach_x(s: dict, xkey: str, target: float) -> float | None:
    """x where series ``s`` first reaches pass rate ``target`` (linear between steps)."""
    y, x = s["pass"], s[xkey]
    for i, v in enumerate(y):
        if v >= target - 1e-9:
            if i == 0:
                return x[0]
            t = (target - y[i - 1]) / (v - y[i - 1])
            return x[i - 1] + t * (x[i] - x[i - 1])
    return None


def speedup(score: dict | None, ours: dict | None, xkey: str) -> float | None:
    """Fig. 3 ratio: x of SCoRe's final point / x where Ours reaches SCoRe's final pass rate."""
    if not score or not ours:
        return None
    xo = reach_x(ours, xkey, score["pass"][-1])
    return None if not xo else score[xkey][-1] / xo


PANELS = (("steps", "Verifier calls / problem"), ("pflops", "Cumulative compute (PFLOPs)"),
          ("hours", "Cumulative GPU time (h)"))


def claims(curves: dict, general: dict) -> str:
    """The paper's quantitative statements, per seed and as mean ± std."""
    rows = ["| Quantity | Paper value (seed 1) | seed 1 | seed 2 | seed 3 | mean ± std |", "|---|---:|---:|---:|---:|---:|"]

    def add(name: str, paper: str, values: list, digits: int = 2):
        rows.append(f"| {name} | {paper} | " + " | ".join(fmt(v, digits) for v in values) + " | "
                    + fmt_ms(values, digits) + " |")

    def best_other(f: dict, keys) -> tuple[str, float] | None:
        """Best method other than Ours; None until every one of them is complete."""
        if not all(f.get(k) for k in keys):
            return None
        return max(((k, f[k]["pass"]) for k in keys), key=lambda kv: kv[1])

    for model, name in MODELS:
        finals = [{k: final(curves[s], model, k) for k, *_ in EXPLORE} for s in SEEDS]
        add(f"Exploration {name}: Ours Pass − SCoRe Pass (pp)", "2.42" if model == "qwen3_8b" else "2.28",
            [f[LEAF]["pass"] - f["score"]["pass"] if f[LEAF] and f["score"] else None for f in finals])
        best = [best_other(f, [k for k, *_ in EXPLORE if k != LEAF]) for f in finals]
        rows.append(f"| Exploration {name}: best method other than Ours (Pass) | SCoRe | "
                    + " | ".join(LABEL[b[0]] if b else "—" for b in best) + " | |")
        add(f"Exploration {name}: Ours Pass − best other Pass (pp)", "2.42" if model == "qwen3_8b" else "2.28",
            [f[LEAF]["pass"] - b[1] if f[LEAF] and b else None for f, b in zip(finals, best)])
    for xkey, xname in PANELS:
        per = [speedup(averaged(curves[s], "score"), averaged(curves[s], LEAF), xkey) for s in SEEDS]
        means = (seed_mean(curves, "score"), seed_mean(curves, LEAF))
        mean_curves = tuple(c if n == len(SEEDS) else None for c, n in means)
        paper = {"pflops": "Fig. 3(b)", "hours": "Fig. 3(c)", "steps": "Fig. 3(a)"}[xkey]
        rows.append(f"| Fig. 3 speedup SCoRe / Ours ({xname}, two-model mean curve) | {paper} | "
                    + " | ".join(fmt(v) for v in per) + f" | {fmt_ms(per)}; on the 3-seed mean curve "
                    + f"{fmt(speedup(*mean_curves, xkey))} |")
    for model, name in MODELS:
        g = [general[s].get(model, {}) for s in SEEDS]
        ov = lambda gs, k: gs.get(k, {}).get("overall")  # noqa: E731
        add(f"Generalization {name}: Ours Overall − SEED Overall (pp)", "1.82" if model == "qwen3_8b" else "1.64",
            [ov(x, "Ours") - ov(x, "SEED") if ov(x, "Ours") is not None and ov(x, "SEED") is not None else None
             for x in g])
        best = []
        for x in g:
            cand = [(k, ov(x, k)) for k in GENERAL if k not in ("Ours", "Base")]
            best.append(max(cand, key=lambda kv: kv[1]) if all(v is not None for _, v in cand) else None)
        rows.append(f"| Generalization {name}: best method other than Ours (Overall) | SEED | "
                    + " | ".join(b[0] if b else "—" for b in best) + " | |")
        add(f"Generalization {name}: Ours Overall − best other Overall (pp)",
            "1.82" if model == "qwen3_8b" else "1.64",
            [ov(x, "Ours") - b[1] if ov(x, "Ours") is not None and b else None for x, b in zip(g, best)])
        add(f"Generalization {name}: SFT Overall − Base Overall (pp)", "−0.77" if model == "qwen3_8b" else "−1.28",
            [ov(x, "SFT") - ov(x, "Base") if ov(x, "SFT") is not None and ov(x, "Base") is not None else None
             for x in g])
        add(f"Generalization {name}: SEED EFLOPs / Ours EFLOPs", "1.86" if model == "qwen3_8b" else "1.91",
            [x["SEED"]["eflops"] / x["Ours"]["eflops"] if x.get("SEED", {}).get("eflops") and
             x.get("Ours", {}).get("eflops") else None for x in g])
    return "\n".join(rows)


# --------------------------------------------------------------------------- generalization
def general_methods(seed: int, model: str) -> dict:
    """Table 2 row name -> UpdateMethod of this seed / model."""
    by_label = {m.label: m for m in update.GENERAL_METHODS}
    rows = {name: by_label[name] for name in ("Base", "GRPO", "SDPO", "SCoRe", "SEED")}
    sft = dataclasses.replace(update.SFT_YSTAR, rollout=True)
    if seed == 1:
        # SFT on y* of seed 1 has its own root (general/seed1/sft); its data rollout is the seed-1 LEAF one
        return {**rows, "SFT": sft, "Ours": update.LEAF_UPDATE}
    baselines, leaf = paths.general_root(seed, "baselines"), paths.general_root(seed, "leaf")
    return {**{k: dataclasses.replace(m, root=baselines) for k, m in rows.items() if k != "SEED"},
            "SEED": dataclasses.replace(rows["SEED"], root=paths.general_root(seed, "seed_baseline")),
            "SFT": dataclasses.replace(sft, root=leaf, rollout_root=None),
            "Ours": dataclasses.replace(update.LEAF_UPDATE, root=leaf)}


def rewrite_cost(slurm: Slurm, root: Path, model: str) -> tuple[float | None, float | None]:
    """y* rewrite GPU h and FLOPs (2N per prompt + completion token; judge excluded, as Table 2)."""
    hours, flops = [], []
    for bench in update.TRAINS:
        ystar = root / model / model / "formal" / "update_data" / bench / "ystar"
        s = read_json(ystar / "rewrite_summary.json")
        # a rewrite job that ends CANCELLED or TIMEOUT after writing its summary did the full work:
        # its elapsed time counts as well
        used = [slurm.state[str(r["job"])][1] for r in slurm.rows
                if r.get("stage") == "formal_rewrite" and same_path(r["output"], ystar)
                and slurm.state.get(str(r["job"]), ("?", 0))[0] in ("COMPLETED", "TIMEOUT", "CANCELLED")]
        hours.append(sum(used) / 3600 if s and used else None)
        flops.append(2 * PARAMS[model] * (s["prompt_tokens"] + s["completion_tokens"]) if s else None)
    total = lambda v: None if None in v else sum(v)  # noqa: E731
    return total(hours), total(flops)


def general_values(seed: int) -> dict:
    """{model: {row: {lcb, math500, sciknoweval, overall (avg@4 %), gpu_h, eflops}}}."""
    out = {}
    methods = {model: general_methods(seed, model) for model, _ in MODELS}
    roots = update.slurm_roots([m for ms in methods.values() for m in ms.values()])
    slurm = Slurm(roots, stages=("formal", "formal_rewrite"))
    for model, _ in MODELS:
        block = out.setdefault(model, {})
        names = list(methods[model])
        ms = [methods[model][k] for k in names]
        costs = {m.label: c for m, c in update.cost_rows(model, slurm, ms)}
        for name, (m, results) in zip(names, update.rows(model, ms)):
            row = {t: (100 * results[t]["avg"] if results.get(t) else None) for t in update.OVERALL}
            total = update.overall(results)
            row["overall"] = None if total is None else 100 * total
            c = costs.get(m.label) if row["overall"] is not None else None  # finished chains only
            if c:
                def tot(values):
                    return None if None in values else sum(values)
                h = [tot(c["train_h"]), tot(c["roll_h"])]
                f = [tot(c["train_f"]), tot(c["roll_f"])]
                if name == "SFT":
                    rh, rf = rewrite_cost(slurm, m.root, model)
                    h.append(rh)
                    f.append(rf)
                row["gpu_h"] = tot(h)
                row["eflops"] = None if tot(f) is None else tot(f) / 1e18
            block[name] = row
    return out


GEN_COLS = (("lcb", "LCB"), ("math500", "MATH"), ("sciknoweval", "SciKE"), ("overall", "Overall"),
            ("gpu_h", "GPU h"), ("eflops", "EFLOPs"))


def general_tables(general: dict) -> str:
    out = []
    for model, name in MODELS:
        rows = ["| Method | " + " | ".join(c for _, c in GEN_COLS) + " |", "|---|" + "---:|" * len(GEN_COLS)]
        for row in GENERAL:
            vals = [[general[s].get(model, {}).get(row, {}).get(k) for s in SEEDS] for k, _ in GEN_COLS]
            rows.append(f"| {row} | " + " | ".join(fmt_ms(v) for v in vals) + " |")
        per = ["| Method | " + " | ".join(f"Overall seed {s}" for s in SEEDS) + " | "
               + " | ".join(f"{c} seed 1 / 2 / 3" for _, c in GEN_COLS[:3]) + " |",
               "|---|" + "---:|" * (len(SEEDS) + 3)]
        for row in GENERAL:
            g = [general[s].get(model, {}).get(row, {}) for s in SEEDS]
            per.append(f"| {row} | " + " | ".join(fmt(x.get("overall")) for x in g) + " | "
                       + " | ".join(" / ".join(fmt(x.get(k)) for x in g) for k, _ in GEN_COLS[:3]) + " |")
        out.append(f"#### {name}: mean ± std (avg@4 %, GPU h, EFLOPs)\n\n" + "\n".join(rows)
                   + f"\n\n#### {name}: per seed\n\n" + "\n".join(per))
    return "\n\n".join(out)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="3-seed main-result tables as JSON + markdown sections.")
    parser.add_argument("--out", type=Path, default=REPORT_TABLES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    curves, errors = {}, []
    for s in SEEDS:
        curves[s], err = exploration.curves(explore_methods(s))
        errors += [{"seed": s, **e} for e in err]
        (args.out / f"curves_seed{s}.json").write_text(json.dumps(curves[s], indent=1) + "\n")
    general = {s: general_values(s) for s in SEEDS}
    (args.out / "general.json").write_text(json.dumps(general, indent=1) + "\n")
    sections = {"updated": time.strftime("%Y-%m-%d %H:%M %Z"), "explore": explore_tables(curves),
                "general": general_tables(general), "claims": claims(curves, general)}
    (args.out / "sections.json").write_text(json.dumps(sections, indent=1, ensure_ascii=False) + "\n")
    for title, key in (("Exploration", "explore"), ("Generalization", "general"), ("Claims", "claims")):
        print(f"\n### {title}\n\n{sections[key]}")
    if errors:
        print("\n".join(json.dumps(e) for e in errors), file=sys.stderr)
    print(f"wrote {args.out}; {len(errors)} exploration cells failed", file=sys.stderr)


if __name__ == "__main__":
    main()
