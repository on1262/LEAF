"""Appendix values over the three run seeds -> ``paths.REPORT_MAIN/appendix.json``.

explore: per (model, method, benchmark | overall) and seed: mean accuracy over the counted steps,
final pass rate (LCB / MBPP+ public AND private), GPU h / PFLOPs / M tokens at the run's common cutoff
(last step all four benchmarks reach), steps.
general: per seed and model: LEAF update problems per training set, y* rewrite statistics (GPU h also counts a
rewrite job that ended CANCELLED or TIMEOUT after writing its summary), test avg@4 / pass@4 per row,
training / rollout cost.

Also writes ``paths.REPORT_MAIN/cost_20_verifier_calls.json``: training GPU h of the seed-1
baseline runs at a budget of 20 verifier calls per problem (GRPO / SDPO / SCoRe / SEED), per
training set and summed (the appendix sentence that motivates the SEED budget of 15 calls per
problem).

Reads the ``curves_seed{1,2,3}.json`` written by :mod:`leaf.report.main_results`.

    python -m leaf.report.appendix_tables
"""
import json
from pathlib import Path

from leaf import paths
from leaf.report import main_results as r, update as u
from leaf.report.common import PARAMS, Slurm, read_json, same_path

OUT = paths.REPORT_MAIN / "appendix.json"
COST_20_CALLS = paths.REPORT_MAIN / "cost_20_verifier_calls.json"
MODELS = [m for m, _ in r.MODELS]
EXP = [k for k, *_ in r.EXPLORE]
# rows of the 20-call budget comparison: row name -> update.GENERAL_METHODS label
COST_20_CALLS_ROWS = {"GRPO": "GRPO", "SDPO": "SDPO", "SCoRe": "SCoRe", "SEED": "SEED (20 verifier calls)"}


def explore(curves: dict) -> dict:
    """Exploration per benchmark: acc (mean over counted steps), pass (last), hours/flops/tokens (last), steps."""
    out = {}
    for m in MODELS:
        for k in EXP:
            for s in r.SEEDS:
                e = curves[s][m][k]
                # every cell is cut at the run's common cutoff: the last step all four benchmarks reach
                # (as the pooled values of Table 1; Olmo SEED seeds 1 / 2 stop at step 16)
                n = len((e["pooled"].get("pass_rate_private") or e["pooled"]["pass_rate"]))
                for c in e["cells"]:
                    a = c.get("accuracy_private") or c["accuracy"]
                    p = c.get("pass_rate_private") or c["pass_rate"]
                    out.setdefault(m, {}).setdefault(k, {}).setdefault(c["benchmark"], []).append(
                        {"acc": 100 * sum(a[:n]) / n, "pass": 100 * p[n - 1], "hours": c["hours"][n - 1],
                         "pflops": c["flops"][n - 1] / 1e15, "mtok": c["tokens"][n - 1] / 1e6, "steps": n})
                pl = e["pooled"]
                a = pl.get("accuracy_private") or pl["accuracy"]
                p = pl.get("pass_rate_private") or pl["pass_rate"]
                out[m][k].setdefault("overall", []).append(
                    {"acc": 100 * sum(a) / len(a), "pass": 100 * p[-1], "steps": len(p)})
    return out


def contexts_summary(seed, m, b):
    d = paths.general_root(seed, "leaf") / m / m / "formal" / "update_data" / b
    return read_json(d / "successful_contexts_summary.json")


def ystar_dir(seed, m, b):
    root = paths.GENERAL_SFT_SEED1 if seed == 1 else paths.general_root(seed, "leaf")
    return root / m / m / "formal" / "update_data" / b / "ystar"


def general() -> dict:
    out = {}
    for s in r.SEEDS:
        methods = {m: r.general_methods(s, m) for m in MODELS}
        roots = u.slurm_roots([x for ms in methods.values() for x in ms.values()])
        slurm = Slurm(sorted(set(roots) | {paths.GENERAL_SFT_SEED1}), stages=("formal", "formal_rewrite"))
        g = out.setdefault(str(s), {})
        for m in MODELS:
            gm = g.setdefault(m, {"data": {}, "rewrite": {}, "tests": {}, "cost": {}})
            for b in u.TRAINS:
                gm["data"][b] = {"problems": contexts_summary(s, m, b)["problems"]}
                y = ystar_dir(s, m, b)
                rs = read_json(y / "rewrite_summary.json")
                jobs = [x for x in slurm.rows if x.get("stage") == "formal_rewrite" and same_path(x["output"], y)]
                secs = sum(slurm.state[str(x["job"])][1] for x in jobs
                           if slurm.state.get(str(x["job"]), ("?", 0))[0] in ("COMPLETED", "TIMEOUT", "CANCELLED"))
                gm["rewrite"][b] = {"pairs": rs["pairs"], "kept": rs["verified"], "dropped": rs["dropped"],
                                    "by_round": [rs["verified_by_round"].get(str(i), 0) for i in range(1, 6)],
                                    "length": rs["ystar_tokens_mean"], "trunc": 100 * rs["truncated_fraction"],
                                    "gpu_h": secs / 3600,
                                    "eflops": 2 * PARAMS[m] * (rs["prompt_tokens"] + rs["completion_tokens"]) / 1e18}
            ms = methods[m]
            rows = dict(zip(ms, u.rows(m, list(ms.values()))))
            for name, (meth, res) in rows.items():
                gm["tests"][name] = {t: {"avg": 100 * res[t]["avg"], "pass": 100 * res[t]["pass"]} for t in u.OVERALL}
            costs = {meth.label: c for meth, c in u.cost_rows(m, slurm, list(ms.values()))}
            for name, meth in ms.items():
                c = costs.get(meth.label)
                if not c:
                    continue
                gm["cost"][name] = {"train_h": c["train_h"], "roll_h": sum(c["roll_h"]),
                                    "roll_h_by": c["roll_h"], "train_f": [f / 1e18 for f in c["train_f"]],
                                    "roll_f": sum(c["roll_f"]) / 1e18}
    return out


def cost_20_calls() -> dict:
    """{model: {row: {"train_h": [LCB, MATH, Sci], "total_h": sum}}} of the seed-1 baseline runs
    (``update.cost_rows`` over ``paths.GENERAL_BASELINES_SEED1``; formal jobs that COMPLETED or hit their time cap)."""
    by_label = {m.label: m for m in u.GENERAL_METHODS}
    methods = [by_label[label] for label in COST_20_CALLS_ROWS.values()]
    slurm = Slurm([paths.GENERAL_BASELINES_SEED1])
    names = {label: name for name, label in COST_20_CALLS_ROWS.items()}
    out = {}
    for m in MODELS:
        for meth, c in u.cost_rows(m, slurm, methods):
            h = c["train_h"]
            out.setdefault(m, {})[names[meth.label]] = {
                "train_h": h, "total_h": None if None in h else sum(h)}
    return out


def main() -> None:
    curves = {s: json.loads((r.REPORT_TABLES / f"curves_seed{s}.json").read_text()) for s in r.SEEDS}
    out = {"explore": explore(curves), "general": general()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(OUT)
    cost = cost_20_calls()
    COST_20_CALLS.write_text(json.dumps(cost, indent=1) + "\n")
    print("Training GPU h at 20 verifier calls per problem (LCB / MATH / Sci; total):")
    for m, rows in cost.items():
        for name, v in rows.items():
            per = " / ".join("—" if x is None else f"{x:.2f}" for x in v["train_h"])
            total = "—" if v["total_h"] is None else f"{v['total_h']:.2f}"
            print(f"  {m:9s} {name:6s} {per}; {total}")
    print(COST_20_CALLS)


if __name__ == "__main__":
    main()
