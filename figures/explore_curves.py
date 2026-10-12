"""Exploration main figure: macro-average (mean of the four benchmarks) cumulative pass rate, averaged over
Qwen3-8B and Olmo-3-7B-Instruct, against (a) verifier calls per problem, (b) cumulative PFLOPs
(log), (c) cumulative GPU wall time.

Source: the three run seeds, $REPORT_ROOT/main/curves_seed{1,2,3}.json (written by
``leaf.report.main_results``), ``<model>.<method>.pooled``. The y value is
``pass_rate_private`` (LCB / MBPP+ graded on public AND private tests; the unweighted mean of
the four benchmarks, as in the report's Overall column). At each aligned step the pass rate and the x coordinate (FLOPs, GPU h) are
averaged over the two models within a seed, then over the seeds (band: +- one standard deviation
across seeds); a method is drawn only up to the last step every model and seed has (Olmo SEED: 16
steps, 5 GPU-h cap).

    python figures/explore_curves.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import IMG_DIR, apply_style, paths, plt, save_figure, style_axes  # noqa: E402

CURVES = [paths.REPORT_MAIN / f"curves_seed{s}.json" for s in (1, 2, 3)]
MODELS = (("qwen3_8b", "Qwen3-8B"), ("olmo3_7b", "Olmo-3-7B-Instruct"))
LEAF = "leaf"
# key, label, color, linestyle, linewidth, marker, zorder
METHODS = (
    ("random_sampling", "Random Sampling", "#8A8B8C", (0, (4, 2)), 1.3, None, 3),
    ("grpo", "GRPO", "#00AEBB", "-", 1.0, "o", 4),
    ("sdpo", "SDPO", "#6A51A3", "-", 1.0, "s", 4),
    ("score", "SCoRe", "#E69F00", "-", 1.0, "^", 4),
    ("seed", "SEED", "#3A3A3A", "-", 1.0, "D", 4),
    (LEAF, "Ours", "#C00000", "-", 2.0, None, 6),
)


def load() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in CURVES]


def model_series(data: dict, model: str, key: str) -> dict:
    """Macro-average per-step series of one model: steps, pass rate (%), GPU h, PFLOPs."""
    p = data[model][key]["pooled"]
    y = p["pass_rate_private"]
    n = len(y)
    return {"steps": p["steps"][:n], "pass": [100 * v for v in y], "hours": p["hours"][:n],
            "pflops": [f / 1e15 for f in p["flops"][:n]]}


def seed_mean(seeds: list[dict], model: str, key: str) -> dict:
    """One model's series averaged over the seeds (steps every seed has)."""
    per = [model_series(data, model, key) for data in seeds]
    n = min(len(p["steps"]) for p in per)
    return {"steps": per[0]["steps"][:n],
            **{f: [statistics.mean(p[f][i] for p in per) for i in range(n)] for f in ("pass", "hours", "pflops")}}


def model_mean(data: dict, key: str) -> dict:
    """Two-model mean of one seed at each aligned step present for both models."""
    series = [model_series(data, m, key) for m, _ in MODELS]
    n = min(len(s["steps"]) for s in series)
    return {"steps": series[0]["steps"][:n],
            **{f: [sum(s[f][i] for s in series) / len(series) for i in range(n)]
               for f in ("pass", "hours", "pflops")}}


def averaged(seeds: list[dict], key: str) -> dict:
    """Mean over the seeds of the two-model curves (steps every seed has); ``pass_std`` across seeds."""
    per = [model_mean(data, key) for data in seeds]
    n = min(len(p["steps"]) for p in per)
    out = {"steps": per[0]["steps"][:n],
           **{f: [statistics.mean(p[f][i] for p in per) for i in range(n)] for f in ("pass", "hours", "pflops")}}
    out["pass_std"] = [statistics.stdev(p["pass"][i] for p in per) for i in range(n)]
    return out


PANELS = (("steps", "Verifier calls / problem", False),
          ("pflops", "Cumulative compute (PFLOPs)", True),
          ("hours", "Cumulative GPU time (h)", False))


def reach_x(s: dict, xkey: str, target: float) -> float | None:
    """x at which series ``s`` first reaches pass rate ``target`` (linear between aligned steps,
    as explore_headline.reach)."""
    y, x = s["pass"], s[xkey]
    for i, v in enumerate(y):
        if v >= target - 1e-9:
            if i == 0:
                return x[0]
            t = (target - y[i - 1]) / (v - y[i - 1])
            return x[i - 1] + t * (x[i] - x[i - 1])
    return None


def mark_speedup(ax, score: dict, ours: dict, xkey: str, ymin: float, ymax: float, logx: bool) -> float:
    """SCoRe's last point and the Ours point at the same pass rate, solid lines from each point
    down to just below a double arrow that spans them (an interval), x_SCoRe / x_leaf above it."""
    xs, ys = score[xkey][-1], score["pass"][-1]
    xo = reach_x(ours, xkey, ys)
    y_arrow = ymin + 0.44 * (ymax - ymin)  # mid-height, below the baselines of panel (b)
    for x in (xo, xs):
        ax.plot([x, x], [y_arrow - 1.5, ys], color="black", linewidth=0.7, zorder=7)
    # the Ours point is 1.5x the diameter of SCoRe's (area x 2.25) so it shows on the thick Ours line
    ax.scatter([xo, xs], [ys, ys], s=[9 * 1.5 ** 2, 9], color=["#C00000", "#E69F00"], edgecolor="black",
               linewidth=0.4, zorder=8)
    ax.annotate("", xy=(xs, y_arrow), xytext=(xo, y_arrow), zorder=8,
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2, mutation_scale=7,
                                shrinkA=0, shrinkB=0))
    xmid = (xo * xs) ** 0.5 if logx else (xo + xs) / 2
    ax.text(xmid, y_arrow + 0.9, f"{xs / xo:.2f}$\\times$", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color="#C00000", zorder=9)
    return xs / xo


def main() -> None:
    apply_style()
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.75), sharey=True)
    ymin, ymax = 58, 85.5
    for i, (ax, (xkey, xlabel, logx)) in enumerate(zip(axes, PANELS)):
        for key, label, color, ls, lw, marker, z in METHODS:
            s = averaged(data, key)
            ax.fill_between(s[xkey], [m - d for m, d in zip(s["pass"], s["pass_std"])],
                            [m + d for m, d in zip(s["pass"], s["pass_std"])], color=color, alpha=0.15,
                            linewidth=0, zorder=z - 1)
            ax.plot(s[xkey], s["pass"], color=color, linestyle=ls, linewidth=lw, label=label, zorder=z,
                    marker=marker, markersize=2.6, markevery=(3, 4), markeredgewidth=0)
        ratio = mark_speedup(ax, averaged(data, "score"), averaged(data, LEAF), xkey, ymin, ymax, logx)
        print(f"{xkey}: SCoRe / Ours = {ratio:.2f}x")
        style_axes(ax, xlabel=f"({'abc'[i]}) {xlabel}", ylabel="Pass rate (%)" if i == 0 else None, grid_axis="both")
        if logx:
            ax.set_xscale("log")
            ticks = [30, 100, 300, 1000, 3000]
            ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
            ax.minorticks_off()
        elif xkey == "steps":
            ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
            ax.set_xlim(0.5, 24.5)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([60, 65, 70, 75, 80, 85])
    handles, labels = axes[0].get_legend_handles_labels()
    order = [5, 0, 1, 2, 3, 4]
    fig.legend([handles[j] for j in order], [labels[j] for j in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, IMG_DIR, "explore_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
