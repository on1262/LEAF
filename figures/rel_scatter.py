"""Per-benchmark relative scatter shared by len_explore.py and len_update.py.

For every benchmark b and method m: x = tokens(m, b) / tokens(ref, b) and y = acc(m, b) - acc(ref, b)
(pp), so every benchmark has the reference at (1, 0). Ours: red stars labelled with the benchmark,
with an arrow from the reference; baselines: smaller faded markers from ``method_style.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_style import METHOD_STYLE  # noqa: E402
from style import IMG_DIR, apply_style, save_figure, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BENCH_LABEL = {"math500": "MATH", "lcb": "LCB", "mbpp": "MBPP+", "sciknoweval": "Sci"}


def relative(points: dict, ref: str) -> dict:
    """{bench: {method: (tokens, acc)}} -> {bench: {method: (ratio, gain pp)}} without the reference."""
    out = {}
    for b, by in points.items():
        t0, a0 = by[ref]
        out[b] = {m: (t / t0, a - a0) for m, (t, a) in by.items() if m != ref}
    return out


def draw(points: dict, ref: str, name: str, xlabel: str, ylabel: str, label_offsets: dict | None = None,
         legend_loc: str = "lower right") -> dict:
    rel = relative(points, ref)
    apply_style()
    fig, ax = plt.subplots(figsize=(3.03, 2.12))
    ax.axvline(1.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    rl, rc, rm, rs = METHOD_STYLE[ref]
    ax.scatter(1.0, 0.0, s=rs, marker=rm, color=rc, edgecolor="white", linewidth=0.5, zorder=5)
    offsets = label_offsets or {}
    for b, by in rel.items():
        for m, (x, y) in by.items():
            label, color, marker, size = METHOD_STYLE[m]
            if m == "leaf":
                ax.annotate("", xy=(x, y), xytext=(1.0, 0.0), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color="#C00000", lw=0.8, alpha=0.55,
                                            mutation_scale=6, shrinkA=2, shrinkB=5))
                ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="black", linewidth=0.5, zorder=6)
                dx, dy, ha = offsets.get(b, (8, 0, "left"))
                ax.annotate(BENCH_LABEL[b], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va="center", fontsize=7.5, fontweight="bold", color="#C00000", zorder=7)
            else:
                ax.scatter(x, y, s=size * 0.6, marker=marker, color=color, alpha=0.55, edgecolor="none", zorder=4)
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, grid_axis="both")
    xs = [x for by in rel.values() for x, _ in by.values()] + [1.0]
    ys = [y for by in rel.values() for _, y in by.values()] + [0.0]
    span = max(xs) - min(xs)
    ax.set_xlim(min(xs) - 0.06 * span, max(xs) + 0.22 * span)  # room for the benchmark labels
    yspan = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.08 * yspan, max(ys) + 0.12 * yspan)
    handles = [Line2D([], [], linestyle="", marker=METHOD_STYLE[k][2], color=METHOD_STYLE[k][1],
                      markersize=7 if k == "leaf" else 4.5, markeredgecolor="black" if k == "leaf" else "none",
                      markeredgewidth=0.5, alpha=1.0 if k in ("leaf", ref) else 0.55,
                      label=METHOD_STYLE[k][0] + (" (ref.)" if k == ref else ""))
               for k in [ref] + [m for m in METHOD_STYLE if any(m in by for by in rel.values())]]
    ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=6.5, handletextpad=0.15,
              labelspacing=0.25, borderaxespad=0.2)
    save_figure(fig, IMG_DIR, name)
    for b, by in rel.items():
        print(b, {m: f"{x:.2f}x/{y:+.1f}" for m, (x, y) in by.items()})
    return rel


def combine_mean(rel: dict) -> dict:
    """{cell: {method: (ratio, gain)}} -> {method: (mean ratio, mean gain pp)}: the length ratio is taken
    per cell first (each model and benchmark), then averaged arithmetically over the cells."""
    methods = next(iter(rel.values())).keys()
    return {m: (sum(rel[c][m][0] for c in rel) / len(rel), sum(rel[c][m][1] for c in rel) / len(rel))
            for m in methods}


def combine(rel: dict) -> dict:
    """{bench: {method: (ratio, gain)}} -> {method: (geometric-mean ratio, mean gain pp)} over benchmarks."""
    methods = next(iter(rel.values())).keys()
    out = {}
    for m in methods:
        ratios = [rel[b][m][0] for b in rel]
        gains = [rel[b][m][1] for b in rel]
        geo = 1.0
        for r in ratios:
            geo *= r
        out[m] = (geo ** (1 / len(ratios)), sum(gains) / len(gains))
    return out


def draw_combined(points: dict, ref: str, name: str, xlabel: str, ylabel: str,
                  legend_loc: str = "lower right") -> dict:
    """One point per method: the per-benchmark length ratios to ``ref`` combined by geometric mean
    (x) and the per-benchmark accuracy gains averaged (y, = the Overall gain). Ours is a red star
    with an arrow from the reference at (1, 0)."""
    pts = combine(relative(points, ref))
    apply_style()
    fig, ax = plt.subplots(figsize=(3.03, 2.12))
    ax.axvline(1.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    rl, rc, rm, rs = METHOD_STYLE[ref]
    ax.scatter(1.0, 0.0, s=rs, marker=rm, color=rc, edgecolor="white", linewidth=0.5, zorder=5,
               label=f"{rl} (ref.)")
    for m in [k for k in METHOD_STYLE if k in pts]:
        x, y = pts[m]
        label, color, marker, size = METHOD_STYLE[m]
        if m == "leaf":
            ax.annotate("", xy=(x, y), xytext=(1.0, 0.0), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="#C00000", lw=0.9, alpha=0.6,
                                        mutation_scale=7, shrinkA=2, shrinkB=6))
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="black" if m == "leaf" else "white",
                   linewidth=0.5, zorder=6 if m == "leaf" else 4, label=label)
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, grid_axis="both")
    xs = [x for x, _ in pts.values()] + [1.0]
    ys = [y for _, y in pts.values()] + [0.0]
    xspan, yspan = max(xs) - min(xs), max(ys) - min(ys)
    ax.set_xlim(min(xs) - 0.1 * xspan, max(xs) + 0.1 * xspan)
    ax.set_ylim(min(ys) - 0.12 * yspan, max(ys) + 0.15 * yspan)
    ax.legend(loc=legend_loc, frameon=False, fontsize=6.5, handletextpad=0.15, labelspacing=0.25,
              borderaxespad=0.2, scatterpoints=1)
    save_figure(fig, IMG_DIR, name)
    for m, (x, y) in pts.items():
        print(f"{m:8s} length x{x:.3f}  gain {y:+.2f} pp")
    return pts
