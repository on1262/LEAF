"""Fig. 4 (single column): response length against accuracy gain, (a) exploration over Random Sampling and
(b) generalization over Base, stacked vertically with shared x and y axes and one legend above.

Each method is one point: lengths and accuracies are means over the three run seeds; the length
ratio and the accuracy gain to the reference are computed per model and benchmark first, then averaged over all model x benchmark cells (``rel_scatter.combine_mean``; the
mean gain equals the two-model mean of the Overall gain), with an arrow from the reference at (1, 0) in
the method's color. Data: ``len_explore.cells`` / ``len_update.cells``. Writes
``$REPORT_ROOT/figures/len_scatter.{pdf,png}``.

    python figures/len_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import len_explore  # noqa: E402
import len_update  # noqa: E402
from method_style import METHOD_STYLE  # noqa: E402
from rel_scatter import combine_mean, relative  # noqa: E402
from style import IMG_DIR, apply_style, save_figure, style_axes  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PANELS = (  # points, reference, panel title
    (len_explore.cells, "random_sampling", "(a) Exploration, reference: Random Sampling"),
    (len_update.cells, "base", "(b) Generalization, reference: Base"),
)


def panel(ax, pts: dict, ref: str, title: str) -> None:
    ax.axvline(1.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    _, rc, rm, rs = METHOD_STYLE[ref]
    ax.scatter(1.0, 0.0, s=rs, marker=rm, color=rc, edgecolor="white", linewidth=0.5, zorder=6)
    for m in [k for k in METHOD_STYLE if k in pts]:
        x, y = pts[m]
        _, color, marker, size = METHOD_STYLE[m]
        ours = m == "leaf"
        ax.annotate("", xy=(x, y), xytext=(1.0, 0.0), zorder=3 if ours else 2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0 if ours else 0.7,
                                    alpha=0.75 if ours else 0.6, mutation_scale=7 if ours else 5,
                                    shrinkA=0, shrinkB=5 if ours else 2.5))
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="black" if ours else "white",
                   linewidth=0.5, zorder=7 if ours else 5)
    style_axes(ax, grid_axis="both")
    ax.set_title(title, loc="left", fontsize=8, pad=3)


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.03, 3.3), sharex=True, sharey=True)
    shown, xs, ys = set(), [1.0], [0.0]
    for ax, (load, ref, title) in zip(axes, PANELS):
        pts = combine_mean(relative(load(), ref))
        panel(ax, pts, ref, title)
        shown |= set(pts) | {ref}
        xs += [x for x, _ in pts.values()]
        ys += [y for _, y in pts.values()]
        for m, (x, y) in pts.items():
            print(f"{ref:8s} {m:6s} length x{x:.3f}  gain {y:+.2f} pp")
    xspan, yspan = max(xs) - min(xs), max(ys) - min(ys)
    axes[0].set_xlim(min(xs) - 0.06 * xspan, max(xs) + 0.06 * xspan)
    axes[0].set_ylim(min(ys) - 0.12 * yspan, max(ys) + 0.15 * yspan)
    axes[1].set_xlabel("Response length / reference", labelpad=2)
    fig.supylabel("Accuracy gain over reference (pp)", fontsize=9.0, fontweight="bold", x=0.02)
    handles = [Line2D([], [], linestyle="", marker=METHOD_STYLE[k][2], color=METHOD_STYLE[k][1],
                      markersize=7.5 if k == "leaf" else 5, markeredgecolor="black" if k == "leaf" else "white",
                      markeredgewidth=0.5, label=METHOD_STYLE[k][0])
               for k in METHOD_STYLE if k in shown]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False,
               fontsize=7, handletextpad=0.2, columnspacing=0.9, labelspacing=0.3, borderaxespad=0.0)
    fig.tight_layout(h_pad=0.4)
    save_figure(fig, IMG_DIR, "len_scatter")


if __name__ == "__main__":
    main()
