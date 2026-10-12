"""Exploration ablations: final macro-average cumulative pass rate (Overall = mean of the four benchmarks, step 24 or
5-GPU-h cutoff) vs total exploration GPU wall time (sum over the four benchmarks), averaged over
Qwen3-8B and Olmo-3-7B-Instruct. Random Sampling is drawn as a reference point.

Sources (written by ``run/report.sh``):
  $REPORT_ROOT/ablation/curves.json                          ``ablation.<model>.<method>`` (leaf.report.ablation_tables)
  $REPORT_ROOT/explore/curves_seed1_with_interleaved.json    ``<model>.random_sampling`` (leaf.report.exploration --set explore)
y = last value of ``pooled.pass_rate_private`` (LCB / MBPP+ graded on public AND private tests,
unweighted mean of the four benchmarks, the report's Overall column); x = sum over cells of the last ``hours`` value (the
total GPU h). Also printed: the mean accuracy over the counted steps (Overall, Table tab:abl-components-mean).
``--check-paper-values`` asserts agreement with the per-model values reported in the paper (``REPORT``).

    python figures/ablation_scatter.py [--check-paper-values]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import IMG_DIR, apply_style, paths, plt, save_figure, style_axes  # noqa: E402

EXPLORE_CURVES = paths.REPORT_EXPLORE / "curves_seed1_with_interleaved.json"
ABLATION_CURVES = paths.REPORT_ABLATION / "curves.json"
MODELS = (("qwen3_8b", "Qwen3-8B"), ("olmo3_7b", "Olmo-3-7B-Instruct"))
LEAF = "leaf"

# key, short label, color, marker, size, label offset (pt), ha
POINTS = (
    (LEAF, "Ours", "#C00000", "*", 230, (0, 7), "center"),
    ("wo_last_response", "w/o resp.", "#5C1010", "v", 70, (-6, -12), "right"),
    ("wo_feedback", "w/o fb", "#D98C8C", "X", 70, (6, -11), "left"),
    ("last_3_rounds", "last 3", "#B03A3A", "D", 52, (6, -11), "left"),
    ("restart_every_2", "restart 2", "#7A0000", "^", 72, (0, 6), "center"),
    ("restart_every_10", "restart 10", "#E06666", "s", 56, (-6, 4), "right"),
    ("no_restart", "no restart", "#F4A3A3", "o", 62, (-7, -3), "right"),
    ("random_sampling", "Random Sampling", "#8A8B8C", "P", 80, (7, -3), "left"),
)
# colors follow explore_curves.py (Fig. 3): Ours #C00000 and Random Sampling #8A8B8C; every ablation is a variant
# of Ours and stays in its red family (no baseline color of Fig. 3), told apart by shade, marker and label
# restart variants get lighter as the restart interval grows: every 2 (#7A0000), every 5 = Ours (#C00000),
# every 10 (#E06666), no restart (#F4A3A3)
# restart variants in order of the restart interval (no restart, every 10, every 5 = Ours, every 2)
RESTART_TREND = ("no_restart", "restart_every_10", LEAF, "restart_every_2")

# Paper values (Overall pass rate %, total GPU h) per model, checked with --check-paper-values.
REPORT = {
    "qwen3_8b": {LEAF: (80.94, 1.47), "wo_last_response": (79.70, 1.47), "wo_feedback": (77.84, 1.56),
                 "restart_every_2": (80.86, 1.68), "restart_every_10": (80.64, 1.43), "no_restart": (79.83, 1.36),
                 "last_3_rounds": (80.63, 1.74), "random_sampling": (77.06, 0.64)},
    "olmo3_7b": {LEAF: (84.49, 6.30), "wo_last_response": (82.49, 5.77), "wo_feedback": (83.72, 7.19),
                 "restart_every_2": (85.35, 8.18), "restart_every_10": (84.94, 5.85), "no_restart": (83.80, 4.97),
                 "last_3_rounds": (84.71, 6.84), "random_sampling": (77.81, 3.39)},
}


def final(entry: dict) -> tuple[float, float, float]:
    """(Overall pass rate %, total GPU h, total PFLOPs) at the cutoff."""
    pr = 100 * entry["pooled"]["pass_rate_private"][-1]
    hours = sum(c["hours"][-1] for c in entry["cells"])
    pflops = sum(c["flops"][-1] for c in entry["cells"]) / 1e15
    return pr, hours, pflops


def mean_accuracy(entry: dict) -> float:
    """Overall mean accuracy (%) over the counted steps (LCB / MBPP+ public AND private)."""
    a = entry["pooled"].get("accuracy_private") or entry["pooled"]["accuracy"]
    return 100 * sum(a) / len(a)


def collect(check: bool = False) -> dict:
    explore = json.loads(EXPLORE_CURVES.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION_CURVES.read_text(encoding="utf-8"))["ablation"]
    out = {}
    for key, *_ in POINTS:
        per = {}
        for m, _ in MODELS:
            entry = explore[m][key] if key == "random_sampling" else ablation[m][key]
            pr, h, pf = final(entry)
            if check:
                rp, rh = REPORT[m][key]
                assert abs(pr - rp) < 0.006 and abs(h - rh) < 0.006, (m, key, pr, h, rp, rh)
            per[m] = (pr, h, pf, mean_accuracy(entry))
        out[key] = per
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-paper-values", action="store_true",
                        help="assert the per-model pass rates / GPU h equal the paper's values")
    args = parser.parse_args()
    data = collect(args.check_paper_values)
    apply_style(font_scale=1.15)
    fig, ax = plt.subplots(figsize=(3.03, 1.72))
    print(f"{'method':16s} {'Qwen PR':>8s} {'Qwen h':>7s} {'Qwen PF':>8s} {'Olmo PR':>8s} {'Olmo h':>7s} "
          f"{'Olmo PF':>8s} {'avg PR':>7s} {'avg h':>6s} {'avg PF':>7s} {'avg acc':>8s}")
    xy = {}
    for key, label, color, marker, size, off, ha in POINTS:
        q, o = data[key]["qwen3_8b"], data[key]["olmo3_7b"]
        y, x, pf, acc = ((q[i] + o[i]) / 2 for i in range(4))
        xy[key] = (x, y)
        print(f"{label:16s} {q[0]:8.2f} {q[1]:7.2f} {q[2]:8.0f} {o[0]:8.2f} {o[1]:7.2f} {o[2]:8.0f} "
              f"{y:7.2f} {x:6.2f} {pf:7.0f} {acc:8.3f}")
        ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolor="black",
                   linewidth=0.9 if key == LEAF else 0.7, zorder=6 if key == LEAF else 5)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", ha=ha, va="bottom",
                    fontsize=8.6, color="#C00000" if key == LEAF else "#222222",
                    fontweight="bold")
    # trend of the restart interval: no restart -> every 10 -> every 5 (Ours) -> every 2
    tx, ty = zip(*(xy[k] for k in RESTART_TREND))
    ax.plot(tx, ty, color="#C00000", lw=1.1, ls=(0, (3, 2)), alpha=0.7, zorder=3)
    style_axes(ax, xlabel="Exploration GPU wall time (h)", ylabel="Final pass rate (%)", grid_axis="both")
    ax.set_xlim(1.6, 5.4)
    ax.set_ylim(76.6, 84.4)
    fig.tight_layout(pad=0.2)
    save_figure(fig, IMG_DIR, "ablation_scatter")


if __name__ == "__main__":
    main()
