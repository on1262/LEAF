"""Generalization: test response length relative to Base against the avg@4 gain over Base (pp) for the
methods of tab:general (GRPO, SDPO, SCoRe, SEED with equal time, SFT, Ours), combined over the three
test sets: per test set ratio and gain, then the geometric mean of the ratios and the mean of the gains
(= the Overall gain). Tokens and
avg@4 are means over the three run seeds and over Qwen3-8B and Olmo-3-7B-Instruct.

Reads ``$REPORT_ROOT/figures/data/len_update.json`` (``len_update_data.py``); writes
``$REPORT_ROOT/figures/len_update.{pdf,png}``.

    python figures/len_update.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rel_scatter import draw_combined  # noqa: E402
from style import DATA_DIR  # noqa: E402

DATA = DATA_DIR / "len_update.json"
MODELS = ("qwen3_8b", "olmo3_7b")
TESTS = ("math500", "lcb", "sciknoweval")
KEYS = ("base", "grpo", "sdpo", "score", "seed", "sft", "leaf")


def seed_runs(data: dict) -> list[dict]:
    """The three run seeds' ``{model: {key: ...}}`` (seed 1 = the top level, seeds 2 / 3 = ``seeds``)."""
    return [data] + [data["seeds"][s] for s in sorted(data.get("seeds", {}))]


def mean_cell(data: dict, m: str, k: str, t: str) -> tuple[float, float]:
    """(test tokens, avg@4 %) of one model / method / test set, averaged over the run seeds."""
    runs = seed_runs(data)
    return (sum(r[m][k]["tests"][t]["tokens"] for r in runs) / len(runs),
            sum(100 * r[m][k]["tests"][t]["avg"] for r in runs) / len(runs))


def cells() -> dict:
    """{"<model>|<test>": {method: (test tokens, avg@4 %)}}, one cell per model and test set (seed means)."""
    data = json.loads(DATA.read_text())
    return {f"{m}|{t}": {k: mean_cell(data, m, k, t) for k in KEYS} for m in MODELS for t in TESTS}


def points() -> dict:
    data = json.loads(DATA.read_text())
    return {t: {k: tuple(sum(v) / len(MODELS) for v in zip(*(mean_cell(data, m, k, t) for m in MODELS)))
                for k in KEYS} for t in TESTS}


def main() -> None:
    draw_combined(points(), "base", "len_update", "Response length / Base", "Overall avg@4 gain (pp)")


if __name__ == "__main__":
    main()
