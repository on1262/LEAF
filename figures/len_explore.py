"""Exploration: response length relative to Random Sampling against the final pass rate gain over Random Sampling
(pp) for the main methods of Table 1 (GRPO, SDPO, SCoRe, SEED, Ours), combined over the four
benchmarks: per benchmark ratio and gain, then the geometric mean of the ratios and the mean of the
gains (= the Overall gain). Tokens per graded
answer and the final cumulative pass rate (LCB / MBPP+ public AND private) are means over Qwen3-8B and
Olmo-3-7B-Instruct and over the three run seeds.

Reads ``$REPORT_ROOT/figures/data/len_explore.json`` (token counts, ``len_explore_data.py``) and the
exploration curves (``curves_seed1_with_interleaved.json`` / ``curves_seed{2,3}.json``, last value of each
benchmark cell);
writes ``$REPORT_ROOT/figures/len_explore.{pdf,png}``.

    python figures/len_explore.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rel_scatter import draw_combined  # noqa: E402
from style import DATA_DIR, paths  # noqa: E402

DATA = DATA_DIR / "len_explore.json"
CURVES = paths.REPORT_EXPLORE / "curves_seed1_with_interleaved.json"
MODELS = ("qwen3_8b", "olmo3_7b")
BENCHES = ("math500", "lcb", "mbpp", "sciknoweval")
KEYS = {"random_sampling": "random_sampling", "grpo": "grpo", "sdpo": "sdpo", "score": "score", "seed": "seed",
        "leaf": "leaf"}


# run seeds: (token tag prefix in len_explore.json, exploration curves)
SEEDS = (("", CURVES),
         *((f"seed{s}|", paths.REPORT_MAIN / f"curves_seed{s}.json") for s in (2, 3)))


def mean_cell(tokens: dict, curves: list[dict], m: str, b: str, method: str) -> tuple[float, float]:
    """(tokens per graded answer, final pass rate %) of one model / benchmark / method, averaged over the
    run seeds."""
    tok = acc = 0.0
    for (prefix, _), data in zip(SEEDS, curves):
        cell = tokens[f"{prefix}{m}|{method}"]["cells"][b]
        c = next(c for c in data[m][method]["cells"] if c["benchmark"] == b)
        tok += cell["resp_tokens"] / cell["answers"] / len(SEEDS)
        acc += 100 * (c.get("pass_rate_private") or c["pass_rate"])[-1] / len(SEEDS)
    return tok, acc


def cells() -> dict:
    """{"<model>|<benchmark>": {method: (tokens per graded answer, final pass rate %)}}, one cell per
    model and benchmark (seed means)."""
    tokens = json.loads(DATA.read_text())
    curves = [json.loads(path.read_text()) for _, path in SEEDS]
    return {f"{m}|{b}": {key: mean_cell(tokens, curves, m, b, method) for key, method in KEYS.items()}
            for m in MODELS for b in BENCHES}


def points() -> dict:
    tokens = json.loads(DATA.read_text())
    curves = [json.loads(path.read_text()) for _, path in SEEDS]
    return {b: {key: tuple(sum(v) / len(MODELS) for v in zip(*(mean_cell(tokens, curves, m, b, method)
                                                               for m in MODELS)))
                for key, method in KEYS.items()} for b in BENCHES}


def main() -> None:
    draw_combined(points(), "random_sampling", "len_explore", "Response length / Random Sampling", "Pass rate gain (pp)",
                  legend_loc="upper left")


if __name__ == "__main__":
    main()
