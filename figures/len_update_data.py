"""Test response length / truncation of the policy-update checkpoints (in-distribution tests).

Rows: the in-distribution generalization table (Base, GRPO, SDPO, SCoRe, SEED (15 instead of 20
verifier calls per problem), SFT = SFT on the verified rewrites y*, Ours = the LEAF update) and the
divergence ablation (reverse KL = the LEAF update run, forward KL, JSD). Test directories come from
``leaf.report.update`` (``GENERAL_METHODS`` / ``LOSS_ABLATION``); each holds the
``leaf.grading.test_eval`` outputs ``summary.json`` (avg@4), ``generations.jsonl``
(``completion_tokens``, ``finish_reason``) and ``generation_summary.json`` (``max_tokens``).

Per (model, method, test): avg@4, mean completion tokens over all 4 x problems responses, and
truncation rate = share of responses with finish_reason == "length" or completion_tokens >=
max_tokens. "overall" = unweighted mean over LCB v6 / MATH-500 / SciKnowEval (as Overall avg@4).

Run seeds 2 / 3 of the Table 2 rows (``leaf.report.main_results.general_methods``) go to
``seeds.<s>.<model>.<key>`` (seed 1 = the top-level ``<model>.<key>`` entries).

Writes ``$REPORT_ROOT/figures/data/len_update.json``.   python figures/len_update_data.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaf import paths  # noqa: E402
from leaf.report import update as u  # noqa: E402
from leaf.report.common import read_json, read_jsonl  # noqa: E402

OUT = paths.FIGURES_OUT / "data" / "len_update.json"
MODELS = ("qwen3_8b", "olmo3_7b")
TESTS = (("lcb", "lcb"), ("math500", "math500"), ("sciknoweval", "sciknoweval"))
BY_LABEL = {m.label: m for m in u.GENERAL_METHODS + u.LOSS_ABLATION + (u.SFT_YSTAR,)}
METHODS = {  # paper key -> report row label
    "base": "Base", "grpo": "GRPO", "sdpo": "SDPO", "score": "SCoRe", "seed": "SEED",
    "sft": "SFT", "leaf": "Ours", "leaf_forward_kl": "Ours w/ forward KL", "leaf_jsd": "Ours w/ JSD",
}


def cell(directory: Path) -> dict:
    s = read_json(directory / "summary.json")
    rows = read_jsonl(directory / "generations.jsonl")
    cap = read_json(directory / "generation_summary.json")["max_tokens"]
    toks = [r["completion_tokens"] for r in rows]
    trunc = [r["finish_reason"] == "length" or r["completion_tokens"] >= cap for r in rows]
    return {"avg": s["avg_at_k"], "pass": s["pass_at_k"], "n": len(rows), "max_tokens": cap,
            "tokens": statistics.mean(toks), "truncated": sum(trunc) / len(rows),
            "finish_length": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            "dir": str(directory)}


def summary(label: str, m, model: str) -> dict:
    cells = {t: cell(m.test_dir(model, t, train)) for t, train in TESTS}
    overall = {k: statistics.mean(c[k] for c in cells.values()) for k in ("avg", "tokens", "truncated")}
    overall["tokens_pooled"] = sum(c["tokens"] * c["n"] for c in cells.values()) / sum(c["n"] for c in cells.values())
    return {"label": label, "tests": cells, "overall": overall}


SEED_ROWS = {"base": "Base", "grpo": "GRPO", "sdpo": "SDPO", "score": "SCoRe", "seed": "SEED", "sft": "SFT",
             "leaf": "Ours"}


def main() -> None:
    from leaf.report import main_results

    out = {}
    for model in MODELS:
        out[model] = {}
        for key, label in METHODS.items():
            m = BY_LABEL[label]
            cells = {t: cell(m.test_dir(model, t, train)) for t, train in TESTS}
            overall = {k: statistics.mean(c[k] for c in cells.values()) for k in ("avg", "tokens", "truncated")}
            overall["tokens_pooled"] = sum(c["tokens"] * c["n"] for c in cells.values()) / sum(
                c["n"] for c in cells.values())
            out[model][key] = {"label": label, "tests": cells, "overall": overall}
            print(f"{model:9s} {key:9s} acc={100 * overall['avg']:.2f} tok={overall['tokens']:.0f} "
                  f"pooled={overall['tokens_pooled']:.0f} trunc={100 * overall['truncated']:.2f} | "
                  + " ".join(f"{t}:{100 * c['avg']:.2f}/{c['tokens']:.0f}/{100 * c['truncated']:.2f}"
                             f"(fl{100 * c['finish_length']:.2f})/cap{c['max_tokens']}" for t, c in cells.items()))
    out["seeds"] = {}
    for seed in (2, 3):
        for model in MODELS:
            rows = main_results.general_methods(seed, model)
            out["seeds"].setdefault(str(seed), {})[model] = {
                key: summary(row, rows[row], model) for key, row in SEED_ROWS.items()}
            for key, v in out["seeds"][str(seed)][model].items():
                o = v["overall"]
                print(f"seed{seed} {model:9s} {key:6s} acc={100 * o['avg']:.2f} tok={o['tokens']:.0f} "
                      f"trunc={100 * o['truncated']:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(OUT)


if __name__ == "__main__":
    main()
