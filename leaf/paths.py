"""Filesystem locations used by the experiment and report code.

Every location is derived from five root directories, each overridable through an
environment variable of the same name:

    DATA_ROOT     frozen datasets and upstream sources      (default <repo>/data)
    RESULTS_ROOT  every run output                           (default <repo>/results)
    MODEL_ROOT    base model checkpoints                     (default <repo>/models)
    REPORT_ROOT   tables, caches and figures of the report   (default $RESULTS_ROOT/report)
    CACHE_ROOT    HF / compile / Ray caches of the runners   (default $RESULTS_ROOT/cache)

Importing this module exports the resolved roots to ``os.environ`` so that child
processes and the ``${VAR}`` placeholders of ``configs/*.json`` (see :func:`load_json`)
see the same values.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _root(name: str, default: Path) -> Path:
    value = os.environ.get(name) or str(default)
    value = os.path.abspath(os.path.expanduser(value))
    os.environ[name] = value
    return Path(value)


DATA_ROOT = _root("DATA_ROOT", REPO / "data")
RESULTS_ROOT = _root("RESULTS_ROOT", REPO / "results")
MODEL_ROOT = _root("MODEL_ROOT", REPO / "models")
REPORT_ROOT = _root("REPORT_ROOT", RESULTS_ROOT / "report")
CACHE_ROOT = _root("CACHE_ROOT", RESULTS_ROOT / "cache")
ROOT_VARIABLES = ("DATA_ROOT", "RESULTS_ROOT", "MODEL_ROOT", "REPORT_ROOT", "CACHE_ROOT")

# --------------------------------------------------------------------------- configs
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_string(value: str) -> str:
    def substitute(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"config placeholder ${{{name}}} is not set in the environment")
        return os.environ[name]

    expanded = _PLACEHOLDER.sub(substitute, value)
    if "${" in expanded:
        raise ValueError(f"unexpanded placeholder left in config value: {value!r}")
    return expanded


def expand(obj):
    """Recursively replace ``${VAR}`` in every string of ``obj`` by ``os.environ[VAR]``.

    Only the braced form is expanded (a bare ``$name`` in prompt text stays as it is);
    a placeholder whose variable is unset raises.
    """
    if isinstance(obj, str):
        return _expand_string(obj)
    if isinstance(obj, dict):
        return {key: expand(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand(value) for value in obj]
    return obj


def load_json(path) -> dict:
    """Read a config file and expand its ``${VAR}`` placeholders."""
    return expand(json.loads(Path(path).read_text()))


def placeholder(path) -> str:
    """Inverse of :func:`expand` for one path: ``<root>/x`` -> ``${ROOT}/x``.

    Used where code writes a config back to ``configs/`` (``leaf.data.prepare_run_seeds``),
    so the written file stays machine independent. The most specific root wins.
    """
    text = os.path.abspath(str(path))
    roots = sorted(((name, os.environ[name]) for name in ROOT_VARIABLES),
                   key=lambda item: len(item[1]), reverse=True)
    for name, root in roots:
        if text == root:
            return f"${{{name}}}"
        if text.startswith(root.rstrip("/") + "/"):
            return f"${{{name}}}/{text[len(root.rstrip('/')) + 1:]}"
    return str(path)


# --------------------------------------------------------------------------- data
# Frozen datasets: <split>/<model>/dataset_<benchmark>.jsonl
EXPLORE_SETS = DATA_ROOT / "explore"
# Generalization test sets <m>/dataset_{lcb,math500,mbpp,sciknoweval}.jsonl: the same files
# as the exploration sets (LCB 131 / MATH-500 / MBPP+ 378 / SciKnowEval 400).
TEST_SETS = EXPLORE_SETS
# Training sets <m>/dataset_<bm>.jsonl (LCB 478 / MATH L5 500 / SciKnowEval 400)
TRAIN_SETS = DATA_ROOT / "train"
# LiveCodeBench exports
LCB_DATA = DATA_ROOT / "lcb"
# the 131-problem LCB split with its public / private test partition
LCB131_SPLIT = LCB_DATA / "lcb131_split.jsonl"
# public tests of the 131 problems: the rollout reward of LCB exploration
LCB_PUBLIC_GROUND_TRUTH = LCB_DATA / "train.parquet"
# every test of the 131 problems: the LCB generalization test
LCB_TEST_GROUND_TRUTH = LCB_DATA / "test.parquet"
# every test of the 478 LCB training problems (the training lcb_ground_truth)
LCB_TRAIN_GROUND_TRUTH = LCB_DATA / "lcb_ground_truth_full.parquet"
# Upstream sources (only needed to rebuild the frozen files or the private test suites)
RAW_DATA = DATA_ROOT / "raw"
# livecodebench/code_generation_lite release_v6 (test.jsonl .. test6.jsonl); the rows'
# verifier.evaluation_ref is resolved by file name under this directory
LCB_RAW_DIR = RAW_DATA / "code_generation_lite"
MATH500_RAW = RAW_DATA / "MATH-500"
MATH_TRAIN_RAW = RAW_DATA / "hendrycks_math"
SCIKNOWEVAL_RAW = RAW_DATA / "SciKnowEval"
MBPPPLUS_RAW = RAW_DATA / "MbppPlus-v0.2.0.jsonl"
MBPP_RAW = RAW_DATA / "mbpp_full_test.jsonl"

# --------------------------------------------------------------------------- results
# Output roots. Each holds <model>/ with jobs.jsonl, logs/, dataset_*.jsonl, rl_data/ and the
# runs under <model>/<model>/<stage>/{leaf,baselines,update,update_data}/<name>/<benchmark>,
# the tests under <model>/<model>/test/<label>/<benchmark>.
EXPLORE_ROOT = RESULTS_ROOT / "explore"
GENERAL_ROOT = RESULTS_ROOT / "general"
ABLATION_ROOT = RESULTS_ROOT / "ablation"
DATA_BUILD_ROOT = RESULTS_ROOT / "data_build"
# parts of a generalization seed: base / GRPO / SDPO / SCoRe runs (+ SEED at 20 verifier calls
# in seed 1), the SEED baseline, LEAF (rollouts, contexts, policy updates, tests; SFT on y* in
# seeds 2 / 3), and SFT on y* of seed 1
GENERAL_PARTS = ("baselines", "seed_baseline", "leaf", "sft")


def explore_root(seed: int) -> Path:
    """Test-time exploration of run seed ``seed`` (Table 1, Fig. 3): <root>/<model>/."""
    return EXPLORE_ROOT / f"seed{int(seed)}"


def general_root(seed: int, part: str) -> Path:
    """One part of the generalization experiment of run seed ``seed`` (Table 2)."""
    if part not in GENERAL_PARTS:
        raise ValueError(f"unknown generalization part {part!r} (one of {GENERAL_PARTS})")
    return GENERAL_ROOT / f"seed{int(seed)}" / part


# seed 1 also holds the interleaved-update runs (update every step)
EXPLORE_SEED1 = explore_root(1)
GENERAL_BASELINES_SEED1 = general_root(1, "baselines")
GENERAL_SEED_BASELINE_SEED1 = general_root(1, "seed_baseline")
GENERAL_LEAF_SEED1 = general_root(1, "leaf")
GENERAL_SFT_SEED1 = general_root(1, "sft")
# exploration ablations (components, interleaved updates every 2 / 5 steps)
ABLATION_EXPLORE = ABLATION_ROOT / "explore"
# history-compression ablation
ABLATION_COMPRESSION = ABLATION_ROOT / "compression"
# policy-update ablations: divergence (forward KL / JSD) and the internalization check
ABLATION_UPDATE = ABLATION_ROOT / "update"
# teacher accuracy under successful / failure / random traces, self-correction after the update
ABLATION_TRACES = ABLATION_ROOT / "traces"

# --------------------------------------------------------------------------- report
# Everything the report / figure code writes lives under REPORT_ROOT, never in a run directory.
REPORT_EXPLORE = REPORT_ROOT / "explore"
REPORT_MAIN = REPORT_ROOT / "main"
REPORT_ABLATION = REPORT_ROOT / "ablation"
REPORT_TRACES = REPORT_ROOT / "traces"
FIGURES_OUT = REPORT_ROOT / "figures"
# FLOPs of the verl runs, keyed on the run directory and its newest mtime
FLOPS_CACHE = REPORT_ROOT / "cache" / "flops_cache.json"
# tokenized SEED SFT-set sizes, keyed on the parquet's path / size / mtime
SFT_TOKENS_CACHE = REPORT_ROOT / "cache" / "sft_tokens.json"
