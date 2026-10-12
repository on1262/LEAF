"""Shared plumbing of the runners: config, paths, records, provenance, resume.

Every experiment runs over two base models and four benchmarks, so what depends on
them is a function of ``(model, mode)`` here rather than a module constant.

The canonical *record* is the one thing every component agrees on.  A
record is what :func:`leaf.verifiers.answer_grader.grade_records` consumes::

    {
      "key":         unique per (condition, trial),
      "problem_uid": dataset row uid,
      "source":      math500 | lcb | mbpp | sciknoweval,
      "task_type":   math | code_lcb | code_mbpp | sciqa,
      "problem":     the frozen problem body,
      "verifier":    the row's whole verifier payload,
      "response":    the sampled text,
      "finish_reason": vLLM stop reason, when known,
    }

Because the payload travels inside the record, the reward function needs no
dataset lookup at rollout time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from leaf import paths
from leaf.core.io_utils import balanced_schedule  # noqa: F401  (re-export)

PROJECT = Path(__file__).resolve().parents[2]
# Every experiment has its own config (prompts, decode, output root); LEAF_CONFIG
# selects it, and every entry point reads the config through this path.
CONFIG_PATH = Path(
    os.environ.get("LEAF_CONFIG") or PROJECT / "configs" / "data" / "test_sets_qwen3_8b.json"
).resolve()
# The models the config knows about, so adding one there is enough to let the
# runners accept it.
MODELS = tuple(paths.load_json(CONFIG_PATH)["models"])
# Every experiment is split into independent jobs, one per benchmark.
BENCHMARKS = ("math500", "lcb", "mbpp", "sciknoweval")
# "smoke_full" is the formal setup -- formal dataset, formal parquets, formal
# batch -- stopped after `smoke_updates` steps. A cheap "smoke" answers "does
# it run"; smoke_full answers "what does a real step cost and produce", so the
# seconds per step it measures extrapolate to the formal run without a
# problem-count correction.
MODES = ("smoke", "smoke_full", "formal")
GPU_TYPES = ("a100", "h200")
METHODS = (
    "random_sampling", "sdpo", "grpo", "score",
    "seed_collection", "seed_sft", "seed_rl",
)


def load_config() -> dict:
    return paths.load_json(CONFIG_PATH)


def root(config: dict | None = None) -> Path:
    config = config or load_config()
    return Path(config["data_root"])


def rl_data_dir(config: dict | None = None) -> Path:
    return root(config) / "rl_data"


def model_path(config: dict, model: str) -> Path:
    if model not in config["models"]:
        raise ValueError(f"Unknown model: {model}")
    return Path(config["models"][model]).resolve()


def benchmark_root(config: dict, benchmark: str) -> Path:
    """The data root one benchmark's frozen files live under.

    ``benchmark_data_roots`` (optional) points single benchmarks at another root.
    """
    override = (config.get("benchmark_data_roots") or {}).get(benchmark)
    return Path(override) if override else root(config)


def benchmark_dataset_path(config: dict, benchmark: str, mode: str) -> Path:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    prefix = "smoke_dataset" if mode == "smoke" else "dataset"
    return benchmark_root(config, benchmark) / f"{prefix}_{benchmark}.jsonl"


def parquet_mode(mode: str) -> str:
    """Which parquet family a mode reads.

    ``smoke_full`` rolls out against the formal parquets -- that is what makes
    it full -- so only the cheap ``smoke`` has parquets of its own.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return "smoke" if mode == "smoke" else "formal"


def benchmark_rl_data_dir(
    config: dict, benchmark: str, model: str | None = None
) -> Path:
    """Where one benchmark's verl parquets live.

    The prompt is baked into the parquet, so a cell whose prompt differs from
    the shared one cannot read the shared tree. Those cells -- the ones listed
    in ``variant_instruction_cells`` -- get ``rl_data/<model>/<benchmark>``
    instead,
    written by the same preparation script off the same schedule and seeds, so
    only the prompt text differs. Every other cell keeps ``rl_data/<benchmark>``
    and its frozen files.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    cells = (config.get("variant_instruction_cells") or {}).get(model or "", ())
    base = benchmark_root(config, benchmark) / "rl_data"
    if benchmark in cells:
        return base / model / benchmark
    return base / benchmark


def benchmark_budget(config: dict, benchmark: str, model: str | None = None) -> dict:
    path = benchmark_rl_data_dir(config, benchmark, model) / "budget_manifest.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing; run leaf.data.prepare_verl_data first"
        )
    return json.loads(path.read_text())


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(value, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries the pid: two processes writing the same output
    # (a manually started grade job racing the scheduled one, say) would
    # otherwise share one temp file and one would rename it out from under
    # the other.
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries the pid: two processes writing the same output
    # (a manually started grade job racing the scheduled one, say) would
    # otherwise share one temp file and one would rename it out from under
    # the other.
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


def verifier_record(row: dict, response: str, key: str, **extra) -> dict:
    """The canonical grading record for one sampled response."""
    record = {
        "key": key,
        "problem_uid": row["uid"],
        "source": row["source"],
        "task_type": row["task_type"],
        "problem": row["problem"],
        "verifier": row["verifier"],
        "response": response,
    }
    record.update(extra)
    return record


def record_from_reward_call(ground_truth: str, extra_info: dict, response: str) -> dict:
    """Rebuild a record from what verl hands a custom reward function.

    ``ground_truth`` is the JSON-encoded verifier payload written by
    :mod:`leaf.data.prepare_verl_data`; ``extra_info`` carries the problem
    identity.  Nothing else is needed, so the reward function never touches the
    dataset on disk.
    """
    payload = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    extra_info = dict(extra_info or {})
    record = {
        "key": str(extra_info.get("key") or extra_info.get("problem_uid") or "reward"),
        "problem_uid": extra_info.get("problem_uid"),
        "source": extra_info.get("source"),
        "task_type": extra_info.get("task_type"),
        "problem": extra_info.get("problem", ""),
        "verifier": payload,
        "response": response,
        "finish_reason": "length" if extra_info.get("truncated") else "stop",
    }
    # open-answer SciKnowEval: the verdict of the fork's judge step
    # (leaf.verifiers.llm_judge), judged on exactly this response.
    if extra_info.get("judge_verdict"):
        record["judge"] = dict(extra_info["judge_verdict"])
    return record


# --------------------------------------------------------------------------- #
# chat templates
# --------------------------------------------------------------------------- #

# Qwen3 reads `enable_thinking` and renders an empty <think></think> block when
# it is False; the Olmo-3 template ignores the kwarg and renders identically
# with or without it.  Passing it unconditionally therefore keeps one code path
# and the intended non-thinking behaviour on both families.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def apply_chat_template(tokenizer, content: str) -> list[int]:
    return apply_chat_messages(tokenizer, [{"role": "user", "content": content}])


def apply_chat_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Token ids for a whole chat, e.g. a system turn plus the user turn."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def lcb_raw_rows(config: dict) -> dict[str, dict]:
    """LiveCodeBench's own rows, by question_id, from the cached HF release.

    The frozen dataset keeps only the problem text of the split; prompts that
    follow LCB's own format need its raw ``question_content`` and
    ``starter_code``.
    """
    snapshot = Path(config["lcb_hf_snapshot"])
    rows = {}
    for path in sorted(snapshot.glob("test*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["question_id"]] = row
    return rows


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


# Upstream SDPO commit the shipped fork patch applies to.
SDPO_BASE_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"


def sdpo_provenance() -> dict:
    """Commit of the vendored fork and how its tree was set up.

    When the tree was snapshotted (``third_party_revisions.json`` present) the
    recorded revision is used, otherwise the live checkout is read.  When
    ``third_party/SDPO`` is not a git checkout or git fails, the pinned upstream
    commit is recorded instead.
    """
    sdpo = PROJECT / "third_party" / "SDPO"
    revisions_path = PROJECT / "third_party_revisions.json"
    if revisions_path.exists():
        revisions = json.loads(revisions_path.read_text())
        commit = revisions.get("SDPO")
        tree_mode = "copied"
    else:
        try:
            if not (sdpo / ".git").exists():  # never read an enclosing repository's HEAD
                raise OSError(f"{sdpo} is not a git checkout")
            commit = subprocess.check_output(
                ["git", "-C", str(sdpo), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            tree_mode = "live"
        except (OSError, subprocess.CalledProcessError):
            commit = SDPO_BASE_COMMIT
            tree_mode = "pinned_patch"
    if (PROJECT / "third_party" / "SDPO.patch").exists():
        tree_mode = "frozen_patch"
    return {
        "sdpo_commit": commit,
        "sdpo_source_tree_mode": tree_mode,
    }


def detect_gpu_type(default: str = "h200") -> str:
    """Pick an execution profile from whatever GPU the job actually got."""
    requested = os.environ.get("LEAF_GPU_TYPE")
    if requested in GPU_TYPES:
        return requested
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        ).splitlines()[0].lower()
    except Exception:
        return default
    for candidate in GPU_TYPES:
        if candidate in name.replace(" ", "").replace("-", ""):
            return candidate
    return default


def execution_profile(config: dict, gpu_type: str, model: str | None = None,
                      benchmark: str | None = None,
                      condition: str | None = None) -> dict:
    """The GPU's profile, with the rollout settings a cell measured for itself.

    One profile per GPU is right only while every cell holds a comparable amount
    of KV per sequence. Olmo-3-7B holds 512 KiB per token -- no GQA -- so on
    LiveCodeBench, where its responses run to 16,384 tokens, the KV cache admits
    roughly a tenth of the profile's `max_num_seqs`; the surplus requests just
    queue. Such a cell carries its own rollout settings under
    `rollout_overrides[model][benchmark]`, optionally split per condition.
    """
    if gpu_type not in GPU_TYPES:
        raise ValueError(f"Unsupported GPU type: {gpu_type}")
    profile = dict(config["sdpo_execution_profiles"][gpu_type])
    profile.setdefault(
        "ppo_micro_batch_size_per_gpu", config["ppo_micro_batch_size_per_gpu"]
    )
    overrides = dict(
        ((config.get("rollout_overrides") or {}).get(model or "", {}) or {})
        .get(benchmark or "", {})
    )
    # Conditions differ in what the trainer keeps resident beside the engine:
    # SDPO and SEED-RL also hold a teacher copy, so they can lend vLLM less of
    # the card than GRPO or SCoRe can.
    by_condition = overrides.pop("by_condition", {}) or {}
    profile.update(overrides)
    profile.update(by_condition.get(condition or "", {}))
    return profile


def launch_manifest(
    *, config: dict, method: str, model: str, benchmark: str, mode: str,
    gpu_type: str, updates: int | None, overrides: list[str],
    extra: dict | None = None,
) -> dict:
    manifest = {
        "revision": config["revision"],
        "experiment": config.get("experiment"),
        "method": method,
        "model": model,
        "benchmark": benchmark,
        "model_path": str(model_path(config, model)),
        "mode": mode,
        "gpu_type": gpu_type,
        "gpus": 1,
        "updates": updates,
        "lora_rank": config["lora_rank"],
        # run/common.sh exports RUN_JOB_ID; otherwise fall back to a scheduler job id
        "slurm_job_id": os.environ.get("RUN_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "overrides": overrides,
    }
    data_manifest = root(config) / "data_manifest.json"
    if data_manifest.exists():
        manifest["data_manifest"] = str(data_manifest)
    benchmark_dir = benchmark_rl_data_dir(config, benchmark, model)
    rl_manifest = benchmark_dir / "manifest.json"
    if rl_manifest.exists():
        manifest["rl_data_manifest"] = str(rl_manifest)
    manifest.update(sdpo_provenance())
    if extra:
        manifest.update(extra)
    return manifest


def write_launch_manifest(output: Path, manifest: dict) -> None:
    """Keep every chunk of a resumed chain, newest one at the canonical name."""
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, output / "launch_manifest.json")
    job = manifest.get("slurm_job_id") or "local"
    atomic_json(manifest, output / "launches" / f"launch_manifest_{job}.json")


def completion_marker(output: Path) -> Path:
    return Path(output) / "COMPLETE.json"


def already_complete(output: Path) -> bool:
    return completion_marker(output).exists()


def mark_complete(output: Path, summary: dict) -> None:
    """Record the run summary (read back to size formal runs from a smoke run).

    A spare chunk that finds the run already finished writes a stub summary
    carrying no tokens and ``generation_seconds: 0``. Clobbering the real
    summary with that erases the run's whole compute record, so merge onto
    whatever is already there and never let a zero or empty value win.
    """
    path = Path(output) / "generation_summary.json"
    merged = dict(summary)
    if path.exists():
        existing = json.loads(path.read_text())
        merged = {**existing, **{
            key: value for key, value in summary.items()
            if value not in (0, 0.0, "", None) or key not in existing
        }}
        # Wall time accumulates across chunks rather than replacing.
        if "generation_seconds" in existing and "generation_seconds" in summary:
            merged["generation_seconds"] = (
                float(existing["generation_seconds"])
                + float(summary["generation_seconds"])
            )
    atomic_json(merged, path)
    atomic_json(merged, completion_marker(output))


# --------------------------------------------------------------------------- #
# override-list editing
# --------------------------------------------------------------------------- #


def replace_prefix(values: list[str], prefix: str, replacement: str) -> None:
    """Replace the single override starting with ``prefix``; fail if ambiguous."""
    matches = [index for index, value in enumerate(values) if value.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one override beginning {prefix!r}, found {len(matches)}")
    values[matches[0]] = replacement


def drop_prefix(values: list[str], prefix: str) -> list[str]:
    """Remove every override starting with ``prefix``; fail if there are none."""
    remaining = [value for value in values if not value.startswith(prefix)]
    if len(remaining) == len(values):
        raise RuntimeError(f"No override begins {prefix!r}")
    return remaining
