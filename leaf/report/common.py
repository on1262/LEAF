"""Shared helpers of the report pipeline: models, FLOPs accounting, file readers, Slurm lookups."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from leaf import paths

MODELS = (("qwen3_8b", "Qwen3-8B"), ("olmo3_7b", "Olmo-3-7B-Instruct"))
# non-embedding parameters N of the FLOPs estimate
PARAMS = {"qwen3_8b": 8.19e9, "olmo3_7b": 7.30e9}
PROJECT = Path(__file__).resolve().parents[2]


def flops(fwd: float, train: float, n: float) -> float:
    """2N per forward token + 6N per trained token (forward + backward)."""
    return 2 * n * fwd + 6 * n * train


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a missing file is empty and unparseable (torn) lines are skipped."""
    rows = []
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def config(name: str) -> dict:
    """A repository config (``configs/<name>``)."""
    return paths.load_json(PROJECT / "configs" / name)


# --------------------------------------------------------------------------- job logs
# the ``date '+%F %T %Z'`` line a job prints first when it starts (run/common.sh run_job)
START_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) [A-Z]{3,4}$", re.M)
# job ids allocated by run/common.sh: "local" + nanoseconds since the epoch (Slurm ids are digits)
LOCAL_JOB = re.compile(r"^local\d+$")


def is_job_id(job: str) -> bool:
    return job.isdigit() or bool(LOCAL_JOB.match(job))


def job_sort_key(job: str) -> tuple:
    """Chronological order within each id family (Slurm ids, then local ids)."""
    return (0, int(job), "") if job.isdigit() else (1, 0, job)


def same_path(a, b) -> bool:
    """Whether two run paths name the same directory (symlinks resolved)."""
    return Path(a).resolve() == Path(b).resolve()


def job_ids(cell: Path, run_root: Path) -> list[str]:
    """Job ids that ran a cell: ``launches/launch_manifest_<job>.json``; cells without
    ``launches/`` (the LEAF rollout cells) fall back to the rows of
    ``<run_root>/jobs.jsonl`` whose ``output`` is the cell and whose ``stage`` is the cell's own
    stage directory (``formal`` / ``smoke_full``: every rollout chunk), so the CPU grading jobs
    (``formal_grade``, ``formal_lcbprivate``) are not counted."""
    ids = []
    for path in sorted((cell / "launches").glob("launch_manifest_*.json")):
        job = path.stem.rsplit("_", 1)[-1]
        if is_job_id(job):
            ids.append(job)
    if not ids:
        stage = cell.parents[2].name  # <root>/<m>/<m>/<stage>/{leaf,baselines}/<name>/<benchmark>
        for row in read_jsonl(run_root / "jobs.jsonl"):
            if row.get("output") and same_path(row["output"], cell) and row.get("job") \
                    and row.get("stage") == stage:
                ids.append(str(row["job"]))
    return sorted(set(ids), key=job_sort_key)


def job_logs(job: str, run_root: Path) -> list[Path]:
    return sorted(p for p in (run_root / "logs").glob(f"*_{job}.*") if p.suffix in (".out", ".err"))


class Slurm:
    """State, elapsed seconds and GPU type of every job in ``<root>/<model>/jobs.jsonl``;
    a run's hours / GPUs count its jobs whose stage is in ``stages`` (default the formal runs; a
    smoke view adds ``smoke_full``: the run directories differ per stage, so the match stays unique).

    Jobs run by ``run/common.sh`` (local ids) carry ``state`` / ``elapsed_seconds`` / ``gpu`` in their
    own row; Slurm job ids are looked up with ``sacct`` and fall back to the row's fields when
    ``sacct`` is unavailable or does not know the job."""

    def __init__(self, roots: list[Path], stages: tuple = ("formal",)):
        self.stages = tuple(stages)
        self.rows = [r for root in roots for m, _ in MODELS for r in read_jsonl(root / m / "jobs.jsonl")]
        ids = sorted({str(r["job"]) for r in self.rows})
        self.state, self.gpu = {}, {}
        slurm_ids = [job for job in ids if job.isdigit()]
        for start in range(0, len(slurm_ids), 300):
            try:
                text = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(slurm_ids[start:start + 300]),
                                       "--format=JobID,State,ElapsedRaw,AllocTRES"],
                                      capture_output=True, text=True).stdout
            except OSError:  # no Slurm on this machine
                break
            for line in text.splitlines():
                job, state, elapsed, tres = line.split("|")
                self.state[job] = (state.split()[0], int(elapsed or 0))
                gpu = [t.split("=")[0].split(":")[-1] for t in tres.split(",") if t.startswith("gres/gpu:")]
                self.gpu[job] = gpu[0] if gpu else None
        for row in self.rows:
            job = str(row["job"])
            if job not in self.state and row.get("state"):
                self.state[job] = (str(row["state"]).split()[0], int(row.get("elapsed_seconds") or 0))
                self.gpu[job] = row.get("gpu")

    def _resolved(self, output) -> Path:
        cache = self.__dict__.setdefault("_resolved_cache", {})
        if output not in cache:
            cache[output] = Path(output).resolve()
        return cache[output]

    def _done(self, output: Path) -> list[str]:
        """Jobs of a run (stage in ``self.stages``) that did work (COMPLETED / TIMEOUT)."""
        target = self._resolved(str(output))
        jobs = [str(r["job"]) for r in self.rows
                if r.get("stage") in self.stages and self._resolved(str(r["output"])) == target]
        return [j for j in jobs if self.state.get(j, ("?", 0))[0] in ("COMPLETED", "TIMEOUT")]

    def hours(self, output: Path) -> float | None:
        """Summed elapsed hours of the run's COMPLETED / TIMEOUT jobs (None: none)."""
        used = [self.state[j][1] for j in self._done(output)]
        return sum(used) / 3600 if used else None

    def gpus(self, output: Path) -> set:
        return {self.gpu.get(j) for j in self._done(output)}
