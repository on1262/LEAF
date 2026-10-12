"""Small helpers shared by the rollout, update and rewrite runners: JSONL reading, the
per-request decode seed, and a device-wide GPU memory monitor."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # split("\n"): splitlines() also breaks on U+0085/U+2028/U+2029, which
    # appear verbatim inside model responses.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def request_seed(config: dict, step: int, problem_index: int, phase: int) -> int:
    return int(config["decode_seed"]) + step * 100_000 + problem_index * 100 + phase


class GpuMemoryMonitor:
    """Track device-wide memory because vLLM owns CUDA in worker processes."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            values = [int(line.strip()) for line in output.splitlines() if line.strip()]
            if values:
                self.peak_mib = max(self.peak_mib, max(values))
                self.samples += 1
        except (OSError, subprocess.SubprocessError, ValueError):
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()
