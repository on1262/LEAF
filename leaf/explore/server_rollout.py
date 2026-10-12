"""LEAF's rollout generating through verl's vLLM server, and the engine KV poller.

:class:`ServerRollout` runs the step logic of :class:`leaf.explore.leaf_rollout.AsyncRollout`
(same prompts, the same per-request seeds) but generates through the vLLM server handle
of a verl trainer, so the policy that generates can be updated between steps (the
interleaved-update runs, :mod:`leaf.explore.interleaved_trainer`). Each generation also
keeps what an update needs (prompt ids, sampled ids, vLLM log-probs).

:class:`KvPoller` samples the server's KV-cache usage, running / waiting requests and
preemptions while a rollout runs (config ``vllm_log_stats``).
"""

from __future__ import annotations

import asyncio
import collections
import json
import time
from pathlib import Path

import numpy as np

import threading

from leaf.explore.leaf_rollout import (
    AsyncRollout,
    request_identifier,
    sample_record_path,
)
from leaf.core.request_seeds import request_seed


def _kv_sample(server) -> dict:
    """Runs inside the verl vLLM server actor (``__ray_call__``): the engine's last
    scheduler stats as its logging stat logger holds them.

    verl's HTTP server exits right after start-up, so neither vLLM's periodic
    stats log nor its /metrics endpoint is served; the stat loggers still record
    every engine step once ``disable_log_stats=False``. ``num_preemptions`` is
    only reset by ``log()``, which nothing calls here, so it is cumulative.
    """
    # verl's runtime env sets VLLM_LOGGING_LEVEL=WARN, so vLLM creates no logging
    # stat logger; its Prometheus stat logger always exists and records every
    # engine step into this process's prometheus_client registry.
    from prometheus_client import REGISTRY

    names = {"vllm:kv_cache_usage_perc": "kv_usage", "vllm:num_requests_running": "running",
             "vllm:num_requests_waiting": "waiting", "vllm:num_preemptions_total": "preemptions"}
    row: dict = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name in names:
                row[names[sample.name]] = row.get(names[sample.name], 0.0) + float(sample.value)
    cache = server.engine.vllm_config.cache_config
    row["capacity_tokens"] = int((cache.num_gpu_blocks or 0) * cache.block_size)
    if "kv_usage" not in row:
        manager = server.engine.logger_manager
        row["error"] = "no vLLM KV gauge in the Prometheus registry"
        row["log_stats"] = bool(getattr(server.engine, "log_stats", False))
        row["stat_loggers"] = [type(x).__name__ for x in getattr(manager, "stat_loggers", [])]
    return row


class KvPoller(threading.Thread):
    """Samples the vLLM engine's scheduler stats every ``interval`` s during rollout.

    Config ``vllm_log_stats`` (which turns vLLM's stat loggers on):
    KV-cache usage, running / waiting requests and cumulative preemptions, read
    inside the server actor by :func:`_kv_sample`; one ``kv_stats.jsonl`` row
    per sample, tagged with the rollout step (``None`` between rollouts, when
    nothing is sampled). ``summary`` gives a step's peak / mean / p90 for
    ``step_timings.jsonl``.
    """

    def __init__(self, server, path: Path, interval: float = 5.0):
        super().__init__(daemon=True, name="kv-poller")
        self.server, self.path, self.interval = server, path, interval
        self.step: int | None = None
        self.samples: dict[int, list[dict]] = collections.defaultdict(list)
        self.capacity_blocks: float | None = None
        self._halt = threading.Event()

    def _scrape(self) -> dict:
        import ray

        row = ray.get(self.server.__ray_call__.remote(_kv_sample), timeout=10)
        if "capacity_tokens" in row:
            self.capacity_blocks = row["capacity_tokens"]
        return row

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            step = self.step
            if step is None:
                continue
            try:
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, **self._scrape()}
            except Exception as error:  # a missed sample must not stop the run
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "error": str(error)}
            if "kv_usage" in row:
                self.samples[step].append(row)
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    def stop(self) -> None:
        self._halt.set()

    def summary(self, step: int) -> dict:
        rows = self.samples.get(step, [])
        if not rows:
            return {}
        kv = sorted(r["kv_usage"] for r in rows)
        return {
            "kv/samples": float(len(rows)),
            "kv/usage_peak": kv[-1], "kv/usage_mean": float(np.mean(kv)),
            "kv/usage_p90": kv[int(0.9 * (len(kv) - 1))],
            "kv/running_mean": float(np.mean([r.get("running", 0) for r in rows])),
            "kv/running_max": float(max(r.get("running", 0) for r in rows)),
            "kv/waiting_share": float(np.mean([r.get("waiting", 0) > 0 for r in rows])),
            "kv/preemptions": rows[-1].get("preemptions", 0) - rows[0].get("preemptions", 0),
            "kv/capacity_tokens": self.capacity_blocks or 0.0,
        }


class ServerRollout(AsyncRollout):
    """The LEAF rollout step logic, generating through verl's vLLM server handle.

    Each generation also keeps what an update needs: the exact prompt token
    ids, the sampled token ids and vLLM's log-probability of each sampled
    token. Those ride in the intermediate rows under ``_train`` and are moved
    out before the rows are written.
    """

    def __init__(self, *, server, **kwargs):
        super().__init__(engine=None, **kwargs)
        self.server = server

    async def _generate(self, *, messages, step, problem_index, phase, max_tokens):
        prompt_ids = list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.template_kwargs,
        ))
        self.context_tokens_max = max(self.context_tokens_max, len(prompt_ids))
        limit = self._generation_limit(len(prompt_ids), max_tokens)
        seed = request_seed(self.config, step, problem_index, phase)
        params = {
            "temperature": self.temperature, "top_p": self.top_p, "top_k": -1,
            "max_tokens": limit, "seed": seed, "logprobs": True,
        }
        request_id = request_identifier(self.condition, step, problem_index, phase)
        started = time.monotonic()
        async with self.semaphore:
            output = await self.server.generate.remote(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=params,
            )
        elapsed = time.monotonic() - started
        token_ids = list(output.token_ids)
        if output.log_probs is None or len(output.log_probs) != len(token_ids):
            raise RuntimeError(f"vLLM returned no per-token log-probs for {request_id}")
        answer = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "response": self.tokenizer.decode(token_ids, skip_special_tokens=True),
            "finish_reason": "length" if len(token_ids) >= limit else "stop",
            "seed": seed,
            "request_id": request_id,
            "_train": {
                "prompt_ids": prompt_ids,
                "token_ids": token_ids,
                "logprobs": [float(value) for value in output.log_probs],
            },
        }
        return answer, elapsed

    async def _judge_generate(self, prompt_ids, params, request_id):
        """Judge requests go to the same server on the base weights, no LoRA."""
        params = dict(params)
        limit = int(params["max_tokens"])
        async with self.semaphore:
            output = await self.server.generate.remote(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=params,
                use_lora=False,
            )
        token_ids = list(output.token_ids)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, len(token_ids), "length" if len(token_ids) >= limit else "stop"

    async def run_one_step(self, step: int, previous_s1: dict) -> dict[str, dict]:
        """Roll every pending problem through ``step``; return uid -> record."""
        # Each step runs under its own asyncio.run, so loop-bound primitives
        # are created here rather than once in __init__.
        self.semaphore = asyncio.Semaphore(self.max_inflight)
        records: dict[str, dict] = {}

        async def one(index: int, problem: dict) -> None:
            uid = problem["uid"]
            if int(self.state["samples"][uid]["next_step"]) != step:
                return
            started = time.monotonic()
            record = await self._run_step(index, problem, step)
            record["sample_step_seconds"] = time.monotonic() - started
            trains = [row.pop("_train") for row in record["intermediate"]]
            if record["restart"]:
                s0, s1 = trains[0], trains[-1]
            else:
                s0, s1 = previous_s1[uid], trains[-1]
            record["train"] = {"s0": s0, "s1": s1}
            append_sample_record(
                sample_record_path(self.output, index, uid), record,
            )
            self.phase_seconds.update(record["phase_seconds"])
            self.sample_seconds[uid] = (
                self.sample_seconds.get(uid, 0.0) + record["sample_step_seconds"]
            )
            self.state["samples"][uid] = {
                "next_step": step + 1, "carry": record["carry"], "notes": record["notes"],
            }
            records[uid] = record

        tasks = [
            asyncio.create_task(one(index, problem), name=f"sample-{index}")
            for index, problem in enumerate(self.problems)
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.flush_state()
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return records


def append_sample_record(path: Path, record: dict) -> None:
    from leaf.explore.leaf_rollout import append_sample_record as append

    append(path, record)
