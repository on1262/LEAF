"""LEAF policy update on top of a finished LEAF rollout (and the SFT-on-y* comparison).

The rollout (condition ``leaf``, :mod:`leaf.explore.leaf_rollout`) is finished before any
update; :mod:`leaf.update.build_contexts` turns its records into
``successful_contexts.jsonl``: per problem with a correct rollout solution, the plain
prompt and ``teacher_messages_list`` (the round prompt -- earlier attempt +
feedback -- of every correct solution). This module trains a LoRA on it, one
optimizer step per training step:

* the current policy samples one student response y per problem from the plain
  prompt x (vLLM, the LoRA synced on wake-up);
* the privileged context c is, per (step, problem), a uniformly drawn entry of
  ``teacher_messages_list`` (``teacher="feedback_random"``);
* loss = full-vocabulary divergence between p(.|x, y<t) and the stop-grad
  p(.|c, y<t), IS-weighted per token by min(exp(logp_fsdp - logp_vllm), 2),
  an exact token mean over the step (``loss_mode`` in the fork's ``dp_actor``).

Every method takes 128 problems per step for 20 steps; the divergence is resolved per
training set by :func:`method_spec`:

==========================  =====================================================
leaf_update                 reverse KL(student || teacher) on LiveCodeBench / SciKnowEval;
                            on MATH the clipped sum_v min(s_v log(s_v / t_v), 1e-3)
leaf_update_forward_kl      forward KL(teacher || student); MATH: sum_v min(t_v log(t_v / s_v), 1e-3)
leaf_update_jsd             JSD_0.5; MATH: sum_v min(d_v, 1e-3), d_v the JSD_0.5 entry
sft_ystar                   SFT (NLL) on the rewritten solutions y* (``sft_ystar.jsonl``),
                            128 rows per step, 20 steps (as the LEAF update), no vLLM sampling
==========================  =====================================================

``x`` is always the plain problem prompt, ``opening_messages(problem, model,
"")`` rendered with ``add_generation_prompt=True`` and the thinking template
kwargs -- the prompt :mod:`leaf.grading.test_eval` tests with.

Problem order: the method's problems sorted by dataset index; epoch ``e`` is the
permutation ``default_rng(schedule_seed + e)``; the epochs form one stream cut
into consecutive batches. A step's batch is a function of the step alone, so a
resumed run replays the same batch, and the per-request seeds depend only on
(step, slot), so the replayed samples come from the restored policy with the
same seeds.

The driver is a :class:`RayPPOTrainer` subclass (``trainer.trainer_class``):
verl owns the FSDP LoRA actor, the colocated vLLM server, checkpoint and resume.
"""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from leaf.core.request_seeds import read_jsonl, request_seed
from leaf.prompts.task_prompts import opening_messages

UPDATE_STEPS = 20
TRAIN_BENCHMARKS = ("lcb", "math500", "sciknoweval")
# MATH-500 takes the clipped divergence (per-entry high clip at tau), the other training sets the
# plain one.
CLIP_TAU = {"math500": 1e-3}
_DISTILL = {"rows": "successful_contexts.jsonl", "teacher": "feedback_random", "batch": 128,
            "steps": UPDATE_STEPS}
UPDATE_METHODS = {
    "leaf_update": {**_DISTILL, "loss": {"math500": "reverse_kl_clip", "lcb": "reverse_kl",
                                         "sciknoweval": "reverse_kl"}, "tau": CLIP_TAU},
    # divergence ablation: the same runs with the divergence swapped
    "leaf_update_forward_kl": {**_DISTILL, "loss": {"math500": "forward_kl_clip", "lcb": "forward_kl",
                                                    "sciknoweval": "forward_kl"}, "tau": CLIP_TAU},
    "leaf_update_jsd": {**_DISTILL, "loss": {"math500": "jsd_clip", "lcb": "jsd", "sciknoweval": "jsd"},
                        "tau": CLIP_TAU},
    # SFT on the rewritten solutions y*: one row per rewritten solution, batch = rows per step
    "sft_ystar": {"loss": "sft", "rows": "sft_ystar.jsonl", "batch": 128, "steps": UPDATE_STEPS},
}
LOSS_MODES = {"reverse_kl": "distill_reverse_kl", "reverse_kl_clip": "distill_reverse_kl_clip",
              "forward_kl": "distill_forward_kl", "forward_kl_clip": "distill_forward_kl_clip",
              "jsd": "distill_jsd", "jsd_clip": "distill_jsd_clip", "sft": "sft_nll"}
SFT_LOSSES = ("sft",)
FEEDBACK_RANDOM_SEED = 200045
IS_CLIP = 2.0
# Seed phase of an update student request (leaf.core.request_seeds.request_seed).
STUDENT_PHASE = 1


def method_spec(method: str, benchmark: str) -> dict:
    """The resolved spec of an update method on one training set:
    ``{loss, rows, teacher, batch, steps[, tau]}`` (``tau`` only for a clipped loss)."""
    if method not in UPDATE_METHODS:
        raise ValueError(f"unknown update method {method!r} (one of {sorted(UPDATE_METHODS)})")
    spec = dict(UPDATE_METHODS[method])
    loss = spec.pop("loss")
    tau = (spec.pop("tau", None) or {}).get(benchmark)
    if isinstance(loss, dict):
        if benchmark not in loss:
            raise ValueError(f"{method}: no loss for training set {benchmark!r}")
        loss = loss[benchmark]
    spec["loss"] = loss
    if tau is not None:
        spec["tau"] = tau
    return spec


# --------------------------------------------------------------------------- #
# data and schedule (CPU, no verl)
# --------------------------------------------------------------------------- #


def is_sft(method: str) -> bool:
    return UPDATE_METHODS[method]["loss"] in SFT_LOSSES


def method_rows(data_dir: Path, method: str) -> dict[str, list[dict]]:
    """Distillation: uid -> [the problem's context row], for every problem that has one.
    SFT: ``<uid>#<j>`` -> [one training row]: the schedule runs over rows, not problems."""
    spec = UPDATE_METHODS[method]
    data_dir = Path(data_dir)
    rows: dict[str, list[dict]] = {}
    for row in read_jsonl(data_dir / spec["rows"]):
        key = f"{row['uid']}#{row['j']}" if spec["loss"] in SFT_LOSSES else row["uid"]
        rows.setdefault(key, []).append(row)
    for uid, value in rows.items():
        if len(value) > 1:
            raise ValueError(f"{data_dir / spec['rows']}: {uid} has {len(value)} rows (> 1)")
    return rows


def problem_order(rows: dict[str, list[dict]]) -> list[str]:
    """The method's problem uids (SFT: row keys) in dataset order: the base of every permutation."""
    return sorted(rows, key=lambda key: (int(rows[key][0]["problem_index"]), int(rows[key][0].get("j", 0)), key))


def batch_problems(config: dict, problems: int, method: str | None = None) -> int:
    """Problems per step: the method's batch (else the config's question batch), fewer only
    if the set is smaller."""
    batch = UPDATE_METHODS[method].get("batch") if method is not None else None
    return max(1, min(int(batch or config["question_batch_size"]), problems))


def epoch_steps(problems: int, batch: int, epochs: int = 3) -> int:
    return math.ceil(epochs * problems / batch)


def plan_steps(config: dict, data_dir: Path, method: str) -> dict:
    """Formal step count of a method on one data directory (20, or the method's ``steps``)."""
    rows = method_rows(data_dir, method)
    problems = len(rows)
    if not problems:
        raise ValueError(f"{method}: no training problems in {data_dir}")
    batch = batch_problems(config, problems, method)
    return {
        "method": method, "problems": problems, "batch_problems": batch,
        "rows": sum(len(value) for value in rows.values()),
        "max_rows_per_problem": max(len(value) for value in rows.values()),
        "epoch_steps": epoch_steps(problems, batch),
        "steps": int(UPDATE_METHODS[method].get("steps", UPDATE_STEPS)),
    }


class Schedule:
    """Deterministic problem stream: permutation of epoch e seeded by ``seed + e``."""

    def __init__(self, uids: list[str], batch: int, seed: int):
        self.uids, self.batch, self.seed = list(uids), int(batch), int(seed)
        self._perms: dict[int, np.ndarray] = {}

    def permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            self._perms[epoch] = np.random.default_rng(self.seed + epoch).permutation(len(self.uids))
        return self._perms[epoch]

    def position(self, step: int) -> int:
        """Stream position of the step's first problem (steps count from 1)."""
        return (int(step) - 1) * self.batch

    def epoch_of(self, step: int) -> int:
        return self.position(step) // len(self.uids)

    def step(self, step: int) -> list[str]:
        start = self.position(step)
        out = []
        for position in range(start, start + self.batch):
            epoch, offset = divmod(position, len(self.uids))
            out.append(self.uids[int(self.permutation(epoch)[offset])])
        return out


def render_prompt(tokenizer, messages: list[dict], template_kwargs: dict) -> list[int]:
    return list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **template_kwargs,
    ))


def sft_response_ids(tokenizer, solution: str) -> list[int]:
    """The solution's tokens plus the end-of-turn token, which is trained."""
    ids = list(tokenizer(solution, add_special_tokens=False)["input_ids"])
    return ids + [int(tokenizer.eos_token_id)]


def teacher_messages(method: str, row: dict, student_messages: list[dict] | None = None,
                     step: int | None = None, seed: int = FEEDBACK_RANDOM_SEED) -> list[dict]:
    """The step's privileged context: a correct rollout solution drawn per (step, problem) with
    ``seed`` (config ``feedback_random_seed``, shifted for run seeds 2 / 3); its round prompt
    (earlier attempt + feedback). ``student_messages`` is unused."""
    if UPDATE_METHODS[method]["teacher"] != "feedback_random":
        raise ValueError(f"{method}: unknown teacher {UPDATE_METHODS[method]['teacher']}")
    choices = row["teacher_messages_list"]
    pick = int(np.random.default_rng([int(seed), int(step), zlib.crc32(row["uid"].encode())])
               .integers(len(choices)))
    messages = choices[pick]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{row['uid']}: teacher messages must end with a user turn")
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _left_pad(rows: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            ids[index, width - len(row):] = torch.tensor(row, dtype=torch.long)
            mask[index, width - len(row):] = 1
    return ids, mask


def _right_pad(rows: list[list], pad, dtype, width: int | None = None):
    width = width or max(len(row) for row in rows)
    values = torch.full((len(rows), width), pad, dtype=dtype)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            values[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = 1
    return values, mask


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """verl.utils.model.compute_position_id_with_mask without importing verl."""
    return torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)


def sequence_tensors(prompts: list[list[int]], responses: torch.Tensor,
                     response_mask: torch.Tensor, pad: int) -> dict[str, torch.Tensor]:
    """Left-padded prompts + right-padded responses, the layout verl's actor expects."""
    prompt_ids, prompt_mask = _left_pad(prompts, pad)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    return {
        "prompts": prompt_ids,
        "input_ids": torch.cat([prompt_ids, responses], dim=1),
        "attention_mask": attention_mask,
        "position_ids": position_ids(attention_mask),
    }


def build_tensors(prompts: list[list[int]], responses: list[list[int]], pad: int,
                  teachers: list[list[int]] | None = None) -> tuple[dict, dict]:
    """The update batch: student (prompt, response) rows, optionally teacher inputs.

    ``advantages`` / ``old_log_probs`` only satisfy the actor's key selection;
    the distillation / SFT loss modes read neither.
    """
    response_ids, response_mask = _right_pad(responses, pad, torch.long)
    tensors = {
        "responses": response_ids, "response_mask": response_mask,
        **sequence_tensors(prompts, response_ids, response_mask, pad),
    }
    tensors["advantages"] = response_mask.float()
    tensors["old_log_probs"] = torch.zeros(response_mask.shape, dtype=torch.float32)
    info = {
        "rows": len(responses),
        "prompt_tokens": int(sum(len(p) for p in prompts)),
        "response_tokens": int(response_mask.sum()),
        "student_tokens": int(tensors["attention_mask"].sum()),
        "max_student_len": int(tensors["attention_mask"].sum(-1).max()),
        "max_response_len": int(response_mask.sum(-1).max()),
    }
    if teachers is not None:
        teacher = sequence_tensors(teachers, response_ids, response_mask, pad)
        tensors["teacher_input_ids"] = teacher["input_ids"]
        tensors["teacher_attention_mask"] = teacher["attention_mask"]
        tensors["teacher_position_ids"] = teacher["position_ids"]
        info["teacher_tokens"] = int(teacher["attention_mask"].sum())
        info["teacher_prompt_tokens"] = int(sum(len(t) for t in teachers))
        info["max_teacher_len"] = int(teacher["attention_mask"].sum(-1).max())
    return tensors, info


def method_tau(method: str, benchmark: str) -> float | None:
    """The clip threshold of a method on one training set (None when unclipped)."""
    tau = method_spec(method, benchmark).get("tau")
    return None if tau is None else float(tau)


def loss_summary(rows: list[dict], schedule: Schedule) -> dict:
    """Per-step loss and the mean over the steps that start in each epoch."""
    by_epoch: dict[int, list[float]] = {}
    steps = []
    for row in sorted(rows, key=lambda r: int(r["step"])):
        loss = row.get("metrics", {}).get("actor/pg_loss")
        if loss is None:
            continue
        epoch = schedule.epoch_of(int(row["step"]))
        by_epoch.setdefault(epoch, []).append(float(loss))
        steps.append({"step": int(row["step"]), "epoch": epoch + 1, "loss": float(loss)})
    return {
        "steps": steps,
        "epoch_mean_loss": {str(e + 1): float(np.mean(v)) for e, v in sorted(by_epoch.items())},
        "first_loss": steps[0]["loss"] if steps else None,
        "last_loss": steps[-1]["loss"] if steps else None,
    }


def read_step_rows(path: Path, upto: int | None = None) -> list[dict]:
    """step_timings rows, the newest row per step (a replayed step overwrites)."""
    if not path.exists():
        return []
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["step"])] = row
    return [rows[s] for s in sorted(rows) if upto is None or s <= upto]


def truncate_step_rows(path: Path, done: int) -> None:
    """Drop rows of steps after the restored checkpoint; they are replayed."""
    rows = read_step_rows(path, upto=done)
    if path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temporary.replace(path)


class GpuMemoryPoller(threading.Thread):
    """Peak ``nvidia-smi`` used MiB (whole card: trainer + vLLM) per step."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True, name="gpu-poller")
        self.interval = interval
        self.peak = 0
        self._halt = threading.Event()

    @staticmethod
    def used_mib() -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
            return max(int(value) for value in out.split())
        except Exception:
            return 0

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.peak = max(self.peak, self.used_mib())

    def take(self) -> int:
        peak, self.peak = max(self.peak, self.used_mib()), 0
        return peak

    def stop(self) -> None:
        self._halt.set()



# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #


class LeafUpdateTrainer(RayPPOTrainer):
    """Replace ``fit`` with the LEAF update step loop: one optimizer step per step."""

    def _update_spec(self) -> dict:
        spec = self.config.get("leaf_update", None)
        if spec is None:
            raise ValueError("LeafUpdateTrainer needs a `leaf_update` config block")
        return dict(spec)

    # ------------------------------------------------------------------ distillation
    async def _sample_students(self, server, step: int, prompts: list[list[int]],
                               limits: list[int], project: dict) -> list[dict]:
        """One sample per problem from the current policy (LoRA synced on wake-up)."""
        semaphore = asyncio.Semaphore(int(self.config.actor_rollout_ref.rollout.max_num_seqs))
        rollout = self.config.actor_rollout_ref.rollout

        async def one(slot: int) -> dict:
            seed = request_seed(project, step, slot, STUDENT_PHASE)
            params = {
                "temperature": float(rollout.temperature), "top_p": float(rollout.top_p),
                "top_k": -1, "max_tokens": int(limits[slot]), "seed": seed, "logprobs": True,
            }
            request_id = f"update:{step}:{slot}"
            started = time.monotonic()
            async with semaphore:
                output = await server.generate.remote(
                    request_id=request_id, prompt_ids=prompts[slot], sampling_params=params,
                )
            token_ids = list(output.token_ids)
            if output.log_probs is None or len(output.log_probs) != len(token_ids):
                raise RuntimeError(f"vLLM returned no per-token log-probs for {request_id}")
            return {
                "slot": slot, "seed": seed, "request_id": request_id,
                "token_ids": token_ids, "logprobs": [float(v) for v in output.log_probs],
                "finish_reason": "length" if len(token_ids) >= limits[slot] else "stop",
                "seconds": time.monotonic() - started,
            }

        return list(await asyncio.gather(*[one(slot) for slot in range(len(prompts))]))

    def _importance_weights(self, batch: DataProto, rollout_logp: torch.Tensor):
        """min(exp(logp_fsdp - logp_vllm), 2) of each sampled token, in x."""
        behaviour = DataProto.from_dict(tensors={
            key: batch.batch[key] for key in
            ("responses", "response_mask", "input_ids", "attention_mask", "position_ids")
        })
        behaviour.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        behaviour.meta_info["skip_entropy"] = True
        behaviour.meta_info["global_token_num"] = torch.sum(
            behaviour.batch["attention_mask"], dim=-1
        ).tolist()
        output, _ = self._compute_old_log_prob(behaviour)
        fsdp_logp = output.batch["old_log_probs"].float().cpu()
        mask = behaviour.batch["response_mask"].float()
        log_ratio = torch.clamp(fsdp_logp - rollout_logp, min=-20.0, max=20.0)
        weights = torch.exp(log_ratio).clamp(max=IS_CLIP) * mask
        valid = mask.sum().clamp(min=1)
        self._last_fsdp_logp = fsdp_logp
        return weights, {
            "rollout_corr/is_weight_mean": float((weights * mask).sum() / valid),
            "rollout_corr/clipped_fraction": float(((torch.exp(log_ratio) > IS_CLIP).float() * mask).sum() / valid),
            "rollout_corr/log_ratio_abs_mean": float((log_ratio.abs() * mask).sum() / valid),
        }

    # ------------------------------------------------------------------ fit
    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        from leaf.core.io_utils import atomic_json, json_line
        from leaf.core.project_config import load_config as load_project_config
        from leaf.explore.leaf_rollout import (
            model_max_model_len, model_response_max_tokens, thinking_template_kwargs,
        )
        from leaf.explore.server_rollout import KvPoller

        spec = self._update_spec()
        method = spec["method"]
        loss = method_spec(method, spec["benchmark"])["loss"]
        project = load_project_config()
        output = Path(spec["output"])
        output.mkdir(parents=True, exist_ok=True)
        data_dir = Path(spec["data"])
        model, benchmark = spec["model"], spec["benchmark"]
        steps = int(spec["steps"])

        rows = method_rows(data_dir, method)
        uids = problem_order(rows)  # SFT: row keys <uid>#<j>
        sft = is_sft(method)
        batch_size = batch_problems(project, len(uids), method)
        schedule = Schedule(uids, batch_size, int(project["schedule_seed"]))
        problems = {row["uid"]: row for row in read_jsonl(Path(spec["dataset"]))}
        uid_of = {key: rows[key][0]["uid"] for key in uids}
        missing = [key for key in uids if uid_of[key] not in problems]
        if missing:
            raise RuntimeError(f"{len(missing)} training uids are not in {spec['dataset']}: {missing[:3]}")

        tokenizer = self.tokenizer
        template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        student_messages = {uid: opening_messages(problems[uid], model, "") for uid in set(uid_of.values())}
        student_prompts = {
            uid: render_prompt(tokenizer, messages, template_kwargs) for uid, messages in student_messages.items()
        }
        teacher_prompts = {}
        tau = method_tau(method, benchmark)
        max_len = model_max_model_len(project, model)
        response_cap = model_response_max_tokens(project, model, benchmark)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        done = int(self.global_steps)
        timings_path = output / "step_timings.jsonl"
        truncate_step_rows(timings_path, done)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        atomic_json({
            "revision": project["revision"], "kind": "leaf_update", "method": method,
            "loss": LOSS_MODES[loss], "model": model, "benchmark": benchmark,
            "data": str(data_dir), "dataset": spec["dataset"], "steps": steps,
            "problems": len(uids), "problems_per_step": batch_size,
            "rows": sum(len(value) for value in rows.values()),
            "epoch_steps": epoch_steps(len(uids), batch_size),
            "schedule_seed": int(project["schedule_seed"]),
            "feedback_random_seed": int(project.get("feedback_random_seed", FEEDBACK_RANDOM_SEED)),
            "learning_rate": float(self.config.actor_rollout_ref.actor.optim.lr),
            "temperature": float(rollout_cfg.temperature), "top_p": float(rollout_cfg.top_p),
            "response_max_tokens": response_cap, "max_model_len": max_len,
            "chat_template_thinking_mode": template_mode, "eos_token_id": int(tokenizer.eos_token_id),
            "is_clip": None if sft else IS_CLIP, "restored_step": done, "clip_tau": tau,
            "teacher_reference_rotation": not sft, "rows_file": UPDATE_METHODS[method]["rows"],
        }, output / "manifest.json")

        manager = self.async_rollout_manager
        server = manager.server_handles[0]
        poller = None
        if project.get("vllm_log_stats") and not sft:
            poller = KvPoller(server, output / "kv_stats.jsonl")
            poller.start()
        gpu = GpuMemoryPoller()
        gpu.start()
        samples_dir = output / "samples"
        samples_dir.mkdir(exist_ok=True)
        try:
            for step in range(done + 1, steps + 1):
                self.global_steps = step
                timing: dict[str, float] = {}
                metrics: dict[str, float] = {}
                batch_uids = schedule.step(step)
                began = time.monotonic()
                if sft:
                    batch, info, record = self._sft_batch(batch_uids, rows, uid_of, student_prompts, pad)
                    timing["build_batch"] = time.monotonic() - began
                    info.update({"problems": len({uid_of[k] for k in batch_uids}),
                                 "epoch": schedule.epoch_of(step) + 1,
                                 "stream_position": schedule.position(step)})
                    memory_before_update = gpu.take()
                else:
                    batch, info, record, memory_before_update = self._distill_batch(
                        step, batch_uids, method, rows, student_messages, student_prompts, teacher_prompts,
                        template_kwargs, response_cap, max_len, manager, server, poller, project, pad,
                        tau, schedule, gpu, timing, metrics)

                began = time.monotonic()
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                actor_output = self._update_actor(batch)
                timing["update_actor"] = time.monotonic() - began
                for key, value in actor_output.meta_info["metrics"].items():
                    metrics[key] = float(np.mean(value)) if isinstance(value, list) else float(value)
                if metrics.get("update/optimizer_steps") != 1.0:
                    raise RuntimeError(f"step {step}: expected one optimizer step, got "
                                       f"{metrics.get('update/optimizer_steps')}")
                metrics["memory/nvidia_smi_peak_mib_before_update"] = float(memory_before_update)
                metrics["memory/nvidia_smi_peak_mib_update"] = float(gpu.take())

                began = time.monotonic()
                self._save_checkpoint()
                timing["save_checkpoint"] = time.monotonic() - began

                temporary = samples_dir / f"step_{step:04d}.jsonl.tmp"
                temporary.write_text("".join(json_line(r) + "\n" for r in record))
                temporary.replace(samples_dir / f"step_{step:04d}.jsonl")
                row = {
                    "step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "timing": timing, "batch": info, "metrics": metrics,
                }
                with timings_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                atomic_json(loss_summary(read_step_rows(timings_path), schedule), output / "loss_summary.json")
                print(json.dumps({"event": "update_step", **row}), flush=True)
                logger.log(data={**{f"timing_s/{k}": v for k, v in timing.items()}, **metrics}, step=step)
        finally:
            gpu.stop()
            if poller is not None:
                poller.stop()

    def _sft_batch(self, keys: list[str], rows: dict, uid_of: dict, student_prompts: dict, pad: int):
        """SFT step: (plain prompt x, y* + end of turn) for each scheduled row."""
        prompts, responses, record = [], [], []
        for slot, key in enumerate(keys):
            row = rows[key][0]
            prompts.append(student_prompts[uid_of[key]])
            responses.append(sft_response_ids(self.tokenizer, row["solution"]))
            record.append({"slot": slot, "uid": uid_of[key], "j": row["j"],
                           "trace_step": row.get("trace_step"), "response_tokens": len(responses[-1])})
        tensors, info = build_tensors(prompts, responses, pad)
        return DataProto.from_dict(tensors=tensors), info, record

    def _distill_batch(self, step, batch_uids, method, rows, student_messages, student_prompts, teacher_prompts,
                    template_kwargs, response_cap, max_len, manager, server, poller, project, pad,
                    tau, schedule, gpu, timing, metrics):
        """One distillation step's batch: student samples from x, teacher prompts from the drawn contexts."""
        tokenizer = self.tokenizer
        began = time.monotonic()
        for uid in dict.fromkeys(batch_uids):
            teacher_prompts[uid] = render_prompt(
                tokenizer,
                teacher_messages(method, rows[uid][0], student_messages[uid], step,
                                 int(project.get("feedback_random_seed", FEEDBACK_RANDOM_SEED))),
                template_kwargs)
        prompts = [student_prompts[uid] for uid in batch_uids]
        teachers = [teacher_prompts[uid] for uid in batch_uids]
        limits = [
            max(1, min(response_cap, max_len - max(len(p), len(t))))
            for p, t in zip(prompts, teachers, strict=True)
        ]
        manager.wake_up()
        timing["wake_up"] = time.monotonic() - began
        began = time.monotonic()
        if poller is not None:
            poller.step = step
        samples = asyncio.run(self._sample_students(server, step, prompts, limits, project))
        if poller is not None:
            poller.step = None
            metrics.update(poller.summary(step))
        timing["rollout"] = time.monotonic() - began
        began = time.monotonic()
        manager.sleep()
        timing["sleep"] = time.monotonic() - began
        began = time.monotonic()
        empty = [s["slot"] for s in samples if not s["token_ids"]]
        if empty:
            raise RuntimeError(f"step {step}: empty student samples at slots {empty}")
        responses = [s["token_ids"] for s in samples]
        tensors, info = build_tensors(prompts, responses, pad, teachers=teachers)
        batch = DataProto.from_dict(tensors=tensors)
        rollout_logp, _ = _right_pad([s["logprobs"] for s in samples], 0.0, torch.float32,
                                     width=tensors["responses"].shape[1])
        weights, is_metrics = self._importance_weights(batch, rollout_logp)
        if tau is not None:
            batch.meta_info["distill_clip_tau"] = tau
        batch.batch["rollout_is_weights"] = weights
        metrics.update(is_metrics)
        timing["behaviour_log_prob"] = time.monotonic() - began
        info.update({
            "generation_prompt_tokens": int(sum(len(p) for p in prompts)),
            "generation_completion_tokens": int(sum(len(r) for r in responses)),
            "mean_student_response_tokens": float(np.mean([len(r) for r in responses])),
            "truncated_fraction": float(np.mean([s["finish_reason"] == "length" for s in samples])),
        })
        record = [{
            "slot": s["slot"], "uid": batch_uids[s["slot"]], "seed": s["seed"],
            "prompt_tokens": len(prompts[s["slot"]]),
            "teacher_prompt_tokens": len(teachers[s["slot"]]),
            "completion_tokens": len(s["token_ids"]), "finish_reason": s["finish_reason"],
            "seconds": s["seconds"],
            "response": tokenizer.decode(s["token_ids"], skip_special_tokens=True),
        } for s in samples]
        info.update({"problems": len(batch_uids), "epoch": schedule.epoch_of(step) + 1,
                     "stream_position": schedule.position(step)})
        memory_before_update = gpu.take()
        return batch, info, record, memory_before_update
