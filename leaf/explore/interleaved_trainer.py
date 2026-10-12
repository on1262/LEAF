"""LEAF exploration interleaved with LEAF policy updates (update every k steps).

Design
------
One run = one (model, benchmark) cell, 20 steps. Step ``t``:

1. **Rollout** -- exactly the LEAF rollout step of condition ``leaf``
   (:data:`leaf.explore.leaf_rollout.SPECS`: restart every 5 steps, one correction round,
   one public-verifier call on the step final, MATH feedback states only the
   verdict, prompt messages recorded). Every problem runs step ``t`` (a barrier,
   as in :mod:`leaf.explore.server_rollout`), the per-request seeds are the LEAF
   rollout's (``request_seed(config, step, problem_index, phase)``), the records go to
   ``samples/NNNNN_<uid>.jsonl`` in the LEAF rollout format, so the pass-rate
   tooling reads the run unchanged. Generation uses the current LoRA (verl syncs
   it into the colocated vLLM server on wake-up); the SciKnowEval judge runs on
   the base weights of the same server (``use_lora=False``).
2. **Batch** -- every problem with at least one correct, non-truncated step
   final among steps <= ``t`` of this run (distinct responses, step order:
   :func:`leaf.update.trace_selection.correct_traces`, the successful-context rule of
   the LEAF update). Its privileged context ``c`` = the recorded prompt messages of one
   of those finals, drawn uniformly per (step, uid) with the ``feedback_random``
   rng (:func:`pick_context` == :func:`leaf.update.leaf_update.teacher_messages`).
3. **Student** -- while vLLM is still awake after the rollout: one sample per
   batch problem from the plain prompt ``x = opening_messages(problem, model, "")``
   (the prompt the test eval uses), rollout temperature / top-p, response cap
   ``min(cap, max_model_len - max(|x|, |c|), ppo_max_token_len - |c|)``, seed
   ``request_seed(config, step, problem_index, STUDENT_PHASE)`` with
   ``STUDENT_PHASE = 70`` (never a rollout phase, so a restart step's opening
   sample is not duplicated). vLLM sleeps afterwards.
4. **Update (LEAF)** -- full-vocabulary reverse KL(p(.|x,y<t) || sg p(.|c,y<t))
   toward the current policy under ``c`` (``distill_reverse_kl``); MATH (task_type
   ``math``) uses the clipped reverse KL ``distill_reverse_kl_clip`` with tau = 1e-3
   (:func:`update_loss`). Every token is weighted by the truncated IS weight
   min(exp(logp_fsdp - logp_vllm), 2) of the sampled token under ``x``. Exactly
   one optimizer step per rollout step, token mean over the step (the distillation
   loss modes normalise by the step's response tokens and refuse a batch larger
   than ``ppo_mini_batch_size``, which the launcher sets to the problem count).
   No solved problem yet: no student sampling, no update (``update_skipped``).
   Update interval ``N`` (``+interleaved.update_interval``, default 1 = the
   above every step): steps 2-4 run only where ``step % N == 0``
   (:func:`update_due`), with the same batch rule (every problem solved at steps
   <= t, a context drawn per (t, uid)) and one optimizer step; the other steps
   roll out only and record ``update_deferred``.
5. **Checkpoint** every step (deferred steps included) (LoRA + optimizer + extra; the fork keeps the
   newest one). ``resume_mode=auto``: a restarted chunk restores step ``done`` and
   moves rollout records of steps > ``done`` aside (``discarded/``), removes
   ``state.json`` (rebuilt from the logs) and drops the ``step_timings`` /
   ``update_samples`` rows of those steps; step ``done + 1`` is then replayed from
   the restored LoRA with the same seeds (:func:`truncate_records`).

Outputs: ``samples/`` (rollout records), ``update_samples/step_NNNN.jsonl`` (student
samples; outside ``samples/`` so the rollout tooling never reads them),
``step_timings.jsonl`` (rollout / student / update timing, batch size, KL, IS and
memory stats), ``kv_stats.jsonl``, ``manifest.json``; after the last step
``steps/``, ``generations.jsonl`` and ``generation_summary.json`` as the LEAF
rollout driver writes them (grading: ``leaf.grading.grade_leaf_rollout``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

from leaf.core.request_seeds import read_jsonl, request_seed
from leaf.explore.leaf_rollout import (
    assemble_steps,
    build_manifest,
    read_sample_records,
    sample_record_path,
    thinking_template_kwargs,
    verifier_worker_count,
    write_summary,
)
from leaf.prompts.task_prompts import opening_messages

CONDITION = "leaf"
METHOD = "interleaved"
STEPS = 20
LEARNING_RATE = 1e-5
# LEAF update: clipped reverse KL (tau 1e-3) on MATH, reverse KL elsewhere
MATH_CLIP_TAU = 1e-3
STUDENT_PHASE = 70
# leaf.update.leaf_update.FEEDBACK_RANDOM_SEED (the feedback_random teacher-context draw)
FEEDBACK_RANDOM_SEED = 200045


# --------------------------------------------------------------------------- #
# pure parts (CPU, no verl)
# --------------------------------------------------------------------------- #


def update_due(step: int, interval: int) -> bool:
    """Whether step ``step`` runs the policy update under update interval ``interval``."""
    return int(interval) <= 1 or int(step) % int(interval) == 0


def update_counts(rows: dict[int, dict]) -> dict:
    """Optimizer steps and the skipped / deferred steps of a run's ``step_timings`` rows."""
    return {
        "optimizer_steps": sum(1 for r in rows.values()
                               if not r.get("update_skipped") and not r.get("update_deferred")),
        "skipped_updates": sorted(s for s, r in rows.items() if r.get("update_skipped")),
        "deferred_updates": sorted(s for s, r in rows.items() if r.get("update_deferred")),
    }


def update_loss(task_type: str) -> tuple[str, float | None]:
    """(verl loss mode, clip tau) of the LEAF update for one task type."""
    if task_type == "math":
        return "distill_reverse_kl_clip", MATH_CLIP_TAU
    return "distill_reverse_kl", None


def compact_row(row: dict) -> dict | None:
    """What batch selection keeps of one rollout record: a correct, untruncated final
    and the generation that produced it (with its recorded messages); None otherwise."""
    from leaf.update.trace_selection import final_generation

    final = row["final"]
    if not final.get("public_correct") or final.get("finish_reason") == "length":
        return None
    generation = final_generation(row)
    keep_final = {key: final[key] for key in ("response", "public_correct", "finish_reason",
                                              "solution_turn") if key in final}
    if "response_with_summary" in final:
        keep_final["response_with_summary"] = final["response_with_summary"]
    return {"step": int(row["step"]), "restart": bool(row.get("restart")), "problem_uid": row["problem_uid"],
            "final": keep_final, "intermediate": [dict(generation)]}


def solved_contexts(rows: list[dict], step: int) -> list[dict]:
    """The distinct correct untruncated finals of steps <= ``step`` whose recorded
    prompt ends with a user turn (the successful-context rule), in step order."""
    from leaf.update.trace_selection import correct_traces

    kept = [row for row in rows if row is not None and int(row["step"]) <= int(step)]
    traces = correct_traces(kept)
    return [t for t in traces if t["messages"] and t["messages"][-1]["role"] == "user"]


def pick_context(uid: str, step: int, contexts: list[list[dict]]) -> tuple[int, list[dict]]:
    """(index, messages) of the context drawn for (step, uid): the
    ``feedback_random`` rng of :func:`leaf.update.leaf_update.teacher_messages`."""
    if not contexts:
        raise ValueError(f"{uid}: no context to draw from")
    pick = int(np.random.default_rng([FEEDBACK_RANDOM_SEED, int(step), zlib.crc32(uid.encode())])
               .integers(len(contexts)))
    messages = contexts[pick]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{uid}: teacher messages must end with a user turn")
    return pick, [{"role": m["role"], "content": m["content"]} for m in messages]


def select_batch(histories: dict[str, list[dict]], problems: list[dict], step: int) -> list[dict]:
    """The step's update batch, in dataset order: one entry per solved problem.

    ``histories`` maps uid -> compacted rows (:func:`compact_row`, ``None`` rows
    allowed). Empty when nothing is solved yet (the update is skipped).
    """
    batch = []
    for index, problem in enumerate(problems):
        uid = problem["uid"]
        traces = solved_contexts(histories.get(uid, []), step)
        if not traces:
            continue
        pick, messages = pick_context(uid, step, [t["messages"] for t in traces])
        batch.append({"uid": uid, "problem_index": index, "contexts": len(traces),
                      "pick": pick, "context_step": traces[pick]["step"],
                      "teacher_messages": messages})
    return batch


def student_limit(*, response_cap: int, max_len: int, ppo_tokens: int,
                  prompt_tokens: int, teacher_tokens: int) -> int:
    """Student completion budget: the rollout cap, the context window under the
    longer of x / c, and the update micro-batch budget under c (packed by c + y)."""
    return max(1, min(response_cap, max_len - max(prompt_tokens, teacher_tokens),
                      ppo_tokens - teacher_tokens))


def truncate_records(output: Path, problems: list[dict], done: int) -> dict:
    """Resume: drop everything of steps after the restored checkpoint ``done``.

    Sample logs keep their records of steps <= ``done`` (the originals are copied
    to ``discarded/<time>/``), ``state.json`` is removed (rebuilt from the logs by
    :func:`leaf.explore.leaf_rollout.load_state`), ``update_samples/step_*`` and the
    ``step_timings.jsonl`` rows of later steps are dropped.
    """
    from leaf.core.io_utils import json_line

    output = Path(output)
    archive = output / "discarded" / time.strftime("%Y%m%d_%H%M%S")
    dropped = 0
    for index, problem in enumerate(problems):
        path = sample_record_path(output, index, problem["uid"])
        records = read_sample_records(path)
        kept = [row for row in records if int(row["step"]) <= done]
        if len(kept) == len(records):
            continue
        dropped += len(records) - len(kept)
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json_line(row) + "\n" for row in kept))
        temporary.replace(path)
    state = output / "state.json"
    if state.exists():
        state.unlink()
    for path in sorted((output / "update_samples").glob("step_*.jsonl")):
        if int(path.stem.split("_")[1]) > done:
            path.unlink()
    timings = output / "step_timings.jsonl"
    if timings.exists():
        rows: dict[int, dict] = {}
        for line in timings.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
        temporary = timings.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(rows[s]) + "\n" for s in sorted(rows) if s <= done))
        temporary.replace(timings)
    return {"dropped_records": dropped, "archive": str(archive) if dropped else None}


# --------------------------------------------------------------------------- #
# the trainer (verl imported lazily so the pure parts load on a login node)
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def _trainer_class():
    from leaf.explore.server_rollout import ServerRollout
    from leaf.update.leaf_update import LeafUpdateTrainer

    class InterleavedRollout(ServerRollout):
        """The server rollout with the token ids the ``tail`` carry transform reads.

        ``tail`` reads ``answer["_token_ids"]``; the per-token training fields of the
        server rollout are not needed (the update samples its own student).
        """

        async def _generate(self, **kwargs):
            answer, elapsed = await super()._generate(**kwargs)
            answer["_token_ids"] = answer.pop("_train")["token_ids"]
            return answer, elapsed

        async def run_one_step(self, step: int) -> dict[str, dict]:
            from leaf.explore.leaf_rollout import append_sample_record

            self.semaphore = asyncio.Semaphore(self.max_inflight)
            records: dict[str, dict] = {}

            async def one(index: int, problem: dict) -> None:
                uid = problem["uid"]
                if int(self.state["samples"][uid]["next_step"]) != step:
                    raise RuntimeError(f"{uid}: next_step {self.state['samples'][uid]['next_step']} != {step}")
                started = time.monotonic()
                record = await self._run_step(index, problem, step)
                record["sample_step_seconds"] = time.monotonic() - started
                append_sample_record(sample_record_path(self.output, index, uid), record)
                self.phase_seconds.update(record["phase_seconds"])
                self.sample_seconds[uid] = self.sample_seconds.get(uid, 0.0) + record["sample_step_seconds"]
                self.state["samples"][uid] = {"next_step": step + 1, "carry": record["carry"],
                                              "notes": record["notes"]}
                records[uid] = record
                self._note_completion()

            tasks = [asyncio.create_task(one(i, p), name=f"sample-{i}") for i, p in enumerate(self.problems)]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self.flush_state()
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return records

    class InterleavedTrainer(LeafUpdateTrainer):
        """``fit`` = 20 x (LEAF rollout step -> LEAF update on the solved set)."""

        def _interleaved_spec(self) -> dict:
            spec = self.config.get("interleaved", None)
            if spec is None:
                raise ValueError("InterleavedTrainer needs an `interleaved` config block")
            return dict(spec)

        async def _sample_students(self, server, step, prompts, limits, indices, concurrency):
            semaphore = asyncio.Semaphore(int(concurrency))
            rollout = self.config.actor_rollout_ref.rollout
            project = self._project

            async def one(slot: int) -> dict:
                seed = request_seed(project, step, indices[slot], STUDENT_PHASE)
                params = {"temperature": float(rollout.temperature), "top_p": float(rollout.top_p),
                          "top_k": -1, "max_tokens": int(limits[slot]), "seed": seed, "logprobs": True}
                request_id = f"interleaved:{step}:{indices[slot]}"
                started = time.monotonic()
                async with semaphore:
                    output = await server.generate.remote(
                        request_id=request_id, prompt_ids=prompts[slot], sampling_params=params)
                token_ids = list(output.token_ids)
                if output.log_probs is None or len(output.log_probs) != len(token_ids):
                    raise RuntimeError(f"vLLM returned no per-token log-probs for {request_id}")
                return {"slot": slot, "seed": seed, "token_ids": token_ids,
                        "logprobs": [float(v) for v in output.log_probs],
                        "finish_reason": "length" if len(token_ids) >= limits[slot] else "stop",
                        "seconds": time.monotonic() - started}

            return list(await asyncio.gather(*[one(slot) for slot in range(len(prompts))]))

        def fit(self):
            import torch
            from omegaconf import OmegaConf

            from verl import DataProto
            from verl.utils.tracking import Tracking

            from leaf.core.io_utils import atomic_json, json_line
            from leaf.core.project_config import load_config as load_project_config
            from leaf.explore.leaf_rollout import load_state, model_max_model_len, model_response_max_tokens
            from leaf.explore.server_rollout import KvPoller
            from leaf.update.leaf_update import GpuMemoryPoller, _right_pad, build_tensors, render_prompt

            spec = self._interleaved_spec()
            project = self._project = load_project_config()
            output = Path(spec["output"])
            output.mkdir(parents=True, exist_ok=True)
            problems = read_jsonl(Path(spec["dataset"]))
            steps = int(spec["steps"])
            model, benchmark = spec["model"], spec["benchmark"]
            task_types = {p["task_type"] for p in problems}
            if len(task_types) != 1:
                raise RuntimeError(f"one task type per cell, got {sorted(task_types)}")
            loss_mode, tau = update_loss(next(iter(task_types)))
            configured = str(self.config.actor_rollout_ref.actor.policy_loss.loss_mode)
            if configured != loss_mode:
                raise RuntimeError(f"loss_mode {configured} != LEAF update {loss_mode} for {task_types}")
            interval = int(spec.get("update_interval", 1))
            rollout_seqs = int(spec["rollout_max_num_seqs"])
            student_seqs = int(spec["student_max_num_seqs"])
            ppo_tokens = int(self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu)
            max_len = model_max_model_len(project, model)
            response_cap = model_response_max_tokens(project, model, benchmark)

            logger = Tracking(project_name=self.config.trainer.project_name,
                              experiment_name=self.config.trainer.experiment_name,
                              default_backend=self.config.trainer.logger,
                              config=OmegaConf.to_container(self.config, resolve=True))
            self.global_steps = 0
            self._load_checkpoint()
            done = int(self.global_steps)
            truncation = truncate_records(output, problems, done)
            state = load_state(output, problems)
            for uid, entry in state["samples"].items():
                if done < steps and int(entry["next_step"]) != done + 1:
                    raise RuntimeError(f"{uid}: rollout state next_step={entry['next_step']} does not "
                                       f"match the restored checkpoint (step {done})")
            histories: dict[str, list[dict]] = {}
            for index, problem in enumerate(problems):
                rows = read_sample_records(sample_record_path(output, index, problem["uid"]))
                histories[problem["uid"]] = [compact_row(row) for row in rows]

            tokenizer = self.tokenizer
            template_kwargs, template_mode = thinking_template_kwargs(tokenizer)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            student_prompts = [render_prompt(tokenizer, opening_messages(p, model, ""), template_kwargs)
                               for p in problems]
            workers = verifier_worker_count(
                {**project, "rollout_verifier_workers": project["lcb_verifier_workers"]}
                if benchmark == "lcb" and "generalization" in project else project)
            shards = int(project.get("leaf_lcb_grading_processes") or 0)
            if benchmark == "lcb" and shards:
                project = {**project, "lcb_grading_processes": shards}
            rollout_cfg = self.config.actor_rollout_ref.rollout
            manifest = build_manifest(
                config=project, condition=CONDITION, model=model, steps=steps, problems=problems,
                dataset=Path(spec["dataset"]), template_mode=template_mode, template_kwargs=template_kwargs,
                max_inflight=rollout_seqs, verifier_workers=workers)
            manifest.update({
                "kind": "interleaved", "method": f"{METHOD}_every_{max(interval, 1)}",
                "benchmark": benchmark, "setting": spec["setting"],
                "engine": "verl vLLM async server (hybrid, LoRA sync on wake-up)",
                "max_num_seqs": int(rollout_cfg.max_num_seqs), "rollout_max_num_seqs": rollout_seqs,
                "student_max_num_seqs": student_seqs,
                "vllm_gpu_memory_utilization": float(rollout_cfg.gpu_memory_utilization),
                "update": {
                    "method": "leaf_update", "loss_mode": loss_mode, "clip_tau": tau,
                    "teacher": "feedback_random over this run's correct untruncated finals of steps <= t",
                    "feedback_random_seed": FEEDBACK_RANDOM_SEED, "student_phase": STUDENT_PHASE,
                    "is_clip": 2.0,
                    **({"optimizer_steps_per_rollout_step": 1} if interval <= 1 else
                       {"optimizer_steps_per_update": 1, "update_interval": interval}),
                    "ppo_mini_batch_size": int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size),
                    "ppo_max_token_len": ppo_tokens,
                    "log_prob_max_token_len": int(rollout_cfg.log_prob_max_token_len_per_gpu),
                    "kl_row_chunk_tokens": int(self.config.actor_rollout_ref.actor.self_distillation.vocab_chunk_size),
                    "learning_rate": float(self.config.actor_rollout_ref.actor.optim.lr),
                    "learning_rate_config": spec.get("learning_rate_config"),
                    "lora_rank": int(self.config.actor_rollout_ref.model.lora_rank),
                    "lora_alpha": int(self.config.actor_rollout_ref.model.lora_alpha),
                    "temperature": float(rollout_cfg.temperature), "top_p": float(rollout_cfg.top_p),
                    "response_max_tokens": response_cap, "max_model_len": max_len,
                },
                "tuning_sources": json.loads(spec["tuning_sources"]) if spec.get("tuning_sources") else None,
                **({"update_interval": interval} if interval > 1 else {}),
                "restored_step": done, "resume_truncation": truncation,
                "eos_token_id": int(tokenizer.eos_token_id),
            })
            atomic_json(manifest, output / "manifest.json")

            manager = self.async_rollout_manager
            server = manager.server_handles[0]
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                             thread_name_prefix="interleaved-verifier")
            rollout = InterleavedRollout(
                server=server, tokenizer=tokenizer, config=project, condition=CONDITION, model=model,
                steps=steps, problems=problems, output=output, state=state, executor=executor,
                template_kwargs=template_kwargs, max_inflight=rollout_seqs, max_model_len=max_len,
                response_max_tokens=response_cap)
            poller = KvPoller(server, output / "kv_stats.jsonl")
            poller.start()
            gpu = GpuMemoryPoller()
            gpu.start()
            timings_path = output / "step_timings.jsonl"
            (output / "update_samples").mkdir(exist_ok=True)
            started_all = time.monotonic()
            try:
                for step in range(done + 1, steps + 1):
                    self.global_steps = step
                    timing: dict[str, float] = {}
                    metrics: dict[str, float] = {}
                    began = time.monotonic()
                    manager.wake_up()
                    timing["wake_up"] = time.monotonic() - began
                    began = time.monotonic()
                    poller.step = step
                    by_uid = asyncio.run(rollout.run_one_step(step))
                    poller.step = None
                    metrics.update(poller.summary(step))
                    timing["rollout"] = time.monotonic() - began
                    for problem in problems:
                        histories[problem["uid"]].append(compact_row(by_uid[problem["uid"]]))
                    accuracy = float(np.mean([bool(by_uid[p["uid"]]["final"].get("public_correct"))
                                              for p in problems]))
                    memory_rollout = gpu.take()

                    due = update_due(step, interval)
                    chosen = select_batch(histories, problems, step) if due else []
                    info: dict = {"problems": len(problems), "batch_problems": len(chosen),
                                  "update_skipped": due and not chosen}
                    if not due:
                        info["update_deferred"] = True
                    record: list[dict] = []
                    if chosen:
                        began = time.monotonic()
                        indices = [c["problem_index"] for c in chosen]
                        prompts = [student_prompts[i] for i in indices]
                        teachers = [render_prompt(tokenizer, c["teacher_messages"], template_kwargs) for c in chosen]
                        limits = [student_limit(response_cap=response_cap, max_len=max_len, ppo_tokens=ppo_tokens,
                                                prompt_tokens=len(p), teacher_tokens=len(t))
                                  for p, t in zip(prompts, teachers, strict=True)]
                        key = f"student_{step}"
                        poller.step = key
                        samples = asyncio.run(self._sample_students(server, step, prompts, limits, indices,
                                                                    student_seqs))
                        poller.step = None
                        metrics.update({f"student_{k}": v for k, v in poller.summary(key).items()})
                        timing["student_rollout"] = time.monotonic() - began
                    began = time.monotonic()
                    manager.sleep()
                    timing["sleep"] = time.monotonic() - began
                    if chosen:
                        began = time.monotonic()
                        empty = [s["slot"] for s in samples if not s["token_ids"]]
                        if empty:
                            raise RuntimeError(f"step {step}: empty student samples at slots {empty}")
                        responses = [s["token_ids"] for s in samples]
                        tensors, batch_info = build_tensors(prompts, responses, pad, teachers=teachers)
                        info.update(batch_info)
                        batch = DataProto.from_dict(tensors=tensors)
                        rollout_logp, _ = _right_pad([s["logprobs"] for s in samples], 0.0, torch.float32,
                                                     width=tensors["responses"].shape[1])
                        weights, is_metrics = self._importance_weights(batch, rollout_logp)
                        batch.batch["rollout_is_weights"] = weights
                        if tau is not None:
                            batch.meta_info["distill_clip_tau"] = tau
                        metrics.update(is_metrics)
                        timing["behaviour_log_prob"] = time.monotonic() - began
                        info.update({
                            "generation_prompt_tokens": int(sum(len(p) for p in prompts)),
                            "generation_completion_tokens": int(sum(len(r) for r in responses)),
                            "mean_student_response_tokens": float(np.mean([len(r) for r in responses])),
                            "truncated_fraction": float(np.mean([s["finish_reason"] == "length" for s in samples])),
                            "contexts_mean": float(np.mean([c["contexts"] for c in chosen])),
                        })
                        memory_before_update = gpu.take()
                        began = time.monotonic()
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        actor_output = self._update_actor(batch)
                        timing["update_actor"] = time.monotonic() - began
                        for name, value in actor_output.meta_info["metrics"].items():
                            metrics[name] = float(np.mean(value)) if isinstance(value, list) else float(value)
                        if metrics.get("update/optimizer_steps") != 1.0:
                            raise RuntimeError(f"step {step}: expected one optimizer step, got "
                                               f"{metrics.get('update/optimizer_steps')}")
                        metrics["memory/nvidia_smi_peak_mib_before_update"] = float(memory_before_update)
                        metrics["memory/nvidia_smi_peak_mib_update"] = float(gpu.take())
                        record = [{
                            "uid": c["uid"], "problem_index": c["problem_index"], "seed": s["seed"],
                            "contexts": c["contexts"], "pick": c["pick"], "context_step": c["context_step"],
                            "prompt_tokens": len(prompts[s["slot"]]), "teacher_prompt_tokens": len(teachers[s["slot"]]),
                            "limit": limits[s["slot"]], "completion_tokens": len(s["token_ids"]),
                            "finish_reason": s["finish_reason"], "seconds": s["seconds"],
                            "response": tokenizer.decode(s["token_ids"], skip_special_tokens=True),
                        } for c, s in zip(chosen, samples, strict=True)]
                    metrics["memory/nvidia_smi_peak_mib_rollout"] = float(memory_rollout)

                    began = time.monotonic()
                    self._save_checkpoint()
                    timing["save_checkpoint"] = time.monotonic() - began
                    temporary = output / "update_samples" / f"step_{step:04d}.jsonl.tmp"
                    temporary.write_text("".join(json_line(r) + "\n" for r in record))
                    temporary.replace(output / "update_samples" / f"step_{step:04d}.jsonl")
                    solved = sum(1 for p in problems if solved_contexts(histories[p["uid"]], step))
                    row = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                           "public_accuracy": accuracy, "solved_so_far": solved,
                           "update_skipped": due and not chosen,
                           "learning_rate": float(self.config.actor_rollout_ref.actor.optim.lr),
                           "timing": timing, "batch": info, "metrics": metrics}
                    if not due:
                        row["update_deferred"] = True
                    with timings_path.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(json.dumps({"event": "interleaved_step", **row}), flush=True)
                    logger.log(data={**{f"timing_s/{k}": v for k, v in timing.items()}, **metrics,
                                     "interleaved/public_accuracy": accuracy, "interleaved/batch_problems": len(chosen)},
                               step=step)
            finally:
                executor.shutdown(wait=True)
                poller.stop()
                gpu.stop()

            assemble_steps(output, problems, steps)
            summary = write_summary(output=output, steps=steps, problems=problems,
                                    phase_seconds=dict(rollout.phase_seconds),
                                    generation_seconds=time.monotonic() - started_all,
                                    sample_seconds=rollout.sample_seconds, peak_memory_mib=0)
            rows = {}
            if timings_path.exists():
                for line in timings_path.read_text().splitlines():
                    if line.strip():
                        rows[int(json.loads(line)["step"])] = json.loads(line)
            counts = update_counts(rows)
            summary["optimizer_steps"] = counts["optimizer_steps"]
            summary["skipped_updates"] = counts["skipped_updates"]
            if interval > 1:
                summary["update_interval"] = interval
                summary["deferred_updates"] = counts["deferred_updates"]
            atomic_json(summary, output / "generation_summary.json")

    InterleavedTrainer.rollout_class = InterleavedRollout
    return InterleavedTrainer


def __getattr__(name: str):
    # ``+trainer.trainer_class=leaf.explore.interleaved_trainer.InterleavedTrainer``
    if name == "InterleavedTrainer":
        return _trainer_class()
    raise AttributeError(name)
