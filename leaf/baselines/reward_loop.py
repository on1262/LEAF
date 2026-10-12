"""Per-sample reward inside verl's agent loop, with the legacy manager's inputs.

The generalization runs' LiveCodeBench reward runs every test case of big AtCoder suites; as
a batch step after generation it left the GPU idle for minutes per update. With
``reward_model.use_reward_loop=True`` each finished trajectory is scored right
away by a ``RewardLoopWorker`` while the rest of the batch is still decoding.

The fork's loop ``NaiveRewardManager`` builds ``extra_info`` differently from
the legacy batch manager (``verl/workers/reward_manager/naive.py``) the other
runs use; this subclass restores the legacy inputs so the reward and the
feedback string stay the same:

* ``truncated`` = no EOS in the valid response (the loop manager never sets it,
  and ``leaf.baselines.verl_reward`` appends the truncation note to the feedback);
* only the SCoRe per-turn keys and judge columns are copied in, not every agent
  ``extra_fields`` entry (e.g. ``raw_prompt``);
* ``verifier_candidate_seconds`` is recorded;
* candidates are scored in a bounded pool: ``lcb_verifier_workers`` split over
  the agent-loop workers (one reward worker each), not asyncio's default
  ``cpu_count + 4`` threads of the whole node.

Enabled only for benchmarks without an LLM-judge phase (the judge must run
before reward): see ``leaf.baselines.sdpo.reward_loop_overrides``.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import time

import numpy as np
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

LEGACY_KEYS = (
    "score_first_response", "score_second_response", "score_turn1_tokens",
    "score_bridge_tokens", "score_turn2_tokens", "score_first_stop_reason",
    "score_second_stop_reason", "judge_verdict", "score_first_judge", "score_second_judge",
)


class LegacyInputsRewardManager(NaiveRewardManager):
    _pool: concurrent.futures.ThreadPoolExecutor | None = None

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if LegacyInputsRewardManager._pool is None:
            budget = int(os.environ.get("LEAF_REWARD_CANDIDATE_WORKERS", "8"))
            agents = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
            LegacyInputsRewardManager._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, math.ceil(budget / max(1, agents))),
                thread_name_prefix="reward-loop",
            )
        return LegacyInputsRewardManager._pool

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        item = data[0]
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = item.batch["attention_mask"][-response_length:].sum()
        valid_ids = response_ids[:valid_length]

        data_source = item.non_tensor_batch["data_source"]
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(item.non_tensor_batch.get("extra_info", {}))
        tool_fields = item.non_tensor_batch.get("tool_extra_fields", None) or {}
        for key in LEGACY_KEYS:
            if key in item.non_tensor_batch:
                extra_info[key] = item.non_tensor_batch[key]
            elif key in tool_fields:
                extra_info[key] = tool_fields[key]
        extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
        extra_info["truncated"] = not (valid_ids == self.tokenizer.eos_token_id).any().item()

        response = self.tokenizer.decode(valid_ids, skip_special_tokens=True)

        def evaluate():
            started = time.monotonic()
            result = self.compute_score(
                data_source=data_source, solution_str=response,
                ground_truth=ground_truth, extra_info=extra_info,
            )
            return result, time.monotonic() - started

        result, seconds = await self.loop.run_in_executor(self._executor(), evaluate)
        if isinstance(result, dict):
            # The agent loop stacks each key into a numpy array; int / bool columns
            # would become np.int64 / np.bool_, which the trainer's rollout dump
            # cannot JSON-encode (np.float64 and np.str_ subclass float / str).
            reward_extra_info = {
                key: float(value) if isinstance(value, (bool, int, np.integer, np.bool_)) else value
                for key, value in result.items()
            }
            reward_extra_info["verifier_candidate_seconds"] = seconds
            score = result["score"]
        else:
            score = result
            reward_extra_info = {"acc": score}
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
