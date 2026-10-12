"""Two-turn SCoRe rollout loop with no correctness signal in turn two."""

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.utils.profiler import simple_timer

from leaf.prompts.correction_prompt import CORRECTION_INSTRUCTION  # noqa: F401
from leaf.prompts.task_prompts import (
    TASK_TYPES,
    correction_instruction_for_prompt,
    correction_instruction_for_task,
)


class ScoreAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_response_length = self.config.actor_rollout_ref.rollout.response_length
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom") or {}
        self.per_turn_response_length = int(
            rollout_custom.get("score_response_length", 8192)
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        # A system turn is allowed ahead of the single user turn: the OLMES
        # LiveCodeBench template carries one.
        roles = [message.get("role") for message in messages]
        if roles[-1:] != ["user"] or any(role not in {"system", "user"} for role in roles):
            raise ValueError("SCoRe expects an ordinary [system,] user input prompt.")
        prompt_ids = await self.apply_chat_template(messages)
        metrics: dict[str, float] = {}
        request_id = uuid4().hex

        first_params = dict(sampling_params)
        first_params["max_tokens"] = self.per_turn_response_length
        with simple_timer("generate_turn1", metrics):
            first = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=first_params,
            )
        first_ids = first.token_ids[: self.per_turn_response_length]
        first_text = self.tokenizer.decode(first_ids, skip_special_tokens=True)

        # Preserve the exact sampled token ids. Re-applying a full chat template
        # can normalize Qwen's prefilled <think> prefix and change those tokens.
        assistant_end_ids = self.tokenizer.encode(
            f"{self.tokenizer.eos_token}\n", add_special_tokens=False
        )
        # The benchmarks have four task types, so the correction turn is worded for
        # whichever one this prompt is; a prompt that matches none of the task
        # instructions falls back to the code wording. The row's task type when the
        # parquet carries it: the OLMES prompts do not tell SciKnowEval filling apart
        # from MATH.
        task_type = (kwargs.get("extra_info") or {}).get("task_type")
        correction_text = (
            correction_instruction_for_task(task_type)
            if task_type in TASK_TYPES
            else correction_instruction_for_prompt(
                "".join(str(row.get("content", "")) for row in messages)
            )
        )
        correction_ids = await self.apply_chat_template([
            {"role": "user", "content": correction_text}
        ])
        bridge_ids = assistant_end_ids + correction_ids
        second_prompt_ids = prompt_ids + first_ids + bridge_ids
        remaining = self.total_response_length - len(first_ids) - len(bridge_ids)
        if remaining <= 0:
            raise ValueError("SCoRe first turn leaves no room for the correction turn.")

        second_params = dict(sampling_params)
        second_params["seed"] = int(sampling_params["seed"]) + 1
        second_params["max_tokens"] = min(self.per_turn_response_length, remaining)
        with simple_timer("generate_turn2", metrics):
            second = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=second_prompt_ids,
                sampling_params=second_params,
            )
        second_ids = second.token_ids[:remaining]
        second_text = self.tokenizer.decode(second_ids, skip_special_tokens=True)

        response_ids = first_ids + bridge_ids + second_ids
        response_mask = [1] * len(first_ids) + [0] * len(bridge_ids) + [1] * len(second_ids)
        response_logprobs = None
        if first.log_probs is not None and second.log_probs is not None:
            response_logprobs = (
                first.log_probs[: len(first_ids)]
                + [0.0] * len(bridge_ids)
                + second.log_probs[: len(second_ids)]
            )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=4,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics.get("generate_turn1", 0.0)
                + metrics.get("generate_turn2", 0.0)
            ),
            extra_fields={
                "score_first_response": first_text,
                "score_second_response": second_text,
                "score_turn1_tokens": len(first_ids),
                "score_bridge_tokens": len(bridge_ids),
                "score_turn2_tokens": len(second_ids),
                "score_first_stop_reason": first.stop_reason,
                "score_second_stop_reason": second.stop_reason,
            },
        )
