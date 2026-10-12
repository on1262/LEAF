"""Helpers of the y* rewrite (:mod:`leaf.update.rewrite_ystar`): the MATH answer of a
solution, the cleanup of a rewrite candidate and vLLM's KV-cache metrics."""

from __future__ import annotations

import re

_SOLUTION_TAGS = re.compile(r"^\s*<solution>\s*(.*?)\s*</solution>\s*$", re.DOTALL)


def math_answer(text: str) -> str | None:
    """The last ``\\boxed{}`` of the graded view of ``text`` (None when there is none)."""
    from leaf.verifiers.answer_grader import extract_last_boxed, strip_think

    view, _ = strip_think(text, "math")
    boxed = extract_last_boxed(view)
    return boxed.strip() if boxed is not None else None


def math_reference(text: str) -> str | None:
    """The original's answer: its last box, else the verifier's fallback extraction."""
    from leaf.verifiers.math_grader import extract_boxed_or_final_answer
    from leaf.verifiers.answer_grader import strip_think

    boxed = math_answer(text)
    if boxed is not None:
        return boxed
    view, _ = strip_think(text, "math")
    return extract_boxed_or_final_answer(view)


def clean_candidate(text: str) -> str:
    """Drop a <solution>...</solution> wrapper the model may echo from the prompt."""
    match = _SOLUTION_TAGS.match(text)
    return (match.group(1) if match else text).strip()


def cache_metrics(llm) -> dict:
    """KV capacity and preemptions from vLLM's metrics (needs disable_log_stats=False)."""
    out: dict = {}
    try:
        for metric in llm.get_metrics():
            name = getattr(metric, "name", "")
            if name == "vllm:cache_config_info":
                labels = getattr(metric, "labels", {}) or {}
                blocks, size = labels.get("num_gpu_blocks"), labels.get("block_size")
                if blocks not in (None, "None") and size not in (None, "None"):
                    out["num_gpu_blocks"] = int(blocks)
                    out["block_size"] = int(size)
                    out["kv_cache_tokens"] = int(blocks) * int(size)
            elif "num_preemptions" in name:
                out["num_preemptions"] = getattr(metric, "value", None)
            elif "prefix_cache_hits" in name:
                out["prefix_cache_hit_tokens"] = getattr(metric, "value", None)
            elif "prefix_cache_queries" in name:
                out["prefix_cache_query_tokens"] = getattr(metric, "value", None)
    except Exception as error:  # noqa: BLE001 - version dependent, never fatal
        out["metrics_error"] = repr(error)
    if "kv_cache_tokens" not in out:
        try:
            cache = llm.llm_engine.vllm_config.cache_config
            if cache.num_gpu_blocks:
                out.update(num_gpu_blocks=int(cache.num_gpu_blocks),
                           block_size=int(cache.block_size),
                           kv_cache_tokens=int(cache.num_gpu_blocks) * int(cache.block_size))
        except Exception:  # noqa: BLE001
            pass
    return out
