"""Successful traces of a finished LEAF rollout and the rewrite helpers that read them.

* :func:`correct_traces`: every correct step-final (``final.public_correct``) not cut by
  the length cap, exact duplicates dropped, in step order; each with the chat messages
  it was generated from (the round prompt: the earlier attempt and its feedback). The
  successful contexts of the LEAF update (:mod:`leaf.update.build_contexts`) and of the
  interleaved updates are these round prompts.
* :func:`transcript` / :data:`ENDINGS`: how the y* rewrite (:mod:`leaf.update.rewrite_ystar`)
  shows a round prompt and words the required ending of a rewritten solution.
"""

from __future__ import annotations

ENDINGS = {
    "math": "The final answer must be exactly the same as in the correct solution: end with \\boxed{{{answer}}}.",
    "code_lcb": ("The final program must be exactly the same as in the correct solution: end with the "
                 "same complete code in a single ```python code block."),
    "sciqa": ('The final answer must be exactly the same as in the correct solution: end with '
              '"Final answer:" followed by exactly the same final answer.'),
}


def final_generation(row: dict) -> dict:
    """The intermediate record that produced a step's final (its S1).

    Kept inside the ``leaf`` package because inside the verl / Ray workers a top-level
    ``scripts`` package is the verl fork's.
    """
    turn = row["final"]["solution_turn"]
    matches = [item for item in row["intermediate"]
               if item.get("solution_turn") == turn and item["phase"] != "carry_summary"]
    if not matches:
        raise ValueError(f"{row['problem_uid']} step {row['step']}: no generation for solution_turn {turn}")
    generation = matches[-1]
    if generation["response"] != row["final"]["response"] and "response_with_summary" not in row["final"]:
        raise ValueError(f"{row['problem_uid']} step {row['step']}: final response differs from its generation")
    return generation


def correct_traces(rows: list[dict]) -> list[dict]:
    """Every distinct untruncated correct step-final with its round prompt, in step order."""

    out, seen = [], set()
    for row in sorted(rows, key=lambda r: int(r["step"])):
        final = row["final"]
        if not final.get("public_correct") or final.get("finish_reason") == "length":
            continue
        if final["response"] in seen:
            continue
        seen.add(final["response"])
        generation = final_generation(row)
        out.append({"step": int(row["step"]), "restart": bool(row.get("restart")),
                    "solution": final["response"],
                    "messages": [dict(m) for m in generation.get("messages") or []]})
    return out


def transcript(round_messages: list[dict], plain_user: str) -> str:
    """The round prompt after the problem: earlier attempt(s) and feedback."""
    parts = []
    for index, message in enumerate(round_messages):
        if message["role"] == "system":
            continue
        if index <= 1 and message["role"] == "user" and message["content"].strip() == plain_user.strip():
            continue
        if message["role"] == "user" and not parts:
            parts.append(f"[Problem prompt]\n{message['content']}")
        elif message["role"] == "assistant":
            parts.append(f"[Earlier attempt]\n{message['content']}")
        else:
            parts.append(f"[Feedback]\n{message['content']}")
    return "\n\n".join(parts)
