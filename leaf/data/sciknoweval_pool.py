"""The SciKnowEval benchmark: 400 open problems graded by an LLM judge.

SciKnowEval's L3 pool is almost all multiple choice and yes/no -- formats where
repeated attempts with verdict feedback can succeed by elimination -- so the benchmark
takes open problems only:

* **pool.** SciKnowEval v2 test (``configs/data/test_sets_qwen3_8b.json`` ``sources.sciknoweval``),
  types ``open-ended-qa`` and ``relation_extraction`` (no ``mcq-*``, no
  ``true_or_false``; L2 / L5 have no ``filling``). Rows whose (instruction,
  question) pair occurs more than once are dropped, so every problem maps back
  from its prompt or verifier payload to exactly one uid.
* **sample.** Per domain (Biology / Chemistry / Material / Physics) 75 rows
  drawn uniformly from L2 and 25 from L5 with ``random.Random(f"{seed}:{domain}:{level}")``,
  ``seed = sciknoweval_sample_seed``: 400 problems, then shuffled with
  ``mix_seed`` like every other frozen benchmark file.
* **row.** ``uid = sciknoweval/o<position in the v2 test file>``; source
  ``sciknoweval``, task_type ``sciqa`` (so the SciKnowEval correction and
  improve instructions apply unchanged); ``problem`` = SciKnowEval's own task
  instruction (``prompt.default``) + the question. Verifier payload
  ``kind = "sciqa_judge"``, ``answer_type = "open"``, carrying the task text,
  the reference answer and the judge criterion (:data:`CRITERIA`), so any
  record can be judged from its payload alone.
* **prompt.** Qwen3-8B (style ``chat``): built at rollout time by
  :func:`leaf.prompts.task_prompts.base_instruction` (``SCIQA_OPEN_INSTRUCTION``).
  Olmo-3-7B (style ``olmes``): frozen ``prompt_user``,
  :func:`leaf.prompts.olmes_prompts.sciqa_open_user`.

The judge is :mod:`leaf.verifiers.llm_judge`.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

CONFIG_FLAG = "sciknoweval_open"
DOMAINS = ("Biology", "Chemistry", "Material", "Physics")
LEVEL_COUNTS = {"L2": 75, "L5": 25}
TYPES = ("open-ended-qa", "relation_extraction")

# Judge criterion per SciKnowEval subtask (``details.subtask``).
SUBTASK_CRITERION = {
    "compound_disease_relation_extraction": "relation_set",
    "drug_drug_relation_extraction": "relation_set",
    "extract_doping": "relation_set",
    "material_component_extraction": "composition",
    "text_summary": "summary",
    "material_text_summary": "summary",
    "physics_text_summary": "summary",
    "reagent_generation": "key_content",
    "procedure_generation": "key_content",
    "Electromagnetic_Derivation": "derivation",
    "Quantum_Mechanics_Derivation": "derivation",
    "General_Physics_Derivation": "derivation",
    "Classical_Mechanics": "final_result",
    "Electromagnetism": "final_result",
    "Quantum_Mechanics": "final_result",
    "Statistical_Mechanics": "final_result",
    "Electrostatics": "final_result",
    "crystal_structure_and_composition_analysis": "final_result",
    "specified_band_gap_material_generation": "final_result",
}


def enabled(config: dict) -> bool:
    return bool(config.get(CONFIG_FLAG))


def read_source(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def open_pool(rows: list[dict]) -> tuple[list[dict], dict]:
    """The eligible rows (with their file position), duplicates removed."""
    eligible = [
        {**row, "_position": position}
        for position, row in enumerate(rows)
        if row["details"]["level"] in LEVEL_COUNTS and row["type"] in TYPES
        and row["domain"] in DOMAINS
    ]
    keys = collections.Counter(
        (row["prompt"]["default"], row["question"]) for row in eligible
    )
    unique = [row for row in eligible
              if keys[(row["prompt"]["default"], row["question"])] == 1]
    unknown = {row["details"]["subtask"] for row in unique} - set(SUBTASK_CRITERION)
    if unknown:
        raise ValueError(f"SciKnowEval subtasks without a judge criterion: {sorted(unknown)}")
    audit = {
        "eligible": len(eligible),
        "dropped_duplicates": len(eligible) - len(unique),
        "pool": {
            f"{domain}/{level}": sum(
                1 for row in unique
                if row["domain"] == domain and row["details"]["level"] == level
            )
            for domain in DOMAINS for level in LEVEL_COUNTS
        },
    }
    return unique, audit


def problem_text(row: dict) -> str:
    return f"{row['prompt']['default'].strip()}\n\n{row['question'].strip()}"


def build_row(row: dict, prompt_style: str) -> dict:
    """One frozen row from an :func:`open_pool` row (no ``index`` yet)."""
    from leaf.prompts.olmes_prompts import sciqa_open_user

    if prompt_style not in ("chat", "olmes"):
        raise ValueError(f"Unknown SciKnowEval prompt style {prompt_style}")
    reference = str(row["answer"]).strip()
    if not reference:
        raise ValueError(f"SciKnowEval row {row['_position']} has no reference answer")
    subtask = row["details"]["subtask"]
    text = problem_text(row)
    out = {
        "uid": f"sciknoweval/o{row['_position']:05d}",
        "source": "sciknoweval",
        "task_type": "sciqa",
        "problem": text,
        "verifier": {
            "kind": "sciqa_judge",
            "answer_type": "open",
            "question": text,
            "reference": reference,
            "criterion": SUBTASK_CRITERION[subtask],
        },
        "meta": {
            "domain": row["domain"],
            "level": row["details"]["level"],
            "type": row["type"],
            "task": row["details"]["task"],
            "subtask": subtask,
            "source_position": row["_position"],
        },
    }
    if prompt_style == "olmes":
        out["prompt_system"] = None
        out["prompt_user"] = sciqa_open_user(text)
    return out


def build_rows(path: Path, seed: int, prompt_style: str) -> tuple[list[dict], dict]:
    """400 frozen rows (unshuffled, no ``index`` yet) and the sampling audit."""
    if prompt_style not in ("chat", "olmes"):
        raise ValueError(f"Unknown SciKnowEval prompt style {prompt_style}")
    pool, audit = open_pool(read_source(path))
    chosen = []
    for domain in DOMAINS:
        for level, count in LEVEL_COUNTS.items():
            stratum = [row for row in pool
                       if row["domain"] == domain and row["details"]["level"] == level]
            rng = random.Random(f"{seed}:{domain}:{level}")
            chosen.extend(sorted(rng.sample(stratum, count), key=lambda r: r["_position"]))
    built = [build_row(row, prompt_style) for row in chosen]
    audit["sampled"] = dict(collections.Counter(
        f"{row['meta']['domain']}/{row['meta']['level']}/{row['meta']['type']}" for row in built
    ))
    audit["criteria"] = dict(collections.Counter(row["verifier"]["criterion"] for row in built))
    audit["seed"] = seed
    return built, audit
