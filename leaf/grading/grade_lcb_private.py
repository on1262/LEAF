"""Regrade existing LiveCodeBench cells with LCB's private tests.

Nothing is regenerated. For every cell the script rebuilds exactly the aligned
24-step grid its ``rollout_summary.json`` was computed from (refusing the cell
if the public numbers do not reproduce), keeps each entry's public verdict, runs
the entry's program on the problem's private tests with the official
``lcb_runner`` grader, and writes ``rollout_summary_lcbprivate.json`` +
``grades_lcbprivate.jsonl`` next to it (see ``leaf.verifiers.lcb_private``).

Cell kinds (from the directory layout ``.../{leaf,baselines}/<condition>/lcb``):
* every condition under ``leaf/``: LEAF rollout driver records;
* ``random_sampling``: records with ``aligned_step``;
* ``sdpo`` / ``grpo`` / ``score`` / ``seed_rl``: verl rollouts through
  ``leaf.grading.grade_baseline.verl_steps`` (SEED's first epoch from
  ``seed_collection``; SCoRe's two turns split and re-extracted, then re-checked
  on the public tests).

Usage (CPU job, one config per invocation -- Qwen cells: configs/data/test_sets_qwen3_8b.json,
Olmo cells: configs/data/test_sets_olmo3_7b.json)::

    python -m leaf.grading.grade_lcb_private --config configs/data/test_sets_qwen3_8b.json \
        --cells DIR [DIR ...] --workers 20
    python -m leaf.grading.grade_lcb_private --config configs/data/test_sets_qwen3_8b.json \
        --sanity --cells DIR [DIR ...] --workers 20
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VERL = ("sdpo", "grpo", "score", "seed_rl")


def cell_kind(cell: Path) -> tuple[str, str]:
    condition = cell.parent.name
    # the kind directory: <stage>/leaf/<condition>/<benchmark> holds LEAF rollout cells
    # (every exploration condition, with or without interleaved updates)
    if cell.parent.parent.name == "leaf":
        return "leaf", condition
    if condition == "random_sampling":
        return "random_sampling", condition
    if condition in VERL:
        return "verl", condition
    raise ValueError(f"Unknown condition directory {cell}")


def load_everything(config_path: str, workers: int):
    os.environ["LEAF_CONFIG"] = str(Path(config_path).resolve())
    sys.path.insert(0, str(PROJECT))
    from leaf.verifiers import lcb_private as lp
    from leaf.core.project_config import load_config, root
    from leaf.verifiers.answer_grader import Grader

    config = load_config()
    dataset_rows = lp.read_jsonl(root(config) / "dataset_lcb.jsonl")
    scratch = lp.ground_truth_scratch()
    ground_truth = lp.build_ground_truth(dataset_rows, Path(config["lcb_hf_snapshot"]), scratch)
    print(json.dumps({"event": "lcbprivate_ground_truth", "counts": ground_truth["counts"],
                      "seconds": round(ground_truth["seconds"], 1),
                      "bytes": sum(t["bytes"] for t in ground_truth["tasks"].values())}),
          flush=True)
    grader = Grader(config, workers)
    return lp, config, {r["uid"]: r for r in dataset_rows}, ground_truth, grader, scratch


def grid_only(cells: list[Path], config_path: str, seed: int) -> list[dict]:
    os.environ["LEAF_CONFIG"] = str(Path(config_path).resolve())
    sys.path.insert(0, str(PROJECT))
    from leaf.verifiers import lcb_private as lp
    from leaf.core.project_config import load_config
    import leaf.grading.grade_baseline as gb

    config = load_config()
    rows = []
    for cell in cells:
        kind, condition = cell_kind(cell)
        grid = lp.build_grid(cell, kind=kind, config=config, condition=condition, mode="formal",
                             seed=seed, model=gb.run_model_key(cell) or None)
        row = {"cell": str(cell), "kind": kind, "entries": len(grid),
               "public_pass": sum(e["public"] for e in grid),
               "empty_code": sum(not e["code"].strip() for e in grid),
               "empty_code_public_pass": sum(e["public"] and not e["code"].strip() for e in grid),
               "unique_programs": len({(e["problem_uid"], lp.sha(e["code"])) for e in grid})}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def resummarize(cells: list[Path], config_path: str, seed: int) -> list[dict]:
    os.environ["LEAF_CONFIG"] = str(Path(config_path).resolve())
    sys.path.insert(0, str(PROJECT))
    from leaf.verifiers import lcb_private as lp
    from leaf.core.project_config import load_config, root
    import leaf.grading.grade_baseline as gb

    config = load_config()
    dataset_rows = lp.read_jsonl(root(config) / "dataset_lcb.jsonl")
    ground_truth = lp.build_ground_truth(dataset_rows, Path(config["lcb_hf_snapshot"]), None)
    rows = []
    for cell in cells:
        kind, condition = cell_kind(cell)
        grid = lp.build_grid(cell, kind=kind, config=config, condition=condition, mode="formal",
                             seed=seed, model=gb.run_model_key(cell) or None)
        result = lp.resummarize_cell(cell, grid, ground_truth, int(config["verifier_timeout"]))
        block = result["lcbprivate"]
        rows.append({"cell": str(cell), "public": block["public_only"],
                     "private": block["private_only"],
                     "public_private": {k: result[k] for k in ("accuracy", "final_pass_rate")},
                     "public_pass_private_fail_records": block["public_pass_private_fail_records"]})
    return rows


def regrade(cells: list[Path], config_path: str, workers: int, seed: int) -> list[dict]:
    lp, config, dataset, ground_truth, grader, scratch = load_everything(config_path, workers)
    import leaf.grading.grade_baseline as gb

    rows = []
    try:
        for cell in cells:
            kind, condition = cell_kind(cell)
            grid = lp.build_grid(cell, kind=kind, config=config, condition=condition,
                                 mode="formal", seed=seed, model=gb.run_model_key(cell) or None)
            print(json.dumps({"event": "grid", "cell": str(cell), "kind": kind,
                              "entries": len(grid), "reproduces_rollout_summary": True}),
                  flush=True)
            result = lp.grade_cell(cell, grid, ground_truth, grader, config, dataset,
                                   condition if kind == "verl" else None)
            block = result["lcbprivate"]
            rows.append({"cell": str(cell),
                         "public": block["public_only"], "private": block["private_only"],
                         "public_private": {k: result[k] for k in ("accuracy", "final_pass_rate")},
                         "public_pass_private_fail_records": block["public_pass_private_fail_records"]})
    finally:
        grader.close()
        lp.remove(scratch)
    return rows


def sanity(cells: list[Path], config_path: str, workers: int) -> dict:
    """Stub / wrong-problem / grader-agreement checks on the official private grader.

    * ``pass`` stub on every problem's private tests: must fail everywhere;
    * a public-passing program of another problem with the same test type: must fail;
    * official grader on the *public* tests vs the rollout verifier's public
      verdict, for up to 4 public-pass and 4 public-fail programs per problem
      from ``cells`` (LEAF / Random Sampling cells);
    * programs passing public AND private (separately) re-run on the full
      private_test_cases list as one sample: must pass.
    """
    lp, config, dataset, ground_truth, grader, scratch = load_everything(config_path, workers)
    timeout = int(config["verifier_timeout"])
    rng = random.Random(0)
    try:
        # Problems with an empty private set pass any program vacuously.
        uids = sorted(u for u, t in ground_truth["tasks"].items() if t["private"])
        stub = lp.run_checks({(u, "stub"): (u, "pass\n") for u in uids}, ground_truth, grader,
                             "private", timeout, label="sanity_stub")
        pool = collections.defaultdict(lambda: {"pass": {}, "fail": {}})
        for cell in cells:
            kind, _ = cell_kind(cell)
            if kind == "verl":
                continue
            grades = lp.read_jsonl(cell / "grades_public.jsonl")
            for g in grades:
                code = g.get("extracted_answer") or ""
                if code.strip():
                    pool[g["problem_uid"]]["pass" if g["correct"] else "fail"].setdefault(
                        lp.sha(code), code)
        agree_jobs, public_verdict = {}, {}
        for uid in uids:
            for label in ("pass", "fail"):
                items = sorted(pool[uid][label].items())
                rng.shuffle(items)
                for digest, code in items[:4]:
                    agree_jobs[(uid, digest)] = (uid, code)
                    public_verdict[(uid, digest)] = label == "pass"
        official_public = lp.run_checks(agree_jobs, ground_truth, grader, "public", timeout,
                                        label="sanity_public_agreement")
        disagreements = [
            {"problem_uid": key[0], "code_hash": key[1], "rollout_public": public_verdict[key],
             "official_public": official_public[key]["status"],
             "error": official_public[key]["error_message"]}
            for key in agree_jobs if official_public[key]["passed"] != public_verdict[key]]
        # wrong problem: a public-passing program of another same-type problem
        by_type = collections.defaultdict(list)
        for uid in uids:
            if pool[uid]["pass"]:
                by_type[ground_truth["tasks"][uid]["testtype"]].append(uid)
        wrong_jobs = {}
        for uid in uids:
            others = [u for u in by_type[ground_truth["tasks"][uid]["testtype"]] if u != uid]
            if others:
                other = rng.choice(others)
                code = sorted(pool[other]["pass"].items())[0][1]
                wrong_jobs[(uid, "wrong:" + other)] = (uid, code)
        wrong = lp.run_checks(wrong_jobs, ground_truth, grader, "private", timeout,
                              label="sanity_wrong_problem")
        passing = {k: v for k, v in agree_jobs.items() if public_verdict[k]}
        private = lp.run_checks(passing, ground_truth, grader, "private", timeout,
                                label="sanity_private_of_public_pass")
        both = {k: passing[k] for k in passing if private[k]["passed"]}
        full = lp.run_checks(both, ground_truth, grader, "full", timeout,
                             label="sanity_full_of_both")
        private_fail_examples = [
            {"problem_uid": k[0], "code_hash": k[1], "status": private[k]["status"],
             "tests_run": private[k]["tests_run"], "tests": private[k]["tests"],
             "error": private[k]["error_message"]}
            for k in passing if not private[k]["passed"]][:12]
        report = {
            "event": "sanity",
            "ground_truth_counts": ground_truth["counts"],
            "stub_pass_private": sum(o["passed"] for o in stub.values()),
            "stub_problems": len(stub),
            "wrong_problem_pass_private": sum(o["passed"] for o in wrong.values()),
            "wrong_problem_checks": len(wrong),
            "wrong_problem_passing": [k for k, o in wrong.items() if o["passed"]],
            "public_agreement_programs": len(agree_jobs),
            "public_agreement_disagree": len(disagreements),
            "public_agreement_by_rollout_verdict": {
                label: {"programs": sum(v == flag for v in public_verdict.values()),
                        "official_pass": sum(official_public[k]["passed"]
                                             for k, v in public_verdict.items() if v == flag)}
                for label, flag in (("rollout_pass", True), ("rollout_fail", False))},
            "public_disagreements": disagreements[:20],
            "public_pass_programs": len(passing),
            "public_pass_private_pass": len(both),
            "private_fail_examples": private_fail_examples,
            "both_full_pass": sum(o["passed"] for o in full.values()),
            "both_full_fail": [(k, o["status"], o["error_message"])
                               for k, o in full.items() if not o["passed"]][:10],
            "problems_with_public_pass_program": sum(bool(pool[u]["pass"]) for u in uids),
        }
        print(json.dumps(report, default=str), flush=True)
        return report
    finally:
        grader.close()
        lp.remove(scratch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="project config (sets LEAF_CONFIG)")
    parser.add_argument("--cells", nargs="+", type=Path, required=True,
                        help="cell directories .../<condition>/lcb")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--allocation-seed", type=int, default=0)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--resummarize", action="store_true",
                        help="rewrite graded cells from their own grades_lcbprivate.jsonl "
                             "(no execution)")
    parser.add_argument("--grid-only", action="store_true",
                        help="only rebuild and check each cell's grid (no private grading)")
    parser.add_argument("--table", type=Path, help="write the per-cell rows / sanity report here")
    args = parser.parse_args()
    started = time.monotonic()
    cells = [c.resolve() for c in args.cells]
    if args.resummarize:
        out = resummarize(cells, args.config, args.allocation_seed)
    elif args.grid_only:
        out = grid_only(cells, args.config, args.allocation_seed)
    elif args.sanity:
        out = sanity(cells, args.config, args.workers)
    else:
        out = regrade(cells, args.config, args.workers, args.allocation_seed)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"event": "done", "seconds": round(time.monotonic() - started, 1)}),
          flush=True)


if __name__ == "__main__":
    main()
