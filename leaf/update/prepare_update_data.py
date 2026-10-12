"""LEAF / SFT training data of one training set from its rollout, run seeds 2 / 3 (CPU job).

``leaf.update.build_contexts`` (``successful_contexts.jsonl`` + ``manifest.json``: the successful
contexts c+ of the rollout) followed by ``leaf.update.build_trace_variants`` on the same directory
(``ystar_pairs.jsonl``: the (c+, y+) pairs the y* rewrite reads, plus the unused failure / random
context files). Runs under the seed's ``configs/general/leaf_<model>_seed<s>.json`` config
(``LEAF_CONFIG`` of the job).

    python -m leaf.update.prepare_update_data --model qwen3_8b --benchmark lcb \\
        --rollout <root>/<m>/<m>/formal/leaf/leaf/lcb \\
        --output <root>/<m>/<m>/formal/update_data/lcb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.rollout / "manifest.json").read_text())
    if Path(args.output / "manifest.json").exists():
        existing = json.loads((args.output / "manifest.json").read_text())
        if Path(existing["rollout"]).resolve() != args.rollout.resolve():
            raise SystemExit(f"{args.output} belongs to another rollout ({existing['rollout']})")
    subprocess.run([sys.executable, "-m", "leaf.update.build_contexts", "--model", args.model,
                    "--benchmark", args.benchmark, "--rollout", str(args.rollout),
                    "--output", str(args.output)], check=True)
    from leaf.update import build_trace_variants

    summary = build_trace_variants.prepare(args.model, args.benchmark, args.output, source_root=args.output.parent)
    print(json.dumps({"event": "update_data", "dataset": manifest["dataset"], **summary}), flush=True)


if __name__ == "__main__":
    main()
