#!/usr/bin/env bash
# Download the upstream datasets into $DATA_ROOT/raw and the base models into $MODEL_ROOT,
# at the revisions the paper used (see leaf/data/fetch.py for the pinned revisions).
#
#   bash run/data_fetch.sh            # datasets and models
#   bash run/data_fetch.sh data       # datasets only  (~4.5 GB, mostly LiveCodeBench)
#   bash run/data_fetch.sh models     # models only    (~31 GB)
#   DRY_RUN=1 bash run/data_fetch.sh  # list what would be downloaded
#
#   $DATA_ROOT/raw/code_generation_lite/test{,2,3,4,5,6}.jsonl    livecodebench/code_generation_lite @ 0fe84c39 (release_v6)
#   $DATA_ROOT/raw/MATH-500/test.jsonl                              HuggingFaceH4/MATH-500 @ ff5b2025
#   $DATA_ROOT/raw/hendrycks_math/<subject>/train-00000-of-00001.parquet   EleutherAI/hendrycks_math @ 21a56338
#   $DATA_ROOT/raw/SciKnowEval/data/v2/sciknoweval_test_v2.jsonl    hicai-zju/SciKnowEval @ 92ef969a
#   $DATA_ROOT/raw/MbppPlus-v0.2.0.jsonl                            evalplus/mbppplus_release v0.2.0
#   $DATA_ROOT/raw/mbpp_full_test.jsonl                             google-research-datasets/mbpp @ 4bb6404f, full/test
#   $MODEL_ROOT/Qwen3-8B                                            Qwen/Qwen3-8B @ 2069b3fa
#   $MODEL_ROOT/Olmo-3-7B-Instruct                                  allenai/Olmo-3-7B-Instruct @ 6e5971d9
#
# Needs network access and the huggingface_hub package. Set HF_TOKEN if the Hub asks for it.
set -euo pipefail
export REQUIRE_THIRD_PARTY=${REQUIRE_THIRD_PARTY:-0}  # imports no third-party code
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

args=("$@")
if [ "$DRY_RUN" = 1 ]; then args+=(--dry-run); fi
"$PYTHON" -m leaf.data.fetch "${args[@]}"
