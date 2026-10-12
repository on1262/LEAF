#!/usr/bin/env bash
# History-compression ablation of LEAF during exploration (appendix table abl-compression).
#
#   bash run/explore_compression.sh MODEL CONDITION BENCHMARK
#
#   MODEL      qwen3_8b | olmo3_7b | all
#   CONDITION  compress_none | compress_llm_summary | compress_tail | compress_head_tail |
#              compress_summary_block | compress_lowest_logprob | all
#              (leaf.explore.leaf_rollout conditions: no compression, LLM summary, tail (LEAF's),
#              head and tail, summary block, lowest log-prob)
#   BENCHMARK  math500 | lcb | mbpp | sciknoweval | all
#   "all" selects the cells of the table only:
#     qwen3_8b  compress_none on math500, lcb; compress_tail and compress_head_tail on all four benchmarks
#     olmo3_7b  compress_tail and compress_head_tail on all four; compress_llm_summary,
#               compress_summary_block, compress_lowest_logprob on lcb, sciknoweval
#
# Examples
#   bash run/explore_compression.sh olmo3_7b compress_lowest_logprob lcb
#   GPUS="0 1 2 3" bash run/explore_compression.sh all all all
#   DRY_RUN=1 bash run/explore_compression.sh all all all
#
# These runs use an earlier engine configuration (configs/ablation/compression_<m>.json) and, as in
# the paper, no 5 GPU-h budget: each cell runs as a chain of resumable chunks (the original chunk count
# and chunk wall clock, CHUNKS below) until its 24 steps are complete; a chunk that finds the run
# complete is skipped. Then CPU grading: leaf.grading.grade_leaf_rollout (public tests, MBPP+ private
# tests) and on LCB leaf.grading.grade_lcb_private. Run seed 1.
# Outputs: $RESULTS_ROOT/ablation/compression/<model>/<model>/formal/leaf/<condition>/<benchmark>/.
# Prerequisite: `bash run/data_prepare.sh roots`. Environment: run/common.sh, run/explore_common.sh.
set -euo pipefail
export _EXPLORE_USER_NUM_CPUS=${_EXPLORE_USER_NUM_CPUS-${NUM_CPUS-}}
# shellcheck source=run/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=run/explore_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/explore_common.sh"

MODELS=(qwen3_8b olmo3_7b)
CONDITIONS=(compress_none compress_llm_summary compress_tail compress_head_tail compress_summary_block
            compress_lowest_logprob)
BENCHMARKS=(math500 lcb mbpp sciknoweval)

# (chunks, hours per chunk) of every cell in the table; no entry = not run
chunks() {  # MODEL CONDITION BENCHMARK -> "CHUNKS HOURS"
  case "$1:$2:$3" in
    qwen3_8b:compress_none:math500|qwen3_8b:compress_none:lcb) echo "2 2" ;;
    qwen3_8b:compress_tail:*|qwen3_8b:compress_head_tail:*) echo "2 2" ;;
    olmo3_7b:compress_llm_summary:lcb|olmo3_7b:compress_tail:lcb|olmo3_7b:compress_head_tail:lcb) echo "2 5" ;;
    olmo3_7b:compress_summary_block:lcb|olmo3_7b:compress_lowest_logprob:lcb) echo "2 5" ;;
    olmo3_7b:compress_llm_summary:sciknoweval|olmo3_7b:compress_tail:sciknoweval) echo "2 2" ;;
    olmo3_7b:compress_head_tail:sciknoweval) echo "2 2" ;;
    olmo3_7b:compress_summary_block:sciknoweval|olmo3_7b:compress_lowest_logprob:sciknoweval) echo "2 2" ;;
    olmo3_7b:compress_tail:mbpp|olmo3_7b:compress_head_tail:mbpp) echo "2 2" ;;
    olmo3_7b:compress_tail:math500|olmo3_7b:compress_head_tail:math500) echo "2 4" ;;
  esac
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; exit 2; }

[ $# -eq 3 ] || usage
[ "$MODE" = formal ] || { echo "explore_compression.sh runs the formal stage only (MODE=formal)" >&2; exit 2; }
member "$1" all "${MODELS[@]}" || usage
member "$2" all "${CONDITIONS[@]}" || usage
member "$3" all "${BENCHMARKS[@]}" || usage

if [ "$1" = all ] || [ "$2" = all ] || [ "$3" = all ]; then
  cells=()
  for model in $(expand_all "$1" "${MODELS[@]}"); do
    for condition in $(expand_all "$2" "${CONDITIONS[@]}"); do
      for bench in $(expand_all "$3" "${BENCHMARKS[@]}"); do
        [ -n "$(chunks "$model" "$condition" "$bench")" ] && cells+=("$model $condition $bench")
      done
    done
  done
  [ ${#cells[@]} -gt 0 ] || { echo "no cell of the table matches" >&2; exit 2; }
  run_cells "${BASH_SOURCE[0]}" "${cells[@]}"
  exit
fi

model=$1 condition=$2 bench=$3
plan=$(chunks "$model" "$condition" "$bench")
[ -n "$plan" ] || { echo "$model $condition $bench is not a cell of the table" >&2; exit 2; }
read -r count hours <<< "$plan"
config=$REPO/configs/ablation/compression_$model.json
root=$(cfg_get "$config" output_root)
workers=$(cap_workers "$(cfg_get "$config" verifier_workers)" "$_EXPLORE_USER_NUM_CPUS")
steps=$(cfg_get "$config" formal_steps)
dataset=$root/dataset_$bench.jsonl
if [ "$DRY_RUN" != 1 ] && [ ! -e "$dataset" ]; then
  echo "missing $dataset: run 'bash run/data_prepare.sh roots' first" >&2
  exit 1
fi
export NUM_CPUS
NUM_CPUS=$(cell_cpus "$bench")
out=$root/$model/$MODE/leaf/$condition/$bench
# job name: ablation_compression_<model>_<condition>_<benchmark>[_chunk<i> | _grade | _private]
name=ablation_compression_${model}_${condition}_$bench
fields=(model="$model" benchmark="$bench" condition="$condition")
# leaf.explore.leaf_rollout writes generation_summary.json once all steps are complete
done_marker=$out/generation_summary.json

status=0
if step_on gpu; then
  for ((i = 0; i < count; i++)); do
    job_done "$done_marker" && break
    status=0
    dry_meta "${name}_chunk$i" "$MODE" $((hours * 60)) "$out"
    run_job "${name}_chunk$i" "$config" "$root" "$MODE" "$out" $((hours * 60)) "${fields[@]}" chunk="$i" chunks="$count" \
      -- "$PYTHON" -u -m leaf.explore.leaf_rollout --condition "$condition" --model "$model" \
         --dataset "$dataset" --steps "$steps" --output "$out" || status=$?
  done
fi
if step_on grade; then
  if [ "$DRY_RUN" != 1 ] && { [ "$status" -ne 0 ] || [ ! -e "$done_marker" ]; }; then
    echo "$model $condition $bench: not complete after $count chunks of $hours h; re-run to continue" >&2
    exit 1
  fi
  grade_cell "$name" "$config" "$root" "$out" "$bench" "${fields[@]}" \
    -- leaf.grading.grade_leaf_rollout --generations "$out/generations.jsonl" \
       --output "$out/grades_public.jsonl" --workers "$workers" || status=$?
fi
exit "$status"
