#!/usr/bin/env bash
# Exploration ablations of LEAF (component ablations: Fig. ablation scatter and the
# abl-components tables) and LEAF with interleaved policy updates (update-interval table).
#
#   bash run/explore_ablations.sh MODEL VARIANT BENCHMARK
#
#   MODEL      qwen3_8b | olmo3_7b | all
#   VARIANT    wo_last_response | wo_feedback | restart_every_2 | restart_every_10 | no_restart |
#              last_3_rounds | interleaved_every_1 | interleaved_every_2 | interleaved_every_5 | all
#              wo_last_response     w/o last response   (leaf.explore.leaf_rollout condition of that name)
#              wo_feedback          w/o feedback
#              restart_every_2      restart every 2 steps
#              restart_every_10     restart every 10 steps
#              no_restart           no restart
#              last_3_rounds        last 3 rounds
#              interleaved_every_1  update every step (leaf.explore.run_interleaved,
#                                   configs/explore/<m>_seed1.json)
#              interleaved_every_2, interleaved_every_5  update every 2 / 5 steps (--update-interval 2 / 5,
#                                   configs/ablation/explore_<m>.json)
#   BENCHMARK  math500 | lcb | mbpp | sciknoweval | all
#
# Examples
#   bash run/explore_ablations.sh qwen3_8b no_restart lcb
#   GPUS="0 1 2 3" bash run/explore_ablations.sh all all all
#   DRY_RUN=1 bash run/explore_ablations.sh all all all
#
# Run seed 1 only. The "No update" rows of the update-interval table and the LEAF rows of the
# component tables are the LEAF cells of run/explore_main.sh (seed 1).
# Every cell: one GPU job capped at 300 minutes (the 5 GPU-h budget), then its CPU grading
# (leaf.grading.grade_explore_cell --kind leaf, then on LCB leaf.grading.grade_lcb_private).
# The ablations and interleaved_every_2 / 5 read the seed-1 exploration datasets (data_root of
# configs/ablation/explore_<m>.json = $RESULTS_ROOT/explore/seed1/<m>) and write to
# $RESULTS_ROOT/ablation/explore/<m>/; interleaved_every_1 writes to $RESULTS_ROOT/explore/seed1/<m>/.
# The interleaved runs also read configs/general/leaf_rollout_math500_<m>_seed1.json (policy-update
# settings).
# Prerequisite: `bash run/data_prepare.sh roots`. Environment: run/common.sh, run/explore_common.sh.
set -euo pipefail
export _EXPLORE_USER_NUM_CPUS=${_EXPLORE_USER_NUM_CPUS-${NUM_CPUS-}}
# shellcheck source=run/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=run/explore_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/explore_common.sh"

MODELS=(qwen3_8b olmo3_7b)
VARIANTS=(wo_last_response wo_feedback restart_every_2 restart_every_10 no_restart last_3_rounds
          interleaved_every_1 interleaved_every_2 interleaved_every_5)
BENCHMARKS=(math500 lcb mbpp sciknoweval)

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; exit 2; }

[ $# -eq 3 ] || usage
[ "$MODE" = formal ] || { echo "explore_ablations.sh runs the formal stage only (MODE=formal)" >&2; exit 2; }
member "$1" all "${MODELS[@]}" || usage
member "$2" all "${VARIANTS[@]}" || usage
member "$3" all "${BENCHMARKS[@]}" || usage

if [ "$1" = all ] || [ "$2" = all ] || [ "$3" = all ]; then
  cells=()
  for model in $(expand_all "$1" "${MODELS[@]}"); do
    for variant in $(expand_all "$2" "${VARIANTS[@]}"); do
      for bench in $(expand_all "$3" "${BENCHMARKS[@]}"); do
        cells+=("$model $variant $bench")
      done
    done
  done
  run_cells "${BASH_SOURCE[0]}" "${cells[@]}"
  exit
fi

model=$1 variant=$2 bench=$3
if [ "$variant" = interleaved_every_1 ]; then
  config=$REPO/configs/explore/${model}_seed1.json
  prefix=explore_s1_$model
else
  config=$REPO/configs/ablation/explore_$model.json
  prefix=ablation_explore_$model
fi
root=$(cfg_get "$config" output_root)
data_root=$(cfg_get "$config" data_root)
workers=$(cap_workers "$(cfg_get "$config" verifier_workers)" "$_EXPLORE_USER_NUM_CPUS")
steps=$(cfg_get "$config" formal_steps)
export NUM_CPUS
NUM_CPUS=$(cell_cpus "$bench")

condition=$variant
case "$variant" in
  interleaved_every_*)
    command=(leaf.explore.run_interleaved --model "$model" --benchmark "$bench" --mode "$MODE") ;;
  *)
    dataset=$data_root/dataset_$bench.jsonl
    if [ "$DRY_RUN" != 1 ] && [ ! -e "$dataset" ]; then
      echo "missing $dataset: run 'bash run/data_prepare.sh roots' first" >&2
      exit 1
    fi
    command=(leaf.explore.leaf_rollout --condition "$condition" --model "$model" --dataset "$dataset" --steps "$steps") ;;
esac
out=$root/$model/$MODE/leaf/$condition/$bench
command+=(--output "$out")
case "$variant" in interleaved_every_2|interleaved_every_5) command+=(--update-interval "${variant##*_}") ;; esac

# job name: <prefix>_<variant>_<benchmark>[_gpu | _grade | _private]
name=${prefix}_${condition}_$bench
fields=(model="$model" benchmark="$bench" condition="$condition")
status=0
if step_on gpu; then
  run_capped "${name}_gpu" "$config" "$root" "$MODE" "$out" 300 "${fields[@]}" \
    -- "$PYTHON" -u -m "${command[@]}" || status=$?
fi
if step_on grade; then
  grade_cell "$name" "$config" "$root" "$out" "$bench" "${fields[@]}" \
    -- leaf.grading.grade_explore_cell --kind leaf --output "$out" --workers "$workers" || status=$?
fi
exit "$status"
