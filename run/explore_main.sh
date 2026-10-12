#!/usr/bin/env bash
# Test-time exploration, main comparison (Table 1, Fig. 3, the per-benchmark exploration tables):
# LEAF and the five baselines on the four exploration benchmarks, run seeds 1-3.
#
#   bash run/explore_main.sh MODEL METHOD BENCHMARK SEED
#
#   MODEL      qwen3_8b | olmo3_7b | all
#   METHOD     leaf | random_sampling | grpo | sdpo | score | seed | all
#              leaf            = LEAF (leaf.explore.leaf_rollout, condition leaf)
#              random_sampling = Random Sampling (leaf.baselines.random_sampling)
#              grpo / sdpo / score = verl batch baselines (leaf.baselines.<method>)
#              seed            = SEED: collection -> SFT -> RL (leaf.baselines.seed_<stage>)
#   BENCHMARK  math500 | lcb | mbpp | sciknoweval | all
#   SEED       1 | 2 | 3 | all     (configs/explore/<model>_seed<seed>.json)
#
# Examples
#   bash run/explore_main.sh qwen3_8b leaf math500 1                  # one cell on the visible GPU
#   CUDA_VISIBLE_DEVICES=2 bash run/explore_main.sh olmo3_7b seed lcb 2
#   GPUS="0 1 2 3" bash run/explore_main.sh all all all 1              # 48 cells, 4 at a time
#   STEPS=gpu GPUS="0 1" bash run/explore_main.sh qwen3_8b all all all # GPU jobs only ...
#   STEPS=grade JOBS=4 bash run/explore_main.sh qwen3_8b all all all   # ... and grade them later
#   DRY_RUN=1 bash run/explore_main.sh all all all all                 # print every command
#
# Prerequisite: the run roots (dataset links and rl_data) from `bash run/data_prepare.sh roots seeds`.
#
# One cell = one GPU job capped at its wall-clock budget, then its CPU grading:
#   * every method but SEED: one job with a 300-minute cap (the 5 GPU-h budget of the protocol);
#     a run that reaches the cap stops there and is graded on the aligned steps it finished;
#   * SEED: three jobs in order, each with its own cap (SEED_MINUTES below), the next stage only
#     after the previous one completed;
#   * grading (CPU): leaf.grading.grade_explore_cell (public tests, MBPP+ private tests), then on LCB the
#     private-test regrade leaf.grading.grade_lcb_private.
# Outputs: $RESULTS_ROOT/explore/seed<s>/<model>/: <model>/formal/{leaf,baselines}/<method>/<benchmark>/,
# jobs.jsonl and logs/.
# Re-running the script skips finished work and resumes interrupted jobs within their budget
# (run/explore_common.sh). MODE=smoke_full runs the formal data for smoke_steps (sizing runs).
# More environment: run/common.sh, run/explore_common.sh (STEPS, GPUS, JOBS, NUM_CPUS, FORCE).
set -euo pipefail
export _EXPLORE_USER_NUM_CPUS=${_EXPLORE_USER_NUM_CPUS-${NUM_CPUS-}}
# shellcheck source=run/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=run/explore_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/explore_common.sh"

MODELS=(qwen3_8b olmo3_7b)
METHODS=(leaf random_sampling grpo sdpo score seed)
BENCHMARKS=(math500 lcb mbpp sciknoweval)
SEEDS=(1 2 3)
LEAF=leaf
SEED_STAGES=(seed_collection seed_sft seed_rl)

# Wall-clock minutes of the SEED stages (collection, SFT, RL) of the paper runs: the 5 GPU-h
# budget of a cell split by the measured collection and SFT times (sum <= 320 min, start-up
# slack included). Exceptions: Olmo seed 1 re-ran its RL stage alone with the longer caps below
# (MATH-500 sum 351 min); Olmo seed 3 MATH-500 reached the 74-minute SFT cap and its SFT was
# resumed with 40 more minutes (114 in total). SEED_MINUTES=COLL,SFT,RL overrides the table.
seed_minutes() {  # MODEL BENCHMARK SEED -> "COLL SFT RL"
  if [ -n "${SEED_MINUTES:-}" ]; then echo "${SEED_MINUTES//,/ }"; return; fi
  case "$1:$2:$3" in
    qwen3_8b:math500:*) echo "16 128 170" ;;
    qwen3_8b:lcb:*) echo "15 36 260" ;;
    qwen3_8b:mbpp:*) echo "14 65 240" ;;
    qwen3_8b:sciknoweval:*) echo "14 85 220" ;;
    olmo3_7b:math500:1) echo "26 105 220" ;;
    olmo3_7b:lcb:1) echo "33 37 250" ;;
    olmo3_7b:mbpp:1) echo "17 78 260" ;;
    olmo3_7b:sciknoweval:1) echo "14 78 260" ;;
    olmo3_7b:math500:3) echo "26 114 220" ;;
    olmo3_7b:math500:*) echo "26 74 220" ;;
    olmo3_7b:lcb:*) echo "33 37 250" ;;
    olmo3_7b:mbpp:*) echo "17 60 240" ;;
    olmo3_7b:sciknoweval:*) echo "14 60 240" ;;
  esac
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; exit 2; }

[ $# -eq 4 ] || usage
case "$MODE" in formal|smoke_full) ;; *) echo "explore_main.sh: MODE must be formal or smoke_full" >&2; exit 2 ;; esac
member "$1" all "${MODELS[@]}" || usage
member "$2" all "${METHODS[@]}" || usage
member "$3" all "${BENCHMARKS[@]}" || usage
member "$4" all "${SEEDS[@]}" || usage

# several cells: one process per cell
if [ "$1" = all ] || [ "$2" = all ] || [ "$3" = all ] || [ "$4" = all ]; then
  cells=()
  for seed in $(expand_all "$4" "${SEEDS[@]}"); do
    for model in $(expand_all "$1" "${MODELS[@]}"); do
      for method in $(expand_all "$2" "${METHODS[@]}"); do
        for bench in $(expand_all "$3" "${BENCHMARKS[@]}"); do
          cells+=("$model $method $bench $seed")
        done
      done
    done
  done
  run_cells "${BASH_SOURCE[0]}" "${cells[@]}"
  exit
fi

model=$1 method=$2 bench=$3 seed=$4
config=$REPO/configs/explore/${model}_seed$seed.json
root=$(cfg_get "$config" output_root)
workers=$(cap_workers "$(cfg_get "$config" verifier_workers)" "$_EXPLORE_USER_NUM_CPUS")
if [ "$MODE" = formal ]; then steps=$(cfg_get "$config" formal_steps); else steps=$(cfg_get "$config" smoke_steps); fi
dataset=$root/dataset_$bench.jsonl
if [ "$DRY_RUN" != 1 ] && [ ! -e "$dataset" ]; then
  echo "missing $dataset: run 'bash run/data_prepare.sh roots seeds' first" >&2
  exit 1
fi
export NUM_CPUS
NUM_CPUS=$(cell_cpus "$bench")
# job name: explore_s<seed>_<model>_<method or SEED stage>_<benchmark>[_gpu | _grade | _private]
name() { printf 'explore_s%s_%s_%s_%s' "$seed" "$model" "$1" "$bench"; }
cap=300
status=0

case "$method" in
  leaf)
    out=$root/$model/$MODE/leaf/$LEAF/$bench
    fields=(model="$model" benchmark="$bench" condition="$LEAF")
    if step_on gpu; then
      run_capped "$(name leaf)_gpu" "$config" "$root" "$MODE" "$out" "$cap" "${fields[@]}" \
        -- "$PYTHON" -u -m leaf.explore.leaf_rollout --condition "$LEAF" --model "$model" \
           --dataset "$dataset" --steps "$steps" --output "$out" || status=$?
    fi
    if step_on grade; then
      grade_cell "$(name leaf)" "$config" "$root" "$out" "$bench" "${fields[@]}" \
        -- leaf.grading.grade_explore_cell --kind leaf --output "$out" --workers "$workers" || status=$?
    fi
    ;;
  random_sampling|grpo|sdpo|score)
    out=$root/$model/$MODE/baselines/$method/$bench
    fields=(model="$model" benchmark="$bench" condition="$method")
    if step_on gpu; then
      run_capped "$(name "$method")_gpu" "$config" "$root" "$MODE" "$out" "$cap" "${fields[@]}" \
        -- "$PYTHON" -u -m "leaf.baselines.$method" --model "$model" --benchmark "$bench" \
           --mode "$MODE" --output "$out" || status=$?
    fi
    if step_on grade; then
      grade_cell "$(name "$method")" "$config" "$root" "$out" "$bench" "${fields[@]}" \
        -- leaf.grading.grade_explore_cell --kind baseline --output "$out" --condition "$method" \
           --benchmark "$bench" --workers "$workers" || status=$?
    fi
    ;;
  seed)
    read -r -a minutes <<< "$(seed_minutes "$model" "$bench" "$seed")"
    if step_on gpu; then
      for i in 0 1 2; do
        stage=${SEED_STAGES[$i]}
        out=$root/$model/$MODE/baselines/$stage/$bench
        run_capped "$(name "$stage")_gpu" "$config" "$root" "$MODE" "$out" "${minutes[$i]}" \
          model="$model" benchmark="$bench" condition="$stage" \
          -- "$PYTHON" -u -m "leaf.baselines.$stage" --model "$model" --benchmark "$bench" \
             --mode "$MODE" --output "$out" || status=$?
        [ "$stage" != seed_rl ] || break
        # collection and SFT feed the next stage: it starts only after they completed, and
        # without the RL stage there is nothing to grade
        if [ "$status" -ne 0 ] || { [ "$DRY_RUN" != 1 ] \
             && [ "$(cell_state "$root" "$out")" != COMPLETED ]; }; then
          echo "$model $bench seed $seed: $stage did not complete within ${minutes[$i]} min;" \
               "re-run to resume a failed stage, SEED_MINUTES=COLL,SFT,RL gives a stage more time" >&2
          exit 1
        fi
      done
    fi
    out=$root/$model/$MODE/baselines/seed_rl/$bench
    if step_on grade; then
      grade_cell "$(name seed_rl)" "$config" "$root" "$out" "$bench" \
        model="$model" benchmark="$bench" condition=seed_rl \
        -- leaf.grading.grade_explore_cell --kind baseline --output "$out" --condition seed_rl \
           --benchmark "$bench" --workers "$workers" || status=$?
    fi
    ;;
esac
exit "$status"
