#!/usr/bin/env bash
# Generalization after a policy update (Table 2 and its per-seed / pass@4 / cost / length tables):
# Base, GRPO, SDPO, SCoRe, SEED, LEAF (rollout -> successful contexts -> policy update) and SFT on y*,
# trained on one training set (LiveCodeBench, MATH, SciKnowEval) and tested on its test set (avg@4
# over 4 samples), run seeds 1-3.
#
#   bash run/general_main.sh MODEL METHOD BENCHMARK SEED
#
#   MODEL      qwen3_8b | olmo3_7b | all
#   METHOD     base | grpo | sdpo | score | seed | leaf | sft | seed_20_calls | all
#              base   = the base model's test evaluation (no training)
#              grpo / sdpo / score = verl baselines (leaf.baselines.<method>), then the test
#              seed   = SEED: collection -> SFT -> RL (leaf.baselines.seed_<stage>), then the test
#              leaf   = LEAF rollout on the training set (leaf.explore.leaf_rollout, 20 steps, no update)
#                       -> successful contexts -> LEAF update (leaf.update.run_update --method
#                       leaf_update) -> test
#              sft    = y* rewrite of every (c+, y+) pair of the LEAF update data (leaf.update.rewrite_ystar)
#                       -> SFT on y* (leaf.update.run_update --method sft_ystar) -> test;
#                       needs the update data of "leaf" (METHOD=all runs every sft cell after the
#                       other cells)
#              seed_20_calls = SEED at 20 verifier calls per problem (configs/general/baselines_<m>_seed1.json):
#                       collection -> SFT -> RL only, no test; seed 1 only. Its training GPU hours give
#                       the appendix SEED cost comparison (run/report.sh appendix ->
#                       main/cost_20_verifier_calls.json)
#   BENCHMARK  lcb | math500 | sciknoweval | all   (training set)
#   SEED       1 | 2 | 3 | all
#
# Examples
#   bash run/general_main.sh qwen3_8b leaf math500 1                 # one cell on the visible GPU
#   CUDA_VISIBLE_DEVICES=1 bash run/general_main.sh olmo3_7b seed lcb 2
#   GPUS="0 1 2 3" bash run/general_main.sh all all all all           # every cell, 4 at a time
#   DRY_RUN=1 bash run/general_main.sh all all all all                # print every command
#   MODE=smoke_full bash run/general_main.sh qwen3_8b grpo lcb 1      # sizing run: smoke_steps updates, no test
#
# Prerequisite: the run roots (dataset links, smoke subsets, rl_data) from
# `bash run/data_prepare.sh lcb roots seeds`.
#
# Configs (configs/general/, <m> = model) and output roots ($RESULTS_ROOT/general/seed<s>/...)
#   base / GRPO / SDPO / SCoRe            baselines_<m>_seed<s>.json        baselines/<m>
#   SEED at 20 verifier calls (seed 1)    baselines_<m>_seed1.json          baselines/<m>
#   SEED (15 verifier calls)              seed_baseline_<m>_seed<s>.json    seed_baseline/<m>
#   LEAF, seed 1                          leaf/<m>: rollouts, successful contexts, updates and tests
#       rollout + contexts, LCB / SciKnowEval  leaf_rollout_lcb_sciknoweval_<m>_seed1.json,
#                                              leaf_contexts_lcb_sciknoweval_<m>_seed1.json
#       rollout + contexts, MATH               leaf_rollout_math500_<m>_seed1.json
#       LEAF update + test                     leaf_update_<m>_seed1.json
#   SFT on y*, seed 1                     sft_<m>_seed1.json                seed1/sft/<m>
#                                         (data: leaf.update.build_trace_variants from seed1/leaf)
#   LEAF and SFT on y*, seeds 2, 3        leaf_<m>_seed<s>.json             leaf/<m>
#   Runs: <root>/<m>/<mode>/{baselines,leaf,update}/<method>/<bm>, update data
#   <root>/<m>/<mode>/update_data/<bm>, tests <root>/<m>/test/<label>/<bm>, jobs.jsonl and logs/ in
#   <root>/<m>. Test labels: base, <method>_<bm> (verl), seed_<bm>, leaf_update_<bm> (LEAF),
#   sft_ystar_<bm> (SFT on y*).
#
# Every step is skipped once its output marker exists (FORCE=1 re-runs it); runners resume from
# their own state, so re-running the script continues an interrupted cell. A step runs only after
# the previous step of its cell completed. More environment: run/common.sh, run/general_common.sh
# (GPUS, NUM_CPUS, TIME_CAPS, MBPP); LEAF_GPU_TYPE selects the a100 / h200 settings of the
# update runner (default: detected from the GPU).
set -euo pipefail
export _GENERAL_USER_NUM_CPUS=${_GENERAL_USER_NUM_CPUS-${NUM_CPUS-}}
# shellcheck source=run/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=run/general_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/general_common.sh"

METHODS=(base grpo sdpo score seed leaf sft seed_20_calls)
R=$RESULTS_ROOT

case "$MODE" in
  formal|smoke_full) ;;
  *) echo "general_main.sh supports MODE=formal and MODE=smoke_full" >&2; exit 2 ;;
esac

# Wall-clock caps (minutes) of the original job chains: chunks x hours of the submitters.
verl_minutes() {  # MODEL METHOD BENCHMARK
  case "$1:$2:$3" in
    qwen3_8b:*) echo 360 ;;
    olmo3_7b:score:lcb) echo 900 ;;
    olmo3_7b:*:sciknoweval) echo 360 ;;
    olmo3_7b:*) echo 600 ;;
  esac
}
seed_minutes() {  # MODEL BENCHMARK -> "COLLECTION SFT RL"
  case "$1:$2" in
    qwen3_8b:*) echo "60 120 360" ;;
    olmo3_7b:lcb) echo "120 240 720" ;;
    olmo3_7b:math500) echo "120 240 600" ;;
    olmo3_7b:sciknoweval) echo "120 240 360" ;;
  esac
}
seed_20_calls_minutes() {  # MODEL BENCHMARK -> "COLLECTION SFT RL" of the seed_20_calls chain
  case "$1:$2" in
    qwen3_8b:math500) echo "240 240 480" ;;
    qwen3_8b:*) echo "240 240 240" ;;
    olmo3_7b:sciknoweval) echo "240 240 480" ;;
    olmo3_7b:*) echo "240 240 900" ;;
  esac
}
rollout_minutes() {  # MODEL BENCHMARK
  case "$1:$2" in
    qwen3_8b:*) echo 120 ;;
    olmo3_7b:lcb) echo 480 ;;
    olmo3_7b:math500) echo 360 ;;
    olmo3_7b:sciknoweval) echo 120 ;;
  esac
}

# roots of one (model, seed): sets CFG_B (base/verl), ROOT_B, CFG_S (SEED), ROOT_S, TEST_MIN_B
# (test-generation cap of base / verl / SEED tests), CFG_L / ROOT_L (LEAF), CFG_SFT / ROOT_SFT and
# PREFIX (job names: general_s<seed>_<model>_...)
roots_of() {
  local m=$1 s=$2
  PREFIX=general_s${s}_$m
  CFG_B=$(cfg "general/baselines_${m}_seed$s"); ROOT_B=$R/general/seed$s/baselines/$m
  CFG_S=$(cfg "general/seed_baseline_${m}_seed$s"); ROOT_S=$R/general/seed$s/seed_baseline/$m
  ROOT_L=$R/general/seed$s/leaf/$m
  if [ "$s" = 1 ]; then
    TEST_MIN_B=60
    CFG_L=$(cfg "general/leaf_update_${m}_seed1")
    CFG_SFT=$(cfg "general/sft_${m}_seed1"); ROOT_SFT=$R/general/seed1/sft/$m
  else
    TEST_MIN_B=120
    CFG_L=$(cfg "general/leaf_${m}_seed$s")
    CFG_SFT=$CFG_L; ROOT_SFT=$ROOT_L
  fi
}

# ----------------------------------------------------------------------------------- cells

cell_base() {  # MODEL BENCHMARK SEED
  local m=$1 bm=$2 s=$3
  roots_of "$m" "$s"
  [ "$MODE" = formal ] || { echo "base: no test under MODE=$MODE" >&2; return 0; }
  local tb grade_cpus=""
  # the seed-1 base tests were graded on the default 16 CPUs (LiveCodeBench included)
  [ "$s" = 1 ] && grade_cpus=$(cpus std)
  for tb in $(tests_of "$bm" "$s"); do
    GRADE_CPUS=$grade_cpus test_eval "$PREFIX" "$CFG_B" "$ROOT_B" "$m" base "$tb" "" "$TEST_MIN_B" test \
      || return
  done
}

cell_verl() {  # MODEL METHOD BENCHMARK SEED
  local m=$1 method=$2 bm=$3 s=$4
  roots_of "$m" "$s"
  local out=$ROOT_B/$m/$MODE/baselines/$method/$bm
  gjob "${PREFIX}_${method}_$bm" "$CFG_B" "$ROOT_B" "$MODE" "$out" \
    "$(verl_minutes "$m" "$method" "$bm")" gpu "$(lcb_cpus "$bm")" "$out/COMPLETE.json" \
    condition="$method" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m "leaf.baselines.$method" --model "$m" --benchmark "$bm" --mode "$MODE" --output "$out" \
    || return
  [ "$MODE" = formal ] || return 0
  local tb
  for tb in $(tests_of "$bm" "$s"); do
    test_eval "$PREFIX" "$CFG_B" "$ROOT_B" "$m" "${method}_$bm" "$tb" "$out" "$TEST_MIN_B" test || return
  done
}

# seed_chain CONFIG ROOT MODEL BENCHMARK "COLLECTION SFT RL minutes" NAME: the three SEED stages
seed_chain() {
  local config=$1 root=$2 m=$3 bm=$4 name=$6 minutes stage i=0 out
  read -r -a minutes <<< "$5"
  for stage in "${SEED_STAGES[@]}"; do
    out=$root/$m/$MODE/baselines/$stage/$bm
    gjob "${name}_${stage}_$bm" "$config" "$root" "$MODE" "$out" \
      "${minutes[$i]}" gpu "$(lcb_cpus "$bm")" "$out/COMPLETE.json" \
      condition="$stage" model="$m" benchmark="$bm" \
      -- "$PYTHON" -u -m "leaf.baselines.$stage" --model "$m" --benchmark "$bm" --mode "$MODE" --output "$out" \
      || return
    i=$((i + 1))
  done
}

cell_seed() {  # MODEL BENCHMARK SEED
  local m=$1 bm=$2 s=$3
  roots_of "$m" "$s"
  seed_chain "$CFG_S" "$ROOT_S" "$m" "$bm" "$(seed_minutes "$m" "$bm")" "$PREFIX" || return
  [ "$MODE" = formal ] || return 0
  local out=$ROOT_S/$m/$MODE/baselines/seed_rl/$bm
  local tb
  for tb in $(tests_of "$bm" "$s"); do
    test_eval "$PREFIX" "$CFG_S" "$ROOT_S" "$m" "seed_$bm" "$tb" "$out" "$TEST_MIN_B" test || return
  done
}

# SEED at 20 verifier calls per problem (seed 1, the baselines config and root): training only, its
# GPU hours are the appendix SEED cost comparison
cell_seed_20_calls() {  # MODEL BENCHMARK SEED
  local m=$1 bm=$2 s=$3
  if [ "$s" != 1 ]; then echo "seed_20_calls: seed 1 only (skipping seed $s)" >&2; return 0; fi
  roots_of "$m" "$s"
  seed_chain "$CFG_B" "$ROOT_B" "$m" "$bm" "$(seed_20_calls_minutes "$m" "$bm")" "${PREFIX}_20_calls"
}

# rollout CONFIG ROOT MODEL BENCHMARK: the LEAF rollout on the training set (no update)
rollout() {
  local config=$1 root=$2 m=$3 bm=$4 steps
  if [ "$MODE" = formal ]; then steps=$(cfgv "$config" 'c["formal_steps"]'); else steps=$(cfgv "$config" 'c["smoke_steps"]'); fi
  local out=$root/$m/$MODE/leaf/$LEAF/$bm
  gjob "${PREFIX}_leaf_rollout_$bm" "$config" "$root" "$MODE" "$out" \
    "$(rollout_minutes "$m" "$bm")" gpu "$(lcb_cpus "$bm")" "$out/generation_summary.json" \
    condition="$LEAF" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m leaf.explore.leaf_rollout --condition "$LEAF" --model "$m" \
       --dataset "$root/dataset_$bm.jsonl" --steps "$steps" --output "$out"
}

# leaf_contexts_seed1 MODEL BENCHMARK: the seed-1 LEAF update data as in the paper runs: the MATH
# rollout and its contexts run under leaf_rollout_math500_<m>_seed1.json, the LCB / SciKnowEval
# rollout under leaf_rollout_lcb_sciknoweval_<m>_seed1.json and its contexts under
# leaf_contexts_lcb_sciknoweval_<m>_seed1.json; everything in general/seed1/leaf/<m>
leaf_contexts_seed1() {
  local m=$1 bm=$2 config data
  if [ "$bm" = math500 ]; then
    config=$(cfg "general/leaf_rollout_math500_${m}_seed1")
  else
    config=$(cfg "general/leaf_rollout_lcb_sciknoweval_${m}_seed1")
  fi
  rollout "$config" "$ROOT_L" "$m" "$bm" || return
  [ "$bm" = math500 ] || config=$(cfg "general/leaf_contexts_lcb_sciknoweval_${m}_seed1")
  data=$ROOT_L/$m/$MODE/update_data/$bm
  gjob "${PREFIX}_leaf_contexts_$bm" "$config" "$ROOT_L" "${MODE}_data" "$data" 60 cpu "$(cpus std)" \
    "$data/successful_contexts_summary.json" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m leaf.update.build_contexts --model "$m" --benchmark "$bm" \
       --rollout "$ROOT_L/$m/$MODE/leaf/$LEAF/$bm" --output "$data"
}

cell_leaf() {  # MODEL BENCHMARK SEED
  local m=$1 bm=$2 s=$3
  roots_of "$m" "$s"
  local data=$ROOT_L/$m/$MODE/update_data/$bm
  if [ "$s" = 1 ]; then
    leaf_contexts_seed1 "$m" "$bm" || return
  else
    rollout "$CFG_L" "$ROOT_L" "$m" "$bm" || return
    gjob "${PREFIX}_leaf_contexts_$bm" "$CFG_L" "$ROOT_L" "${MODE}_data" "$data" 60 cpu "$(cpus std)" \
      "$data/contexts_summary.json" model="$m" benchmark="$bm" \
      -- "$PYTHON" -u -m leaf.update.prepare_update_data --model "$m" --benchmark "$bm" \
         --rollout "$ROOT_L/$m/$MODE/leaf/$LEAF/$bm" --output "$data" || return
  fi
  local method=leaf_update minutes out
  minutes=$(hours_min "$(cfgv "$CFG_SFT" "c['update_ablation']['train_hours']['$bm']")")
  update_run "${PREFIX}_${method}_$bm" "$CFG_L" "$ROOT_L" "$m" "$bm" "$method" "$data" "$minutes" || return
  [ "$MODE" = formal ] || return 0
  out=$ROOT_L/$m/$MODE/update/$method/$bm
  local tb
  for tb in $(tests_of "$bm" "$s"); do
    test_eval "$PREFIX" "$CFG_L" "$ROOT_L" "$m" "$(label_of "$method" "$bm")" "$tb" "$out" 120 test \
      || return
  done
}

cell_sft() {  # MODEL BENCHMARK SEED
  local m=$1 bm=$2 s=$3
  roots_of "$m" "$s"
  local data=$ROOT_SFT/$m/$MODE/update_data/$bm limit=()
  if [ "$s" = 1 ]; then
    # leaf.update.build_trace_variants: the (c+, y+) pairs of the seed-1 LEAF update data
    local prep=("$PYTHON" -u -m leaf.update.build_trace_variants --model "$m" --benchmark "$bm")
    if [ "$MODE" != formal ]; then
      prep+=(--output-root "$ROOT_SFT/$m/$MODE/update_data"
             --source-root "$ROOT_L/$m/$MODE/update_data")
    fi
    gjob "${PREFIX}_trace_variants_$bm" "$CFG_SFT" "$ROOT_SFT" "${MODE}_data" "$data" 60 cpu \
      "$(cpus std)" "$data/contexts_summary.json" model="$m" benchmark="$bm" -- "${prep[@]}" || return
  elif [ "$DRY_RUN" != 1 ] && [ ! -e "$data/ystar_pairs.jsonl" ]; then
    echo "sft $m $bm seed $s: no $data/ystar_pairs.jsonl; run the 'leaf' cell first" >&2
    return 1
  fi
  [ "$MODE" = formal ] || limit=(--limit "$(cfgv "$CFG_SFT" 'c["sft_and_traces"]["smoke_problems"]')")
  gjob "${PREFIX}_ystar_rewrite_$bm" "$CFG_SFT" "$ROOT_SFT" "${MODE}_rewrite" "$data/ystar" \
    "$(hours_min "$(cfgv "$CFG_SFT" "c['sft_and_traces']['rewrite_hours']['$bm']")")" gpu "$(lcb_cpus "$bm")" \
    "$data/sft_ystar.jsonl" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m leaf.update.rewrite_ystar --model "$m" --benchmark "$bm" --data "$data" "${limit[@]}" || return
  update_run "${PREFIX}_sft_ystar_$bm" "$CFG_SFT" "$ROOT_SFT" "$m" "$bm" sft_ystar "$data" \
    "$(hours_min "$(cfgv "$CFG_SFT" "c['sft_and_traces']['sft_hours']['$bm']")")" || return
  [ "$MODE" = formal ] || return 0
  local out=$ROOT_SFT/$m/$MODE/update/sft_ystar/$bm tb
  for tb in $(tests_of "$bm" "$s"); do
    test_eval "$PREFIX" "$CFG_SFT" "$ROOT_SFT" "$m" "$(label_of sft_ystar "$bm")" "$tb" "$out" 120 test \
      || return
  done
}

run_cell() {  # METHOD MODEL BENCHMARK SEED
  local method=$1 m=$2 bm=$3 s=$4
  echo "== general_main $method $m $bm seed $s ($MODE)" >&2
  case "$method" in
    base) cell_base "$m" "$bm" "$s" ;;
    grpo|sdpo|score) cell_verl "$m" "$method" "$bm" "$s" ;;
    seed) cell_seed "$m" "$bm" "$s" ;;
    seed_20_calls) cell_seed_20_calls "$m" "$bm" "$s" ;;
    leaf) cell_leaf "$m" "$bm" "$s" ;;
    sft) cell_sft "$m" "$bm" "$s" ;;
  esac
}

# ------------------------------------------------------------------------------------ main
if [ "${1:-}" = _cell ]; then
  shift
  run_cell "$@"
  exit
fi
if [ $# -ne 4 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
models=$(expand_all "$1" "${MODELS[@]}")
methods=$(expand_all "$2" "${METHODS[@]}")
benches=$(expand_all "$3" "${TRAIN_BENCHMARKS[@]}")
seeds=$(expand_all "$4" "${SEEDS[@]}")

first=() later=()
for s in $seeds; do
  for m in $models; do
    for bm in $benches; do
      for method in $methods; do
        if [ "$method" = sft ]; then later+=("sft $m $bm $s"); else first+=("$method $m $bm $s"); fi
      done
    done
  done
done
status=0
run_cells "$0" ${first[@]+"${first[@]}"} || status=1
run_cells "$0" ${later[@]+"${later[@]}"} || status=1
exit "$status"
