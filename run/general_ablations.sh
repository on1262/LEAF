#!/usr/bin/env bash
# Policy-update analyses of the paper (run seed 1): the divergence ablation (reverse KL vs forward
# KL vs JSD), the internalization check (train-set accuracy), the teacher accuracy under successful /
# failure / random rollout traces, and the self-correction of the trained policies.
#
#   bash run/general_ablations.sh MODEL PART BENCHMARK
#
#   MODEL      qwen3_8b | olmo3_7b | all
#   PART       loss | trainset | teacher | selfcorr | all
#              loss      the LEAF update with the divergence swapped: leaf_update_forward_kl,
#                        leaf_update_jsd (MATH keeps the clip tau = 1e-3), on the seed-1 LEAF update
#                        data, then the test (labels leaf_update_{forward_kl,jsd}_<bm>; the reverse-KL
#                        row is the seed-1 LEAF run of run/general_main.sh)
#              trainset  the base model on the full training sets (leaf.grading.test_eval --test-set,
#                        LiveCodeBench graded on all tests of the training problems), then
#                        leaf.analysis.internalization_eval: GPU judge of the LEAF update runs' last-step
#                        and last-rollout SciKnowEval samples, CPU summarize ->
#                        <root>/<m>/trainset/trainacc/summary.json
#                        (TRAINSET_LABELS="base leaf_update" also evaluates the trained LEAF update runs)
#              teacher   the base model as teacher on the update problems under successful / failure /
#                        random traces (leaf.analysis.teacher_accuracy generate -> grade)
#              selfcorr  LEAF (condition leaf, 10 steps) on the training set with the trained LEAF update
#                        and SFT-on-y* adapters, then leaf.analysis.self_correction_eval summaries
#                        (also of the base model = steps 1-10 of the update data's rollout) into
#                        $REPORT_ROOT/traces/selfcorr/<model>/<label>_<benchmark>.json
#   BENCHMARK  lcb | math500 | sciknoweval | all   (training set; trainset runs all three at once)
#
# Examples
#   bash run/general_ablations.sh qwen3_8b loss math500
#   GPUS="0 1 2 3" bash run/general_ablations.sh all all all
#   DRY_RUN=1 bash run/general_ablations.sh all all all
#
# Prerequisites: run/general_main.sh seed 1, METHOD leaf (update data and runs: loss, trainset,
# selfcorr) and sft (leaf.update.build_trace_variants contexts and the SFT runs: teacher, selfcorr).
# Configs and output roots: configs/ablation/update_<m>.json -> $RESULTS_ROOT/ablation/update/<m>
# (loss, trainset), configs/general/sft_<m>_seed1.json -> $RESULTS_ROOT/ablation/traces/<m>
# (teacher, selfcorr). Every step is skipped once its output marker exists (FORCE=1 re-runs it).
# More environment: run/common.sh, run/general_common.sh (GPUS, NUM_CPUS, TIME_CAPS, MBPP).
set -euo pipefail
export _GENERAL_USER_NUM_CPUS=${_GENERAL_USER_NUM_CPUS-${NUM_CPUS-}}
# shellcheck source=run/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=run/general_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/general_common.sh"

PARTS=(loss trainset teacher selfcorr)
LOSSES=(forward_kl jsd)
CONDITIONS=(successful failure random)
SELFCORR_LABELS=(leaf_update sft_ystar)
R=$RESULTS_ROOT

case "$MODE" in
  formal) ;;
  smoke_full) echo "MODE=smoke_full: only the loss part runs (smoke_steps updates, no test)" >&2 ;;
  *) echo "general_ablations.sh supports MODE=formal and MODE=smoke_full" >&2; exit 2 ;;
esac

# seed-1 locations shared with run/general_main.sh
leaf_data() { printf '%s/general/seed1/leaf/%s/%s/formal/update_data/%s\n' "$R" "$1" "$1" "$2"; }
leaf_run() { printf '%s/general/seed1/leaf/%s/%s/formal/update/leaf_update/%s\n' "$R" "$1" "$1" "$2"; }
sft_run() { printf '%s/general/seed1/sft/%s/%s/formal/update/sft_ystar/%s\n' "$R" "$1" "$1" "$2"; }
context_data() { printf '%s/general/seed1/sft/%s/%s/formal/update_data/%s\n' "$R" "$1" "$1" "$2"; }
# dataset of the update data's source rollout (its manifest "dataset"): same problem order and seeds
rollout_dataset() { printf '%s/general/seed1/leaf/%s/dataset_%s.jsonl\n' "$R" "$1" "$2"; }

cell_loss() {  # MODEL BENCHMARK LOSS
  local m=$1 bm=$2 loss=$3 config root method prefix
  config=$(cfg "ablation/update_$m"); root=$R/ablation/update/$m
  prefix=ablation_update_$m
  method=leaf_update_$loss
  update_run "${prefix}_${method}_$bm" "$config" "$root" "$m" "$bm" "$method" \
    "$(leaf_data "$m" "$bm")" "$(hours_min "$(cfgv "$config" "c['update_ablation']['train_hours']['$bm']")")" || return
  [ "$MODE" = formal ] || return 0
  local out=$root/$m/$MODE/update/$method/$bm tb
  for tb in $(tests_of "$bm" 1); do
    test_eval "$prefix" "$config" "$root" "$m" "$(label_of "$method" "$bm")" "$tb" "$out" 120 test || return
  done
}

cell_trainset() {  # MODEL BENCHMARK...
  local m=$1 config root bm label status=0 prefix
  shift
  config=$(cfg "ablation/update_$m"); root=$R/ablation/update/$m
  prefix=ablation_update_$m
  local kind=trainset trainacc=$root/$m/trainset/trainacc
  for bm in "$@"; do
    local eval_args=(--test-set "$DATA_ROOT/train/$m/dataset_$bm.jsonl"
                     --max-num-seqs "$(cfgv "$config" "c['update_ablation']['trainset_max_num_seqs']['$bm']")")
    [ "$bm" = lcb ] && eval_args+=(--ground-truth "$DATA_ROOT/lcb/lcb_ground_truth_full.parquet")
    for label in ${TRAINSET_LABELS:-base}; do
      local run=""
      [ "$label" = leaf_update ] && run=$(leaf_run "$m" "$bm")
      TEST_GRADE_MINUTES=$(hours_min "$(cfgv "$config" "c['update_ablation']['trainset_grade_hours']['$bm']")") \
        test_eval "$prefix" "$config" "$root" "$m" "$label" "$bm" "$run" \
        "$(hours_min "$(cfgv "$config" "c['update_ablation']['trainset_hours']['$bm']")")" "$kind" "${eval_args[@]}" \
        || status=1
    done
  done
  # judge (GPU) the update runs' SciKnowEval samples, then summarize (CPU) once every grade ended
  local bench_args=()
  for bm in "$@"; do bench_args+=(--benchmark "$bm"); done
  if member sciknoweval "$@"; then
    gjob "${prefix}_internalization_judge" "$config" "$root" "${kind}_trainacc_judge" "$trainacc" 30 gpu "$(cpus std)" \
      "$trainacc/update_samples_sciknoweval_judged.jsonl" model="$m" \
      -- "$PYTHON" -u -m leaf.analysis.internalization_eval judge --model "$m" --output "$trainacc" || status=1
    gjob "${prefix}_internalization_judge_last" "$config" "$root" "${kind}_trainacc_last_judge" "$trainacc" 60 gpu \
      "$(cpus std)" "$trainacc/update_last_rollout_sciknoweval_judged.jsonl" model="$m" \
      -- "$PYTHON" -u -m leaf.analysis.internalization_eval judge --model "$m" --output "$trainacc" \
         --samples last_rollout --trainset-root "$root/$m/$kind" || status=1
  fi
  # summary.json is the summary's completion marker: write it only from complete inputs, so a
  # re-run after a failed evaluation or judge step does not keep "missing" cells
  if [ "$status" -ne 0 ]; then
    echo "trainset $m: an evaluation or judge step did not complete; summary not written (re-run to resume)" >&2
    return "$status"
  fi
  local wargs
  read -r -a wargs <<< "$(grade_worker_args "$config")"
  gjob "${prefix}_internalization_summary" "$config" "$root" "${kind}_trainacc_summary" "$trainacc" 120 cpu \
    "$(cpus big)" "$trainacc/summary.json" model="$m" \
    -- "$PYTHON" -u -m leaf.analysis.internalization_eval summarize --model "$m" --output "$trainacc" \
       --trainset-root "$root/$m/$kind" "${bench_args[@]}" ${wargs[@]+"${wargs[@]}"}
}

cell_teacher() {  # MODEL BENCHMARK CONDITION
  local m=$1 bm=$2 cond=$3 config root
  config=$(cfg "general/sft_${m}_seed1"); root=$R/ablation/traces/$m
  local data out
  data=$(context_data "$m" "$bm")
  out=$root/$m/teacher/$cond/$bm
  local common=(--model "$m" --benchmark "$bm" --data "$data" --condition "$cond" --output "$out")
  local name=ablation_traces_${m}_teacher_${cond}_$bm
  gjob "${name}_generate" "$config" "$root" formal_teacher_generate "$out" \
    "$(hours_min "$(cfgv "$config" "c['sft_and_traces']['teacher_hours']['$bm']")")" gpu "$(cpus std)" \
    "$out/generation_summary.json" condition="$cond" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m leaf.analysis.teacher_accuracy generate "${common[@]}" \
       --max-num-seqs "$(cfgv "$config" "c['sft_and_traces']['teacher_max_num_seqs']['$bm']")" || return
  local wargs
  read -r -a wargs <<< "$(grade_worker_args "$config")"
  gjob "${name}_grade" "$config" "$root" formal_teacher_grade "$out" 120 cpu "$(lcb_cpus "$bm")" \
    "$out/summary.json" condition="$cond" model="$m" benchmark="$bm" \
    -- "$PYTHON" -u -m leaf.analysis.teacher_accuracy grade "${common[@]}" ${wargs[@]+"${wargs[@]}"}
}

# selfcorr_summary MODEL BENCHMARK LABEL RUN: leaf.analysis.self_correction_eval on the update problems
selfcorr_summary() {
  local m=$1 bm=$2 label=$3 run=$4 steps=$5
  local target=$REPORT_ROOT/traces/selfcorr/$m/${label}_$bm.json
  if job_done "$target"; then return 0; fi
  if [ "$DRY_RUN" != 1 ] && [ ! -e "$run" ]; then echo "selfcorr $label $m $bm: no run $run" >&2; return 1; fi
  local cmd=("$PYTHON" -u -m leaf.analysis.self_correction_eval summarize --run "$run" --steps "$steps"
             --uids-file "$(context_data "$m" "$bm")/successful_contexts.jsonl" --output "$target")
  if [ "$DRY_RUN" = 1 ]; then _do "${cmd[@]}"; return 0; fi
  ( cd "$REPO" && "${cmd[@]}" > /dev/null ) && echo "selfcorr summary: $target" >&2
}

cell_selfcorr() {  # MODEL BENCHMARK LABEL
  local m=$1 bm=$2 label=$3 config root run
  config=$(cfg "general/sft_${m}_seed1"); root=$R/ablation/traces/$m
  local steps cond out minutes
  steps=$(cfgv "$config" 'c["sft_and_traces"]["selfcorr_steps"]')
  cond=$(cfgv "$config" 'c["sft_and_traces"]["selfcorr_condition"]')
  minutes=$(hours_min "$(cfgv "$config" "c['sft_and_traces']['selfcorr_hours']['$bm']")" \
                      "$(cfgv "$config" "c['sft_and_traces']['selfcorr_chunks']['$bm']")")
  if [ "$label" = leaf_update ]; then run=$(leaf_run "$m" "$bm"); else run=$(sft_run "$m" "$bm"); fi
  out=$root/$m/selfcorr/$label/$bm
  if [ "$DRY_RUN" != 1 ] && [ ! -e "$run/COMPLETE.json" ]; then
    echo "selfcorr $label $m $bm: $run has no COMPLETE.json (run/general_main.sh seed 1 first)" >&2
    return 1
  fi
  gjob "ablation_traces_${m}_selfcorr_${label}_$bm" "$config" "$root" selfcorr "$out" "$minutes" gpu \
    "$(lcb_cpus "$bm")" "$out/generation_summary.json" label="$label" model="$m" benchmark="$bm" source_run="$run" \
    -- "$PYTHON" -u -m leaf.explore.leaf_rollout --condition "$cond" --model "$m" --steps "$steps" \
       --dataset "$(rollout_dataset "$m" "$bm")" --output "$out" --run-dir "$run" || return
  selfcorr_summary "$m" "$bm" "$label" "$out" "$steps" || return
  # the base model: steps 1-10 of the rollout that produced the update data
  selfcorr_summary "$m" "$bm" base "$R/general/seed1/leaf/$m/$m/formal/leaf/$LEAF/$bm" "$steps"
}

run_cell() {  # PART MODEL ARGS...
  local part=$1
  shift
  echo "== general_ablations $part $* ($MODE)" >&2
  case "$part" in
    loss) cell_loss "$@" ;;
    trainset) cell_trainset "$@" ;;
    teacher) cell_teacher "$@" ;;
    selfcorr) cell_selfcorr "$@" ;;
  esac
}

# ------------------------------------------------------------------------------------ main
if [ "${1:-}" = _cell ]; then
  shift
  run_cell "$@"
  exit
fi
if [ $# -ne 3 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
models=$(expand_all "$1" "${MODELS[@]}")
parts=$(expand_all "$2" "${PARTS[@]}")
benches=$(expand_all "$3" "${TRAIN_BENCHMARKS[@]}")

cells=()
for part in $parts; do
  if [ "$MODE" != formal ] && [ "$part" != loss ]; then continue; fi
  for m in $models; do
    case "$part" in
      trainset) cells+=("trainset $m ${benches//$'\n'/ }") ;;
      *)
        for bm in $benches; do
          case "$part" in
            loss) for x in "${LOSSES[@]}"; do cells+=("loss $m $bm $x"); done ;;
            teacher) for x in "${CONDITIONS[@]}"; do cells+=("teacher $m $bm $x"); done ;;
            selfcorr) for x in "${SELFCORR_LABELS[@]}"; do cells+=("selfcorr $m $bm $x"); done ;;
          esac
        done ;;
    esac
  done
done
run_cells "$0" ${cells[@]+"${cells[@]}"}
