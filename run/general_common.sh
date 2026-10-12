# shellcheck shell=bash disable=SC2034  # sourced: the constants are used by the run scripts
# Helpers shared by run/general_main.sh and run/general_ablations.sh. Source it after
# run/common.sh, do not execute it.
#
# Environment read here (all optional):
#   GPUS       GPU ids for a multi-cell run, e.g. "0 1 2 3": one cell per GPU at a time
#              (CUDA_VISIBLE_DEVICES of each cell = one id); unset = cells run one after another
#              on the GPU(s) CUDA_VISIBLE_DEVICES already selects
#   NUM_CPUS   CPUs for Ray / verifier pools; default per job = the original allocation
#              (48 for LiveCodeBench jobs and the train-set summary, 16 otherwise), capped at
#              the CPUs of this machine
#   TIME_CAPS  1 (default): every job runs under the wall-clock cap of the original job chain
#              (the sum of its chunks); 0: no cap. No runner of these experiments changes its
#              result with the cap: a capped run that stops early resumes on the next run of
#              the script, and the steps after it wait until it has completed.
#   MBPP       1: also evaluate the seed-1 base model and LiveCodeBench-trained checkpoints on
#              MBPP+ (the out-of-distribution test of the seed-1 runs; not part of the paper
#              tables). Default 0.

# NUM_CPUS as the user gave it, captured by each script before run/common.sh sets its default
_GENERAL_USER_NUM_CPUS=${_GENERAL_USER_NUM_CPUS-}

MODELS=(qwen3_8b olmo3_7b)
TRAIN_BENCHMARKS=(lcb math500 sciknoweval)
SEEDS=(1 2 3)
LEAF=leaf
SEED_STAGES=(seed_collection seed_sft seed_rl)

# cfg NAME: configs/NAME.json
cfg() { printf '%s/configs/%s.json\n' "$REPO" "$1"; }

# test label of an update method on a training set: <method>_<training set> (the directory names
# the report reads)
label_of() { printf '%s_%s\n' "$1" "$2"; }
# tests_of BENCHMARK SEED: test sets of a training set: the in-distribution one, plus MBPP+ for
# LiveCodeBench under MBPP=1 (seed 1 only, as in the paper runs)
tests_of() {
  if [ "$1" = lcb ] && [ "${MBPP:-0}" = 1 ] && [ "$2" = 1 ]; then echo "lcb mbpp"; else echo "$1"; fi
}

# cfgv CONFIG EXPR: a value of a config (placeholders expanded); EXPR is a python expression of c
cfgv() {
  "$PYTHON" -c 'import sys; from leaf import paths; c = paths.load_json(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"
}
# hours_min HOURS [CHUNKS]: minutes of CHUNKS chunks of HOURS hours
hours_min() { "$PYTHON" -c 'import sys; print(round(60 * float(sys.argv[1]) * int(sys.argv[2])))' "$1" "${2:-1}"; }

# cpus KIND: CPUs of a job; KIND = big (48: LiveCodeBench jobs, train-set summary) | std (16)
cpus() {
  local want=16 have
  [ "$1" = big ] && want=48
  if [ -n "$_GENERAL_USER_NUM_CPUS" ]; then printf '%s\n' "$_GENERAL_USER_NUM_CPUS"; return; fi
  have=$(nproc 2>/dev/null || echo "$want")
  if [ "$have" -lt "$want" ]; then want=$have; fi
  printf '%s\n' "$want"
}
lcb_cpus() { if [ "$1" = lcb ]; then cpus big; else cpus std; fi; }

# _do CMD...: run a plain (non-job) command, or print it under DRY_RUN=1
_do() {
  if [ "$DRY_RUN" = 1 ]; then printf 'DRY:'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

# gjob NAME CONFIG LOGROOT STAGE OUTPUT MINUTES KIND CPUS MARKER [KEY=VALUE ...] -- COMMAND...
# run_job (run/common.sh) of one step: skipped when MARKER exists (FORCE=1 re-runs it), run with
# RUN_KIND=KIND (gpu | cpu) and NUM_CPUS=CPUS; returns non-zero when the step failed or stopped
# before MARKER was written (a capped run: the next run of the script resumes it), so the steps
# that depend on it do not start. Under DRY_RUN=1 it prints a "DRYJOB:" line, then the command.
gjob() {
  local name=$1 config=$2 logroot=$3 stage=$4 output=$5 minutes=$6 kind=$7 ncpu=$8 marker=$9
  shift 9
  if [ -n "$marker" ] && job_done "$marker"; then
    echo "$name: done ($marker)" >&2
    return 0
  fi
  [ "${TIME_CAPS:-1}" = 1 ] || minutes=0
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRYJOB: name=%s stage=%s kind=%s cpus=%s minutes=%s output=%s\n' \
      "$name" "$stage" "$kind" "$ncpu" "$minutes" "$output"
  fi
  NUM_CPUS=$ncpu RUN_KIND=$kind run_job "$name" "$config" "$logroot" "$stage" "$output" "$minutes" "$@" \
    || return
  if [ "$DRY_RUN" != 1 ] && [ -n "$marker" ] && [ ! -e "$marker" ]; then
    echo "$name: stopped before completion ($marker missing); run the script again to resume" >&2
    return 3
  fi
}

# grade_worker_args CONFIG: "--workers N" when the config's verifier_workers exceeds the CPUs
# available (cap_workers), nothing otherwise (the grader then uses verifier_workers, as in the paper runs)
grade_worker_args() {
  local want have
  want=$(cfgv "$1" 'c["verifier_workers"]')
  have=$(cap_workers "$want" "$_GENERAL_USER_NUM_CPUS")
  if [ "$have" -lt "$want" ]; then printf -- '--workers %s\n' "$have"; fi
}

# test_eval PREFIX CONFIG ROOT MODEL LABEL TEST_BENCHMARK RUN_DIR GEN_MINUTES KIND [EVAL_ARG ...]
# leaf.grading.test_eval generate (GPU) -> grade (CPU) of one checkpoint (RUN_DIR; "" = the base
# model) into ROOT/MODEL/KIND/LABEL/TEST_BENCHMARK (KIND = test | trainset). EVAL_ARGs go to both.
# Grading cap: 120 minutes (TEST_GRADE_MINUTES overrides, e.g. the train-set evaluations); grading
# CPUs: 48 for LiveCodeBench, 16 otherwise (GRADE_CPUS overrides).
test_eval() {
  local prefix=$1 config=$2 root=$3 model=$4 label=$5 bench=$6 run=$7 minutes=$8 kind=$9
  shift 9
  local out=$root/$model/$kind/$label/$bench name
  name=${prefix}_${kind}_${label}_$bench
  local common=(--model "$model" --benchmark "$bench" --output "$out" "$@")
  local gen=("$PYTHON" -u -m leaf.grading.test_eval generate "${common[@]}")
  local fields=(condition="$label" model="$model" benchmark="$bench" source_run=null adapter=null)
  if [ -n "$run" ]; then
    gen+=(--run-dir "$run")
    fields=(condition="$label" model="$model" benchmark="$bench" source_run="$run" adapter=null)
  fi
  gjob "${name}_generate" "$config" "$root" "${kind}_generate" "$out" "$minutes" gpu "$(cpus std)" \
    "$out/generation_summary.json" "${fields[@]}" -- "${gen[@]}" || return
  local grade_cpus wargs
  grade_cpus=${GRADE_CPUS:-$(lcb_cpus "$bench")}
  read -r -a wargs <<< "$(grade_worker_args "$config")"
  gjob "${name}_grade" "$config" "$root" "${kind}_grade" "$out" "${TEST_GRADE_MINUTES:-120}" cpu "$grade_cpus" \
    "$out/summary.json" "${fields[@]}" \
    -- "$PYTHON" -u -m leaf.grading.test_eval grade "${common[@]}" ${wargs[@]+"${wargs[@]}"}
}

# update_run NAME CONFIG ROOT MODEL BENCHMARK METHOD DATA MINUTES: leaf.update.run_update (a LEAF
# update method or SFT on y*) into ROOT/MODEL/MODE/update/METHOD/BENCHMARK
update_run() {
  local name=$1 config=$2 root=$3 model=$4 bench=$5 method=$6 data=$7 minutes=$8
  local out=$root/$model/$MODE/update/$method/$bench
  gjob "$name" "$config" "$root" "$MODE" "$out" "$minutes" gpu "$(cpus std)" "$out/COMPLETE.json" \
    kind=update method="$method" model="$model" benchmark="$bench" data="$data" init_run=null \
    -- "$PYTHON" -u -m leaf.update.run_update --model "$model" --benchmark "$bench" \
       --method "$method" --data "$data" --mode "$MODE" --output "$out"
}

# member VALUE ITEM...: VALUE is one of the ITEMs
member() {
  local value=$1 item
  shift
  for item in "$@"; do [ "$item" = "$value" ] && return 0; done
  return 1
}
# expand_all VALUE ALL...: "all" -> every item of ALL, otherwise VALUE itself (checked)
expand_all() {
  local value=$1
  shift
  if [ "$value" = all ]; then printf '%s\n' "$@"; return; fi
  member "$value" "$@" || { echo "unknown value '$value' (one of: $* all)" >&2; return 2; }
  printf '%s\n' "$value"
}

# run_cells SCRIPT CELL...: "bash SCRIPT _cell <cell words>" for every cell, GPUS of them at a
# time (xargs); each cell gets its own GPU while it runs. Returns non-zero if a cell failed.
run_cells() {
  local script=$1
  shift
  [ $# -gt 0 ] || return 0
  local gpus status=0 cell
  read -r -a gpus <<< "${GPUS:-}"
  if [ "$DRY_RUN" = 1 ] || [ ${#gpus[@]} -le 1 ]; then
    for cell in "$@"; do
      if [ ${#gpus[@]} -eq 1 ]; then
        # shellcheck disable=SC2086  # a cell is a list of words
        CUDA_VISIBLE_DEVICES=${gpus[0]} bash "$script" _cell $cell || status=1
      else
        # shellcheck disable=SC2086
        bash "$script" _cell $cell || status=1
      fi
    done
    return "$status"
  fi
  # several cells at once: run_job's Ray clean-up only stops the finished job's own cluster
  # shellcheck disable=SC2016  # expanded by the inner shell
  printf '%s\n' "$@" | xargs -P "${#gpus[@]}" -L 1 --process-slot-var=GENERAL_SLOT bash -c '
    gpus=(${GPUS:-})
    export CUDA_VISIBLE_DEVICES=${gpus[$GENERAL_SLOT]}
    exec bash "$0" _cell "$@"' "$script"
}
