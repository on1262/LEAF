# shellcheck shell=bash
# Helpers shared by run/explore_main.sh, run/explore_ablations.sh and run/explore_compression.sh.
# Source it after run/common.sh, do not execute it.
#
# Environment read here (all optional):
#   STEPS     gpu,grade (default) | gpu | grade: run the GPU job of a cell, its CPU grading, or both
#   GPUS      GPU ids for a multi-cell run, e.g. "0 1 2 3": one cell per GPU at a time
#             (CUDA_VISIBLE_DEVICES of each cell = one id); unset = cells run one after another
#             on the GPUs CUDA_VISIBLE_DEVICES already selects (one GPU per cell)
#   JOBS      cells in parallel when STEPS=grade (CPU only; default 1)
#   NUM_CPUS  CPUs for Ray / verifier pools; default per cell = the original allocation
#             (48 on LCB and MBPP+, 16 otherwise), capped at the CPUs of this machine
#
# Wall-clock budget of a cell. The exploration protocol gives every cell a fixed GPU wall clock
# (min(24 steps, 5 GPU h)); a run that reaches it stops there and is graded on the steps it
# finished. run_capped keeps the seconds a cell has used in LOGROOT/wallclock/<cell>.tsv, so a
# re-run of a script resumes an interrupted (FAILED) cell with the rest of its budget, never
# continues a cell that reached its budget (TIMEOUT), and skips a COMPLETED one.
# FORCE=1 ignores (and restarts) that record.

# NUM_CPUS as the user gave it, captured by each script before run/common.sh sets its default
_EXPLORE_USER_NUM_CPUS=${_EXPLORE_USER_NUM_CPUS-}

# cfg_get CONFIG KEY: one top-level value of a config (placeholders expanded)
cfg_get() {
  "$PYTHON" -c 'import sys; from leaf import paths; print(paths.load_json(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}

# step_on gpu|grade: whether STEPS selects it
step_on() {
  case ",${STEPS:-gpu,grade}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# cell_cpus BENCHMARK: CPUs of the original job (48 on LCB / MBPP+, 16 otherwise), capped at nproc
cell_cpus() {
  local want=16 have
  case "$1" in lcb|mbpp) want=48 ;; esac
  if [ -n "$_EXPLORE_USER_NUM_CPUS" ]; then printf '%s\n' "$_EXPLORE_USER_NUM_CPUS"; return; fi
  have=$(nproc 2>/dev/null || echo "$want")
  if [ "$have" -lt "$want" ]; then want=$have; fi
  printf '%s\n' "$want"
}

_wallclock_file() {  # LOGROOT OUTPUT -> the budget record of the cell
  local rel=${2#"$1"/}
  printf '%s/wallclock/%s.tsv\n' "$1" "${rel//\//__}"
}

# cell_state LOGROOT OUTPUT: COMPLETED | TIMEOUT | FAILED | NONE (last recorded GPU state)
cell_state() {
  local file
  file=$(_wallclock_file "$1" "$2")
  if [ "${FORCE:-0}" = 1 ] || [ ! -s "$file" ]; then echo NONE; return; fi
  tail -n 1 "$file" | cut -f 2
}

# dry_meta NAME STAGE MINUTES OUTPUT: under DRY_RUN, the job line printed before run_job's DRY line
dry_meta() {
  [ "$DRY_RUN" = 1 ] || return 0
  printf 'DRYJOB: name=%s stage=%s minutes=%s output=%s\n' "$1" "$2" "$3" "$4"
}

# run_capped NAME CONFIG LOGROOT STAGE OUTPUT BUDGET_MINUTES [KEY=VALUE ...] -- COMMAND...
# run_job with a cell budget kept across re-runs (see above). Returns 0 when the cell is
# COMPLETED or used its budget (TIMEOUT), non-zero when the job failed.
run_capped() {
  local name=$1 config=$2 logroot=$3 stage=$4 output=$5 budget=$6
  if [ "$DRY_RUN" = 1 ]; then
    dry_meta "$name" "$stage" "$budget" "$output"
    run_job "$@"
    return
  fi
  shift 6
  local file used=0 remaining state start end status=0
  file=$(_wallclock_file "$logroot" "$output")
  mkdir -p "$(dirname "$file")"
  if [ "${FORCE:-0}" = 1 ]; then rm -f "$file"; fi
  state=$(cell_state "$logroot" "$output")
  case "$state" in
    COMPLETED) echo "$name: completed earlier ($file)" >&2; return 0 ;;
    TIMEOUT) echo "$name: used its ${budget}-minute budget earlier ($file)" >&2; return 0 ;;
  esac
  if [ -s "$file" ]; then used=$(awk -F '\t' '{s += $1} END {print s + 0}' "$file"); fi
  remaining=$(( (budget * 60 - used) / 60 ))
  if [ "$remaining" -lt 1 ]; then
    printf '%s\t%s\n' 0 TIMEOUT >> "$file"
    echo "$name: no budget left (${used}s of ${budget} min used)" >&2
    return 0
  fi
  start=$(date +%s)
  run_job "$name" "$config" "$logroot" "$stage" "$output" "$remaining" "$@" || status=$?
  end=$(date +%s)
  state=COMPLETED
  if [ "$status" -ne 0 ]; then
    state=FAILED
  elif [ $((end - start)) -ge $((remaining * 60)) ]; then
    state=TIMEOUT
  fi
  printf '%s\t%s\n' $((end - start)) "$state" >> "$file"
  return "$status"
}

# fresh MARKER LOGROOT OUTPUT: MARKER exists and is newer than the cell's last GPU job
# (a resumed GPU run makes its old grading stale). Always false under DRY_RUN or FORCE=1.
fresh() {
  local marker=$1 file
  [ "$DRY_RUN" = 1 ] && return 1
  job_done "$marker" || return 1
  file=$(_wallclock_file "$2" "$3")
  [ ! -e "$file" ] || [ "$marker" -nt "$file" ]
}

# grade_cell NAME_PREFIX CONFIG LOGROOT OUTPUT BENCHMARK KEY=VALUE... -- GRADE_ARGS...
# CPU grading of one exploration cell (jobs NAME_PREFIX_grade, NAME_PREFIX_private): leaf.grading.grade_explore_cell
# (public verifier, MBPP+ private tests, and the aligned steps of a run stopped by its budget),
# then on LCB the private-test regrade (leaf.grading.grade_lcb_private, 20 workers) when the
# grade succeeded. Worker counts are capped at the CPUs available (cap_workers in run/common.sh).
grade_cell() {
  local prefix=$1 config=$2 logroot=$3 output=$4 bench=$5
  shift 5
  local fields=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do fields+=("$1"); shift; done
  shift
  if ! fresh "$output/rollout_summary.json" "$logroot" "$output"; then
    dry_meta "${prefix}_grade" "${MODE}_grade" 0 "$output"
    RUN_KIND=cpu run_job "${prefix}_grade" "$config" "$logroot" "${MODE}_grade" "$output" 0 "${fields[@]}" \
      -- "$PYTHON" -u -m "$@" || return
  fi
  if [ "$bench" = lcb ] && ! fresh "$output/rollout_summary_lcbprivate.json" "$logroot" "$output"; then
    dry_meta "${prefix}_private" "${MODE}_lcbprivate" 0 "$output"
    RUN_KIND=cpu run_job "${prefix}_private" "$config" "$logroot" "${MODE}_lcbprivate" "$output" 0 "${fields[@]}" \
      -- "$PYTHON" -u -m leaf.grading.grade_lcb_private --config "$config" --cells "$output" --workers "$(cap_workers 20 "$_EXPLORE_USER_NUM_CPUS")"
  fi
}

# run_cells SCRIPT CELL...: run "bash SCRIPT <cell words>" for every cell, in parallel over GPUS
# (or JOBS for STEPS=grade) with xargs; each cell gets its own GPU while it runs.
run_cells() {
  local script=$1
  shift
  local gpus slots status=0 cell
  read -r -a gpus <<< "${GPUS:-}"
  if step_on gpu; then slots=${#gpus[@]}; else slots=${JOBS:-1}; gpus=(); fi
  if [ "$DRY_RUN" = 1 ] || [ "$slots" -le 1 ]; then
    for cell in "$@"; do
      if [ ${#gpus[@]} -eq 1 ]; then
        # shellcheck disable=SC2086  # a cell is a list of words
        CUDA_VISIBLE_DEVICES=${gpus[0]} bash "$script" $cell || status=1
      else
        # shellcheck disable=SC2086
        bash "$script" $cell || status=1
      fi
    done
    return "$status"
  fi
  # several cells at once: run_job's Ray clean-up only stops the finished job's own cluster
  # xargs numbers its parallel slots 0..slots-1 (EXPLORE_SLOT): slot i runs on GPU i of the list
  # shellcheck disable=SC2016  # expanded by the inner shell
  printf '%s\n' "$@" | _EXPLORE_GPUS="${gpus[*]}" xargs -P "$slots" -L 1 --process-slot-var=EXPLORE_SLOT bash -c '
    read -r -a gpus <<< "$_EXPLORE_GPUS"
    if [ ${#gpus[@]} -gt 0 ]; then export CUDA_VISIBLE_DEVICES=${gpus[$EXPLORE_SLOT]}; fi
    exec bash "$0" "$@"' "$script"
}

# expand_all VALUE ITEM...: "all" -> every ITEM, otherwise VALUE itself
expand_all() {
  local value=$1
  shift
  if [ "$value" = all ]; then printf '%s\n' "$@"; else printf '%s\n' "$value"; fi
}

# member VALUE ITEM...: VALUE is one of the ITEMs
member() {
  local value=$1 item
  shift
  for item in "$@"; do [ "$item" = "$value" ] && return 0; done
  return 1
}
