# Shared environment and job runner of the run/*.sh scripts. Source it, do not execute it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Environment (all optional):
#   PYTHON        python interpreter                           (python)
#   DATA_ROOT     frozen datasets / upstream sources           ($REPO/data)
#   RESULTS_ROOT  run outputs                                  ($REPO/results)
#   MODEL_ROOT    base models                                  ($REPO/models)
#   REPORT_ROOT   report tables / caches / figures             ($RESULTS_ROOT/report)
#   CACHE_ROOT    HF / compile / Ray caches                    ($RESULTS_ROOT/cache)
#   MODE          formal | smoke | smoke_full, forwarded as --mode by the run scripts (formal)
#   DRY_RUN=1     print every command ("DRY: ...") instead of running it
#   FORCE=1       run a job even if job_done finds its completion marker
#   NUM_CPUS      CPUs given to Ray / verifier pools            (nproc)
#   KILL_GRACE    seconds between TERM and KILL at a time cap   (120)
#   RUN_KIND      gpu | cpu: environment of the job (gpu)
#   RAY_STOP      auto | 1 | all | 0: after the job, stop the Ray processes of THIS job (auto = only
#                 when the job started a Ray cluster, 1 = always); all = `ray stop --force`, which
#                 stops every Ray cluster of the user on the machine; 0 = never
#   REQUIRE_THIRD_PARTY  1 | 0: abort when the patched verl fork is missing (1; the data and
#                 report scripts set 0 because they import no third-party code)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
export PYTHON="${PYTHON:-python}"

_abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

export DATA_ROOT="$(_abspath "${DATA_ROOT:-$REPO/data}")"
export RESULTS_ROOT="$(_abspath "${RESULTS_ROOT:-$REPO/results}")"
export MODEL_ROOT="$(_abspath "${MODEL_ROOT:-$REPO/models}")"
export REPORT_ROOT="$(_abspath "${REPORT_ROOT:-$RESULTS_ROOT/report}")"
export CACHE_ROOT="$(_abspath "${CACHE_ROOT:-$RESULTS_ROOT/cache}")"
export MODE="${MODE:-formal}"
case "$MODE" in
  formal|smoke|smoke_full) ;;
  *) echo "MODE must be formal, smoke or smoke_full (got '$MODE')" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="$REPO:$REPO/third_party/SDPO:$REPO/third_party/LiveCodeBench${PYTHONPATH:+:$PYTHONPATH}"
# Without the fork, python would silently import a stock verl wheel (if one is installed).
if [ "${REQUIRE_THIRD_PARTY:-1}" = 1 ] && [ "$DRY_RUN" != 1 ] && [ ! -d "$REPO/third_party/SDPO/verl" ]; then
  echo "$REPO/third_party/SDPO/verl is missing: run 'bash setup.sh' first" >&2
  return 1 2>/dev/null || exit 1
fi
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TMPDIR="${TMPDIR:-/tmp}"
export NUM_CPUS="${NUM_CPUS:-$(nproc 2>/dev/null || echo 1)}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export VLLM_USE_V1=1

# The job environment of the GPU and the CPU (grading) jobs of the original runs.
_job_env() {
  if [ "${RUN_KIND:-gpu}" = cpu ]; then
    export TOKENIZERS_PARALLELISM=true
  else
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
  fi
}

# job_done MARKER...: true when every marker file exists (and FORCE is not 1). Every python
# runner resumes from its own state, so re-running a script continues unfinished jobs.
# cap_workers N [LIMIT]: N verifier workers, capped at the CPUs this process may use (nproc, which
# follows the CPU affinity) and at LIMIT when given. The paper's grading jobs had at least as many
# CPUs as workers, so on such a machine this returns N unchanged; on a smaller one it keeps the
# pool from oversubscribing the CPUs, which would make time-limited verdicts fire spuriously.
cap_workers() {
  local n=$1 have
  have=$(nproc 2>/dev/null || echo "$n")
  [ "$have" -ge "$n" ] || n=$have
  if [ -n "${2:-}" ] && [ "$2" -lt "$n" ]; then n=$2; fi
  printf '%s\n' "$n"
}

job_done() {
  if [ "${FORCE:-0}" = 1 ]; then return 1; fi
  local marker
  for marker in "$@"; do
    [ -e "$marker" ] || return 1
  done
  return 0
}

_new_job_id() {
  # sortable and unique per machine: "local" + nanoseconds since the epoch
  local id
  id="local$(date +%s%N)"
  while [ "$id" = "${_LAST_JOB_ID:-}" ]; do id="local$(date +%s%N)"; done
  _LAST_JOB_ID=$id
  printf '%s\n' "$id"
}

_gpu_name() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1
}

# _jobs_row FILE JSON: append a row (a new job) or replace the row of the same job.
_jobs_row() {
  "$PYTHON" - "$1" "$2" <<'PY'
import fcntl, json, os, sys
path, row = sys.argv[1], json.loads(sys.argv[2])
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a+", encoding="utf-8") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.seek(0)
    lines = handle.read().splitlines()
    out, replaced = [], False
    for line in lines:
        try:
            old = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if str(old.get("job")) == row["job"]:
            line, replaced = json.dumps(row), True
        out.append(line)
    if not replaced:
        out.append(json.dumps(row))
    handle.seek(0)
    handle.truncate()
    handle.write("".join(item + "\n" for item in out))
PY
}

# _row_json KEY=VALUE...: a JSON object; "null" is null, a value that parses as JSON (numbers,
# booleans, lists) keeps its type, anything else -- and every value of the text fields -- is a string.
_row_json() {
  "$PYTHON" - "$@" <<'PY'
import json, sys
TEXT = {"stage", "job", "name", "output", "config", "command", "model", "benchmark", "condition",
        "method", "label", "kind", "revision", "gpu", "gpu_name", "start", "end", "submitted_at",
        "state", "source_run", "adapter", "data", "init_run", "snapshot", "dependency"}
row = {}
for item in sys.argv[1:]:
    key, _, value = item.partition("=")
    if value == "null":
        row[key] = None
    elif key in TEXT:
        row[key] = value
    else:
        try:
            row[key] = json.loads(value)
        except ValueError:
            row[key] = value
print(json.dumps(row))
PY
}

# run_job NAME CONFIG LOGROOT STAGE OUTPUT MINUTES [KEY=VALUE ...] -- COMMAND...
#
#   NAME     job name; logs go to LOGROOT/logs/NAME_<RUN_JOB_ID>.{out,err}
#   CONFIG   config file exported as LEAF_CONFIG ("" leaves LEAF_CONFIG unset)
#   LOGROOT  the per-model output root <root>/<model>: holds jobs.jsonl and logs/
#   STAGE    the jobs.jsonl "stage" (formal, smoke_full, formal_grade, test, ...)
#   OUTPUT   the run directory the job fills (jobs.jsonl "output", matched by the report)
#   MINUTES  wall-clock cap (TERM, then KILL after KILL_GRACE seconds); 0 = no cap
#   KEY=VALUE extra jobs.jsonl fields (model=, benchmark=, condition=, method=, kind=, ...)
#
# Returns the command's exit status; a capped run that hit its limit returns 0 (state TIMEOUT),
# like a time-limited chunk that ran out of time: the next run of the script resumes it.
run_job() {
  local name=$1 config=$2 logroot=$3 stage=$4 output=$5 minutes=$6
  shift 6 || { echo "run_job: missing arguments" >&2; return 2; }
  local fields=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do fields+=("$1"); shift; done
  [ "${1:-}" = "--" ] || { echo "run_job: expected -- before the command" >&2; return 2; }
  shift
  [ $# -gt 0 ] || { echo "run_job: empty command" >&2; return 2; }
  logroot=$(_abspath "$logroot")
  output=$(_abspath "$output")
  if [ -n "$config" ]; then config=$(_abspath "$config"); fi

  if [ "$DRY_RUN" = 1 ]; then
    local line="DRY:"
    [ -z "$config" ] || line+=" LEAF_CONFIG=$(printf '%q' "$config")"
    line+="$(printf ' %q' "$@")"
    printf '%s\n' "$line"
    return 0
  fi

  local job
  job=$(_new_job_id)
  local logs="$logroot/logs"
  mkdir -p "$logs" "$output"
  local out="$logs/${name}_${job}.out" err="$logs/${name}_${job}.err"
  local revision=null
  if [ -n "$config" ]; then
    revision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("revision") or "null")' "$config")
  fi
  local gpu_name gpu=null
  gpu_name=$(_gpu_name)
  if [ -n "$gpu_name" ]; then
    gpu=$(printf '%s' "$gpu_name" | tr 'A-Z' 'a-z' | tr -d ' -' | grep -oE 'h200|h100|a100|a6000|l40s|l40|a40|v100' | head -n 1)
    gpu=${gpu:-$(printf '%s' "$gpu_name" | tr 'A-Z ' 'a-z_')}
  fi
  local started submitted
  submitted=$(date '+%Y-%m-%dT%H:%M:%S')
  local base=(stage="$stage" job="$job" name="$name" chunk=0 chunks=1 dependency=null
              gpu="$gpu" gpu_name="${gpu_name:-null}" snapshot=null output="$output"
              config="${config:-null}" revision="$revision" time_minutes="$minutes"
              submitted_at="$submitted")
  _jobs_row "$logroot/jobs.jsonl" "$(_row_json "${base[@]}" "${fields[@]}" state=RUNNING \
            start="$submitted" command="$(printf '%q ' "$@")")"

  local zone
  zone=$(date +%Z)
  [[ $zone =~ ^[A-Z]{3,4}$ ]] || zone=LCL
  local compile_cache="$CACHE_ROOT/compile/job_$job"
  local ray_tmp="$CACHE_ROOT/ray/$job"
  # Ray's socket paths must stay short (AF_UNIX limit of 107 bytes)
  [ ${#ray_tmp} -le 60 ] || ray_tmp="/tmp/ray_$job"
  local start_s end_s status=0
  start_s=$(date +%s)
  {
    printf '%s %s\n' "$(date '+%F %T')" "$zone"
    printf 'RUN_JOB_ID=%s\n' "$job"
    printf 'LEAF_CONFIG=%s\n' "${config:-<default>}"
    printf 'command:%s\n' "$(printf ' %q' "$@")"
  } > "$out"
  (
    _job_env
    export RUN_JOB_ID=$job
    if [ -n "$config" ]; then export LEAF_CONFIG=$config; fi
    # per-job compile caches: parallel jobs racing on one inductor cache fail at engine start
    export VLLM_CACHE_ROOT=$compile_cache/vllm
    export TORCHINDUCTOR_CACHE_DIR=$compile_cache/torchinductor
    export TRITON_CACHE_DIR=$compile_cache/triton
    export RAY_TMPDIR=$ray_tmp
    mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$RAY_TMPDIR"
    cd "$REPO"
    if [ "$minutes" -gt 0 ] 2>/dev/null; then
      exec timeout --signal=TERM --kill-after="${KILL_GRACE:-120}" "${minutes}m" "$@"
    else
      exec "$@"
    fi
  ) >> "$out" 2> "$err" || status=$?
  end_s=$(date +%s)

  local state=COMPLETED
  if [ "$status" -ne 0 ]; then
    state=FAILED
    if [ "$minutes" -gt 0 ] 2>/dev/null && { [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; } \
       && [ $((end_s - start_s)) -ge $((minutes * 60)) ]; then
      state=TIMEOUT
    fi
  fi
  case "${RAY_STOP:-auto}" in
    1) _stop_ray "$ray_tmp" ;;
    all) _stop_ray_all ;;
    auto) if [ -n "$(ls -A "$ray_tmp" 2>/dev/null)" ]; then _stop_ray "$ray_tmp"; fi ;;
  esac
  rm -rf "$compile_cache" "$ray_tmp" 2>/dev/null || true

  _jobs_row "$logroot/jobs.jsonl" "$(_row_json "${base[@]}" "${fields[@]}" state="$state" \
            exit_code="$status" start="$(date -d "@$start_s" '+%Y-%m-%dT%H:%M:%S')" \
            end="$(date -d "@$end_s" '+%Y-%m-%dT%H:%M:%S')" elapsed_seconds=$((end_s - start_s)) \
            command="$(printf '%q ' "$@")")"
  printf '[%s] %s %s %s (%ss) -> %s\n' "$(date '+%F %T')" "$job" "$name" "$state" \
         $((end_s - start_s)) "$out" >&2
  if [ "$state" = TIMEOUT ]; then return 0; fi
  return "$status"
}

# _stop_ray RAY_TMPDIR: leftover Ray processes of one verl job (the original jobs were cleaned up
# by the scheduler). The raylet / gcs / dashboard / worker command lines carry the session
# directory under the job's own RAY_TMPDIR, so other jobs' clusters on the machine keep running.
_stop_ray() {
  local ray_tmp=$1
  [ -n "$ray_tmp" ] || return 0
  if ! command -v pkill >/dev/null 2>&1; then
    echo "pkill not found: Ray processes under $ray_tmp may still be running" >&2
    return 0
  fi
  pkill -TERM -f -- "$ray_tmp" 2>/dev/null || return 0
  sleep 5
  pkill -KILL -f -- "$ray_tmp" 2>/dev/null || true
}

# RAY_STOP=all: every Ray cluster of the user on the machine (safe only when nothing else runs).
_stop_ray_all() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  else
    "$PYTHON" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  fi
}
