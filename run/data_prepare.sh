#!/usr/bin/env bash
# Build everything the run scripts read, from the shipped datasets (data/) and the raw
# upstream files fetched by run/data_fetch.sh. CPU only.
#
#   bash run/data_prepare.sh             # all steps
#   bash run/data_prepare.sh lcb         # LiveCodeBench ground-truth parquets only
#   bash run/data_prepare.sh roots       # run roots of seed 1 (links, smoke subsets, rl_data)
#   bash run/data_prepare.sh seeds       # run roots of seeds 2 and 3 (leaf.data.prepare_run_seeds data)
#   DRY_RUN=1 bash run/data_prepare.sh   # print the commands instead of running them
#   FORCE=1 bash run/data_prepare.sh     # rebuild outputs that already exist
#
# Steps
#   lcb    $DATA_ROOT/lcb/{train,test}.parquet         public / all tests of the 131-problem split
#          $DATA_ROOT/lcb/lcb_ground_truth_full.parquet all tests of the 478 training problems (~2.2 GB)
#   roots  every seed-1 run root under $RESULTS_ROOT that the configs read (configs/*/*.json data_root):
#          dataset_<benchmark>.jsonl symlinks to $DATA_ROOT/{explore,train}/<model>/, the
#          smoke subsets (first smoke_source_sizes rows) of the training roots, and the verl
#          parquets / schedules / budgets in <root>/rl_data (leaf.data.prepare_verl_data);
#          the output-only roots are created empty.
#   seeds  the same for the seed-2/3 roots ($RESULTS_ROOT/explore/seed<s>, general/seed<s>; the
#          configs/{explore,general}/*_seed{2,3}.json).
set -euo pipefail
export REQUIRE_THIRD_PARTY=${REQUIRE_THIRD_PARTY:-0}  # imports no third-party code
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(lcb roots seeds)
for step in "${steps[@]}"; do
  case "$step" in lcb|roots|seeds) ;; *) echo "unknown step '$step' (lcb, roots, seeds)" >&2; exit 2 ;; esac
done
want() { local s; for s in "${steps[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# _do CMD...: run, or print it under DRY_RUN=1
_do() {
  if [ "$DRY_RUN" = 1 ]; then printf 'DRY:'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

R=$RESULTS_ROOT
BENCH_EXPLORE=(lcb math500 mbpp sciknoweval)
BENCH_TRAIN=(lcb math500 sciknoweval)

# link_root ROOT SOURCE_DIR BENCHMARK...: ROOT/dataset_<b>.jsonl -> SOURCE_DIR/dataset_<b>.jsonl
link_root() {
  local root=$1 source=$2 b
  shift 2
  _do mkdir -p "$root"
  for b in "$@"; do
    _do ln -sfn "$source/dataset_$b.jsonl" "$root/dataset_$b.jsonl"
  done
}

# smoke_root ROOT CONFIG BENCHMARK...: ROOT/smoke_dataset_<b>.jsonl = first smoke_source_sizes[b] rows
smoke_root() {
  local root=$1 config=$2 b n
  shift 2
  for b in "$@"; do
    job_done "$root/smoke_dataset_$b.jsonl" && continue
    n=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["smoke_source_sizes"][sys.argv[2]])' "$config" "$b")
    if [ "$DRY_RUN" = 1 ]; then
      printf 'DRY: head -n %s %q > %q\n' "$n" "$root/dataset_$b.jsonl" "$root/smoke_dataset_$b.jsonl"
    else
      head -n "$n" "$root/dataset_$b.jsonl" > "$root/smoke_dataset_$b.jsonl"
    fi
  done
}

# rl_data ROOT CONFIG BENCHMARK...: ROOT/rl_data/<b>/ for every benchmark (CONFIG's data_root is ROOT)
rl_data() {
  local root=$1 config=$2 b markers=() args=()
  shift 2
  for b in "$@"; do markers+=("$root/rl_data/$b/budget_manifest.json"); args+=(--benchmark "$b"); done
  job_done "${markers[@]}" && { echo "$root/rl_data exists"; return 0; }
  if [ "$DRY_RUN" = 1 ]; then
    _do env LEAF_CONFIG="$config" "$PYTHON" -m leaf.data.prepare_verl_data "${args[@]}"
  else
    LEAF_CONFIG="$config" "$PYTHON" -m leaf.data.prepare_verl_data "${args[@]}" > /dev/null
    echo "$root/rl_data"
  fi
}

# ------------------------------------------------------------------------------------- lcb
if want lcb; then
  job_done "$DATA_ROOT/lcb/train.parquet" "$DATA_ROOT/lcb/test.parquet" \
    || _do "$PYTHON" -m leaf.data.build_lcb_ground_truth split131
  # write_lcb_ground_truth keeps an existing file after checking its problem set
  job_done "$DATA_ROOT/lcb/lcb_ground_truth_full.parquet" \
    || _do "$PYTHON" -m leaf.data.build_lcb_ground_truth train478
fi

# ----------------------------------------------------------------------------------- roots
if want roots; then
  for m in qwen3_8b olmo3_7b; do
    explore=$DATA_ROOT/explore/$m
    train=$DATA_ROOT/train/$m

    # exploration (seed 1); the exploration ablations read the same root
    link_root "$R/explore/seed1/$m" "$explore" "${BENCH_EXPLORE[@]}"
    rl_data "$R/explore/seed1/$m" "$REPO/configs/explore/${m}_seed1.json" "${BENCH_EXPLORE[@]}"
    link_root "$R/ablation/compression/$m" "$explore" "${BENCH_EXPLORE[@]}"

    # generalization: the baselines root is the data root of every seed-1 generalization config
    link_root "$R/general/seed1/baselines/$m" "$train" "${BENCH_TRAIN[@]}"
    smoke_root "$R/general/seed1/baselines/$m" "$REPO/configs/general/baselines_${m}_seed1.json" "${BENCH_TRAIN[@]}"
    rl_data "$R/general/seed1/baselines/$m" "$REPO/configs/general/baselines_${m}_seed1.json" "${BENCH_TRAIN[@]}"
    # LEAF rollouts read their training sets from the LEAF root
    link_root "$R/general/seed1/leaf/$m" "$train" "${BENCH_TRAIN[@]}"

    # SEED baseline
    seed_config=$REPO/configs/general/seed_baseline_${m}_seed1.json
    link_root "$R/general/seed1/seed_baseline/$m" "$train" "${BENCH_TRAIN[@]}"
    smoke_root "$R/general/seed1/seed_baseline/$m" "$seed_config" "${BENCH_TRAIN[@]}"
    rl_data "$R/general/seed1/seed_baseline/$m" "$seed_config" "${BENCH_TRAIN[@]}"

    # output-only roots
    for out in general/seed1/sft ablation/explore ablation/update ablation/traces; do
      _do mkdir -p "$R/$out/$m"
    done
  done
fi

# ----------------------------------------------------------------------------------- seeds
if want seeds; then
  # links (from $DATA_ROOT) and rl_data of every seed-2/3 root; skips roots that have rl_data
  _do "$PYTHON" -m leaf.data.prepare_run_seeds data
fi
