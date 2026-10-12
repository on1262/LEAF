#!/usr/bin/env bash
# Regenerate every table and figure number of the paper from the run outputs ($RESULTS_ROOT) into
# $REPORT_ROOT (CPU only; nothing is written into a run directory).
#
#   bash run/report.sh [STEP...]          (default: all steps, in this order)
#
# Steps and outputs (under $REPORT_ROOT):
#   explore_seed1  leaf.report.exploration --set explore -> explore/curves_seed1_with_interleaved.json
#                  (seed-1 exploration of Table 1 / Fig. 3 incl. the interleaved-update runs)
#   main        leaf.report.main_results    -> main/curves_seed{1,2,3}.json, general.json, sections.json
#               (Table 1 / Fig. 3 over 3 seeds, Table 2 over 3 seeds, the quantitative claims)
#   appendix    leaf.report.appendix_tables -> main/appendix.json (per-benchmark appendix tables),
#               main/cost_20_verifier_calls.json (GPU h of the baselines at 20 verifier calls, SEED
#               budget sentence)
#   ablation    leaf.report.ablation_tables -> ablation/{curves,update,repeat}.json
#               (exploration / divergence ablations, history compression, internalization, repeat
#               and fix rates)
#   traces      leaf.report.trace_tables    -> traces/tables_formal.json (SFT on y*, teacher accuracy,
#               self-correction)
#   repeat      leaf.report.repeat          -> ablation/repeat_random.json (Table tab:repeat-random)
#   lendata     figures/len_{explore,update}_data.py -> figures/data/len_{explore,update}.json
#   figures     figures/{explore_curves,explore_headline,ablation_scatter,len_scatter}.py
#               -> figures/*.pdf|png (Fig. 3 speedups, headline numbers, ablation scatter, Table tab:length)
#
# Each step's stdout (the English markdown tables / printed numbers) is also saved to
# $REPORT_ROOT/logs/<script>.txt (one file per script a step runs). CHECK_PAPER_VALUES=1 makes ablation_scatter assert the paper's values.
# Caches: $REPORT_ROOT/cache/{flops_cache,sft_tokens}.json (token counts need the tokenizers under
# $MODEL_ROOT). Jobs are read from <run root>/<model>/jobs.jsonl: numeric scheduler job ids are looked
# up with sacct when it is available, local job ids (run/common.sh) use the row's own state / elapsed time.
set -euo pipefail
export REQUIRE_THIRD_PARTY=${REQUIRE_THIRD_PARTY:-0}  # imports no third-party code
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

ALL_STEPS=(explore_seed1 main appendix ablation traces repeat lendata figures)
if [ "$#" -gt 0 ]; then STEPS=("$@"); else STEPS=("${ALL_STEPS[@]}"); fi
mkdir -p "$REPORT_ROOT/logs"

run_step() {
  local name=$1
  shift
  echo "== $name: $*" >&2
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY: $*"
    return 0
  fi
  "$@" | tee "$REPORT_ROOT/logs/$name.txt"
}

for step in "${STEPS[@]}"; do
  case "$step" in
    explore_seed1)
      run_step explore_seed1 "$PYTHON" -u -m leaf.report.exploration --set explore \
        --json "$REPORT_ROOT/explore/curves_seed1_with_interleaved.json" ;;
    main)
      run_step main "$PYTHON" -u -m leaf.report.main_results ;;
    appendix)
      run_step appendix "$PYTHON" -u -m leaf.report.appendix_tables ;;
    ablation)
      run_step ablation "$PYTHON" -u -m leaf.report.ablation_tables ;;
    traces)
      run_step traces "$PYTHON" -u -m leaf.report.trace_tables ;;
    repeat)
      run_step repeat "$PYTHON" -u -m leaf.report.repeat ;;
    lendata)
      run_step len_explore_data "$PYTHON" -u figures/len_explore_data.py
      run_step len_update_data "$PYTHON" -u figures/len_update_data.py ;;
    figures)
      run_step explore_curves "$PYTHON" -u figures/explore_curves.py
      run_step explore_headline "$PYTHON" -u figures/explore_headline.py
      if [ "${CHECK_PAPER_VALUES:-0}" = 1 ]; then
        run_step ablation_scatter "$PYTHON" -u figures/ablation_scatter.py --check-paper-values
      else
        run_step ablation_scatter "$PYTHON" -u figures/ablation_scatter.py
      fi
      run_step len_scatter "$PYTHON" -u figures/len_scatter.py ;;
    *)
      echo "unknown step '$step' (steps: ${ALL_STEPS[*]})" >&2
      exit 2 ;;
  esac
done
echo "report written to $REPORT_ROOT" >&2
