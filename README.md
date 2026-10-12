# LEAF: Explore with Memory, Learn without It

Code for the paper *Explore with Memory, Learn without It: In-Context Exploration and
Self-Distillation for LLM Reasoning* by Yutong Chen, Zhichao Xu, Rui Liu and Kunpeng Liu.

This repository contains the code behind every experiment in the paper. LEAF explores each problem
with a memory of its last answer and the verifier's feedback, and then distills the successful
attempts into the model weights by self-distillation. The code covers:

- exploration on MATH-500, LiveCodeBench (LCB), MBPP+ and SciKnowEval with LEAF and the baselines
  Random Sampling, GRPO, SDPO, SCoRe and SEED (Table 1, Figure 3);
- the LEAF policy update and in-distribution generalization (Table 2);
- the ablations and analyses of Section 3.3 (Figure 4, Tables 3 to 6) and the appendix tables.

All experiments use Qwen3-8B and Olmo-3-7B-Instruct. Every job is a bash script that runs on a
single GPU.

## Repository layout

| Path | Contents |
|---|---|
| `leaf/explore/` | LEAF exploration rollout (`leaf_rollout.py`); LEAF with updates interleaved into exploration |
| `leaf/update/` | LEAF policy update (`leaf_update.py`, `run_update.py`), successful-context construction, SFT on y* |
| `leaf/baselines/` | Random Sampling, GRPO, SDPO, SCoRe and SEED; their verl reward functions |
| `leaf/prompts/` | task prompts, memory turns and restart instructions |
| `leaf/verifiers/` | public-test verifiers and feedback, held-out-test graders, the SciKnowEval LLM judge |
| `leaf/grading/`, `leaf/analysis/` | grading of runs, test-set evaluation, teacher accuracy, self-correction, internalization |
| `leaf/data/`, `leaf/core/` | data preparation; shared config, record and seed helpers; `leaf/paths.py` defines every location |
| `leaf/report/`, `figures/` | compute the numbers of the paper's tables and draw its figures from the run outputs |
| `configs/` | one JSON config per experiment, model and seed (`explore/`, `general/`, `ablation/`, `data/`) |
| `data/` | the evaluation and training sets used in the paper |
| `run/` | entry points for all experiments (described below) |
| `patches/`, `setup.sh` | our changes to verl (the SDPO code base) and LiveCodeBench, and the script that applies them |

## Setup

Python 3.11, CUDA 12.8, torch 2.9.0 and vLLM 0.12.0.

    pip install -r requirements.txt
    pip install flash-attn==2.8.3 --no-build-isolation
    bash setup.sh

`setup.sh` clones three upstream repositories into `third_party/` and applies our patches:

| Directory | Upstream | Commit | Patch |
|---|---|---|---|
| `third_party/SDPO` (verl) | https://github.com/lasgroup/SDPO | `7c457fc1b1f636ae794eb0362ba37d4743b06fbc` | `patches/SDPO-7c457fc.patch` |
| `third_party/LiveCodeBench` | https://github.com/LiveCodeBench/LiveCodeBench | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | `patches/LiveCodeBench-28fef95.patch` |
| `third_party/EvalPlus` | https://github.com/evalplus/evalplus | `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` | none |

The verl patch adds the self-distillation losses (reverse KL, forward KL and JSD, each with an
optional per-entry clip), the SCoRe and SEED training objectives, LoRA checkpoint resumption and
the SciKnowEval judge inside the training loop, and per-sample decoding seeds. The LiveCodeBench
patch passes test results through a pipe so that large results cannot block the runner, and treats
a worker that exits without a result as a timeout.

The run scripts set `PYTHONPATH=$REPO:$REPO/third_party/SDPO:$REPO/third_party/LiveCodeBench`.
The patched verl imports modules from `leaf/`, so the repository root has to come first, and a
separately installed verl package should be removed.

## Data and models

    bash run/data_fetch.sh       # upstream datasets into $DATA_ROOT/raw, models into $MODEL_ROOT
    bash run/data_prepare.sh     # LCB test parquets and the per-experiment run directories

The evaluation and training sets are included in `data/`:

| Files | Contents |
|---|---|
| `data/explore/<model>/dataset_{math500,lcb,mbpp,sciknoweval}.jsonl` | exploration sets, also the generalization test sets: MATH-500 (500), LCB release v6 problems from 2025-02-01 to 2025-04-30 (131), MBPP+ (378), SciKnowEval open-ended questions (400) |
| `data/train/<model>/dataset_{lcb,math500,sciknoweval}.jsonl` | training sets of the policy update: LCB v2 to v5 (478), MATH Level 5 (500), SciKnowEval L5 (400) |
| `data/lcb/lcb131_split.jsonl` | the 131 LCB problems with their split into public and held-out tests |

The Olmo files also contain the OLMES chat prompts used for that model. LCB rows refer to the
upstream release files by name and offset, so `run/data_fetch.sh` downloads those files at the
same revision. `run/data_prepare.sh lcb` then builds the LCB test parquets in `data/lcb/`.

`run/data_fetch.sh` downloads these sources at fixed revisions:

| Source | Revision |
|---|---|
| livecodebench/code_generation_lite (release_v6) | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| HuggingFaceH4/MATH-500 | `ff5b20257d8185524591543f8ff5993951537bb8` |
| EleutherAI/hendrycks_math | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| hicai-zju/SciKnowEval | `92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2` |
| EvalPlus MbppPlus | release v0.2.0 |
| Qwen/Qwen3-8B | `2069b3fae1114555f3c020c81410e51fa0f656f2` |
| allenai/Olmo-3-7B-Instruct | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |

The scripts that built the files in `data/` from these sources are in `leaf/data/`
(`prepare_test_sets.py`, `prepare_olmes_test_sets.py`, `prepare_mbppplus.py`,
`prepare_sciknoweval.py`, `prepare_train_sets.py`, with the configs in `configs/data/`).

## Reproducing the paper

Run `bash run/data_prepare.sh` once, then the experiment scripts, then `bash run/report.sh` to
compute the tables and figures. Every script documents its arguments in its header, accepts `all`
for any argument, and prints its commands without running them when `DRY_RUN=1` is set.

| Paper result | Runs | Report step |
|---|---|---|
| Table 1, Figure 3 (exploration) and the per-benchmark and per-seed appendix tables | `run/explore_main.sh MODEL METHOD BENCHMARK SEED` | `main`, `appendix`, `figures` |
| Table 2 (generalization) and its appendix tables | `run/general_main.sh MODEL METHOD TRAINING_SET SEED` | `main`, `appendix` |
| Figure 4 and the component-ablation tables | `run/explore_ablations.sh MODEL VARIANT BENCHMARK` | `ablation`, `figures` |
| Table 3 (response length) | the runs of Tables 1 and 2 | `lendata`, `figures` |
| Table 4 (interleaved updates) | `run/explore_ablations.sh` with `interleaved_every_{1,2,5}` | `explore_seed1`, `ablation` |
| Table 5 (teacher accuracy under different traces) and the self-correction table | `run/general_ablations.sh MODEL {teacher,selfcorr} TRAINING_SET` | `traces` |
| Table 6 (loss types) | `run/general_ablations.sh MODEL loss TRAINING_SET` | `ablation`, `lendata` |
| Internalization table | `run/general_ablations.sh MODEL trainset TRAINING_SET` | `ablation` |
| History-compression table | `run/explore_compression.sh MODEL CONDITION BENCHMARK` | `ablation` |
| Repeated wrong answers (appendix) | the Random Sampling, LEAF and ablation runs | `repeat`, `ablation` |

Arguments of the experiment scripts:

- `explore_main.sh`: MODEL `qwen3_8b | olmo3_7b`; METHOD `leaf | random_sampling | grpo | sdpo |
  score | seed`; BENCHMARK `math500 | lcb | mbpp | sciknoweval`; SEED `1 | 2 | 3`. Each run is
  capped at 5 GPU hours and 24 verifier calls per problem, and is graded on CPU afterwards.
- `explore_ablations.sh` (seed 1): VARIANT `wo_last_response | wo_feedback | restart_every_2 |
  restart_every_10 | no_restart | last_3_rounds | interleaved_every_1 | interleaved_every_2 |
  interleaved_every_5`.
- `explore_compression.sh` (seed 1): CONDITION `compress_none | compress_llm_summary |
  compress_tail | compress_head_tail | compress_summary_block | compress_lowest_logprob`.
- `general_main.sh`: METHOD `base | grpo | sdpo | score | seed | leaf | sft | seed_20_calls`;
  TRAINING_SET `lcb | math500 | sciknoweval`. `leaf` runs the LEAF rollout on the training set,
  collects the successful contexts and runs the LEAF update (reverse KL; on MATH with a per-entry
  clip at tau = 1e-3), then evaluates on the matching test set. `sft` trains on the rewritten
  solutions y* from the same rollouts. `seed_20_calls` gives the SEED cost reported in the appendix.
- `general_ablations.sh` (seed 1): PART `loss | trainset | teacher | selfcorr`. It uses the outputs
  of `general_main.sh`.

`GPUS="0 1 2 3"` runs independent cells in parallel, one GPU each.

The report reads `$RESULTS_ROOT` and writes JSON tables, figures and logs to `$REPORT_ROOT`.
`bash run/report.sh` without arguments runs all steps in order: `explore_seed1 main appendix
ablation traces repeat lendata figures`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `data/` | datasets and upstream sources |
| `MODEL_ROOT` | `models/` | `Qwen3-8B/`, `Olmo-3-7B-Instruct/` |
| `RESULTS_ROOT` | `results/` | run outputs |
| `REPORT_ROOT` | `$RESULTS_ROOT/report` | tables and figures |
| `PYTHON` | `python` | interpreter |
| `GPUS` | unset | GPUs for parallel cells |
| `DRY_RUN` | `0` | `1` prints the commands instead of running them |
| `FORCE` | `0` | `1` re-runs steps that have finished |
| `TIME_CAPS` | `1` | `0` removes the per-job time limits of the generalization runs |
| `LEN_WORKERS` | all CPUs | processes for the `lendata` report step |

Configs refer to these locations as `${DATA_ROOT}`, `${RESULTS_ROOT}` and `${MODEL_ROOT}`. Each job
appends its state and wall time to `jobs.jsonl` in its run directory, and the report computes GPU
hours from these records. Re-running a script resumes unfinished jobs and skips finished ones.

## Compute

The paper's runs used one H200 (141 GB) per job. Each exploration run is limited to 5 GPU hours, so
on slower GPUs some runs stop at fewer than 24 verifier calls per problem, as the SEED baseline did
for Olmo-3-7B-Instruct on MATH-500. The code verifiers enforce time limits, and grading used 24 CPU
cores (48 for LCB test grading). On fewer cores, some programs may hit these limits.

Figure 2 is a hand-drawn diagram and is not generated by this code.

## Citation

    @misc{chen2026leaf,
      title  = {Explore with Memory, Learn without It: In-Context Exploration and
                Self-Distillation for LLM Reasoning},
      author = {Chen, Yutong and Xu, Zhichao and Liu, Rui and Liu, Kunpeng},
      year   = {2026},
      url    = {https://github.com/on1262/LEAF}
    }
