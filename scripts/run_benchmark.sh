#!/bin/bash
# Run one benchmark sample for an experiment, then score it.
#
# Usage:
#   scripts/run_benchmark.sh <EXPERIMENT_ID> <RUN_INDEX>              # Basic holdout: 11 families x variants 8-10 = 33 cases
#   scripts/run_benchmark.sh <EXPERIMENT_ID> <RUN_INDEX> --advanced   # Advanced: 16 out-of-distribution cases
# EXPERIMENT_ID: a key of configs/experiments.json, or SPLIT_<NAME> for a split-adapter map (run scripts/localize.py first).
# Output: code/experiment-finetuned[-advanced]/<ID>_holdout_run<k>/ (results/ = transcripts, runs/ = OpenFOAM cases).
# Re-running the same command resumes: only never-started cases run (an aborted case counts as a failure, not a redo).
# Needs 4 GPUs with >= 80 GB each (tensor parallel 4, 100k context). Set CUDA_VISIBLE_DEVICES to pick them.
set -euo pipefail
EXP="${1:?usage: run_benchmark.sh <EXPERIMENT_ID> <RUN_INDEX> [--advanced]}"
RUN="${2:?usage: run_benchmark.sh <EXPERIMENT_ID> <RUN_INDEX> [--advanced]}"
ADV="${3:-}"
KIT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$KIT/env.sh" ] && [ -f "$KIT/configs/$EXP.json" ] || { echo "run scripts/localize.py first ($EXP)" >&2; exit 1; }
source "$KIT/env.sh"
source "$KIT/configs/$EXP.env"          # VENV_ACTIVATE and the experiment's environment
set +eu; source "$FOAMGPT_OPENFOAM_BASHRC"; set -eu
source "$VENV_ACTIVATE"

# ---- Protocol constants (identical for every run in the paper). Do not change them for comparable results. ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FOAMAGENT_HF_TP="${FOAMAGENT_HF_TP:-4}"
export FOAMAGENT_HF_MAX_MODEL_LEN=100000
export FOAMAGENT_HF_MAX_OUTPUT_TOKENS=16384
export FOAMAGENT_HF_MAX_NUM_SEQS=1
export FOAMAGENT_HF_ENFORCE_EAGER=1
export FOAMAGENT_HF_INPLACE_SWAP=0
export FOAMAGENT_HF_FORCE_LORA_ENGINE="${FOAMAGENT_HF_FORCE_LORA_ENGINE:-1}"
export FOAMAGENT_HF_MAX_LORA_RANK=32
export FOAMAGENT_RUN_MODE=local
export FOAMAGENT_CASES_BASIC="8 9 10"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # models and embedder are local after scripts/fetch_models.sh
export PYTHONUNBUFFERED=1

LABEL="${EXP}_holdout_run${RUN}"
cd "$KIT/code"
if [ "$ADV" = "--advanced" ]; then
  python benchmark_finetuned_advanced.py --finetuned_config "$KIT/configs/$EXP.json" --all_finetuned --workers 1 \
    --resume --run_label "$LABEL"
  python "$KIT/scripts/score_run.py" "$LABEL" --advanced
else
  python benchmark_finetuned.py --finetuned_config "$KIT/configs/$EXP.json" --all_finetuned --workers 1 \
    --resume --run_label "$LABEL"
  python "$KIT/scripts/score_run.py" "$LABEL"
fi
