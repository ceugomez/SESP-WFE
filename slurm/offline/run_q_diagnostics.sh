#!/bin/bash
#SBATCH --job-name=q_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/q_diag_%A_%a.log

# Out-of-sample check of the calibrated process noise Q̂α against the held-out
# truth member's own POD-coefficient increments, per LOO fold.
# Requires: basis + models files in Bases/ and truth member outputs on disk.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

FOLDS=(00 01 02 03 04 05 06 07 08 09 10 11 truth)
export TRUTH_ID=${FOLDS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) starting on $(hostname) at $(date)"

precompile_julia

run_jl offline/run_q_diagnostics.jl

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) finished at $(date)"
