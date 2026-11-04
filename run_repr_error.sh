#!/bin/bash
#SBATCH --job-name=repr_error
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/repr_error_%A_%a.log

# Time-resolved representation error (held-out truth minus its rank-r projection), every
# rank 1..50, per LOO fold -> results/repr/repr_error_<fold>.nc. One fold per array task.
# Feeds paper figure `repr_error_time` (src/viz/paper_figures.py).
#   sbatch run_repr_error.sh                 # all 13 folds
#   FOLDS=03 julia --threads=8 scripts/run_repr_error.jl   # one fold, interactively

set -euo pipefail
FOLD_LIST=(00 01 02 03 04 05 06 07 08 09 10 11 truth)
export FOLDS=${FOLD_LIST[$SLURM_ARRAY_TASK_ID]}
mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID (fold $FOLDS) on $(hostname) at $(date)"
julia --startup-file=no --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_repr_error.jl
echo "Finished at $(date)"
