#!/bin/bash
#SBATCH --job-name=fold_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/fold_sweep_%A_%a.log

# LOO-fold campaign for R1: per fold, run the Θ-baseline (conditional-mean map
# fit on that fold's 11 basis members) and then the KF rank sweep against the
# same truth member. Both scripts seed 20260713 before drawing measurement
# noise, so the comparison stays paired within each fold.
# Index 12 = the original truth member (rerun fills the full-domain column that
# the Float32-cancellation bug zeroed in jobs 5053/5054).
# %4 caps concurrency at 4 tasks (~128G each); adjust to taste.

set -euo pipefail

FOLDS=(00 01 02 03 04 05 06 07 08 09 10 11 truth)
export TRUTH_ID=${FOLDS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) starting on $(hostname) at $(date)"

julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_theta_baseline.jl

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_kf_rank_sweep.jl

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) finished at $(date)"
