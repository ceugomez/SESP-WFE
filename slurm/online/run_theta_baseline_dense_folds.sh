#!/bin/bash
#SBATCH --job-name=theta_dense_folds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/theta_dense_folds_%A_%a.log

# Θ-baseline + Θ-oracle leg ONLY, across all 13 LOO folds, on the dense sensor
# lattice. The paired dense KF sweeps already exist (job 5085, Jul 16) at the
# default config, so this backfills the missing
# results/<tag>/kf/theta_baseline_dense_truth_<ID>.csv files without redoing the
# ~1.5 h/fold KF rank-sweep leg. Same folds and env knobs as
# run_fold_sweep_dense.sh — keep SENSOR_STRIDE_XY/Z and the seed untouched or
# the comparison to kf_rank_sweep_dense_truth_<ID>.csv is no longer paired.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

FOLDS=(00 01 02 03 04 05 06 07 08 09 10 11 truth)
export TRUTH_ID=${FOLDS[$SLURM_ARRAY_TASK_ID]}

export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) starting on $(hostname) at $(date)"
echo "TRUTH_ID=$TRUTH_ID XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"

precompile_julia

run_jl online/run_theta_baseline_dense.jl

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) finished at $(date)"
