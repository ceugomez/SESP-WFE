#!/bin/bash
#SBATCH --job-name=fold_sweep_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-12%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/fold_sweep_dense_%A_%a.log

# LOO-fold campaign of the DENSE-array rank sweep: the same folds as
# run_fold_sweep.sh, but with the 175-sensor / 525-obs-row array spanning the
# downwind half of the analysis domain instead of the 15-sensor tower rake.
# Isolates rake under-determination from basis/rank effects across all folds.
# Writes results/<tag>/kf/kf_rank_sweep_dense_truth_<ID>.csv AND
# results/<tag>/kf/theta_baseline_dense_truth_<ID>.csv (one pair per fold) — the
# Θ-baseline leg (run_theta_baseline_dense.jl, added 2026-07-20) is scored on
# the SAME dense array so it is a matched comparison, unlike the 45-obs-row
# theta_baseline_truth_<ID>.csv rake files.
#
# Single-fold runtime was ~1.5 h at the default spacing (job 5076) for the KF
# leg alone; with %4 concurrency the 13-fold array should land in ~5-6 h.
#
# Env knobs (defaults shown, same as run_kf_rank_sweep_dense.sh):
#   OBS_STRIDE=1             assimilate every OBS_STRIDE-th snapshot (6 s apart)
#   SENSOR_STRIDE_XY=50      sensor spacing in x,y (grid cells; dx = 10 m -> 500 m)
#   SENSOR_STRIDE_Z=10       sensor spacing in vertical levels
#
# DO NOT tighten SENSOR_STRIDE_XY below ~40 without reading check_R_pd in
# src/experiment.jl: R_rep is indefinite at all
# densities (nonstationary midpoint-parameter kernel) and is only PD-safe while
# the sensor-noise floor masks it. Job 5069 died this way at 100 m spacing.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

FOLDS=(00 01 02 03 04 05 06 07 08 09 10 11 truth)
export TRUTH_ID=${FOLDS[$SLURM_ARRAY_TASK_ID]}

export OBS_STRIDE="${OBS_STRIDE:-1}"
export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) starting on $(hostname) at $(date)"
echo "TRUTH_ID=$TRUTH_ID OBS_STRIDE=$OBS_STRIDE XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"

precompile_julia

run_jl online/run_theta_baseline_dense.jl

run_jl online/run_kf_rank_sweep_dense.jl

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (fold $TRUTH_ID) finished at $(date)"
