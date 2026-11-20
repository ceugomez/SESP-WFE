#!/bin/bash
#SBATCH --job-name=kf_rank_sweep_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_rank_sweep_dense_%j.log

# Rank sweep of the reduced-order KF with a sensor array spanning the downwind
# half of the analysis domain (every SENSOR_STRIDE_XY grid cells in x/y, every
# SENSOR_STRIDE_Z-th vertical level below 250 m AGL) instead of the 15-sensor
# tower rake. Removes rake under-determination as a variable: 175 sensors /
# 525 obs rows at the default spacing, vs 15 / 45 for the rake.
#
# Requires: basis, models, rep-error, and truth outputs on disk.
#
# Env knobs (defaults shown):
#   TRUTH_ID=truth           which member is held out as truth
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

export TRUTH_ID="${TRUTH_ID:-truth}"
export OBS_STRIDE="${OBS_STRIDE:-1}"
export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

echo "TRUTH_ID=$TRUTH_ID OBS_STRIDE=$OBS_STRIDE XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"

precompile_julia

run_jl online/run_kf_rank_sweep_dense.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
