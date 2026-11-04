#!/bin/bash
#SBATCH --job-name=theta_baseline_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/theta_baseline_dense_%j.log

# R1: the Θ-regression baseline scored against the DENSE sensor lattice (same
# array as run_kf_rank_sweep_dense.jl) instead of the 15-sensor tower rake.
# Requires: POD basis + rep-error model for TRUTH_ID on disk; kf_result
# optional (NaNs the kf columns if missing).
#
# Env knobs (defaults shown, must match run_kf_rank_sweep_dense.sh for a
# paired comparison):
#   TRUTH_ID=truth           which member is held out as truth
#   SENSOR_STRIDE_XY=50      sensor spacing in x,y (grid cells; dx = 10 m -> 500 m)
#   SENSOR_STRIDE_Z=10       sensor spacing in vertical levels

set -euo pipefail

mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

export TRUTH_ID="${TRUTH_ID:-truth}"
export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "TRUTH_ID=$TRUTH_ID XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')"

julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_theta_baseline_dense.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
