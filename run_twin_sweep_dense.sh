#!/bin/bash
#SBATCH --job-name=twin_sweep_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/twin_sweep_dense_%A_%a.log

# 12-FOLD identical-twin campaign (rung 1 of the experiment ladder), the twin counterpart
# of run_fold_sweep_dense.sh. Each task assimilates one prior member 00..11 against the
# truth_truth pool basis — which contains all twelve — so every truth lies in its own
# subspace and the only error left is tracking (random walk + Q) plus the L=50 truncation.
#
# The pairing is the whole point: fold k here and fold k in run_fold_sweep_dense.sh share
# the basis rank list, the sensor lattice, the wake mask, the Q, and the RNG seed. They
# differ ONLY in whether member k was in the basis. Their difference at fixed r is the
# measured closure price — the cost of a held-out realization not being spanned. Neither
# the twin numbers nor the difference license a standalone skill claim: the twin is a
# deliberate inverse crime and must be labeled as one in the paper.
#
# NOTE: the "truth" member itself is NOT a fold here. It is in no basis, so it cannot be a
# twin; it stays the subject of run_fold_sweep_dense.sh (fold "truth"), which is why this
# array is 0-11 while that one is 0-12.
#
# No new LES runs are needed — this reuses the existing members and the existing pool basis.
#
# Env knobs (defaults shown, same as run_kf_twin_dense.sh):
#   BASIS_ID=truth           pool basis containing all 12 prior members
#   REP_MODE=none            R = sensor floor + tail clutter; see the R_rep note in
#                            scripts/run_kf_twin_dense.jl before setting this to "loo"
#   OBS_STRIDE=1             assimilate every OBS_STRIDE-th snapshot (6 s apart)
#   SENSOR_STRIDE_XY=50      sensor spacing in x,y (grid cells; dx = 10 m -> 500 m)
#   SENSOR_STRIDE_Z=10       sensor spacing in vertical levels
#
# Writes results/kf/kf_twin_dense_basistruth_truth_<ID>.csv (one per fold).
# Single-fold runtime tracks the LOO dense sweep (~1.5 h); with %4 the array should land
# in ~5 h.

set -euo pipefail

MEMBERS=(00 01 02 03 04 05 06 07 08 09 10 11)
export TRUTH_ID=${MEMBERS[$SLURM_ARRAY_TASK_ID]}

export BASIS_ID="${BASIS_ID:-truth}"
export REP_MODE="${REP_MODE:-none}"
export OBS_STRIDE="${OBS_STRIDE:-1}"
export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (twin fold $TRUTH_ID) starting on $(hostname) at $(date)"
echo "BASIS_ID=$BASIS_ID TRUTH_ID=$TRUTH_ID REP_MODE=$REP_MODE OBS_STRIDE=$OBS_STRIDE XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"

julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

julia --startup-file=no \
      --threads=$SLURM_CPUS_PER_TASK \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/run_kf_twin_dense.jl

echo "Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID (twin fold $TRUTH_ID) finished at $(date)"
