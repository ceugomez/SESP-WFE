#!/bin/bash
#SBATCH --job-name=kf_twin_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_twin_dense_%j.log

# SINGLE-FOLD identical-twin dense rank sweep (rung 1). Assimilates one prior member
# against the 12-member pool basis that member helped build, so the truth lies in the
# subspace by construction and only tracking error remains. See the header of
# scripts/online/run_kf_twin_dense.jl — in particular, these scores are a deliberate inverse
# crime and may only be quoted as a difference against the matched LOO fold from
# run_fold_sweep_dense.sh.
#
# Env knobs (defaults shown):
#   BASIS_ID=truth           basis to estimate in (truth = all 12 prior members)
#   TRUTH_ID=00              assimilated member; MUST be in the basis (script enforces it)
#   REP_MODE=none            "none" = R_sensor + tail clutter (the defensible twin);
#                            "loo"  = also add the out-of-sample LOO R_rep (diagnostic only)
#   OBS_STRIDE=1             assimilate every OBS_STRIDE-th snapshot (6 s apart)
#   SENSOR_STRIDE_XY=50      sensor spacing in x,y (grid cells; dx = 10 m -> 500 m)
#   SENSOR_STRIDE_Z=10       sensor spacing in vertical levels
#
# Keep SENSOR_STRIDE_XY=50 for the headline result: the rung1-vs-rung2 difference is only
# meaningful if both rungs see the same sensors. (The 500 m PD floor that binds the LOO
# sweep is an R_rep-kernel constraint and does NOT bind at REP_MODE=none, so a tighter
# lattice is available here for a separate sensor-density study.)
#
# Writes results/<tag>/kf/kf_twin_dense_basis<BASIS_ID>_truth_<TRUTH_ID>.csv

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

export BASIS_ID="${BASIS_ID:-truth}"
export TRUTH_ID="${TRUTH_ID:-00}"
export REP_MODE="${REP_MODE:-none}"
export OBS_STRIDE="${OBS_STRIDE:-1}"
export SENSOR_STRIDE_XY="${SENSOR_STRIDE_XY:-50}"
export SENSOR_STRIDE_Z="${SENSOR_STRIDE_Z:-10}"

echo "BASIS_ID=$BASIS_ID TRUTH_ID=$TRUTH_ID REP_MODE=$REP_MODE OBS_STRIDE=$OBS_STRIDE XY=$SENSOR_STRIDE_XY Z=$SENSOR_STRIDE_Z"

precompile_julia

run_jl online/run_kf_twin_dense.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
