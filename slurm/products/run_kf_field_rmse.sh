#!/bin/bash
#SBATCH --job-name=kf_field_rmse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_field_rmse_%j.log

# Spatial (per-cell) RMSE map of the KF reconstruction, truth fold only.
# Writes results/<tag>/fields/kf_field_rmse_truth_truth.nc for viz/plot_kf_field_rmse.py.
# Mem: truth_U (state_dim=36.65M x K=268, f32) ~39G + basis.modes ~7G; field_rmse_map
# chunks 32 cols at a time (no full-size reconstruction), so 150G is ~3x the measured peak.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

run_jl products/make_kf_field_rmse.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
