#!/bin/bash
#SBATCH --job-name=cov_rep_arms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/cov_rep_arms_%j.log

# Rep-error calibration study, Riccati (offline) side: three R-model arms on a FIXED
# config (fold 03, LOO basis) so the rep treatment is the ONLY thing that varies —
# unlike the legacy cov_fold03_* files, where REP=auto tied rep to basis/sensor choice.
#   none = R_sensor only (rep error ignored -> overconfident, claims to observe past the edge)
#   diag = R_sensor + diag(R_rep) (variance inflated, correlations dropped)
#   full = R_sensor + R_rep (the proper model)
# Runs both sensor configs (dense 500 m lattice = PD floor for full R_rep; rake 15).
# Each arm records eigmin(R_sensor + R_rep) as the PD check — do not quote the full-arm
# results if that comes back negative.
#
# Writes results/kf/cov_analysis/cov_fold03_<sensor>_loo_rep<arm>.jld2
# Plot with: src/viz/plot_rep_calibration_riccati.py

set -euo pipefail
mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

export FOLD="${FOLD:-03}"
export BASIS="${BASIS:-loo}"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

for SENSOR in dense rake; do
  for REP in none diag full; do
    export SENSOR REP
    echo "=== FOLD=$FOLD SENSOR=$SENSOR BASIS=$BASIS REP=$REP ==="
    julia --startup-file=no \
          --threads=$SLURM_CPUS_PER_TASK \
          /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/analyze_filter_covariance.jl
  done
done

echo "Job $SLURM_JOB_ID finished at $(date)"
