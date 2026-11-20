#!/bin/bash
#SBATCH --job-name=theta_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/theta_baseline_%j.log

# R1: the Θ-regression baseline (2-parameter forcing fit + conditional-mean
# lookup) vs the saved KF run, full-domain RMSE.
# Requires: POD basis + kf_result for TRUTH_ID already on disk.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

run_jl online/run_theta_baseline.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
