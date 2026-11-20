#!/bin/bash
#SBATCH --job-name=kf_rank_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_rank_sweep_%j.log

# Rank sweep of the reduced-order KF vs the Θ-baseline (paired noise).
# Requires: basis, models, rep-error, and truth outputs on disk.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

run_jl online/run_kf_rank_sweep.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
