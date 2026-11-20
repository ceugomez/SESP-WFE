#!/bin/bash
#SBATCH --job-name=simple_kf_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=728G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/simpleKF_%j.log

# run a simple kalman filter assimilation test, to v&v backend

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

run_jl online/run_simple_KF_assimilation.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
