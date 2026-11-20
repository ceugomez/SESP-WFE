#!/bin/bash
#SBATCH --job-name=rep_lowrank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/rep_lowrank_%j.log

# ONE-OFF (2026-11-20): low-rank + diagonal R_rep vs parametric kernel, POD-KF r=10
# (run_rep_lowrank.jl). Memory: cache 108 GB + 11 fold bases (r=10) 16 GB + truth/U4 27 GB
# + L=50 basis 7 GB → ~170 GB peak.
#   SIGMAS=0.5,0.1 sbatch run_rep_lowrank.sh
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$HERE/run_rep_lowrank.jl"
echo "Job finished at $(date)"
