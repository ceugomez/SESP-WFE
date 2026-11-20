#!/bin/bash
#SBATCH --job-name=r10_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/r10_compare_%j.log

# ONE-OFF (2026-11-20): POD-KF r=10 vs unlocalized EnOI vs unlocalized EnKF at σ = 0.5, 0.1 m/s
# (run_r10_compare.jl), then the figures. Memory ≈ run_enoi.sh (~180 GB peak).
#   SIGMAS=0.5,0.1 sbatch run_r10_compare.sh
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$HERE/run_r10_compare.jl"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasteddy
python "$HERE/plot_r10_compare.py" --out "$HERE/out_n67${OUT_TAG:-_all11}"
echo "Job finished at $(date)"
