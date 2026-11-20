#!/bin/bash
#SBATCH --job-name=partial_kf_ranksweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/rank_sweep_%j.log

# ONE-OFF (2026-11-10): rank sweep on the bases saved by run_partial_kf.jl, then its figure.
# Needs out_n<N_SNAP>/bases/ — submit after the main job:
#   N_SNAP=67 sbatch --dependency=afterok:<main job id> run_rank_sweep.sh
# Memory (N_SNAP=67): basis ≈ 7.3 GB + truth ≈ 10 GB + full-grid truth copy ≈ 17 GB.
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
export N_SNAP="${N_SNAP:-67}"
OUTDIR=$HERE/out_n$N_SNAP
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date), N_SNAP=$N_SNAP"
julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$HERE/run_rank_sweep.jl"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasteddy
python "$HERE/plot_rank_sweep.py" --out "$OUTDIR"
echo "Job finished at $(date)"
