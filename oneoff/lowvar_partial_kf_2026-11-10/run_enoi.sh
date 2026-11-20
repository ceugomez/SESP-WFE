#!/bin/bash
#SBATCH --job-name=enoi_lowvar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/enoi_%j.log

# ONE-OFF (2026-11-18): EnOI / localized EnOI / cycled EnKF baselines vs the POD-KF (run_enoi.jl).
# Same POOL / N_SNAP / OUT_TAG env as run_partial_kf.sh; run AFTER it so the POD rows can be binned:
#   POOL=00,01,02,03,04,05,06,07,08,10,11 OUT_TAG=_all11 N_SNAP=67 \
#     sbatch --dependency=afterok:<partial_kf job> run_enoi.sh
# Memory (11 members, n67): cache ≈ 108 GB + truth 10 + full-grid truth copy 17 + BHᵀ/K/ρ∘BHᵀ
# ≈ 20 + EnKF ensemble (Ne=50) ≈ 7 + POD modes ≈ 15 → ~180 GB peak.
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$HERE/run_enoi.jl"

# 2026-11-20: distance-bin figure (the separate enoi_plot job died under sh: no `source`)
OUTDIR=$HERE/out${N_SNAP:+_n$N_SNAP}${OUT_TAG:-}
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasteddy
mkdir -p "$OUTDIR/figures"
python /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/viz/plot_enoi_bins.py "$OUTDIR/enoi/bins.csv" --out "$OUTDIR/figures/enoi_bins.png"
echo "Job finished at $(date)"
