#!/bin/bash
#SBATCH --job-name=partial_kf_lowvar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/partial_kf_%j.log

# ONE-OFF (2026-11-10): basis from finished lowvar members {03,10,11} + a few KF experiments,
# then the figures. Memory: 3-member cache ≈ 118 GB + truth ≈ 39 GB + full-grid truth copy for
# the measurement sampler ≈ 67 GB + bases ≈ 15 GB → ~250 GB peak.
#   sbatch run_partial_kf.sh
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

julia --startup-file=no --threads=1 -e 'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'
# N_SNAP=67 sbatch run_partial_kf.sh  -> quick run on the first quarter (outputs in out_n67/)
# POOL=03,04,10,11 N_SNAP=67 sbatch run_partial_kf.sh -> 4-member pool (outputs in out_n67_pool03041011/)
# POOL=00,...,11 OUT_TAG=_all11 N_SNAP=67 sbatch run_partial_kf.sh -> 11 finished members (out_n67_all11/)
POOL_TAG=""; [ -n "${POOL:-}" ] && [ "$POOL" != "03,10,11" ] && POOL_TAG="_pool${POOL//,/}"
[ -n "${OUT_TAG:-}" ] && POOL_TAG="$OUT_TAG"
OUTDIR=$HERE/out${N_SNAP:+_n$N_SNAP}$POOL_TAG
julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$HERE/run_partial_kf.jl"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasteddy
python "$HERE/plot_partial_kf.py" --out "$OUTDIR"
echo "Job finished at $(date)"
