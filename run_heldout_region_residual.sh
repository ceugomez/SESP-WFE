#!/bin/bash
#SBATCH --job-name=heldout_region
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/heldout_region_%j.log

# Leave-one-member-out residual energy vs. truncation rank, split by region
# (full / wake / ambient) and component (all / uv / w-only). Reuses the LOO bases
# already on disk (Bases/truth_XX_basis.jld2) — no POD rebuild — and projects
# each held-out member's fluctuation field onto its basis. Writes
# results/heldout/heldout_region_residual.csv (+ _headline.txt). BLAS threads come from
# SLURM_CPUS_PER_TASK (read inside the script).

set -euo pipefail

mkdir -p /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')"

# precompile single-threaded to avoid the multi-threaded precompile memory spike
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, JLD2, Statistics, Printf; @info "precompile done"'

julia --startup-file=no \
      /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/scripts/make_heldout_region_residual.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
