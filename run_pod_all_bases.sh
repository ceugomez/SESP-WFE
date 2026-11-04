#!/bin/bash
#SBATCH --job-name=pod_all_bases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/pod_all_%j.log

# Build every POD basis for the FULL 12-member ensemble in ONE Julia process:
#   truth                -> basis on all 12 prior members (assimilates the nature run)
#   00,01,...,11         -> LOO bases, each on the other 11 prior members
# All 12 members are read from disk exactly once into a RAM cache, the full-pool Gram
# matrix is built exactly once, and every fold is derived from that one Gram matrix via
# gram_slice / loo_basis_from_gram (see run_pod_all_bases.jl). This replaces the old
# design of one fresh Julia process per fold, which re-streamed each member's snapshot
# matrix from disk independently per fold (O(M^2) total disk reads across folds).
#   sbatch run_pod_all_bases.sh                 # truth + 00..11
#   sbatch run_pod_all_bases.sh truth 09 10 11  # only these folds

set -euo pipefail
ROOT=/home/cego6160/workspace/ensemble_runs_paper
SCRIPT=$ROOT/SESP-WFE/scripts/run_pod_all_bases.jl
mkdir -p "$ROOT/SESP-WFE/log"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')   folds: ${*:-truth 00 01 02 03 04 05 06 07 08 09 10 11}"

# Precompile once, single-threaded (avoids the multi-threaded precompile memory spike).
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

THREADS=${SLURM_CPUS_PER_TASK:-64}
julia --startup-file=no --threads="$THREADS" "$SCRIPT" "$@"
echo "Job finished at $(date)"
