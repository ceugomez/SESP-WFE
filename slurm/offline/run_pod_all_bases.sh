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
# gram_slice / loo_basis_from_gram (see scripts/offline/run_pod_all_bases.jl). This replaces the old
# design of one fresh Julia process per fold, which re-streamed each member's snapshot
# matrix from disk independently per fold (O(M^2) total disk reads across folds).
#   sbatch slurm/offline/run_pod_all_bases.sh                 # truth + 00..11
#   sbatch slurm/offline/run_pod_all_bases.sh truth 09 10 11  # only these folds

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh
echo "Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')   folds: ${*:-truth 00 01 02 03 04 05 06 07 08 09 10 11}"

precompile_julia

run_jl offline/run_pod_all_bases.jl "$@"
echo "Job finished at $(date)"
