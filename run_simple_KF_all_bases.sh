#!/bin/bash
#SBATCH --job-name=simple_kf_allbases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=728G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/simpleKF_allbases_%j.log

# Run the simple Kalman filter over every POD basis in Bases/, one fold at a time.
# Each basis is named truth_<ID>_basis.jld2; ID is passed to the driver via $TRUTH_ID.
# Override the set of IDs by passing them as args:  sbatch run_simple_KF_all_bases.sh 00 03 truth

set -euo pipefail

ROOT=/home/cego6160/workspace/ensemble_runs_paper
BASES_DIR=$ROOT/SESP-WFE/Bases
SCRIPT=$ROOT/SESP-WFE/scripts/run_simple_KF_assimilation.jl
LOGDIR=$ROOT/SESP-WFE/log
mkdir -p "$LOGDIR"

# IDs: explicit args win; otherwise auto-discover from the basis files on disk.
if [ "$#" -gt 0 ]; then
  IDS=("$@")
else
  mapfile -t IDS < <(ls "$BASES_DIR"/truth_*_basis.jld2 2>/dev/null \
                     | sed -E 's#.*/truth_(.*)_basis\.jld2#\1#')
fi

if [ "${#IDS[@]}" -eq 0 ]; then
  echo "No bases found in $BASES_DIR (truth_*_basis.jld2)." >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}   Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')"
echo "Bases to process: ${IDS[*]}"

# Precompile once, single-threaded, to avoid the multi-threaded precompile memory spike.
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

THREADS=${SLURM_CPUS_PER_TASK:-8}
fail=0
for id in "${IDS[@]}"; do
  echo "=============================================================="
  echo ">>> KF over basis truth=$id  ($(date))"
  echo "=============================================================="
  if TRUTH_ID="$id" julia --startup-file=no --threads="$THREADS" "$SCRIPT"; then
    echo ">>> done: $id"
  else
    echo ">>> FAILED: $id" >&2
    fail=1
  fi
done

echo "Job finished at $(date) (fail=$fail)"
exit $fail
