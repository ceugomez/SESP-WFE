#!/bin/bash
#SBATCH --job-name=pod_all_bases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=48:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/pod_all_%j.log

# Build every POD basis for the FULL 12-member ensemble, one fold at a time:
#   truth                -> basis on all 12 prior members (assimilates the nature run)
#   00,01,...,11         -> LOO bases, each on the other 11 prior members
# Each build is a fresh Julia process (frees the ~500 GB snapshot matrix between folds).
# Member set is fixed in run_pod.jl (INCLUDE_MEMBERS = 00..11); the held-out id is passed via
# TRUTH_ID ("" => config default = member_truth). Overwrites the stale 9-member bases on disk.
#   sbatch run_pod_all_bases.sh                 # truth + 00..11
#   sbatch run_pod_all_bases.sh truth 09 10 11  # only these folds

set -euo pipefail
ROOT=/home/cego6160/workspace/ensemble_runs_paper
SCRIPT=$ROOT/SESP-WFE/scripts/run_pod.jl
mkdir -p "$ROOT/SESP-WFE/log"

if [ "$#" -gt 0 ]; then
  IDS=("$@")
else
  IDS=(truth 00 01 02 03 04 05 06 07 08 09 10 11)
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Mem: $(grep MemTotal /proc/meminfo | awk '{print $2/1048576 " GB"}')   folds: ${IDS[*]}"

# Precompile once, single-threaded (avoids the multi-threaded precompile memory spike).
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

THREADS=${SLURM_CPUS_PER_TASK:-64}
fail=0
for id in "${IDS[@]}"; do
  echo "=============================================================="
  echo ">>> POD basis: held-out truth=$id   ($(date))"
  echo "=============================================================="
  # "truth" => config default (TRUTH_ID unset); any member id => that LOO fold.
  ENVID=""; [ "$id" != "truth" ] && ENVID="$id"
  if TRUTH_ID="$ENVID" julia --startup-file=no --threads="$THREADS" "$SCRIPT"; then
    echo ">>> done: $id"
  else
    echo ">>> FAILED: $id" >&2; fail=1
  fi
done
echo "Job finished at $(date) (fail=$fail)"
exit $fail
