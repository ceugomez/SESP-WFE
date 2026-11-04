#!/bin/bash
#SBATCH --job-name=kf_field_rmse_allbases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_field_rmse_allbases_%j.log

# Build the per-cell KF field-RMSE NetCDF (results/fields/kf_field_rmse_truth_<ID>.nc) for
# every fold, one at a time, via make_kf_field_rmse.jl with $TRUTH_ID overridden per fold.
# Each fold needs its kf_result_truth_<ID>.jld2 to exist first (run_simple_KF_all_bases.sh).
# Override the set of IDs by passing them as args:  sbatch run_kf_field_rmse_all_bases.sh 00 03
#
# Mem note: field_rmse chunks columns, so peak is ~50G/fold; 150G is headroom. Folds run
# sequentially in one job so peak memory does not stack.

set -euo pipefail

ROOT=/home/cego6160/workspace/ensemble_runs_paper
BASES_DIR=$ROOT/SESP-WFE/Bases
KF_DIR=$ROOT/SESP-WFE/results/kf
SCRIPT=$ROOT/SESP-WFE/scripts/make_kf_field_rmse.jl
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
echo "Folds to process: ${IDS[*]}"

# Precompile once, single-threaded, to avoid the multi-threaded precompile memory spike.
julia --startup-file=no --threads=1 -e \
  'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'

THREADS=${SLURM_CPUS_PER_TASK:-8}
fail=0
for id in "${IDS[@]}"; do
  if [ ! -f "$KF_DIR/kf_result_truth_$id.jld2" ]; then
    echo ">>> SKIP $id: kf_result_truth_$id.jld2 missing (run the assimilation first)" >&2
    fail=1
    continue
  fi
  echo "=============================================================="
  echo ">>> field-RMSE for basis truth=$id  ($(date))"
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
