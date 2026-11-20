#!/bin/bash
#SBATCH --job-name=kf_field_rmse_allbases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_field_rmse_allbases_%j.log

# Build the per-cell KF field-RMSE NetCDF (results/<tag>/fields/kf_field_rmse_truth_<ID>.nc) for
# every fold, one at a time, via make_kf_field_rmse.jl with $TRUTH_ID overridden per fold.
# Each fold needs its kf_result_truth_<ID>.jld2 to exist first (slurm/online/run_simple_KF_all_bases.sh).
# Override the set of IDs by passing them as args:  sbatch slurm/products/run_kf_field_rmse_all_bases.sh 00 03
#
# Mem note: field_rmse chunks columns, so peak is ~50G/fold; 150G is headroom. Folds run
# sequentially in one job so peak memory does not stack.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

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

echo "Folds to process: ${IDS[*]}"

precompile_julia

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
  if TRUTH_ID="$id" run_jl products/make_kf_field_rmse.jl; then
    echo ">>> done: $id"
  else
    echo ">>> FAILED: $id" >&2
    fail=1
  fi
done

echo "Job finished at $(date) (fail=$fail)"
exit $fail
