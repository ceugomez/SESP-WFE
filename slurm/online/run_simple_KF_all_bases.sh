#!/bin/bash
#SBATCH --job-name=simple_kf_allbases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=728G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/simpleKF_allbases_%j.log

# Run the single-rank Kalman filter over every POD basis in Bases-<tag>/, one fold at a time.
# Each basis is named truth_<ID>_basis.jld2; ID is passed to the driver via $TRUTH_ID.
# Override the set of IDs by passing them as args:  sbatch slurm/online/run_simple_KF_all_bases.sh 00 03 truth

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

echo "Bases to process: ${IDS[*]}"

precompile_julia

fail=0
for id in "${IDS[@]}"; do
  echo "=============================================================="
  echo ">>> KF over basis truth=$id  ($(date))"
  echo "=============================================================="
  if TRUTH_ID="$id" run_jl online/run_simple_KF_assimilation.jl; then
    echo ">>> done: $id"
  else
    echo ">>> FAILED: $id" >&2
    fail=1
  fi
done

echo "Job finished at $(date) (fail=$fail)"
exit $fail
