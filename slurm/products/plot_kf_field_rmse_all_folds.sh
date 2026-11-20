#!/bin/bash
# plot_kf_field_rmse_all_folds.sh
# Render figure/<tag>/kf_field_rmse_truth_<ID>.png (with sensor rake overlaid) for every fold
# that has a results/<tag>/fields/kf_field_rmse_truth_<ID>.nc. Light Python job — run directly,
# no SLURM:
#   conda activate fasteddy && slurm/products/plot_kf_field_rmse_all_folds.sh   [ID ...]
set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

mkdir -p "$FIGURE_DIR"

if [ "$#" -gt 0 ]; then
  IDS=("$@")
else
  mapfile -t IDS < <(ls "$FIELDS_DIR"/kf_field_rmse_truth_*.nc 2>/dev/null \
                     | sed -E 's#.*/kf_field_rmse_truth_(.*)\.nc#\1#')
fi

for id in "${IDS[@]}"; do
  nc="$FIELDS_DIR/kf_field_rmse_truth_$id.nc"
  if [ ! -f "$nc" ]; then echo ">>> SKIP $id: $nc missing" >&2; continue; fi
  echo ">>> plotting fold $id"
  python "$VIZ_DIR/plot_kf_field_rmse.py" "$nc" --out "$FIGURE_DIR/kf_field_rmse_truth_$id.png"
done
