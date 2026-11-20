#!/bin/bash
#SBATCH --job-name=partial_kf_members
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10/members_fig_%j.log

# ONE-OFF (2026-11-10): per-member top-down figures straight from the FastEddy output
# (I/O-bound: reads levels kmin..kmax of u,v for 4 members × NSNAP snapshots).
#   NSNAP=67 sbatch run_members_fig.sh
set -euo pipefail
HERE=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/oneoff/lowvar_partial_kf_2026-11-10
NSNAP="${NSNAP:-67}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasteddy
python "$HERE/plot_members.py" --nsnap "$NSNAP" --out "$HERE/out_n$NSNAP"
