#!/bin/bash
#SBATCH --job-name=kf_field_snapshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_field_snapshot_%j.log

# Single-time xy field snapshot (truth, filter estimate, oracle, prior mean), truth fold only.
# Writes results/<tag>/fields/kf_field_snapshot_truth_truth.nc for viz/plot_kf_field_snapshot.py.
# FOLDS restricts to the truth fold only -- the other folds' bases/KF results aren't built yet.
# Mem: same truth_U (~39G) + basis.modes (~7G) as kf_field_rmse; only single-column
# reconstructions beyond that, so 150G is ~3x the measured peak.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

precompile_julia

FOLDS="truth" run_jl products/make_kf_field_snapshot.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
