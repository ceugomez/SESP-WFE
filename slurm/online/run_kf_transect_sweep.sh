#!/bin/bash
#SBATCH --job-name=kf_transect_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/log/kf_transect_sweep_%j.log

# Rank sweep of the reduced-order KF observed by a TRANSITING POINT OBSERVER: one sensor
# at fixed x and height AGL, translating up and down along y across the wake. Companion to
# the static rake (run_kf_rank_sweep.sh) and the downwind array (run_kf_rank_sweep_dense.sh).
#
# 3 obs rows per update vs the array's 525: this asks what a single moving point can pin
# down by accumulating over time, not whether the geometry determines the subspace.
#
# One sensor per update => build_rep_covariance collapses to a 3x3 diagonal and the
# non-PSD kernel is never evaluated, so there is no spacing floor here. The cost is that
# rep-error correlation moves into TIME and goes unmodeled (the filter will be somewhat
# overconfident; expect NIS < 1) — but at the default 78 m/s the observer outruns it: the
# 468 m along-track step ~ L_perp, so consecutive samples largely decorrelate. See the
# script header before trusting the variances.
#
# Requires: basis, models, rep-error, and truth outputs on disk.
#
# Env knobs (defaults shown):
#   TRUTH_ID=truth           which member is held out as truth
#   OBS_STRIDE=1             assimilate every OBS_STRIDE-th snapshot (6 s apart)
#                            NB: stride 5 (30 s) is dt ~ T_meander/2, where persistence is
#                            at its WORST. See the cadence note in the script header.
#   TRANSECT_X_M=3000        fixed x of the transect (matches the static rake)
#   TRANSECT_Z_AGL_M=100     fixed height above ground (matches the rake's top sensor)
#   TRANSECT_HALFY_M=1900    transect runs crest_y +/- this (full interior y, 635-4435 m)
#   TRANSECT_SPEED=78        scan speed (m/s) => ~33 traverses, 468 m along-track per update
#                            SPEED ALIASES: the triangle wave resonates with the cadence at
#                            some speeds (v=100 visits only 20 distinct y cells, v=50/150
#                            only 39). 78 gives 256. The script's coverage guard fails loudly
#                            rather than sweeping a sparse comb. See the script header.

set -euo pipefail
source /home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/slurm/common.sh

export TRUTH_ID="${TRUTH_ID:-truth}"
export OBS_STRIDE="${OBS_STRIDE:-1}"
export TRANSECT_X_M="${TRANSECT_X_M:-3000}"
export TRANSECT_Z_AGL_M="${TRANSECT_Z_AGL_M:-100}"
export TRANSECT_HALFY_M="${TRANSECT_HALFY_M:-1900}"
export TRANSECT_SPEED="${TRANSECT_SPEED:-78}"

echo "TRUTH_ID=$TRUTH_ID OBS_STRIDE=$OBS_STRIDE x=$TRANSECT_X_M z_agl=$TRANSECT_Z_AGL_M" \
     "halfy=$TRANSECT_HALFY_M speed=$TRANSECT_SPEED"

precompile_julia

run_jl online/run_kf_transect_sweep.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
