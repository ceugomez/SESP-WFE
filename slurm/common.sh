#!/bin/bash
# common.sh — sourced by every launcher in slurm/. One place for the repo location, the
# ensemble selection, the output layout (mirrors scripts/paths.jl) and the Julia invocation.
#
# Ensemble: export SESP_ENSEMBLE before sbatch to analyze another ensemble, e.g.
#   SESP_ENSEMBLE=ensemble-lowvar sbatch slurm/online/run_fold_sweep_dense.sh
# (sbatch exports the submitting environment by default, so the variable reaches the job.)
#
# #SBATCH --output paths cannot use variables, so each launcher still spells out log/.

SESP_ROOT=/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE
export SESP_ENSEMBLE="${SESP_ENSEMBLE:-ensemble-highvar}"
ENSEMBLE_TAG="${SESP_ENSEMBLE#ensemble-}"

SCRIPTS_DIR=$SESP_ROOT/scripts
VIZ_DIR=$SESP_ROOT/viz
BASES_DIR=$SESP_ROOT/Bases-$ENSEMBLE_TAG
RESULTS_DIR=$SESP_ROOT/results/$ENSEMBLE_TAG
KF_DIR=$RESULTS_DIR/kf
FIELDS_DIR=$RESULTS_DIR/fields
FIGURE_DIR=$SESP_ROOT/figure/$ENSEMBLE_TAG
LOG_DIR=$SESP_ROOT/log
mkdir -p "$LOG_DIR"

echo "Job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "Ensemble: $SESP_ENSEMBLE   CPUs: ${SLURM_CPUS_PER_TASK:-?}"

# Precompile packages single-threaded first (avoids the multi-threaded precompile memory spike).
precompile_julia() {
    julia --startup-file=no --threads=1 -e \
        'using LinearAlgebra, NCDatasets, Statistics, JSON3, JLD2; @info "precompile done"'
}

# run_jl <stage>/<driver>.jl [args...] — run a driver from scripts/ on the job's CPUs.
run_jl() {
    local script=$1; shift
    julia --startup-file=no --threads="${SLURM_CPUS_PER_TASK:-8}" "$SCRIPTS_DIR/$script" "$@"
}
