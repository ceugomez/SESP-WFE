# paths.jl — filesystem layout for the SESP-WFE estimation pipeline.
# Every driver includes this first:  include(joinpath(@__DIR__, "..", "paths.jl"))
# All locations derive from this file's position (SESP-WFE/scripts), so the repo can be
# moved or cloned without editing paths. viz/paths.py mirrors this file for Python.
#
# Every output location is NAMESPACED BY ENSEMBLE, so bases/results from different
# ensembles can never be mixed (the fold bases are reused across runs — see
# heldout_residual_curve — so a shared directory would silently cross-load them):
#
#   Bases-<tag>/            POD bases truth_<id>_basis.jld2 + companions (truth_<id>_models.jld2,
#                           rep_error_truth_<id>.jld2), one set per fold
#   results/<tag>/kf/       filter sweeps, Θ-baselines, Q diagnostics, cov_analysis/
#   results/<tag>/fields/   NetCDF field products (kf_field_*, rep_error_field_*, modes_*, cov/corrmap)
#   results/<tag>/heldout/  closure CSVs (residual curves, eigenvalue spectra)
#   results/<tag>/repr/     time-resolved representation error (run_repr_error.jl)
#   figure/<tag>/           every figure (paper figures in figure/<tag>/paper/)
#
# <tag> is the ensemble directory name minus "ensemble-" (ensemble-highvar -> highvar).

const SCRIPTS_DIR = @__DIR__
const PKG_ROOT    = normpath(joinpath(SCRIPTS_DIR, ".."))          # SESP-WFE/
const SRC_DIR     = joinpath(PKG_ROOT, "src")
const REPO_ROOT   = normpath(joinpath(PKG_ROOT, ".."))            # ensemble_runs_paper/

# Which ensemble to analyze. Override per run with `SESP_ENSEMBLE=ensemble-lowvar julia ...`
# (the slurm/ launchers pass it through).
const ENSEMBLE_NAME = get(ENV, "SESP_ENSEMBLE", "ensemble-highvar")
const ENSEMBLE_TAG  = replace(ENSEMBLE_NAME, r"^ensemble-" => "")
const ENSEMBLE_DIR  = joinpath(REPO_ROOT, ENSEMBLE_NAME)
const CONFIG_PATH   = joinpath(ENSEMBLE_DIR, "ensemble_config.json")
isfile(CONFIG_PATH) || error("ensemble config not found: $CONFIG_PATH (set SESP_ENSEMBLE)")

const BASES_DIR   = joinpath(PKG_ROOT, "Bases-$(ENSEMBLE_TAG)")
const RESULTS_DIR = joinpath(PKG_ROOT, "results", ENSEMBLE_TAG)
const KF_DIR      = joinpath(RESULTS_DIR, "kf")
const FIELDS_DIR  = joinpath(RESULTS_DIR, "fields")
const HELDOUT_DIR = joinpath(RESULTS_DIR, "heldout")
const REPR_DIR    = joinpath(RESULTS_DIR, "repr")
const FIGURE_DIR  = joinpath(PKG_ROOT, "figure", ENSEMBLE_TAG)

for d in (BASES_DIR, KF_DIR, FIELDS_DIR, HELDOUT_DIR, REPR_DIR, FIGURE_DIR)
    mkpath(d)
end
