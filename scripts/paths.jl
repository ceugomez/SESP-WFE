# paths.jl — filesystem layout for the SESP-WFE estimation pipeline.
# Include from any driver: include(joinpath(@__DIR__, "paths.jl"))
# All locations derive from this file's position (SESP-WFE/scripts), so the repo
# can be moved or cloned elsewhere without editing paths.
#
#   Bases/        POD bases (truth_<id>_basis.jld2) and their companion products, saved by
#                 default alongside the basis: prior/process-noise model
#                 (truth_<id>_models.jld2) and representativeness-error model
#                 (rep_error_truth_<id>.jld2)
#   results/kf/   KF coeff/diagnostic outputs
#   results/fields/  NetCDF field products (kf_field_*, rep_error_field_*, modes_*, covmap/corrmap)
#   results/heldout/ closure CSVs (residual curves, eigenvalue spectra)
#   figure/       every non-field figure (PNG)

const SCRIPTS_DIR = @__DIR__
const SRC_DIR     = normpath(joinpath(SCRIPTS_DIR, "..", "src"))
const PKG_ROOT    = normpath(joinpath(SCRIPTS_DIR, ".."))          # SESP-WFE/
const REPO_ROOT   = normpath(joinpath(PKG_ROOT, ".."))            # ensemble_runs_paper/

const CONFIG_PATH = joinpath(REPO_ROOT, "ensemble", "ensemble_config.json")

const BASES_DIR   = joinpath(PKG_ROOT, "Bases")
const RESULTS_DIR = joinpath(PKG_ROOT, "results")
const KF_DIR      = joinpath(RESULTS_DIR, "kf")
const FIELDS_DIR  = joinpath(RESULTS_DIR, "fields")
const HELDOUT_DIR = joinpath(RESULTS_DIR, "heldout")
const FIGURE_DIR  = joinpath(PKG_ROOT, "figure")

for d in (BASES_DIR, KF_DIR, FIELDS_DIR, HELDOUT_DIR, FIGURE_DIR)
    mkpath(d)
end
