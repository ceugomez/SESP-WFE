# make_rep_error_fields.jl
# cgf  2026-07-08
# Spatial representability-error (σ_rep) fields, one NetCDF per truth id, read
# directly from the saved RepError statistical model (rep_error_truth_XX.jld2,
# built offline by build_rep_error/save_rep_error in reperror.jl — the
# leave-one-out, Gram-matrix-sliced model, not a per-member recomputation).
#
# Writes FIELDS_DIR/rep_error_field_XX.nc in the standard FastEddy NetCDF
# layout (write_fasteddy_nc), so plot_xy_slices.py / plot_xz_slices.py-style
# plotters can read it unchanged. u/v/w here are σ_rep(ζ) per component
# (sqrt of the stored per-cell variance), not a wind field.
using JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, shared setup

include(joinpath(SRC_DIR, "fasteddy_io.jl"))
include(joinpath(SRC_DIR, "pod.jl"))
include(joinpath(SRC_DIR, "measurement.jl"))
include(joinpath(SRC_DIR, "reperror.jl"))

rep_files = sort(filter(f -> occursin(r"^rep_error_truth_.*\.jld2$", basename(f)),
                         readdir(BASES_DIR; join=true)))
isempty(rep_files) && error("No rep_error_truth_*.jld2 files found in $BASES_DIR; " *
                             "run scripts/offline/run_pod_all_bases.jl first")

for filepath in rep_files
    mid = replace(basename(filepath), r"^rep_error_truth_" => "", r"\.jld2$" => "")
    @info "=== Truth $mid ==="
    rep    = load_rep_error(filepath)
    domain = AnalysisDomain(rep.grid; margin_m=MARGIN_M)
    domain.n_grid == rep.n_grid ||
        error("MARGIN_M=$MARGIN_M gives domain.n_grid=$(domain.n_grid) ≠ " *
              "rep.n_grid=$(rep.n_grid); rep_error was built with a different margin")

    σ_rep = Float32.(sqrt.(max.(rep.σ², 0.0)))     # per-cell std, [u;v;w] masked layout
    u3, v3, w3 = unpack_snapshot(σ_rep, rep.grid, domain)

    outpath = joinpath(FIELDS_DIR, "rep_error_field_$(mid).nc")
    write_fasteddy_nc(outpath, u3, v3, w3, rep.grid)
    @info "  σ_rep: mean=$(round(sum(σ_rep)/rep.n_grid, digits=4)) max=$(round(maximum(σ_rep), digits=4)) m/s"
end
@info "=== Done ==="
