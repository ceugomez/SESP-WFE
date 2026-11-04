# make_kf_field_rmse.jl
# cgf cego6160@colorado.edu 6.25.26
# Resolve the KF total reconstruction error in space: load the saved filter coefficients
# (kf_result_*.jld2), re-stream the truth field, and write a per-cell RMSE-over-time map to
# NetCDF (rmse_speed/u/v/w + positions) for the Python slice plotters. Constants must match
# the run that produced the basis + kf_result.
using LinearAlgebra, JLD2, NCDatasets
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, FIELDS_DIR, SRC_DIR, ...
const WORKDIR     = SRC_DIR
const T_SPINUP    = 900.0
const MARGIN_M    = 600f0
const TRUTH_ID    = get(ENV, "TRUTH_ID", "truth")  # held-out/truth fold id; override via env to sweep folds
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

# filter coefficients α (L × K) and the oracle truth_coeffs (= project(basis, truth) over TIDX),
# both saved by save_kf_result.
α, truth_coeffs, sensor_locs = load(kf_result_filename(KF_DIR, TRUTH_ID), "α", "truth_coeffs", "sensor_locs")
# sensor rake is static, so the per-step locations are identical; take the first set (3 × nm)
sensor_xyz = sensor_locs[1]

# truth field over the analysis domain (state_dim × K), columns aligned with α
truth_U = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
size(truth_U, 2) == size(α, 2) ||
    error("truth has $(size(truth_U,2)) snapshots but α has $(size(α,2)) steps")

# Two spatial error maps (per-cell RMSE over time, total velocity-error magnitude):
#   filt — Prα − x_truth : the filter's total reconstruction error (carries the floor)
#   proj — Pr·truth_coeffs − x_truth : the IRREDUCIBLE projection/reconstruction floor of the
#          subspace (best in-subspace fit; independent of the filter). Where this is large the
#          subspace can't represent the truth (expect it in the wake/shear). filt − proj ≥ 0
#          localizes the online filter's own added error.
filt = field_rmse_map(basis, α,            truth_U, grid, domain)
proj = field_rmse_map(basis, truth_coeffs, truth_U, grid, domain)
indom = count(domain.mask)
@info "filter total RMSE: mean(in-domain)=$(round(sum(filt.speed)/indom, digits=4)) max=$(round(maximum(filt.speed),digits=4)) m/s"
@info "projection floor:  mean(in-domain)=$(round(sum(proj.speed)/indom, digits=4)) max=$(round(maximum(proj.speed),digits=4)) m/s"

outpath = joinpath(FIELDS_DIR, "kf_field_rmse_truth_$(TRUTH_ID).nc")
isfile(outpath) && rm(outpath)
NCDataset(outpath, "c") do ds
    defDim(ds, "xIndex", grid.Nx); defDim(ds, "yIndex", grid.Ny)
    defDim(ds, "zIndex", grid.Nz); defDim(ds, "time", 1)
    d4 = ("xIndex", "yIndex", "zIndex", "time")
    vars = (("rmse_speed", filt.speed), ("rmse_u", filt.u), ("rmse_v", filt.v), ("rmse_w", filt.w),
            ("proj_speed", proj.speed), ("proj_u", proj.u), ("proj_v", proj.v), ("proj_w", proj.w),
            ("xPos", grid.xPos), ("yPos", grid.yPos), ("zPos", grid.zPos))
    for (name, arr) in vars
        v = defVar(ds, name, Float32, d4); v[:, :, :, 1] = arr
    end
    tp = defVar(ds, "topoPos", Float32, ("xIndex", "yIndex", "time")); tp[:, :, 1] = grid.zPos[:, :, 1]
    # static sensor rake positions (m), for the plotters to overlay
    defDim(ds, "sensor", size(sensor_xyz, 2))
    for (name, row) in (("sensor_x", 1), ("sensor_y", 2), ("sensor_z", 3))
        sv = defVar(ds, name, Float32, ("sensor",)); sv[:] = sensor_xyz[row, :]
    end
end
@info "Wrote KF field-RMSE NetCDF (filter total + projection floor): $outpath"
