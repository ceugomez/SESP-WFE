# run_covariance.jl — point-referenced spatial covariance map from a saved POD basis.
# cgf cego6160@colorado.edu
#
# Loads a POD basis, computes the centered spatial covariance of one reference DOF
# (component COMP at physical point REF_X, REF_Y, REF_Z_AGL) with the u/v/w fields
# everywhere, and writes a FastEddy-format NetCDF. Render with:
#   python src/viz/plot_xy_slices.py results/fields/covmap --zagl <Z> --ulim .. --vlim .. --wlim ..
# (the u/v/w fields hold cov(ref, u|v|w); set colorbar limits to the covariance scale.)
using LinearAlgebra

include(joinpath(@__DIR__, "paths.jl"))  # BASES_DIR, FIELDS_DIR, SRC_DIR, ...
const WORKDIR    = SRC_DIR
const TRUTH_ID   = "truth"        # which saved basis (truth_<id>_basis.jld2) to load
const MARGIN_M   = 600f0          # MUST match the value used to build the basis

# Reference point (domain is 506×506×82 @ 10 m → ~5060 m square; center ≈ 2530 m).
const REF_X      = 2530.0         # metres
const REF_Y      = 2530.0         # metres
const REF_Z_AGL  = 50.0           # metres above local terrain
const REF_COMP   = :u             # reference component: :u, :v, or :w
const NORMALIZE  = true           # true → correlation map in [−1,1]; false → covariance

include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))

basis_file = basis_filename(BASES_DIR, TRUTH_ID)
@info "Loading POD basis: $basis_file"
basis  = load_pod_basis(basis_file)
domain = AnalysisDomain(basis.grid; margin_m = MARGIN_M)

kind = NORMALIZE ? "correlation" : "covariance"
@info "Computing $kind map (ref comp=$REF_COMP @ x=$REF_X y=$REF_Y z_agl=$REF_Z_AGL)..."
res = covariance_map(basis, domain;
    x_m = REF_X, y_m = REF_Y, z_agl_m = REF_Z_AGL, comp = REF_COMP, normalize = NORMALIZE)

ncf   = res.ncf
i,j,k = res.cell
covdir = joinpath(FIELDS_DIR, NORMALIZE ? "corrmap" : "covmap")
mkpath(covdir)
# Name starts with FE_ so the Python plotters' glob picks it up; tag with ref cell.
tag     = NORMALIZE ? "corr" : "cov"
outfile = joinpath(covdir, "FE_$(tag)_$(REF_COMP)_i$(i)_j$(j)_k$(k).0")
write_fasteddy_nc(outfile,
    ncf.u[:,:,:,1], ncf.v[:,:,:,1], ncf.w[:,:,:,1], basis.grid)

@info "=== Done. $kind NetCDF: $outfile ==="
ulim = NORMALIZE ? " --ulim 1 --vlim 1 --wlim 1" : ""
@info "Render: python src/viz/plot_xy_slices.py $covdir --zagl $(REF_Z_AGL)$ulim"
