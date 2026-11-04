# export_loo_figs.jl — make figure intermediates from the LOO bases already on disk.
# For each completed truth_XX_basis.jld2 this writes:
#   output/eigenvalue_spectrum_XX.csv      (for plot_eigenvalue_spectrum.py)
#   output/modes_truth_XX/FE_mode_*.0      (for plot_pod_modes.py)
# Loads only the fields each product needs (not the full basis) so it stays light
# while the heldout_residual job is running on the same node.
# cgf 2026-06-26
using LinearAlgebra, JLD2

include(joinpath(@__DIR__, "paths.jl"))  # BASES_DIR, FIELDS_DIR, HELDOUT_DIR, SRC_DIR, ...
const WORKDIR    = SRC_DIR
const MARGIN_M   = 600f0
const N_MODES    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const MEMBER_IDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))

for mid in MEMBER_IDS
    bfile = basis_filename(BASES_DIR, mid)
    isfile(bfile) || (@warn "missing $bfile, skip"; continue)
    @info "=== $mid ==="

    # --- eigenvalue spectrum (cheap: load only the eigenvalues) ---
    λ = jldopen(bfile, "r") do f; f["eigenvalues"]; end
    save_eigenvalue_spectrum_csv(λ, joinpath(HELDOUT_DIR, "eigenvalue_spectrum_$(mid).csv"))

    # --- leading modes (load only modes + grid, never coeffs) ---
    modes, Nx, Ny, Nz, dx, dy, xP, yP, zP = jldopen(bfile, "r") do f
        f["modes"], f["grid_Nx"], f["grid_Ny"], f["grid_Nz"],
        f["grid_dx"], f["grid_dy"], f["grid_xPos"], f["grid_yPos"], f["grid_zPos"]
    end
    grid   = GridInfo(Nx, Ny, Nz, dx, dy, xP, yP, zP, Nx*Ny*Nz, 3*Nx*Ny*Nz)
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)
    3*domain.n_grid == size(modes, 1) ||
        error("MARGIN_M mismatch: state_dim=$(3*domain.n_grid) ≠ $(size(modes,1))")

    modedir = joinpath(FIELDS_DIR, "modes_truth_$(mid)")
    mkpath(modedir)
    nm = min(N_MODES, size(modes, 2))
    for k in 1:nm
        u3, v3, w3 = unpack_snapshot(modes[:, k], grid, domain)
        outfile = joinpath(modedir, "FE_mode_$(lpad(k-1, 2, '0')).0")
        write_fasteddy_nc(outfile, u3, v3, w3, grid; t=Float32(λ[k]))
    end
    @info "  wrote spectrum csv + $nm modes → $modedir"
    modes = nothing; GC.gc()
end
@info "=== Done ==="
