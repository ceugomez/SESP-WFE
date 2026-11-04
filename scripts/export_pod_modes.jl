# export_pod_modes.jl — write the leading POD modes as FastEddy-format NetCDFs.
# Each mode P_r (domain-masked state vector) is unpacked to full u/v/w 3D fields
# (zero-filled in the sponge margin) and written via write_fasteddy_nc, so the
# Python slice/panel plotters read them directly. cgf cego6160@colorado.edu
using LinearAlgebra

include(joinpath(@__DIR__, "paths.jl"))  # BASES_DIR, FIELDS_DIR, SRC_DIR, ...
const WORKDIR    = SRC_DIR
# Accept basis file as optional CLI arg: julia export_pod_modes.jl [basis.jld2] [n_modes]
const BASIS_FILE = length(ARGS) >= 1 ? ARGS[1] : joinpath(BASES_DIR, "truth_truth_basis.jld2")
const MARGIN_M   = 600f0          # Sponge layer size
const N_MODES    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))

@info "Loading POD basis: $BASIS_FILE"
basis  = load_pod_basis(BASIS_FILE)
domain = AnalysisDomain(basis.grid; margin_m = MARGIN_M)
3 * domain.n_grid == size(basis.modes, 1) ||
    error("MARGIN_M=$MARGIN_M gives state_dim=$(3*domain.n_grid) ≠ basis $(size(basis.modes,1)); fix MARGIN_M")

n = min(N_MODES, basis.L)
basis_tag = replace(splitext(basename(BASIS_FILE))[1], "pod_basis_" => "")
modedir = joinpath(FIELDS_DIR, "modes_$(basis_tag)")
mkpath(modedir)
@info "Exporting $n modes to $modedir"
for k in 1:n
    u3, v3, w3 = unpack_snapshot(basis.modes[:, k], basis.grid, domain)
    # zero-pad index so the Python glob sorts modes correctly; store λ in the time slot.
    outfile = joinpath(modedir, "FE_mode_$(lpad(k-1, 2, '0')).0")
    write_fasteddy_nc(outfile, u3, v3, w3, basis.grid; t = Float32(basis.eigenvalues[k]))
    @info "  mode $(k-1): λ=$(basis.eigenvalues[k])  →  $outfile"
end
@info "=== Done. Panel plot: python src/viz/plot_pod_modes.py $modedir --zagl 50 ==="
