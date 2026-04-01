# visualize_modes.jl
# Export POD modes to NetCDF and render a static 4-panel PNG for each.
# Run from /home/cego6160/workspace/:
#
#   julia --startup-file=no prediction/visualize_modes.jl
#
# Outputs: prediction/pod_modes/mode_01.png … mode_09.png

using Printf

include(joinpath(@__DIR__, "pod_ensemble.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────
const BASIS_FILE  = "prediction/pod_basis_timemean.jld2"
const MODES_DIR   = "prediction/pod_modes"
const REF_FILE    = "runs/member_00/output/FE_Member00.180000"
const ANIMATE_PY  = "runs/ensemble_fp/setup/animate_output.py"
const PYTHON      = `conda run -n fasteddy python3`

# ── Load basis ────────────────────────────────────────────────────────────────
@info "Loading basis from $BASIS_FILE..."
basis = load_pod_basis(BASIS_FILE)
@info "  L=$(basis.L) modes, mode=$(basis.snapshot_mode), state_dim=$(basis.grid.state_dim)"

# ── Export modes to NetCDF ────────────────────────────────────────────────────
mkpath(MODES_DIR)
@info "Exporting $(basis.L) modes to $MODES_DIR..."
export_all_modes(basis, MODES_DIR, REF_FILE)

# ── Render one PNG per mode ───────────────────────────────────────────────────
@info "Rendering images..."
for k in 1:basis.L
    frac = basis.energy_frac[k]
    incr = k == 1 ? frac : frac - basis.energy_frac[k-1]
    title = @sprintf("POD mode %d  (%.1f%% / cumul. %.1f%%)", k, incr*100, frac*100)
    outimg = joinpath(MODES_DIR, @sprintf("mode_%02d.png", k))

    modefile = joinpath(MODES_DIR, "FE_Pod_mode.$k")
    cmd = `$PYTHON $ANIMATE_PY
        --file      $modefile
        --Nx        $(basis.grid.Nx)
        --Ny        $(basis.grid.Ny)
        --dx        $(basis.grid.dx)
        --dy        $(basis.grid.dy)
        --static
        --symmetric_h
        --no_quiver
        --out_image $outimg
        --title     $title`

    @info "  Mode $k: $title"
    run(cmd)
    @info "  → $outimg"
end

@info "Done. Images in $MODES_DIR/"
