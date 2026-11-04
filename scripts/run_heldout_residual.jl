# run_heldout_residual.jl — leave-one-member-out residual-energy curve (closure diagnostic)
# Builds the basis from M-1 members and measures the projection residual on the
# held-out member, looped over all members. Output is the family-closure curve.
# cgf cego6160@colorado.edu 4.1.26
using LinearAlgebra

# ---- Configuration ----
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, HELDOUT_DIR, SRC_DIR, ...
const WORKDIR          = SRC_DIR
const OUTPUT_DIR       = HELDOUT_DIR            # residual-curve CSV lands here
const BASIS_DIR        = BASES_DIR              # LOO bases saved + reused (shared with run_pod / KF)
const L                = 50          # max truncation rank evaluated (curve covers 1:L)
const T_SPINUP         = 900.0       # seconds — exclude spin-up transient
const MARGIN_M         = 600f0       # lateral exclusion margin in metres (sponge + buffer)
# Members cycled as held-out (nothing = all prior members in config).
const MEMBER_IDS       = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
const EXCLUDE_MEMBERS  = nothing

include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))

# allocate threads for the eigendecomposition / projections
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"

@info "=== Held-out residual curve (LOO, L=$L, t_spinup=$(T_SPINUP) s) ==="
hr = heldout_residual_curve(CONFIG_PATH;
    L           = L,
    t_spinup    = T_SPINUP,
    margin_m    = MARGIN_M,
    member_ids  = MEMBER_IDS,
    exclude_ids = EXCLUDE_MEMBERS,
    basis_dir   = BASIS_DIR,   # save + reuse each fold's LOO basis (shared with KF)
    reuse       = true,
)

mkpath(OUTPUT_DIR)
csv_file = joinpath(OUTPUT_DIR, "heldout_residual_curve.csv")
save_heldout_residual_csv(hr, csv_file)

# concise summary to stdout / log
@info "=== Summary (residual energy fraction, averaged over held-out members) ==="
for r in (1, 5, 10, 20, 30, 40, hr.L)
    r <= hr.L && @info "  r=$r: $(round(hr.resid_mean[r]*100, digits=3))%"
end
@info "Saved curve to $csv_file"
@info "=== Done ==="
