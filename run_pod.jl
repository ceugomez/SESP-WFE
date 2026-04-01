# run_pod.jl — Driver script for POD basis construction and GMF prior initialisation.
# Called by run_pod.sh; configure the parameters below then submit as SLURM job.

using LinearAlgebra

# ---- Configuration ----
const CONFIG_PATH  = "/home/cego6160/workspace/runs/ensemble_fp/ensemble_config.json"
const WORKDIR      = "/home/cego6160/workspace/prediction"

const MODE         = :all   # :timemean (N=10, fast) or :all (N≈530, ~95 GB peak)
const L            = 50          # number of POD modes to retain
const T_SPINUP     = 900.0       # seconds — exclude spin-up transient (default 15 min)

const BASIS_FILE   = joinpath(WORKDIR, "pod_basis_$(MODE).jld2")
const PRIOR_FILE   = joinpath(WORKDIR, "gmf_prior_$(MODE).jld2")

# ---- BLAS threads ----
# Set to match --cpus-per-task in the SLURM script for optimal matrix performance.
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"

# ---- Load POD functions ----
include(joinpath(WORKDIR, "pod_ensemble.jl"))

# ---- Build basis ----
@info "=== Building POD basis (mode=$MODE, L=$L, t_spinup=$(T_SPINUP) s) ==="
basis = build_pod_basis(CONFIG_PATH; mode=MODE, L=L, t_spinup=T_SPINUP,
                         output_path=BASIS_FILE)

# ---- Verify ----
@info "=== Verifying basis ==="
verify_pod_basis(basis)

# ---- Project all members and truth ----
@info "=== Initialising GMF prior ==="
Z_prior, Z_truth = initialize_gmf_prior(CONFIG_PATH, BASIS_FILE; t_spinup=T_SPINUP)

# ---- Save projected coefficients ----
using JLD2
jldsave(PRIOR_FILE;
    Z_prior = Z_prior,    # Vector of (L, T) matrices, one per prior member
    Z_truth = Z_truth,    # (L, T) held-out truth coefficients
    L       = basis.L,
    mode    = String(MODE),
    t_spinup = T_SPINUP,
)
@info "Saved GMF prior coefficients to $PRIOR_FILE"

# ---- Quick reconstruction check on truth ----
@info "=== Reconstruction check (truth member, time-mean) ==="
config  = load_ensemble_config(CONFIG_PATH)
truth   = config.truth_member
snaps   = load_member_snapshots(truth; t_spinup=T_SPINUP)
x_mean  = vec(mean(snaps, dims=2))
zeta    = project_snapshot(x_mean, basis)
x_hat   = reconstruct_snapshot(zeta, basis)
abs_err = abs.(x_hat .- x_mean)
@info "  MAE: $(round(mean(abs_err),    digits=4)) m/s"
@info "  Min: $(round(minimum(abs_err), digits=4)) m/s"
@info "  Max: $(round(maximum(abs_err), digits=4)) m/s"

# Reconstruction quality across the full post-spinup truth trajectory
@info "=== Reconstruction check (truth member, all $(size(snaps,2)) post-spinup timesteps) ==="
acc_mae = 0.0; acc_min = Inf32; acc_max = -Inf32
for t in 1:size(snaps, 2)
    xt    = snaps[:, t]
    ζt    = project_snapshot(xt, basis)
    x̂t   = reconstruct_snapshot(ζt, basis)
    err_t = abs.(x̂t .- xt)
    acc_mae += mean(err_t)
    acc_min  = min(acc_min, minimum(err_t))
    acc_max  = max(acc_max, maximum(err_t))
end
@info "  MAE: $(round(acc_mae / size(snaps,2), digits=4)) m/s"
@info "  Min: $(round(acc_min,                 digits=4)) m/s"
@info "  Max: $(round(acc_max,                 digits=4)) m/s"
@info "=== Done ==="
