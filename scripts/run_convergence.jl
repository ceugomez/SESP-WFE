# cgf cego6160@colorado.edu
# Dynamic field convergence run.
# 2 flight-track vehicles, 50 assimilation steps advancing through time.
# Random-walk propagation via process noise Q between steps.

const WORKDIR = "/home/cego6160/workspace/prediction/src"
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "GMF.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "plotting.jl"))
include(joinpath(WORKDIR, "comparisons.jl"))

const OUTDIR = "/home/cego6160/workspace/prediction/output"

# ── Data ──────────────────────────────────────────────────────────────────────
basis       = load_pod_basis(joinpath(OUTDIR, "pod_basis.jld2"))                                    # POD basis reduced from ensemble 
encfg       = load_ensemble_config("/home/cego6160/workspace/runs_lad/ensemble_config.json")        # ensemble config json, describes members & truth field
truth_field = reconstruct_NCfield(load_member_snapshots(encfg.truth_member), encfg.truth_member, basis.grid)    # truth field in NCfield struct
prior       = init_prior(basis, encfg.n_members)                                                                # multimodal ensemble prior, as GM

# ── POD reconstruction floor (evaluated at first truth snapshot) ──────────────
truth_tidx_start = 2
max_iter         = 50
tidx_seq         = collect(truth_tidx_start : truth_tidx_start + max_iter - 1)
@assert tidx_seq[end] <= length(truth_field.t) "truth field has only $(length(truth_field.t)) timesteps; reduce max_iter"

truth_state0  = get_truth_state(truth_field, truth_tidx_start)
truth_coeffs0 = Vector{Float32}(basis.modes' * truth_state0)
pod_rmse      = Float64(sqrt(mean((basis.modes * truth_coeffs0 .- truth_state0).^2)))
@info "POD floor (L=$(basis.L)): RMSE = $(round(pod_rmse, digits=4)) m/s"

# ── Time-varying truth coefficients (for coeff tracking plot) ─────────────────
truth_coeffs_seq = Matrix{Float32}(undef, max_iter, basis.L)
for (i, tidx) in enumerate(tidx_seq)
    s = get_truth_state(truth_field, tidx)
    truth_coeffs_seq[i, :] = basis.modes' * s
end

# ── Process noise Q (random-walk propagation) ─────────────────────────────────
# Fraction of the prior variance added per step; tune q_frac to match field dynamics.
q_frac = 0.015
σ²_prior = vec(mean(prior.vars, dims=1))
Q = diagm(Float64.(q_frac .* σ²_prior))

# ── Measurements ──────────────────────────────────────────────────────────────
mvar = diagm(Float32[0.1, 0.1, 0.1])
Y    = get_flighttrack_sequence(truth_field, basis.grid, 2, mvar, tidx_seq; step_frac=0.05)

# ── Filter runs ───────────────────────────────────────────────────────────────
filter_history = runtime_loop_resample(basis, prior, Y, mvar*5, Q)
kf_history     = kf(basis, prior, Y, mvar, Q)

# ── Compare ───────────────────────────────────────────────────────────────────
rmse_gmf, rmse_kf, rmse_ls = compare_estimators(
    filter_history, kf_history, basis, truth_field, tidx_seq, Y)[1:3]

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_rmse_convergence(rmse_gmf, rmse_kf, rmse_ls;
                      pod_floor=pod_rmse,
                      title="Dynamic field tracking  (2 sensors, flight track, q=$(q_frac))",
                      outdir=OUTDIR)
plot_prior(prior, truth_coeffs0; outdir=OUTDIR, label="ensemble")
plot_uncertainty_field(filter_history[end], basis, Y; outdir=OUTDIR)
plot_coeff_tracking(filter_history, kf_history, basis, Y, truth_coeffs_seq; outdir=OUTDIR)
plot_weight_evolution(filter_history; outdir=OUTDIR)
plot_mode_shapes_xy(basis, 10)