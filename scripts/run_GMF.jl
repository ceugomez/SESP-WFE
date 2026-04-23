# cgf cego6160@colorado.edu
const WORKDIR = "/home/cego6160/workspace/prediction/src"
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "GMF.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "plotting.jl"))
include(joinpath(WORKDIR, "comparisons.jl"))

# ── Data ──────────────────────────────────────────────────────────────────────
basis       = load_pod_basis("/home/cego6160/workspace/prediction/output/pod_basis.jld2")
grid        = basis.grid
encfg       = load_ensemble_config("/home/cego6160/workspace/runs_lad/ensemble_config.json")
truth_field = reconstruct_NCfield(load_member_snapshots(encfg.truth_member), encfg.truth_member, grid)
#β = 0.1         # variance inflation to apply to prior coefficients
prior       = init_prior(basis, encfg.n_members)

# ── POD reconstruction floor ──────────────────────────────────────────────────
# Irreducible error from basis truncation — filter cannot beat this
truth_tidx = 2
truth_state  = get_truth_state(truth_field, truth_tidx)
truth_coeffs = basis.modes' * truth_state
pod_rmse = sqrt(mean((basis.modes * truth_coeffs .- truth_state).^2))
@info "POD floor (L=$(basis.L)): RMSE = $(round(pod_rmse, digits=4)) m/s"

# ── Level 1: static field estimation ─────────────────────────────────────────\
@info "starting static field estimation"
max_iter = 50                               # run 50 timesteps
num_measurements = 2                        # 1 sensors in the environment 
mvar = diagm(Float32[0.1, 0.1, 0.1])     # sensor noise characteristics
tidx_seq = fill(truth_tidx, max_iter)       # static field — same snapshot every iter
Y =  get_flighttrack_sequence(truth_field, grid, 2, mvar, tidx_seq, step_frac=0.05)   #get_measurement_sequence(truth_field, grid, num_measurements, mvar, tidx_seq)  
filter_history = runtime_loop(basis, prior, Y, mvar)
kf_history = kf(basis, prior, Y, mvar)
rmse_gmf, rmse_kf, rmse_ls = compare_estimators(filter_history, kf_history, basis, truth_field, tidx_seq, Y)[1:3]


#  ── Level 2: Dynamic field estimation ─────────────────────────────────────────
@info "starting dynamic field estimation"
max_iter_dyn    = 50
num_meas_dyn    = 2
tidx_seq_dyn    = collect(1:max_iter_dyn)
Y_dyn           =  get_flighttrack_sequence(truth_field, grid, 2, mvar, tidx_seq, step_frac=0.1)   #get_measurement_sequence(truth_field, grid, num_meas_dyn, mvar, tidx_seq_dyn)
Q               = diagm(vec(mean(prior.vars, dims=1)) .* 0.01)
filter_hist_dyn = runtime_loop(basis, prior, Y_dyn, mvar, Q)
kf_hist_dyn     = kf(basis, prior, Y_dyn, mvar, Q)
rmse_gmf_dyn, rmse_kf_dyn, rmse_ls_dyn = compare_estimators(filter_hist_dyn, kf_hist_dyn, basis, truth_field, tidx_seq_dyn, Y_dyn)[1:3]

# ── Visualization & evaluation ────────────────────────────────────────────────
@info "plotting"
for i in 1:10
    plot_gaussian_mixture(prior, i, truth_val=truth_coeffs[i])
end
plot_gaussian_mixture_ridgeline.(Ref(filter_history), 1:10, Ref(truth_coeffs))
plot_field_reconstruction(filter_history[end], basis, truth_field, truth_tidx)

# ── Level 1 animation ────────────────────────────────────────────────────────
@info "saving L1 filter history for animation"
save_filter_history(filter_history, kf_history, rmse_ls,
                    basis, truth_field, tidx_seq,
                    outpath="/home/cego6160/workspace/prediction/output/filter_history_L1.jld2",
                    sensor_locs=Y)
run(`conda run -n fasteddy python /home/cego6160/workspace/prediction/preprocessing/animate_filter.py
     --history /home/cego6160/workspace/prediction/output/filter_history_L1.jld2
     --outdir  /home/cego6160/workspace/prediction/output/L1`)

# ── Level 2 animation ────────────────────────────────────────────────────────
@info "saving L2 filter history for animation"
save_filter_history(filter_hist_dyn, kf_hist_dyn, rmse_ls_dyn,
                    basis, truth_field, tidx_seq_dyn,
                    outpath="/home/cego6160/workspace/prediction/output/filter_history_L2.jld2",
                    sensor_locs=Y_dyn)
run(`conda run -n fasteddy python /home/cego6160/workspace/prediction/preprocessing/animate_filter.py
     --history /home/cego6160/workspace/prediction/output/filter_history_L2.jld2
     --outdir  /home/cego6160/workspace/prediction/output/L2`)
