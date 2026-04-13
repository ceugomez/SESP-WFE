# cgf cego6160@colorado.edu
# Monte Carlo experiment runner for GMF vs KF vs LS comparison.
# Runs N trials with different random seeds, saves all per-trial data to a
# single JLD2 file for later analysis.
#
# Usage:
#   julia run_montecarlo.jl                    # run fresh experiment
#   julia run_montecarlo.jl --analyze results.jld2   # analyze saved results

const WORKDIR = "/home/cego6160/workspace/prediction/src"
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "GMF.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "plotting.jl"))
include(joinpath(WORKDIR, "comparisons.jl"))
using Random
using Dates

# ══════════════════════════════════════════════════════════════════════════════
# Configuration — edit these to define your experiment
# ══════════════════════════════════════════════════════════════════════════════
const N_TRIALS       = 100
const MAX_ITER       = 50
const NUM_SENSORS    = 2
const MVAR           = diagm(Float32[0.1, 0.1, 0.1])
const BETA           = 0.1
const Q_SCALE        = 0.01        # process noise = Q_SCALE * mean prior variance
const TRUTH_TIDX     = 1           # truth snapshot for L1
const SENSOR_MODE    = :random     # :random or :flighttrack
const STEP_FRAC      = 0.1        # only used if SENSOR_MODE == :flighttrack
const OUTPATH        = "/home/cego6160/workspace/prediction/output/montecarlo_results.jld2"

# ══════════════════════════════════════════════════════════════════════════════
# Analysis mode — load and summarize a saved results file
# ══════════════════════════════════════════════════════════════════════════════
if length(ARGS) >= 2 && ARGS[1] == "--analyze"
    fpath = ARGS[2]
    @info "Analyzing saved results from $fpath"
    d = load(fpath)

    config = d["config"]
    @info "Experiment config:" config

    for level in ["L1", "L2"]
        gmf = d["$(level)_rmse_gmf_final"]
        kf  = d["$(level)_rmse_kf_final"]
        ls  = d["$(level)_rmse_ls_final"]
        n   = length(gmf)

        println("\n── $level ($n trials) ──────────────────────────────")
        for (name, vals) in [("GMF", gmf), ("KF", kf), ("LS", ls)]
            μ = mean(vals)
            σ = std(vals)
            lo, hi = quantile(vals, [0.05, 0.95])
            println("  $name:  mean=$(round(μ, digits=4))  std=$(round(σ, digits=4))  " *
                    "90% CI=[$(round(lo, digits=4)), $(round(hi, digits=4))]")
        end

        # pairwise win rates
        gmf_beats_kf = sum(gmf .< kf) / n
        gmf_beats_ls = sum(gmf .< ls) / n
        kf_beats_ls  = sum(kf .< ls)  / n
        println("  Win rates:  GMF<KF $(round(100*gmf_beats_kf, digits=1))%  " *
                "GMF<LS $(round(100*gmf_beats_ls, digits=1))%  " *
                "KF<LS $(round(100*kf_beats_ls, digits=1))%")
    end
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════════
# Run experiment
# ══════════════════════════════════════════════════════════════════════════════
@info "Loading data..."
basis       = load_pod_basis("/home/cego6160/workspace/prediction/output/pod_basis.jld2")
grid        = basis.grid
encfg       = load_ensemble_config("/home/cego6160/workspace/runs_lad/ensemble_config.json")
truth_field = reconstruct_NCfield(load_member_snapshots(encfg.truth_member), encfg.truth_member, grid)
prior       = init_prior(basis, encfg.n_members, β=BETA)

truth_state  = get_truth_state(truth_field, TRUTH_TIDX)
truth_coeffs = basis.modes' * truth_state
pod_rmse = sqrt(mean((basis.modes * truth_coeffs .- truth_state).^2))
@info "POD floor (L=$(basis.L)): RMSE = $(round(pod_rmse, digits=4)) m/s"

Q = diagm(vec(mean(prior.vars, dims=1)) .* Q_SCALE)

function make_measurements(grid, truth_field, n_sensors, mvar, tidx_seq)
    if SENSOR_MODE == :flighttrack
        return get_flighttrack_sequence(truth_field, grid, n_sensors, mvar, tidx_seq,
                                        step_frac=STEP_FRAC)
    else
        return get_measurement_sequence(truth_field, grid, n_sensors, mvar, tidx_seq)
    end
end

# Preallocate result storage
# Per-trial: full RMSE time series (MAX_ITER) and final scalar
L1_rmse_gmf_all = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_rmse_kf_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_rmse_ls_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_weights_final = Matrix{Float64}(undef, N_TRIALS, prior.K)

L2_rmse_gmf_all = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_rmse_kf_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_rmse_ls_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_weights_final = Matrix{Float64}(undef, N_TRIALS, prior.K)

seeds = collect(1:N_TRIALS)

@info "Running $N_TRIALS trials ($(NUM_SENSORS) sensors, β=$BETA, mode=$SENSOR_MODE)..."
t_start = now()

for trial in 1:N_TRIALS
    Random.seed!(seeds[trial])

    # ── L1: static ───────────────────────────────────────────────────────
    tidx_seq_L1 = fill(TRUTH_TIDX, MAX_ITER)
    Y_L1 = make_measurements(grid, truth_field, NUM_SENSORS, MVAR, tidx_seq_L1)
    fh_L1  = runtime_loop(basis, prior, Y_L1, MVAR)
    kfh_L1 = kf(basis, prior, Y_L1, MVAR)
    ζ_ls_L1 = leastsquares(basis, Y_L1)
    rg, rk, rl = compare_estimators(fh_L1, kfh_L1, ζ_ls_L1, basis, truth_field, tidx_seq_L1)
    L1_rmse_gmf_all[trial, :] = rg
    L1_rmse_kf_all[trial, :]  = rk
    L1_rmse_ls_all[trial, :]  = rl
    L1_weights_final[trial, :] = fh_L1[end].weights

    # ── L2: dynamic ──────────────────────────────────────────────────────
    tidx_seq_L2 = collect(1:MAX_ITER)
    Y_L2 = make_measurements(grid, truth_field, NUM_SENSORS, MVAR, tidx_seq_L2)
    fh_L2  = runtime_loop(basis, prior, Y_L2, MVAR, Q)
    kfh_L2 = kf(basis, prior, Y_L2, MVAR, Q)
    ζ_ls_L2 = leastsquares(basis, Y_L2)
    rg2, rk2, rl2 = compare_estimators(fh_L2, kfh_L2, ζ_ls_L2, basis, truth_field, tidx_seq_L2)
    L2_rmse_gmf_all[trial, :] = rg2
    L2_rmse_kf_all[trial, :]  = rk2
    L2_rmse_ls_all[trial, :]  = rl2
    L2_weights_final[trial, :] = fh_L2[end].weights

    if trial % 10 == 0
        elapsed = Dates.value(now() - t_start) / 1000
        per_trial = elapsed / trial
        remaining = per_trial * (N_TRIALS - trial)
        @info "Trial $trial/$N_TRIALS  ($(round(elapsed, digits=0))s elapsed, ~$(round(remaining, digits=0))s remaining)"
    end
end

elapsed_total = Dates.value(now() - t_start) / 1000
@info "All trials complete in $(round(elapsed_total, digits=1))s"

# ── Save everything ──────────────────────────────────────────────────────────
config = Dict{String,Any}(
    "n_trials"      => N_TRIALS,
    "max_iter"      => MAX_ITER,
    "num_sensors"   => NUM_SENSORS,
    "beta"          => BETA,
    "q_scale"       => Q_SCALE,
    "truth_tidx"    => TRUTH_TIDX,
    "sensor_mode"   => string(SENSOR_MODE),
    "step_frac"     => STEP_FRAC,
    "pod_L"         => basis.L,
    "pod_rmse"      => pod_rmse,
    "n_members"     => encfg.n_members,
    "timestamp"     => string(now()),
)

jldsave(OUTPATH;
    config = config,
    seeds  = seeds,
    L1_rmse_gmf_all    = L1_rmse_gmf_all,
    L1_rmse_kf_all     = L1_rmse_kf_all,
    L1_rmse_ls_all     = L1_rmse_ls_all,
    L1_rmse_gmf_final  = L1_rmse_gmf_all[:, end],
    L1_rmse_kf_final   = L1_rmse_kf_all[:, end],
    L1_rmse_ls_final   = L1_rmse_ls_all[:, end],
    L1_weights_final   = L1_weights_final,
    L2_rmse_gmf_all    = L2_rmse_gmf_all,
    L2_rmse_kf_all     = L2_rmse_kf_all,
    L2_rmse_ls_all     = L2_rmse_ls_all,
    L2_rmse_gmf_final  = L2_rmse_gmf_all[:, end],
    L2_rmse_kf_final   = L2_rmse_kf_all[:, end],
    L2_rmse_ls_final   = L2_rmse_ls_all[:, end],
    L2_weights_final   = L2_weights_final,
)
@info "Results saved to $OUTPATH"

# ── Print summary ────────────────────────────────────────────────────────────
# Re-use the analysis logic inline
for (level, gmf, kf_r, ls) in [
    ("L1", L1_rmse_gmf_all[:, end], L1_rmse_kf_all[:, end], L1_rmse_ls_all[:, end]),
    ("L2", L2_rmse_gmf_all[:, end], L2_rmse_kf_all[:, end], L2_rmse_ls_all[:, end]),
]
    n = length(gmf)
    println("\n── $level ($n trials) ──────────────────────────────")
    for (name, vals) in [("GMF", gmf), ("KF", kf_r), ("LS", ls)]
        μ = mean(vals)
        σ = std(vals)
        lo, hi = quantile(vals, [0.05, 0.95])
        println("  $name:  mean=$(round(μ, digits=4))  std=$(round(σ, digits=4))  " *
                "90% CI=[$(round(lo, digits=4)), $(round(hi, digits=4))]")
    end
    gmf_beats_kf = sum(gmf .< kf_r) / n
    gmf_beats_ls = sum(gmf .< ls) / n
    kf_beats_ls  = sum(kf_r .< ls) / n
    println("  Win rates:  GMF<KF $(round(100*gmf_beats_kf, digits=1))%  " *
            "GMF<LS $(round(100*gmf_beats_ls, digits=1))%  " *
            "KF<LS $(round(100*kf_beats_ls, digits=1))%")
end
