# cgf cego6160@colorado.edu
# Monte Carlo experiment runner for GMF vs KF vs LS comparison.
# Runs N trials with different random seeds, saves all per-trial data to a
# single JLD2 file for later analysis.
#
# Usage:
#   # Run with defaults
#   julia run_montecarlo.jl
#
#   # Override parameters via CLI
#   julia run_montecarlo.jl --n_sensors 5 --sensor_mode flighttrack --q_scale 0.05
#
#   # Analyze saved results
#   julia run_montecarlo.jl --analyze path/to/results.jld2
#
#   # Analyze all results in a directory
#   julia run_montecarlo.jl --analyze-all path/to/sweep_dir/

const WORKDIR = "/home/cego6160/workspace/prediction/src"
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "GMF.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "plotting.jl"))
include(joinpath(WORKDIR, "comparisons.jl"))
using Random
using Dates

# ══════════════════════════════════════════════════════════════════════════════
# Parse CLI arguments
# ══════════════════════════════════════════════════════════════════════════════
function parse_arg(args, key, default)
    idx = findfirst(==(key), args)
    isnothing(idx) && return default
    return args[idx + 1]
end

# ── Analysis modes ───────────────────────────────────────────────────────────
if length(ARGS) >= 2 && ARGS[1] == "--analyze"
    fpath = ARGS[2]
    @info "Analyzing saved results from $fpath"
    d = load(fpath)
    config = d["config"]
    @info "Experiment config:" config

    for level in ["L1", "L2"]
        gmf  = d["$(level)_rmse_gmf_final"]
        kf   = d["$(level)_rmse_kf_final"]
        ls   = d["$(level)_rmse_ls_final"]
        gmfr = haskey(d, "$(level)_rmse_gmfr_final") ? d["$(level)_rmse_gmfr_final"] : nothing
        n    = length(gmf)

        println("\n── $level ($n trials) ──────────────────────────────")
        estimators = [("GMF", gmf), ("KF", kf), ("LS", ls)]
        if !isnothing(gmfr)
            insert!(estimators, 2, ("GMFR", gmfr))
        end
        for (name, vals) in estimators
            μ = mean(vals); σ = std(vals)
            lo, hi = quantile(vals, [0.05, 0.95])
            println("  $name:  mean=$(round(μ, digits=4))  std=$(round(σ, digits=4))  " *
                    "90% CI=[$(round(lo, digits=4)), $(round(hi, digits=4))]")
        end
        gmf_beats_kf = sum(gmf .< kf) / n
        if !isnothing(gmfr)
            gmfr_beats_gmf = sum(gmfr .< gmf) / n
            gmfr_beats_kf  = sum(gmfr .< kf) / n
            println("  Win rates:  GMFR<GMF $(round(100*gmfr_beats_gmf, digits=1))%  " *
                    "GMFR<KF $(round(100*gmfr_beats_kf, digits=1))%  " *
                    "GMF<KF $(round(100*gmf_beats_kf, digits=1))%")
        else
            gmf_beats_ls = sum(gmf .< ls) / n
            kf_beats_ls  = sum(kf .< ls)  / n
            println("  Win rates:  GMF<KF $(round(100*gmf_beats_kf, digits=1))%  " *
                    "GMF<LS $(round(100*gmf_beats_ls, digits=1))%  " *
                    "KF<LS $(round(100*kf_beats_ls, digits=1))%")
        end
    end
    exit(0)
end

if length(ARGS) >= 2 && ARGS[1] == "--analyze-all"
    dir = ARGS[2]
    files = filter(f -> endswith(f, ".jld2"), readdir(dir, join=true))
    sort!(files)
    @info "Found $(length(files)) result files in $dir"

    # Print a compact comparison table
    # Check if any file has GMFR data
    has_gmfr = any(haskey(load(f), "L1_rmse_gmfr_final") for f in files)

    if has_gmfr
        println("\n" * "="^140)
        println(rpad("Config", 40) *
                "  ──── L1 Final RMSE ──────────   ──── L2 Final RMSE ──────────")
        println(rpad("", 40) *
                "  GMF     GMFR    KF      LS      " *
                "  GMF     GMFR    KF      LS")
        println("─"^140)
    else
        println("\n" * "="^110)
        println(rpad("Config", 40) *
                "  ──── L1 Final RMSE ────   ──── L2 Final RMSE ────")
        println(rpad("", 40) *
                "  GMF     KF      LS      " *
                "  GMF     KF      LS")
        println("─"^110)
    end

    for fpath in files
        d = load(fpath)
        c = d["config"]
        label = "$(c["sensor_mode"]) n=$(c["num_sensors"]) q=$(c["q_scale"])"
        l1g = round(mean(d["L1_rmse_gmf_final"]), digits=4)
        l1k = round(mean(d["L1_rmse_kf_final"]),  digits=4)
        l1l = round(mean(d["L1_rmse_ls_final"]),  digits=4)
        l2g = round(mean(d["L2_rmse_gmf_final"]), digits=4)
        l2k = round(mean(d["L2_rmse_kf_final"]),  digits=4)
        l2l = round(mean(d["L2_rmse_ls_final"]),  digits=4)
        if has_gmfr && haskey(d, "L1_rmse_gmfr_final")
            l1r = round(mean(d["L1_rmse_gmfr_final"]), digits=4)
            l2r = round(mean(d["L2_rmse_gmfr_final"]), digits=4)
            println(rpad(label, 40) *
                    "  $(lpad(l1g, 6))  $(lpad(l1r, 6))  $(lpad(l1k, 6))  $(lpad(l1l, 6))" *
                    "  $(lpad(l2g, 6))  $(lpad(l2r, 6))  $(lpad(l2k, 6))  $(lpad(l2l, 6))")
        else
            println(rpad(label, 40) *
                    "  $(lpad(l1g, 6))  $(lpad(l1k, 6))  $(lpad(l1l, 6))" *
                    "  $(lpad(l2g, 6))  $(lpad(l2k, 6))  $(lpad(l2l, 6))")
        end
    end
    println("="^(has_gmfr ? 140 : 110))
    exit(0)
end

# ══════════════════════════════════════════════════════════════════════════════
# Configuration — defaults, overridable via CLI
# ══════════════════════════════════════════════════════════════════════════════
N_TRIALS     = parse(Int,     parse_arg(ARGS, "--n_trials",    "100"))
MAX_ITER     = parse(Int,     parse_arg(ARGS, "--max_iter",    "50"))
NUM_SENSORS  = parse(Int,     parse_arg(ARGS, "--n_sensors",   "2"))
BETA         = parse(Float64, parse_arg(ARGS, "--beta",        "0.1"))
Q_SCALE      = parse(Float64, parse_arg(ARGS, "--q_scale",     "0.01"))
TRUTH_TIDX   = parse(Int,     parse_arg(ARGS, "--truth_tidx",  "1"))
SENSOR_MODE  = Symbol(        parse_arg(ARGS, "--sensor_mode", "random"))
STEP_FRAC    = parse(Float64, parse_arg(ARGS, "--step_frac",   "0.1"))
PMAX_FRAC    = parse(Float64, parse_arg(ARGS, "--pmax_frac",  "0.0"))
K_MAX        = parse(Int,     parse_arg(ARGS, "--k_max",      "10"))
TAU          = parse(Float64, parse_arg(ARGS, "--tau",        "0.1"))
OUTPATH      =                parse_arg(ARGS, "--outpath",
    "/home/cego6160/workspace/prediction/output/mc_$(SENSOR_MODE)_n$(NUM_SENSORS)_q$(Q_SCALE).jld2")

MVAR = diagm(Float32[0.1, 0.1, 0.1])

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
L1_rmse_gmf_all   = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_rmse_kf_all    = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_rmse_ls_all    = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_rmse_gmfr_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L1_weights_final  = Matrix{Float64}(undef, N_TRIALS, prior.K)

L2_rmse_gmf_all   = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_rmse_kf_all    = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_rmse_ls_all    = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_rmse_gmfr_all  = Matrix{Float64}(undef, N_TRIALS, MAX_ITER)
L2_weights_final  = Matrix{Float64}(undef, N_TRIALS, prior.K)

seeds = collect(1:N_TRIALS)

@info "Running $N_TRIALS trials (n_sensors=$NUM_SENSORS, β=$BETA, q=$Q_SCALE, mode=$SENSOR_MODE)"
@info "Output: $OUTPATH"
t_start = now()

for trial in 1:N_TRIALS
    Random.seed!(seeds[trial])

    # ── L1: static ───────────────────────────────────────────────────────
    tidx_seq_L1 = fill(TRUTH_TIDX, MAX_ITER)
    Y_L1 = make_measurements(grid, truth_field, NUM_SENSORS, MVAR, tidx_seq_L1)
    fh_L1    = runtime_loop(basis, prior, Y_L1, MVAR)
    fhr_L1   = runtime_loop_resample(basis, prior, Y_L1, MVAR;
                                     τ=TAU, P_max_frac=PMAX_FRAC, K_max=K_MAX)
    kfh_L1   = kf(basis, prior, Y_L1, MVAR)
    ζ_ls_L1  = leastsquares(basis, Y_L1)
    rg, rk, rl, rr = compare_estimators(fh_L1, kfh_L1, ζ_ls_L1, basis, truth_field,
                                         tidx_seq_L1; gmfr_history=fhr_L1)
    L1_rmse_gmf_all[trial, :]  = rg
    L1_rmse_kf_all[trial, :]   = rk
    L1_rmse_ls_all[trial, :]   = rl
    L1_rmse_gmfr_all[trial, :] = rr
    L1_weights_final[trial, :] = fh_L1[end].weights

    # ── L2: dynamic ──────────────────────────────────────────────────────
    tidx_seq_L2 = collect(1:MAX_ITER)
    Y_L2 = make_measurements(grid, truth_field, NUM_SENSORS, MVAR, tidx_seq_L2)
    fh_L2    = runtime_loop(basis, prior, Y_L2, MVAR, Q)
    fhr_L2   = runtime_loop_resample(basis, prior, Y_L2, MVAR, Q;
                                     τ=TAU, P_max_frac=PMAX_FRAC, K_max=K_MAX)
    kfh_L2   = kf(basis, prior, Y_L2, MVAR, Q)
    ζ_ls_L2  = leastsquares(basis, Y_L2)
    rg2, rk2, rl2, rr2 = compare_estimators(fh_L2, kfh_L2, ζ_ls_L2, basis, truth_field,
                                              tidx_seq_L2; gmfr_history=fhr_L2)
    L2_rmse_gmf_all[trial, :]  = rg2
    L2_rmse_kf_all[trial, :]   = rk2
    L2_rmse_ls_all[trial, :]   = rl2
    L2_rmse_gmfr_all[trial, :] = rr2
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
    "pmax_frac"     => PMAX_FRAC,
    "k_max"         => K_MAX,
    "tau"           => TAU,
    "pod_L"         => basis.L,
    "pod_rmse"      => pod_rmse,
    "n_members"     => encfg.n_members,
    "timestamp"     => string(now()),
)

jldsave(OUTPATH;
    config = config,
    seeds  = seeds,
    L1_rmse_gmf_all     = L1_rmse_gmf_all,
    L1_rmse_kf_all      = L1_rmse_kf_all,
    L1_rmse_ls_all      = L1_rmse_ls_all,
    L1_rmse_gmfr_all    = L1_rmse_gmfr_all,
    L1_rmse_gmf_final   = L1_rmse_gmf_all[:, end],
    L1_rmse_kf_final    = L1_rmse_kf_all[:, end],
    L1_rmse_ls_final    = L1_rmse_ls_all[:, end],
    L1_rmse_gmfr_final  = L1_rmse_gmfr_all[:, end],
    L1_weights_final    = L1_weights_final,
    L2_rmse_gmf_all     = L2_rmse_gmf_all,
    L2_rmse_kf_all      = L2_rmse_kf_all,
    L2_rmse_ls_all      = L2_rmse_ls_all,
    L2_rmse_gmfr_all    = L2_rmse_gmfr_all,
    L2_rmse_gmf_final   = L2_rmse_gmf_all[:, end],
    L2_rmse_kf_final    = L2_rmse_kf_all[:, end],
    L2_rmse_ls_final    = L2_rmse_ls_all[:, end],
    L2_rmse_gmfr_final  = L2_rmse_gmfr_all[:, end],
    L2_weights_final    = L2_weights_final,
)
@info "Results saved to $OUTPATH"

# ── Print summary ────────────────────────────────────────────────────────────
for (level, gmf, kf_r, ls, gmfr) in [
    ("L1", L1_rmse_gmf_all[:, end], L1_rmse_kf_all[:, end], L1_rmse_ls_all[:, end], L1_rmse_gmfr_all[:, end]),
    ("L2", L2_rmse_gmf_all[:, end], L2_rmse_kf_all[:, end], L2_rmse_ls_all[:, end], L2_rmse_gmfr_all[:, end]),
]
    n = length(gmf)
    println("\n── $level ($n trials) ──────────────────────────────")
    for (name, vals) in [("GMF", gmf), ("GMFR", gmfr), ("KF", kf_r), ("LS", ls)]
        μ = mean(vals); σ = std(vals)
        lo, hi = quantile(vals, [0.05, 0.95])
        println("  $name:  mean=$(round(μ, digits=4))  std=$(round(σ, digits=4))  " *
                "90% CI=[$(round(lo, digits=4)), $(round(hi, digits=4))]")
    end
    gmfr_beats_gmf = sum(gmfr .< gmf) / n
    gmfr_beats_kf  = sum(gmfr .< kf_r) / n
    gmf_beats_kf   = sum(gmf .< kf_r) / n
    println("  Win rates:  GMFR<GMF $(round(100*gmfr_beats_gmf, digits=1))%  " *
            "GMFR<KF $(round(100*gmfr_beats_kf, digits=1))%  " *
            "GMF<KF $(round(100*gmf_beats_kf, digits=1))%")
end
