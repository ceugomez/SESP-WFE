# run_theta_baseline_dense.jl
# cgf cego6160@colorado.edu 7.20.26
#
# THE Θ-REGRESSION BASELINE (design doc R1), scored against the SAME dense
# sensor lattice as run_kf_rank_sweep_dense.jl instead of the 15-sensor tower
# rake in run_theta_baseline.jl. run_fold_sweep_dense.sh flagged this gap: the
# existing theta_baseline_truth_<ID>.csv files are a 45-obs-row reference and
# are NOT a matched comparison for the 525-obs-row dense KF sweep. This script
# closes that gap so R1 (does the KF beat the Θ-baseline?) can be
# answered at the sensor density where under-determination is no longer a
# confound.
#
# Conditional-mean map and per-step scoring are UNCHANGED from
# run_theta_baseline.jl (same members, same regression, full-domain RMSE). The
# GLS weighting for Θ̂ uses R_base = R_sensor + R_rep (sensor + stored
# representativeness covariance), matching the R the dense KF sweep uses —
# see check_R_pd in src/experiment.jl before changing the sensor spacing.
#
# The kf_* CSV columns are always NaN here (kept so the schema matches the rake file):
# score the dense KF from kf_rank_sweep_dense_truth_<ID>.csv instead. Pairing against the
# rake-sensor kf_result_truth_<ID>.jld2 would silently compare mismatched sensors and R.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, sensors, seed

const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
# Sensor lattice spacing, in GRID CELLS. Must match run_kf_rank_sweep_dense.jl for the
# comparison to be paired.
const SENSOR_STRIDE_XY = parse(Int, get(ENV, "SENSOR_STRIDE_XY", "50"))
const SENSOR_STRIDE_Z  = parse(Int, get(ENV, "SENSOR_STRIDE_Z",  "10"))

Random.seed!(NOISE_SEED)   # same seed in every driver -> paired noise

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = resolve_member(config, TRUTH_ID)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
rep    = load_rep_error(rep_error_filename(BASES_DIR, TRUTH_ID))

# --- Conditional-mean map from member time-means (same members as the basis) ---
members = filter(m -> m.id in basis.member_ids, config.prior_members)
M = length(members)
M >= 3 || error("need ≥3 members to fit [1,Ug,Vg] regression, got $M")
Xbar = Matrix{Float32}(undef, domain.state_dim, M)
D    = Matrix{Float64}(undef, M, 3)
for (i, m) in enumerate(members)
    @info "  time-mean of member $(m.id) ($i/$M)..."
    Xi = load_member_snapshots(m; t_spinup=T_SPINUP, domain)
    Xbar[:, i] = vec(mean(Xi; dims=2))
    D[i, :]    = [1.0, m.U_g_ms, m.V_g_ms]
    Xi = nothing; GC.gc()
end
# least squares across members, pointwise: B = X̄ D (DᵀD)⁻¹   (state_dim × 3)
B = Xbar * Float32.(D * inv(D' * D))
Xbar = nothing; GC.gc()

# --- Truth field ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = sensor_mvar()
TIDX = collect(1:length(truth_t))

# --- Dense lattice: identical to run_kf_rank_sweep_dense.jl ---
n    = domain.n_grid
lin  = _masked_linear_index(domain, grid)
locs = dense_lattice_locations(grid, domain; stride_xy=SENSOR_STRIDE_XY, stride_z=SENSOR_STRIDE_Z)
NM   = length(locs)
@info "Sensor array: NM=$NM sensors ($(3NM) obs rows); xy stride $(SENSOR_STRIDE_XY) cells " *
      "($(SENSOR_STRIDE_XY * grid.dx) m), z stride $(SENSOR_STRIDE_Z) levels"

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)

# --- Sample the map at the sensors (static array -> build once) ---
Ha  = Vector{Float32}(undef, 3NM)        # intercept response at sensors
Hb  = Matrix{Float32}(undef, 3NM, 2)     # [∂/∂Ug  ∂/∂Vg] at sensors
for (k, m) in enumerate(meas_seq[1].measurements)
    p = lin[m.locidx[1], m.locidx[2], m.locidx[3]]
    p == 0 && error("sensor $k lies outside the analysis domain")
    for (r, q) in enumerate((p, n + p, 2n + p))
        Ha[3k-3+r]    = B[q, 1]
        Hb[3k-3+r, :] = B[q, 2:3]
    end
end

# R_base = sensor + rep-error covariance (matches the dense KF sweep's weighting).
R_base = Float64.(build_measurement_covariance(NM, MVAR) +
                  build_rep_covariance(meas_seq[1], rep, domain))
check_R_pd(R_base; context="Sensor spacing $(SENSOR_STRIDE_XY * grid.dx) m.")
Rinv = inv(Symmetric(R_base))
A    = Symmetric(Hb' * Rinv * Hb)        # 2×2 GLS normal matrix

truth_coeffs = basis.modes' * truth_U

# --- Per-step scoring ---
θ_true = Float32[truth_member.U_g_ms, truth_member.V_g_ms]
xstar  = B * Float32[1f0, θ_true...]     # Θ-oracle field (constant in time)
K = length(TIDX)
Ug_hat = zeros(Float32, K); Vg_hat = zeros(Float32, K)
res = Dict(s => zeros(Float64, K) for s in
    (:theta_full, :thor_full, :kf_full, :orc_full, :const_full))
xconst = Vector{Float32}(undef, 3n)
for k in 1:K
    xt = @view truth_U[:, k]
    y  = stack_measurements(meas_seq[k])
    θ̂  = A \ (Hb' * (Rinv * (y - Ha)))
    Ug_hat[k], Vg_hat[k] = θ̂
    xhat = B * Float32[1f0, θ̂...]
    for c in 1:3   # obs stacked [u,v,w] per sensor; state stacked u-, v-, w-block
        xconst[(c-1)*n+1 : c*n] .= mean(@view y[c:3:end])
    end
    xorc = basis.modes * truth_coeffs[:, k]
    res[:theta_full][k] = sqrt(mean(abs2, xhat .- xt))
    res[:thor_full][k]  = sqrt(mean(abs2, xstar .- xt))
    res[:orc_full][k]   = sqrt(mean(abs2, xorc .- xt))
    res[:const_full][k] = sqrt(mean(abs2, xconst .- xt))
    res[:kf_full][k] = NaN   # see header: the dense KF is scored by the rank sweep
end

csvpath = joinpath(KF_DIR, "theta_baseline_dense_truth_$(TRUTH_ID).csv")
open(csvpath, "w") do io
    println(io, "step,time,Ug_hat,Vg_hat,theta_rmse_full,thetaoracle_rmse_full," *
                "kf_rmse_full,oracle_rmse_full,const_rmse_full")
    for k in 1:K
        println(io, "$k,$(truth_t[k]),$(Ug_hat[k]),$(Vg_hat[k])," *
            "$(res[:theta_full][k]),$(res[:thor_full][k])," *
            "$(res[:kf_full][k]),$(res[:orc_full][k]),$(res[:const_full][k])")
    end
end

m(s) = round(mean(res[s]), digits=4)
@info "Θ truth = ($(θ_true[1]), $(θ_true[2]));  Θ̂ mean = " *
      "($(round(mean(Ug_hat),digits=3)), $(round(mean(Vg_hat),digits=3)))"
@info "FULL domain RMSE  —  Θ-baseline: $(m(:theta_full))  Θ-oracle: $(m(:thor_full))  " *
      "KF: $(m(:kf_full))  subspace oracle: $(m(:orc_full))  const-wind: $(m(:const_full))"
@info "R1 verdict: compare against kf_rank_sweep_dense_truth_$(TRUTH_ID).csv"
@info "Saved $csvpath"
