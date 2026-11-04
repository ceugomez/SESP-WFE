# run_theta_baseline_dense.jl
# cgf cego6160@colorado.edu 7.20.26
#
# THE Θ-REGRESSION BASELINE (design doc R1), scored against the SAME dense
# sensor lattice as run_kf_rank_sweep_dense.jl instead of the 15-sensor tower
# rake in run_theta_baseline.jl. run_fold_sweep_dense.sh flagged this gap: the
# existing theta_baseline_truth_<ID>.csv files are a 45-obs-row reference and
# are NOT a matched comparison for the 525-obs-row dense KF sweep. This script
# closes that gap so R1 (does the KF beat the Θ-baseline in the wake?) can be
# answered at the sensor density where under-determination is no longer a
# confound.
#
# Conditional-mean map, wake-region mask, and per-step scoring are UNCHANGED
# from run_theta_baseline.jl (same members, same regression, same box). The
# GLS weighting for Θ̂ uses R_base = R_sensor + R_rep (sensor + stored
# representativeness covariance), matching the R the dense KF sweep factors
# into H — see the PD note in run_kf_rank_sweep_dense.jl before changing the
# sensor spacing.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, SRC_DIR, ...
const WORKDIR  = SRC_DIR
const T_SPINUP = 900.0   # must match the basis + KF run
const MARGIN_M = 600f0   # must match the basis + KF run
const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const RTOT_MIN_EIG = 0.05   # PD guard: require min eig(R_base) above this (same floor as the dense rank sweep)
# Sensor lattice spacing, in GRID CELLS (dx = dy = 10 m). Must match
# run_kf_rank_sweep_dense.jl for the comparison to be paired.
const SENSOR_STRIDE_XY = parse(Int, get(ENV, "SENSOR_STRIDE_XY", "50"))
const SENSOR_STRIDE_Z  = parse(Int, get(ENV, "SENSOR_STRIDE_Z",  "10"))
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

Random.seed!(20260713)   # SAME seed as run_theta_baseline.jl / run_kf_rank_sweep_dense.jl -> paired noise

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
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

MVAR = Matrix{Float32}(0.25f0 * I, 3, 3)
TIDX = collect(1:length(truth_t))

crest = argmax(grid.zPos[:, :, 1])
crest_x, crest_y = grid.xPos[crest[1], crest[2], 1], grid.yPos[crest[1], crest[2], 1]

# --- Wake-region mask (box downwind of the crest, FULL analysis y-width) ---
# Identical to run_theta_baseline.jl / run_kf_rank_sweep_dense.jl.
WAKE_DX_M, WAKE_AGL_M = 2500f0, 250f0
n   = domain.n_grid
lin = _masked_linear_index(domain, grid)
terrain3 = repeat(grid.zPos[:, :, 1:1], 1, 1, grid.Nz)
wake3 = (grid.xPos .>= crest_x) .& (grid.xPos .<= crest_x + WAKE_DX_M) .&
        ((grid.zPos .- terrain3) .<= WAKE_AGL_M) .& domain.mask
ps   = lin[wake3]
widx = vcat(ps, ps .+ n, ps .+ 2n)

# --- DENSE sensor array: same lattice as run_kf_rank_sweep_dense.jl, spanning
#     the downwind half of the analysis domain (crest to interior x edge, full
#     interior y), every SENSOR_STRIDE_XY cells in x,y and every
#     SENSOR_STRIDE_Z-th level below WAKE_AGL_M. ---
ix_dom_lo = findfirst(i -> any(domain.mask[i, :, :]), 1:grid.Nx)
ix_hi     = findlast( i -> any(domain.mask[i, :, :]), 1:grid.Nx)
iy_lo     = findfirst(j -> any(domain.mask[:, j, :]), 1:grid.Ny)
iy_hi     = findlast( j -> any(domain.mask[:, j, :]), 1:grid.Ny)
ix_lo     = max(ix_dom_lo, findfirst(i -> grid.xPos[i, 1, 1] >= crest_x, 1:grid.Nx))
locs = Vector{Vector{Float32}}()
for i in ix_lo:SENSOR_STRIDE_XY:ix_hi, j in iy_lo:SENSOR_STRIDE_XY:iy_hi
    terr = grid.zPos[i, j, 1]
    for k in 2:SENSOR_STRIDE_Z:grid.Nz
        (grid.zPos[i, j, k] - terr) <= WAKE_AGL_M || break
        domain.mask[i, j, k] || continue
        push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
    end
end
NM = length(locs)
NM > 0 || error("sensor array is empty — check the downwind span against the domain margin")
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
# PD guard identical to run_kf_rank_sweep_dense.jl: at this spacing R_rep's
# negative eigenvalues must sit below the sensor-noise floor. See the header
# note in run_kf_rank_sweep_dense.jl before tightening SENSOR_STRIDE_XY.
R_base = Float64.(build_measurement_covariance(NM, MVAR) +
                  build_rep_covariance(meas_seq[1], rep, domain))
let e = eigvals(Symmetric(R_base))
    λmin = minimum(e)
    @info "min eig(R_sensor + R_rep) = $(round(λmin, digits=4))  cond = $(round(maximum(e)/max(λmin,1e-12), digits=1))"
    λmin >= RTOT_MIN_EIG || error(
        "R_base min eigenvalue $(round(λmin, digits=4)) < $(RTOT_MIN_EIG): the rep-error " *
        "kernel is not PSD and at $(SENSOR_STRIDE_XY * grid.dx) m spacing its negative " *
        "eigenvalues are no longer masked by the $(MVAR[1,1]) sensor-noise floor. " *
        "Increase SENSOR_STRIDE_XY (500 m / stride 50 is the calibrated default). " *
        "See run_kf_rank_sweep_dense.jl.")
end
Rinv = inv(Symmetric(R_base))
A    = Symmetric(Hb' * Rinv * Hb)        # 2×2 GLS normal matrix

# --- No saved single-rank KF run to pair against here: kf_result_filename()
# is written by run_simple_KF_assimilation.jl (the RAKE-sensor fixed-rank
# run), and pairing it against this dense-array baseline would silently
# compare mismatched sensor configs/R. The kf_* columns below are always NaN;
# score the dense KF instead against kf_rank_sweep_dense_truth_<ID>.csv. ---
have_kf = false
global truth_coeffs = basis.modes' * truth_U
Prw = basis.modes[widx, :]

# --- Per-step scoring ---
θ_true = Float32[truth_member.U_g_ms, truth_member.V_g_ms]
xstar  = B * Float32[1f0, θ_true...]     # Θ-oracle field (constant in time)
K = length(TIDX)
Ug_hat = zeros(Float32, K); Vg_hat = zeros(Float32, K)
res = Dict(s => zeros(Float64, K) for s in
    (:theta_full, :theta_wake, :thor_full, :thor_wake,
     :kf_full, :kf_wake, :orc_full, :orc_wake,
     :const_full, :const_wake))
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
    xtw = @view xt[widx]
    res[:theta_wake][k] = sqrt(mean(abs2, xhat[widx]  .- xtw))
    res[:thor_wake][k]  = sqrt(mean(abs2, xstar[widx] .- xtw))
    res[:const_wake][k] = sqrt(mean(abs2, xconst[widx] .- xtw))
    res[:orc_wake][k]   = sqrt(mean(abs2, Prw * truth_coeffs[:, k] .- xtw))
    if have_kf
        xkf = basis.modes * α[:, k]
        res[:kf_full][k] = sqrt(mean(abs2, xkf .- xt))
        res[:kf_wake][k] = sqrt(mean(abs2, Prw * α[:, k] .- xtw))
    else
        res[:kf_full][k] = NaN
        res[:kf_wake][k] = NaN
    end
end

csvpath = joinpath(KF_DIR, "theta_baseline_dense_truth_$(TRUTH_ID).csv")
open(csvpath, "w") do io
    println(io, "step,time,Ug_hat,Vg_hat,theta_rmse_full,theta_rmse_wake," *
                "thetaoracle_rmse_full,thetaoracle_rmse_wake," *
                "kf_rmse_full,kf_rmse_wake,oracle_rmse_full,oracle_rmse_wake," *
                "const_rmse_full,const_rmse_wake")
    for k in 1:K
        println(io, "$k,$(truth_t[k]),$(Ug_hat[k]),$(Vg_hat[k])," *
            "$(res[:theta_full][k]),$(res[:theta_wake][k])," *
            "$(res[:thor_full][k]),$(res[:thor_wake][k])," *
            "$(res[:kf_full][k]),$(res[:kf_wake][k])," *
            "$(res[:orc_full][k]),$(res[:orc_wake][k])," *
            "$(res[:const_full][k]),$(res[:const_wake][k])")
    end
end

m(s) = round(mean(res[s]), digits=4)
@info "Θ truth = ($(θ_true[1]), $(θ_true[2]));  Θ̂ mean = " *
      "($(round(mean(Ug_hat),digits=3)), $(round(mean(Vg_hat),digits=3)))"
@info "FULL domain RMSE  —  Θ-baseline: $(m(:theta_full))  Θ-oracle: $(m(:thor_full))  " *
      "KF: $(m(:kf_full))  subspace oracle: $(m(:orc_full))  const-wind: $(m(:const_full))"
@info "WAKE region RMSE  —  Θ-baseline: $(m(:theta_wake))  Θ-oracle: $(m(:thor_wake))  " *
      "KF: $(m(:kf_wake))  subspace oracle: $(m(:orc_wake))  const-wind: $(m(:const_wake))"
if have_kf
    verdict = m(:kf_wake) < m(:theta_wake) ? "KF BEATS the Θ-baseline in the wake" :
                                             "KF DOES NOT beat the Θ-baseline in the wake"
    @info "R1 verdict (dense array): $verdict (ratio KF/Θ = $(round(m(:kf_wake)/m(:theta_wake), digits=3)))"
else
    @info "R1 verdict: (no saved KF run — compare against kf_rank_sweep_dense_truth_$(TRUTH_ID).csv)"
end
@info "Saved $csvpath"
