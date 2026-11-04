# run_theta_baseline.jl
# cgf cego6160@colorado.edu 7.13.26
#
# THE Θ-REGRESSION BASELINE (design doc R1, second_case_valley_design.md Part II).
# The skeptic's estimator: forget the POD filter — estimate only the 2-vector
# forcing Θ = (U_g, V_g) from the SAME rake observations, and reconstruct the
# field as the conditional-mean response to that Θ. If the KF does not beat
# this in the wake region, the mid-rank modes aren't earning their keep.
#
# Conditional-mean map: pointwise linear regression of member POST-SPINUP
# TIME-MEAN fields on their forcing, x̄(ζ) ≈ a(ζ) + b(ζ)·U_g + c(ζ)·V_g,
# fit across the same ensemble members the basis was built from. Per step,
# Θ̂ is a 2-parameter GLS fit of the rake obs through that map; the field
# estimate is B·[1, Θ̂]. Two variants are scored:
#   theta      — Θ̂ estimated from the obs (the honest baseline)
#   thetaoracle— Θ set to the truth member's TRUE forcing (the baseline's
#                ceiling; separates Θ-estimation error from conditional-mean
#                representation error)
#   const      — constant-wind floor (WindSeer-style): per step the field is a
#                uniform (u,v,w) equal to the component-wise mean of the SAME
#                noisy rake obs. What the obs buy you with NO prior at all.
# Both are compared per-step against the saved KF run (kf_result_truth_<id>.jld2)
# and the subspace-oracle floor, on full-domain AND wake-region RMSE.
#
# Rake geometry, noise variance, spinup, and margin REPLICATE
# run_simple_KF_assimilation.jl exactly (fresh noise draws, seeded).
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, SRC_DIR, ...
const WORKDIR  = SRC_DIR
const T_SPINUP = 900.0   # must match the basis + KF run
const MARGIN_M = 600f0   # must match the basis + KF run
const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

Random.seed!(20260713)   # measurement-noise reproducibility

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

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

# --- Truth field + the SAME rake as run_simple_KF_assimilation.jl ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = Matrix{Float32}(0.25f0 * I, 3, 3)
TIDX = collect(1:length(truth_t))

crest = argmax(grid.zPos[:, :, 1])
crest_x, crest_y = grid.xPos[crest[1], crest[2], 1], grid.yPos[crest[1], crest[2], 1]

TOWER_X_M         = 3000f0
TOWER_Y_OFFSETS_M = Float32[-500, -250, 0, 250, 500]
TOWER_HEIGHTS_M   = Float32[10, 50, 100]

locs = Vector{Vector{Float32}}()
for dy in TOWER_Y_OFFSETS_M, z in TOWER_HEIGHTS_M
    i, j, k = nearest_cell(grid, domain, TOWER_X_M, crest_y + dy, z)
    push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
end
NM = length(locs)
meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)

# --- Sample the map at the sensors (static rake -> build once) ---
n   = domain.n_grid
lin = _masked_linear_index(domain, grid)
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
R    = build_measurement_covariance(NM, MVAR)
Rinv = inv(R)
A    = Symmetric(Hb' * Rinv * Hb)        # 2×2 GLS normal matrix

# --- Wake-region mask (box downwind of the crest, FULL analysis y-width) ---
# Full y so the scoring area is identical for every LOO fold: the old ±750 m
# strip (270°-aligned) let far-direction truths sweep their wake out of the
# box, scoring ambient flow and biasing those folds LOW (found 2026-07-15).
WAKE_DX_M, WAKE_AGL_M = 2500f0, 250f0
terrain3 = repeat(grid.zPos[:, :, 1:1], 1, 1, grid.Nz)
wake3 = (grid.xPos .>= crest_x) .& (grid.xPos .<= crest_x + WAKE_DX_M) .&
        ((grid.zPos .- terrain3) .<= WAKE_AGL_M) .& domain.mask
ps   = lin[wake3]
widx = vcat(ps, ps .+ n, ps .+ 2n)
@info "wake box: x∈[$(crest_x), $(crest_x+WAKE_DX_M)] m, full analysis y, " *
      "AGL≤$(WAKE_AGL_M) m — $(length(ps)) cells"

# --- Saved KF run for the head-to-head (optional: fold runs are scored against
# the rank sweep instead, so a missing kf_result just NaNs the kf columns) ---
kfpath  = kf_result_filename(KF_DIR, TRUTH_ID)
have_kf = isfile(kfpath)
if have_kf
    kf = load(kfpath)
    global α, truth_coeffs = kf["α"], kf["truth_coeffs"]
    kf_times = kf["times"]
    length(kf_times) == length(truth_t) && all(isapprox.(kf_times, truth_t[TIDX])) ||
        error("KF result times don't match truth snapshots — rerun run_simple_KF_assimilation.jl first")
else
    @info "no saved KF result at $kfpath — kf columns will be NaN; oracle by direct projection"
    global truth_coeffs = basis.modes' * truth_U
end
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

csvpath = joinpath(KF_DIR, "theta_baseline_truth_$(TRUTH_ID).csv")
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
    @info "R1 verdict: $verdict (ratio KF/Θ = $(round(m(:kf_wake)/m(:theta_wake), digits=3)))"
else
    @info "R1 verdict: (no saved KF run — compare against kf_rank_sweep_truth_$(TRUTH_ID).csv)"
end
@info "Saved $csvpath"
