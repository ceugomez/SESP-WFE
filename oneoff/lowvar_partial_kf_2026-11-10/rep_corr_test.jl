# rep_corr_test.jl — ONE-OFF (2026-11-11): is the representation error white in time?
# Follow-up to ar1_test.jl on the out_n67 (pool {03,10,11}, 67 snapshots) run. No FastEddy I/O.
#
# Hypothesis: loo11 mode 2 sits 598 off with σ = 26 (z = −23) under BOTH RW and AR(1)
# dynamics, so the over-confidence comes from the observation side: R_rep is a static
# covariance applied as if the representation error were independent at every 6 s step, but
# the out-of-subspace residual is a slowly evolving field → the filter counts ~K copies of the
# same error as K independent pieces of evidence.
#
# Diagnostic (needs only saved quantities):
#   d_k = y_k − H a_k,   a_k = oracle coefficients (projection of truth)
#       = (out-of-subspace truth at the sensor cells) + (white sensor noise)
#   - per channel: UNCENTERED autocorrelation of the rep part (what the filter assumes is zero
#     for lag ≥ 1), lag-0 has the known sensor-noise variance removed;
#     integrated timescale τ = 1 + 2 Σ_{ℓ≥1} ρ_ℓ (summed to the first non-positive lag);
#     calibration: rep-part mean square vs diag(R_rep)
#   - mode forcing b_k = Hᵀ R⁻¹ d_k: the rep error as the filter feels it, per mode
#     (persistence ⇒ steady push; mean/std ⇒ bias)
# Remedy test: R(c) = R_sensor + R_rep + (c−1)·R_rep⁺ for c ≥ 1 (R_sensor alone for c = 0),
# c ∈ C_SWEEP and c = τ̂ (median channel τ), under RW and AR(1) dynamics (same φ fit as
# ar1_test.jl). R_rep⁺ = PSD part of R_rep: the rep kernel is indefinite (min eig of R goes
# negative by c = 20 if R_rep itself is scaled), so c = 1 is exactly production and the
# extra inflation only adds a PSD term.
#
# Writes out_n67/repcorr/: acf_<exp>.csv, channel_<exp>.csv, forcing_acf_<exp>.csv,
# sweep.csv (one row per exp × dynamics × c), steps.csv and coeffs.csv (per step, all variants),
# inputs_<exp>.jld2 (H, R_sensor, R_rep, Y, a, oracle_rmse, d — so later tests skip the 7.6 GB basis).
#
#   julia --threads=16 rep_corr_test.jl      (~16 GB peak)
using LinearAlgebra, Statistics, JLD2, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kf_adapters.jl"))
include(joinpath(SESP_ROOT, "src", "experiment.jl"))     # MARGIN_M, sensor_mvar

const RUN = joinpath(@__DIR__, "out_n67")
const OUT = joinpath(RUN, "repcorr"); mkpath(OUT)
const CASES   = [(name="truth_rake_full_rep", fold="truth"), (name="loo11_rake_full_rep", fold="11")]
const C_SWEEP = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
const MAXLAG  = 30
const NMODES_OUT = 6
BLAS.set_num_threads(16)

function fit_phi(coeffs::AbstractMatrix, M::Int, μ::AbstractVector)
    L, N = size(coeffs); T = N ÷ M
    num = zeros(L); den = zeros(L)
    for i in 1:M
        a = Float64.(coeffs[:, (i-1)*T+1:i*T]) .- μ
        num .+= vec(sum(a[:, 2:end] .* a[:, 1:end-1]; dims=2))
        den .+= vec(sum(a[:, 1:end-1] .^ 2; dims=2))
    end
    return clamp.(num ./ den, -0.999, 0.999)
end

# Linear-Gaussian filter α_k = μ + Φ(α_{k-1} − μ) + η; Φ = I is exactly kalman.jl's run_filter.
function run_lin(μ, Σ0, Φ, Q, Y, obs::StaticObs)
    r = length(μ); m, K = size(Y)
    α = copy(μ); P = copy(Σ0)
    A = zeros(r, K); S = zeros(r, K); N = zeros(K)
    for k in 1:K
        α = μ + Φ * (α - μ)
        P = Matrix(Symmetric(Φ * P * Φ' + Q))
        α, P, ν, c2 = _update_information(α, P, view(Y, :, k), obs)
        A[:, k] = α; S[:, k] = sqrt.(max.(diag(P), 0)); N[k] = c2 / m
    end
    return A, S, N
end

# Uncentered autocorrelation of each row of X (lags 0:maxlag); lag-0 power reduced by `nvar`.
function uacf(X::AbstractMatrix, nvar::AbstractVector, maxlag)
    n, K = size(X)
    p0 = vec(mean(X .^ 2; dims=2)) .- nvar
    ρ = zeros(n, maxlag + 1)
    for ℓ in 0:maxlag
        c = vec(mean(X[:, 1+ℓ:K] .* X[:, 1:K-ℓ]; dims=2))
        ρ[:, ℓ+1] = ℓ == 0 ? ones(n) : c ./ p0
    end
    return ρ, p0
end
function tau_int(ρrow)
    s = 0.0
    for ℓ in 2:length(ρrow)
        ρrow[ℓ] <= 0 && break
        s += ρrow[ℓ]
    end
    return 1 + 2s
end

fieldrmse(a, α, oracle, d) = [sqrt(oracle[k]^2 + sum(abs2, a[:, k] .- α[:, k]) / d) for k in eachindex(oracle)]

sweep_io = open(joinpath(OUT, "sweep.csv"), "w")
println(sweep_io, "experiment,dynamics,c,field_mean,field_2ndhalf,field_final,nis_mean,oracle_mean,prior_mean,mode2_err_final,mode2_sigma_final,mode2_z_final,maxabs_z_final_modes1to6")
steps_io = open(joinpath(OUT, "steps.csv"), "w")
println(steps_io, "experiment,dynamics,c,step,field,nis,oracle,prior")
coef_io = open(joinpath(OUT, "coeffs.csv"), "w")
println(coef_io, "experiment,dynamics,c,step,mode,alpha,sigma,truth")

for cs in CASES
    @info "==== $(cs.name) (fold $(cs.fold)) ===="; flush(stderr)
    basis  = load_pod_basis(joinpath(RUN, "bases", "truth_$(cs.fold)_basis.jld2"))
    mdl    = load_models(joinpath(RUN, "bases", "truth_$(cs.fold)_models.jld2"))
    rep    = load_rep_error(joinpath(RUN, "bases", "rep_error_truth_$(cs.fold).jld2"))
    kf     = load(joinpath(RUN, "kf", "kf_result_$(cs.name).jld2"))
    grid   = basis.grid
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)
    dstate = domain.state_dim
    MV     = sensor_mvar()

    locs = kf["sensor_locs"][1]
    ms   = measSet([meas(Vector{Float32}(locs[:, j]), loc2idx(Vector{Float32}(locs[:, j]), grid),
                         zeros(Float32, 3)) for j in 1:size(locs, 2)])
    H, Rs = obs_operator_and_base_R(ms, basis, domain, MV)               # sensor only
    _, Rf = obs_operator_and_base_R(ms, basis, domain, MV; rep=rep)      # sensor + rep
    Rr    = Rf - Rs
    λr    = eigen(Symmetric(Rr))
    Rrp   = Matrix(Symmetric(λr.vectors * Diagonal(max.(λr.values, 0)) * λr.vectors'))
    @info @sprintf("R_rep: %d/%d negative eigs (min %.3g, max %.3g)", count(<(0), λr.values), length(λr.values),
                   minimum(λr.values), maximum(λr.values))
    Y     = Float64.(kf["innov"]) .+ H * Float64.(kf["α_pred"])
    a     = Float64.(kf["truth_coeffs"]); oracle = Float64.(kf["oracle_rmse"])
    μ, Σ0 = Float64.(mdl.μ_α0), Matrix(Symmetric(Float64.(mdl.Σ_α0)))
    Qrw   = Matrix(Symmetric(Float64.(mdl.Qα)))
    m, K  = size(Y)
    φ     = fit_phi(basis.coeffs, length(basis.member_ids), μ)
    Φ     = Matrix(Diagonal(φ))
    Qa    = Matrix(Symmetric(Σ0 - Φ * Σ0 * Φ'))
    λ     = eigen(Symmetric(Qa))
    Qa    = Matrix(Symmetric(λ.vectors * Diagonal(max.(λ.values, 1e-8 * maximum(λ.values))) * λ.vectors'))
    basis = nothing; rep = nothing; GC.gc()

    st = readlines(joinpath(RUN, "kf", "$(cs.name)_steps.csv"))
    ip = findfirst(==("field_prior"), split(st[1], ","))
    fprior = [parse(Float64, split(l, ",")[ip]) for l in st[2:end]]

    # sanity: RW with the full R reproduces the saved run
    Arw, _, _ = run_lin(μ, Σ0, Matrix(1.0I, length(μ), length(μ)), Qrw, Y, StaticObs(H, Rf))
    rel = norm(Arw .- kf["α"]) / norm(kf["α"])
    @info @sprintf("check: RW rel‖Δα‖ = %.2e", rel); rel < 1e-3 || error("reconstruction mismatch")

    # ---- diagnostic: residual at the sensors ----
    d    = Y .- H * a                                           # (m, K)
    nvar = repeat(Float64.(diag(MV)), m ÷ 3)
    ρ, p0 = uacf(d, nvar, MAXLAG)
    τ    = [tau_int(ρ[i, :]) for i in 1:m]
    dmean = vec(mean(d; dims=2)); dsd = vec(std(d; dims=2))
    open(joinpath(OUT, "acf_$(cs.name).csv"), "w") do io
        println(io, "channel,comp," * join(["lag$(ℓ)" for ℓ in 0:MAXLAG], ","))
        for i in 1:m
            println(io, i, ",", "uvw"[mod1(i, 3)], ",", join(round.(ρ[i, :]; digits=5), ","))
        end
    end
    open(joinpath(OUT, "channel_$(cs.name).csv"), "w") do io
        println(io, "channel,comp,rep_ms,Rrep_diag,Rsensor_diag,mean,sd,tau")
        for i in 1:m
            println(io, join([i, "uvw"[mod1(i, 3)], p0[i], Rr[i, i], Rs[i, i], dmean[i], dsd[i], τ[i]], ","))
        end
    end
    τhat = median(τ)
    calib = median(p0 ./ diag(Rr))
    @info @sprintf("residual: median τ_int = %.1f steps (IQR %.1f–%.1f), median rep_ms/diag(R_rep) = %.2f, median |mean|/sd = %.2f",
                   τhat, quantile(τ, 0.25), quantile(τ, 0.75), calib, median(abs.(dmean) ./ dsd))

    # mode forcing b_k = Hᵀ R⁻¹ d_k (white-noise part included; its ACF is ~0 for lag ≥ 1)
    B = H' * (Rf \ d)
    ρb, _ = uacf(B, zeros(size(B, 1)), MAXLAG)
    open(joinpath(OUT, "forcing_acf_$(cs.name).csv"), "w") do io
        println(io, "mode,mean_over_sd," * join(["lag$(ℓ)" for ℓ in 0:MAXLAG], ","))
        for i in 1:NMODES_OUT
            println(io, i, ",", mean(B[i, :]) / std(B[i, :]), ",", join(round.(ρb[i, :]; digits=5), ","))
        end
    end
    @info @sprintf("mode-2 forcing: mean/sd = %.2f, lag-1/5/20 uncentered ACF = %.2f/%.2f/%.2f",
                   mean(B[2, :]) / std(B[2, :]), ρb[2, 2], ρb[2, 6], ρb[2, 21])

    jldsave(joinpath(OUT, "inputs_$(cs.name).jld2"); H, R_sensor=Rs, R_rep=Rr, Y, a, oracle, d,
            μ, Σ0, Q_rw=Qrw, φ, Q_ar1=Qa, field_prior=fprior)

    # ---- remedy sweep ----
    h = (K ÷ 2 + 1):K
    for c in sort(unique(vcat(C_SWEEP, round(τhat; digits=1))))
        obs = StaticObs(H, c == 0 ? Rs : Rs + Rr + (c - 1) * Rrp)
        for (dyn, Φd, Qd) in (("RW", Matrix(1.0I, length(μ), length(μ)), Qrw), ("AR1", Φ, Qa))
            A, S, N = run_lin(μ, Σ0, Φd, Qd, Y, obs)
            f = fieldrmse(a, A, oracle, dstate)
            z = (A[:, K] .- a[:, K]) ./ S[:, K]
            @printf(sweep_io, "%s,%s,%g,%.5f,%.5f,%.5f,%.4f,%.5f,%.5f,%.2f,%.2f,%.3f,%.3f\n",
                    cs.name, dyn, c, mean(f), mean(f[h]), f[end], mean(N), mean(oracle), mean(fprior),
                    A[2, K] - a[2, K], S[2, K], z[2], maximum(abs.(z[1:NMODES_OUT])))
            for k in 1:K
                @printf(steps_io, "%s,%s,%g,%d,%.5f,%.4f,%.5f,%.5f\n", cs.name, dyn, c, k, f[k], N[k], oracle[k], fprior[k])
                for i in 1:NMODES_OUT
                    @printf(coef_io, "%s,%s,%g,%d,%d,%.4f,%.4f,%.4f\n", cs.name, dyn, c, k, i, A[i, k], S[i, k], a[i, k])
                end
            end
            @info @sprintf("  %-4s c=%5.1f  field %.4f (2nd half %.4f, final %.4f)  NIS %.2f  mode2 z %+.1f",
                           dyn, c, mean(f), mean(f[h]), f[end], mean(N), z[2])
        end
    end
    flush(sweep_io); flush(stderr); GC.gc()
end
close(sweep_io); close(steps_io); close(coef_io)
@info "done → $OUT"
