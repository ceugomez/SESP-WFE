# prior_test.jl — ONE-OFF (2026-11-11): is the PRIOR the reason loo11 mode 2 locks on the
# wrong value? Follow-up to rep_corr_test.jl, which showed the representation residual is ~white
# (τ_int ≈ 2 steps) and R_rep inflation up to 50× leaves mode-2 z ≈ −20.
#
# Uses out_n67/repcorr/inputs_<exp>.jld2 only (seconds; no basis load).
#
# 1. Cost decomposition at the TRUE trajectory a vs the filter estimate α (AR1 dynamics, c = 1):
#      J_obs  = Σ_k ‖y_k − H x_k‖²_{R⁻¹} / (mK)          (≈ 1 if x explains y as R expects)
#      J_init = (x_1 − μ)ᵀ Σ0⁻¹ (x_1 − μ) / r            (≈ 1 if x_1 is a typical prior draw)
#      J_dyn  = Σ_k ‖x_k − μ − Φ(x_{k−1} − μ)‖²_{Q⁻¹} / (r(K−1))
#    plus the prior Mahalanobis of a_1 along each eigenvector of Σ0 (which directions does the
#    prior rule the truth out in?).
# 2. Prior inflation sweep: Σ0 → s·Σ0 (RW), and Σ0, Q → s·Σ0, s·Q (AR1: stationary cov s·Σ0),
#    s ∈ S_SWEEP; s = 1e4 under RW ≈ the "uniform"/flat prior.
# 3. Shrinkage toward the diagonal (the fix suggested by 1: Σ0 has a near-null direction ≈ mode 2
#    because 2–3 members give rank-(M−1) between-member covariance → spurious exact correlation):
#      Σ0(γ) = (1−γ)Σ0 + γ·Diag(Σ0);  AR1: Q(γ) = Σ0(γ) − ΦΣ0(γ)Φ (PSD-clipped);
#      RW: Σ0(γ), and the same shrinkage on Q_rw.  γ ∈ G_SWEEP (dynamics label "RW"/"AR1", s = −γ
#      in sweep.csv/steps.csv is avoided: γ rows are written with dynamics "RW_shrink"/"AR1_shrink",
#      column s holding γ).
#
# Writes out_n67/priortest/: cost.csv, sigma0_eig_<exp>.csv, sweep.csv, steps.csv,
#   coeffs.csv (modes 1–6, α, σ, truth per step, every run)
#   julia prior_test.jl
using LinearAlgebra, Statistics, JLD2, Printf
const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kalman.jl"))

const RUN  = joinpath(@__DIR__, "out_n67")
const OUT  = joinpath(RUN, "priortest"); mkpath(OUT)
const EXPS = ["truth_rake_full_rep", "loo11_rake_full_rep"]
const S_SWEEP = [1.0, 2.0, 4.0, 10.0, 100.0, 1e4]
const G_SWEEP = [0.0, 0.01, 0.1, 0.5, 1.0]
shrink(C, γ) = Matrix(Symmetric((1 - γ) * C + γ * Diagonal(diag(C))))
function psdclip(C)
    E = eigen(Symmetric(C))
    return Matrix(Symmetric(E.vectors * Diagonal(max.(E.values, 1e-8 * maximum(E.values))) * E.vectors'))
end

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

# state_dim from the saved run: field² − oracle² = ‖a − α‖²/d  (median over steps)
function state_dim(name)
    kf = load(joinpath(RUN, "kf", "kf_result_$(name).jld2"))
    e2 = vec(sum(abs2, Float64.(kf["α"]) .- Float64.(kf["truth_coeffs"]); dims=1))
    return median(e2 ./ (Float64.(kf["field_rmse"]) .^ 2 .- Float64.(kf["oracle_rmse"]) .^ 2))
end
fieldrmse(a, α, oracle, d) = [sqrt(oracle[k]^2 + sum(abs2, a[:, k] .- α[:, k]) / d) for k in eachindex(oracle)]
wnorm2(x, C) = dot(x, C \ x)

cost_io = open(joinpath(OUT, "cost.csv"), "w")
println(cost_io, "experiment,traj,J_obs,J_init,J_dyn")
sw_io = open(joinpath(OUT, "sweep.csv"), "w")
println(sw_io, "experiment,dynamics,s,field_mean,field_2ndhalf,field_final,nis_mean,oracle_mean,prior_mean,mode2_err_final,mode2_sigma_final,mode2_z_final")
st_io = open(joinpath(OUT, "steps.csv"), "w")
println(st_io, "experiment,dynamics,s,step,field,nis")
co_io = open(joinpath(OUT, "coeffs.csv"), "w")
println(co_io, "experiment,dynamics,s,step,mode,alpha,sigma,truth")
write_coeffs(name, dyn, s, A, S, a) = for k in axes(A, 2), i in 1:6
    @printf(co_io, "%s,%s,%g,%d,%d,%.4f,%.4f,%.4f\n", name, dyn, s, k, i, A[i, k], S[i, k], a[i, k])
end

for name in EXPS
    @info "==== $name ===="
    D = load(joinpath(RUN, "repcorr", "inputs_$(name).jld2"))
    H, Y, a, oracle = D["H"], D["Y"], D["a"], D["oracle"]
    R  = D["R_sensor"] + D["R_rep"]
    μ, Σ0, Qrw, Qa, φ = D["μ"], D["Σ0"], D["Q_rw"], D["Q_ar1"], D["φ"]
    Φ  = Matrix(Diagonal(φ)); r = length(μ); m, K = size(Y)
    obs = StaticObs(H, R)
    d  = state_dim(name); @info @sprintf("state_dim (recovered) = %.0f", d)

    # ---- 1. cost decomposition ----
    Aar, _, _ = run_lin(μ, Σ0, Φ, Qa, Y, obs)
    for (lab, X) in (("truth_a", a), ("estimate_AR1", Aar))
        Jo = sum(wnorm2(Y[:, k] - H * X[:, k], R) for k in 1:K) / (m * K)
        Ji = wnorm2(X[:, 1] - μ, Σ0) / r
        Jd = sum(wnorm2(X[:, k] - μ - Φ * (X[:, k-1] - μ), Qa) for k in 2:K) / (r * (K - 1))
        @printf(cost_io, "%s,%s,%.4f,%.4f,%.4f\n", name, lab, Jo, Ji, Jd)
        @info @sprintf("  %-13s J_obs %.3f   J_init %.3g   J_dyn %.3g", lab, Jo, Ji, Jd)
    end
    # truth's prior Mahalanobis along Σ0 eigenvectors
    E = eigen(Symmetric(Σ0)); ord = sortperm(E.values; rev=true)
    λ, V = E.values[ord], E.vectors[:, ord]
    w  = V' * (a[:, 1] - μ)                      # truth deviation in Σ0 eigen-coords
    contrib = w .^ 2 ./ λ
    open(joinpath(OUT, "sigma0_eig_$(name).csv"), "w") do io
        println(io, "eig_rank,lambda,truth_dev,mahal_contrib,mode2_loading")
        for j in 1:r
            println(io, join([j, λ[j], w[j], contrib[j], V[2, j]], ","))
        end
    end
    top = sortperm(contrib; rev=true)[1:5]
    @info "  truth (step 1) Mahalanobis² = $(round(sum(contrib), sigdigits=4)) over r=$r; top eig-directions:"
    for j in top
        @info @sprintf("    eig #%2d  λ = %.3g (sd %.3g)  dev %.3g  → %.3g  (mode-2 loading %.2f)",
                       j, λ[j], sqrt(λ[j]), w[j], contrib[j], V[2, j])
    end
    @info @sprintf("  Σ0 condition number %.3g; eigs < 1e-6·max: %d", λ[1] / λ[end], count(<(1e-6 * λ[1]), λ))

    # ---- 2. prior inflation sweep ----
    fprior = D["field_prior"]; h = (K ÷ 2 + 1):K
    for s in S_SWEEP, (dyn, Φd, Qd) in (("RW", Matrix(1.0I, r, r), Qrw), ("AR1", Φ, s * Qa))
        A, S, N = run_lin(μ, s * Σ0, Φd, Qd, Y, obs)
        f = fieldrmse(a, A, oracle, d); z2 = (A[2, K] - a[2, K]) / S[2, K]
        @printf(sw_io, "%s,%s,%g,%.5f,%.5f,%.5f,%.4f,%.5f,%.5f,%.2f,%.2f,%.3f\n", name, dyn, s, mean(f), mean(f[h]),
                f[end], mean(N), mean(oracle), mean(fprior), A[2, K] - a[2, K], S[2, K], z2)
        for k in 1:K
            @printf(st_io, "%s,%s,%g,%d,%.5f,%.4f\n", name, dyn, s, k, f[k], N[k])
        end
        write_coeffs(name, dyn, s, A, S, a)
        @info @sprintf("  %-4s s=%7g  field %.4f (2nd half %.4f, final %.4f)  NIS %.2f  mode2 err %+.0f σ %.0f z %+.1f",
                       dyn, s, mean(f), mean(f[h]), f[end], mean(N), A[2, K] - a[2, K], S[2, K], z2)
    end

    # ---- 3. shrinkage toward diagonal ----
    for γ in G_SWEEP
        Σg = shrink(Σ0, γ)
        for (dyn, Φd, Qd) in (("RW_shrink", Matrix(1.0I, r, r), shrink(Qrw, γ)),
                              ("AR1_shrink", Φ, psdclip(Σg - Φ * Σg * Φ')))
            A, S, N = run_lin(μ, Σg, Φd, Qd, Y, obs)
            f = fieldrmse(a, A, oracle, d); z2 = (A[2, K] - a[2, K]) / S[2, K]
            @printf(sw_io, "%s,%s,%g,%.5f,%.5f,%.5f,%.4f,%.5f,%.5f,%.2f,%.2f,%.3f\n", name, dyn, γ, mean(f), mean(f[h]),
                    f[end], mean(N), mean(oracle), mean(fprior), A[2, K] - a[2, K], S[2, K], z2)
            for k in 1:K
                @printf(st_io, "%s,%s,%g,%d,%.5f,%.4f\n", name, dyn, γ, k, f[k], N[k])
            end
            write_coeffs(name, dyn, γ, A, S, a)
            @info @sprintf("  %-10s γ=%4g  field %.4f (2nd half %.4f, final %.4f)  NIS %.2f  mode2 err %+.0f σ %.0f z %+.1f",
                           dyn, γ, mean(f), mean(f[h]), f[end], mean(N), A[2, K] - a[2, K], S[2, K], z2)
        end
    end
end
close(cost_io); close(sw_io); close(st_io); close(co_io)
@info "done → $OUT"
