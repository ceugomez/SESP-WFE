# ar1_test.jl — ONE-OFF (2026-11-11): mean-reverting AR(1) dynamics vs the random walk, on the
# saved out_n67 (3-member pool {03,10,11}, 67 snapshots) bases + KF results. No FastEddy I/O.
#
#   RW  : α_k = α_{k-1} + η,                    η ~ N(0, Q_inc)       (production, F = I)
#   AR1 : α_k = μ0 + Φ(α_{k-1} − μ0) + η,       η ~ N(0, Σ0 − ΦΣ0Φᵀ)  (stationary cov = Σ0)
#         Φ = diag(φ_i), φ_i = lag-1 autocorrelation of mode i about μ0, pooled over the
#         basis members (within-member pairs only), clipped to [-0.999, 0.999].
#
# Reconstruction (exact, nothing re-sampled):
#   H, R  — production obs_operator_and_base_R at the saved sensor cells (rep=true)
#   Y     — y_k = ν_k + H α⁻_k from the saved innovations and forecast means
#   field RMSE — modes orthonormal and x − Pr a ⟂ span(Pr), so
#                ‖x − Pr α‖² = ‖x − Pr a‖² + ‖a − α‖²  →  √(oracle² + ‖a−α‖²/d)
#                (agrees with the saved Float32 field_rmse to ~0.4%: Float32 modes/sums)
# Both identities are checked against the saved RW run before the AR1 numbers are trusted.
#
#   julia --threads=16 ar1_test.jl      (~16 GB peak: one 7.6 GB basis at a time)
using LinearAlgebra, Statistics, JLD2, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kf_adapters.jl"))
include(joinpath(SESP_ROOT, "src", "experiment.jl"))     # MARGIN_M, sensor_mvar

const RUN = joinpath(@__DIR__, "out_n67")
const OUT = joinpath(RUN, "ar1"); mkpath(OUT)
const CASES = [(name="truth_rake_full_rep", fold="truth"), (name="loo11_rake_full_rep", fold="11")]
BLAS.set_num_threads(16)

# Pooled within-member lag-1 autocorrelation per mode, about the global mean μ.
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

# Same loop as kalman.jl run_filter, with F = Φ and a mean μ to revert to.
function run_ar1(μ, Σ0, Φ, Q, Y, obs::StaticObs)
    r = length(μ); m, K = size(Y)
    α = copy(μ); P = copy(Σ0)
    A = zeros(r, K); C = zeros(K); Ps = zeros(r, K)
    for k in 1:K
        α = μ + Φ * (α - μ)
        P = Symmetric(Φ * P * Φ' + Q) |> Matrix
        α, P, ν, c2 = _update_information(α, P, view(Y, :, k), obs)
        A[:, k] = α; C[k] = c2 / m; Ps[:, k] = sqrt.(max.(diag(P), 0))
    end
    return A, C, Ps
end

fieldrmse(a, α, oracle, d) = [sqrt(oracle[k]^2 + sum(abs2, a[:, k] .- α[:, k]) / d) for k in eachindex(oracle)]

rows = String[]
for c in CASES
    @info "==== $(c.name) (fold $(c.fold)) ===="; flush(stderr)
    basis  = load_pod_basis(joinpath(RUN, "bases", "truth_$(c.fold)_basis.jld2"))
    mdl    = load_models(joinpath(RUN, "bases", "truth_$(c.fold)_models.jld2"))
    rep    = load_rep_error(joinpath(RUN, "bases", "rep_error_truth_$(c.fold).jld2"))
    kf     = load(joinpath(RUN, "kf", "kf_result_$(c.name).jld2"))
    grid   = basis.grid
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)
    d      = domain.state_dim

    locs = kf["sensor_locs"][1]                                   # (3, nm)
    ms   = measSet([meas(Vector{Float32}(locs[:, j]), loc2idx(Vector{Float32}(locs[:, j]), grid),
                         zeros(Float32, 3)) for j in 1:size(locs, 2)])
    H, R = obs_operator_and_base_R(ms, basis, domain, sensor_mvar(); rep=rep)
    obs  = StaticObs(H, R)
    Y    = Float64.(kf["innov"]) .+ H * Float64.(kf["α_pred"])
    a    = Float64.(kf["truth_coeffs"]); oracle = Float64.(kf["oracle_rmse"])
    μ, Σ0 = Float64.(mdl.μ_α0), Float64.(Symmetric(Float64.(mdl.Σ_α0)))

    # --- checks: RW reproduces the saved run; field-RMSE identity holds ---
    rw = run_filter(KFModel(mdl, basis.L), Y, obs)
    relα = norm(rw.α_post .- kf["α"]) / norm(kf["α"])
    frw  = fieldrmse(a, rw.α_post, oracle, d)
    relf = maximum(abs.(frw .- kf["field_rmse"]) ./ kf["field_rmse"])
    @info @sprintf("check: RW rel‖Δα‖ = %.2e, max rel field-RMSE diff = %.2e", relα, relf)
    (relα < 1e-3 && relf < 1e-2) || error("reconstruction does not reproduce the saved run")

    # --- AR(1) ---
    φ  = fit_phi(basis.coeffs, length(basis.member_ids), μ)
    Φ  = Diagonal(φ)
    Qa = Matrix(Symmetric(Σ0 - Φ * Σ0 * Φ'))
    λ  = eigen(Symmetric(Qa)); nneg = count(<(0), λ.values)
    Qa = λ.vectors * Diagonal(max.(λ.values, 1e-8 * maximum(λ.values))) * λ.vectors'
    Qa = Matrix(Symmetric(Qa))
    @info @sprintf("φ: min %.3f median %.3f max %.3f | Q_ar1 negative eigs clipped: %d", minimum(φ), median(φ), maximum(φ), nneg)
    Aar, Car, Sar = run_ar1(μ, Σ0, Matrix(Φ), Qa, Y, obs)
    far = fieldrmse(a, Aar, oracle, d)

    # prior baseline straight from the saved steps CSV (constant μ0 field)
    st = readlines(joinpath(RUN, "kf", "$(c.name)_steps.csv"))
    hdr = split(st[1], ","); ip = findfirst(==("field_prior"), hdr)
    fprior = [parse(Float64, split(l, ",")[ip]) for l in st[2:end]]

    K = size(Y, 2); h = (K ÷ 2 + 1):K
    for (lab, f, nisv) in (("RW", frw, nis(rw)), ("AR1", far, Car))
        push!(rows, @sprintf("%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f", c.name, lab, mean(f), mean(f[h]), f[end],
                             mean(oracle), mean(fprior), mean(nisv)))
    end
    open(joinpath(OUT, "$(c.name)_steps.csv"), "w") do io
        println(io, "step,field_rw,field_ar1,field_oracle,field_prior,nis_rw,nis_ar1")
        for k in 1:K
            println(io, join([k, frw[k], far[k], oracle[k], fprior[k], nis(rw)[k], Car[k]], ","))
        end
    end
    println("\n", c.name, ": per-mode at final step (error / filter σ)")
    for i in 1:6
        Prw = sqrt(rw.P_post[i, i, end])
        @printf("  mode %d  φ=%.4f | RW err %8.1f σ %7.1f z %+6.2f | AR1 err %8.1f σ %7.1f z %+6.2f | prior sd %7.1f\n",
                i, φ[i], rw.α_post[i, end] - a[i, end], Prw, (rw.α_post[i, end] - a[i, end]) / Prw,
                Aar[i, end] - a[i, end], Sar[i, end], (Aar[i, end] - a[i, end]) / Sar[i, end], sqrt(Σ0[i, i]))
    end
    jldsave(joinpath(OUT, "ar1_$(c.name).jld2"); φ, Q_ar1=Qa, α_ar1=Aar, σ_ar1=Sar, nis_ar1=Car, field_ar1=far)
    basis = nothing; rep = nothing; GC.gc()
end

open(joinpath(OUT, "summary.csv"), "w") do io
    println(io, "experiment,dynamics,field_mean,field_2ndhalf,field_final,oracle_mean,prior_mean,nis_mean")
    foreach(r -> println(io, r), rows)
end
println("\nexperiment,dynamics,field_mean,field_2ndhalf,field_final,oracle_mean,prior_mean,nis_mean")
foreach(println, rows)
