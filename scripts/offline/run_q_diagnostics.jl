# run_q_diagnostics.jl
# cgf cego6160@colorado.edu 7.22.26
#
# How closely does the calibrated process noise Q̂α (statmodel.jl: per-member
# increment covariance of the 11 basis members, averaged) track the REAL
# process noise of the held-out truth member, in POD coefficient space?
#
# "Real" process noise for fold <id> = the truth member's own oracle-coefficient
# increments Δα*_k = Pr'x(t_{k+1}) − Pr'x(t_k). Q̂ never saw the truth member,
# so this is an honest out-of-sample check. Diagnostics:
#   1. per-mode variance ratio  Q_true,ii / Q̂_ii            (6 s increments)
#   2. whitened increments z = chol(Q̂)⁻¹ Δα*: per-mode var/mean, and
#      mean‖z‖²/L (process-noise analogue of NIS; ≈1 if Q̂ is right)
#   3. full-matrix: generalized eigenvalues of (Q_true, Q̂) + Förstner distance
#      d = sqrt(Σ ln²λ_i) — catches wrong correlation structure
#   4. whiteness: per-mode autocorrelation of 6 s increments, lags 1..10.
#      RW assumes white; meander (τ≈60 s) predicts structure near lag 5.
#   5. stride test: non-overlapping 30 s truth increments vs the filter's
#      variance-adds rule 5·Q̂ (QSCALE = OBS_STRIDE in run_kf_rank_sweep.jl).
using LinearAlgebra, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, shared setup
const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const MAXLAG   = 10
include(joinpath(SRC_DIR, "fasteddy_io.jl"))
include(joinpath(SRC_DIR, "pod.jl"))
include(joinpath(SRC_DIR, "statmodel.jl"))

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = resolve_member(config, TRUTH_ID)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
mdl    = load_models(models_filename(BASES_DIR, TRUTH_ID))
L      = basis.L

# Oracle coefficients of the held-out truth (same as astar in run_kf_rank_sweep.jl)
truth_U = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
astar   = Float64.(basis.modes' * truth_U)          # (L × Ktot)
truth_U = nothing; GC.gc()
Ktot    = size(astar, 2)

Qhat = Float64.(Symmetric(mdl.Qα))

# Increments at a given stride, NON-overlapping (what the filter experiences at
# that cadence), against the variance-adds model s·Q̂.
increments(s::Int) = astar[:, 1+s:s:end] .- astar[:, 1:s:end-s]

# Whitened + full-matrix diagnostics of raw (not demeaned) increments Δ against
# the model covariance Qm — the RW assumes zero-mean noise, so bias belongs in
# the misfit, reported separately as zmean.
function diag_against(Δ::Matrix{Float64}, Qm::AbstractMatrix{Float64})
    n     = size(Δ, 2)
    C     = cholesky(Symmetric(Qm))
    Z     = C.L \ Δ                                   # (L × n) whitened
    zvar  = vec(sum(abs2, Z; dims=2)) ./ n            # raw 2nd moment per mode
    zmean = vec(mean(Z; dims=2))
    z2L   = sum(zvar) / L                             # mean ‖z‖²/(L·n)... = tr/L
    Qtrue = Symmetric(cov(Δ; dims=2, corrected=true)) # demeaned sample cov
    λ     = eigvals(Qtrue, Symmetric(Matrix(Qm)))     # generalized eigenvalues
    λ     = max.(λ, eps())
    forst = sqrt(sum(abs2, log.(λ)))
    return (; Qtrue, zvar, zmean, z2L, λmin=minimum(λ), λmed=median(λ),
              λmax=maximum(λ), forst, n)
end

Δ6  = increments(1);  d6  = diag_against(Δ6,  Qhat)
Δ30 = increments(5);  d30 = diag_against(Δ30, 5.0 .* Qhat)

# Per-mode autocorrelation of the 6 s increments (demeaned per mode)
Δc = Δ6 .- mean(Δ6; dims=2)
n6 = size(Δc, 2)
ac = Matrix{Float64}(undef, L, MAXLAG)
for i in 1:L
    v = @view Δc[i, :]
    s0 = sum(abs2, v) / n6
    for τ in 1:MAXLAG
        ac[i, τ] = (dot(@view(v[1+τ:end]), @view(v[1:end-τ])) / (n6 - τ)) / s0
    end
end

permode = joinpath(KF_DIR, "q_diag_permode_truth_$(TRUTH_ID).csv")
open(permode, "w") do io
    println(io, "mode,qhat,qtrue6,ratio6,zvar6,zmean6,qtrue30,ratio30,zvar30," *
                join(("ac$τ" for τ in 1:MAXLAG), ","))
    for i in 1:L
        println(io, join((i, Qhat[i, i],
            d6.Qtrue[i, i],  d6.Qtrue[i, i] / Qhat[i, i],        d6.zvar[i],  d6.zmean[i],
            d30.Qtrue[i, i], d30.Qtrue[i, i] / (5 * Qhat[i, i]), d30.zvar[i],
            ac[i, :]...), ","))
    end
end

summary = joinpath(KF_DIR, "q_diag_summary_truth_$(TRUTH_ID).csv")
open(summary, "w") do io
    println(io, "stride,dt_s,n_incr,mean_z2_per_L,forstner,geneig_min,geneig_med,geneig_max")
    println(io, join((1, 6,  d6.n,  d6.z2L,  d6.forst,  d6.λmin,  d6.λmed,  d6.λmax), ","))
    println(io, join((5, 30, d30.n, d30.z2L, d30.forst, d30.λmin, d30.λmed, d30.λmax), ","))
end

@info "fold=$TRUTH_ID  Ktot=$Ktot  L=$L"
@info "6 s : mean z²/L=$(round(d6.z2L,  digits=3))  Förstner=$(round(d6.forst,  digits=2))  genλ ∈ [$(round(d6.λmin,  sigdigits=3)), $(round(d6.λmax,  sigdigits=3))]"
@info "30 s: mean z²/L=$(round(d30.z2L, digits=3))  Förstner=$(round(d30.forst, digits=2))  genλ ∈ [$(round(d30.λmin, sigdigits=3)), $(round(d30.λmax, sigdigits=3))]"
@info "Saved $permode"
@info "Saved $summary"
