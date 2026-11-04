# kalman.jl
# cgf cego6160@colorado.edu 11.4.26
#
# THE Kalman filter for SESP-WFE. Every driver that filters (rake / dense / twin / transect
# rank sweeps, the single-run assimilation, the Riccati covariance analysis) calls into this
# file; none carries its own predict/update loop. Pure linear algebra — no FastEddy IO, no
# POD structs — so it can be unit-tested in isolation (test/test_kalman.jl). The glue that
# turns a PODBasis / models / measSet into these types lives in kf_adapters.jl.
#
# Model: random walk in the r-dim POD coefficient subspace,
#     α_k = α_{k-1} + η_k,   η_k ~ N(0, Q)          (F = I)
#     y_k = H_k α_k + ε_k,   ε_k ~ N(0, R_k)
# with prior α ~ N(μ0, Σ0). Convention (unchanged from every pre-consolidation loop): the
# filter PREDICTS FIRST, so the first update sees P⁻_1 = Σ0 + Q, not Σ0.
#
# Two algebraically identical update forms:
#   :gain         S = H P⁻ Hᵀ + R,  G = P⁻ Hᵀ S⁻¹,  α⁺ = α⁻ + Gν,
#                 P⁺ = (I − GH) P⁻ (I − GH)ᵀ + G R Gᵀ          (Joseph form)
#                 Cost per step O(m³) in the obs dimension m. Use for small m / varying H.
#   :information  M = P⁻⁻¹ + HᵀR⁻¹H,  α⁺ = α⁻ + M⁻¹ HᵀR⁻¹ν,  P⁺ = M⁻¹,
#                 νᵀS⁻¹ν = νᵀR⁻¹ν − bᵀM⁻¹b  (b = HᵀR⁻¹ν; Woodbury)
#                 R is factorized ONCE (static obs only); per-step cost O(r³). Use for m ≫ r.
# test/test_kalman.jl checks the two agree to round-off.
#
# Everything is carried in Float64. (The pre-consolidation rake sweep and runtime_loop ran in
# Float32; their numbers differ from this library at the ~1e-5 relative level.)

using LinearAlgebra
using Statistics: mean

# ---------------------------------------------------------------------------
# Model and observation types
# ---------------------------------------------------------------------------

"Prior + process noise of the rank-r random-walk state-space model."
struct KFModel
    μ0 :: Vector{Float64}   # (r,)   prior mean
    Σ0 :: Matrix{Float64}   # (r, r) prior covariance
    Q  :: Matrix{Float64}   # (r, r) process-noise covariance per assimilation step
    function KFModel(μ0::AbstractVector, Σ0::AbstractMatrix, Q::AbstractMatrix)
        r = length(μ0)
        size(Σ0) == (r, r) || error("KFModel: Σ0 is $(size(Σ0)), expected ($r, $r)")
        size(Q)  == (r, r) || error("KFModel: Q is $(size(Q)), expected ($r, $r)")
        new(Float64.(μ0), _sym(Float64.(Σ0)), _sym(Float64.(Q)))
    end
end
state_rank(m::KFModel) = length(m.μ0)

abstract type ObsModel end

"""
Static observation model: same H and R at every step (fixed sensor array). Precomputes the
Cholesky factor of R and HᵀR⁻¹H so the information-form update does only r×r work per step.
Throws if R is not positive definite (e.g. an indefinite R_rep not masked by R_sensor).
"""
struct StaticObs <: ObsModel
    H     :: Matrix{Float64}             # (m, r)
    R     :: Matrix{Float64}             # (m, m)
    Rchol :: Cholesky{Float64,Matrix{Float64}}
    RiH   :: Matrix{Float64}             # R⁻¹H    (m, r)
    HtRiH :: Matrix{Float64}             # HᵀR⁻¹H  (r, r)
end
function StaticObs(H::AbstractMatrix, R::AbstractMatrix)
    Hd, Rd = Float64.(H), _sym(Float64.(R))
    size(Rd, 1) == size(Hd, 1) || error("StaticObs: H has $(size(Hd,1)) rows, R is $(size(Rd))")
    C = cholesky(Symmetric(Rd); check=false)
    issuccess(C) || error("StaticObs: R ($(size(Rd,1))×$(size(Rd,1))) is not positive " *
        "definite (min eig $(round(eigmin(Symmetric(Rd)), sigdigits=3))). If R includes " *
        "R_rep, the rep-error kernel's negative eigenvalues are not masked by R_sensor.")
    RiH = C \ Hd
    return StaticObs(Hd, Rd, C, RiH, _sym(Hd' * RiH))
end

"Time-varying observation model (moving sensor): H_k, R_k per step, fixed obs dimension m."
struct VaryingObs <: ObsModel
    H :: Vector{Matrix{Float64}}
    R :: Vector{Matrix{Float64}}
    function VaryingObs(H::AbstractVector, R::AbstractVector)
        length(H) == length(R) || error("VaryingObs: $(length(H)) H's but $(length(R)) R's")
        m = size(H[1], 1)
        all(h -> size(h, 1) == m, H) || error("VaryingObs: obs dimension must be constant")
        new([Float64.(h) for h in H], [_sym(Float64.(x)) for x in R])
    end
end

obs_dim(o::StaticObs)  = size(o.H, 1)
obs_dim(o::VaryingObs) = size(o.H[1], 1)
obs_rank(o::StaticObs)  = size(o.H, 2)
obs_rank(o::VaryingObs) = size(o.H[1], 2)
n_steps(::StaticObs)    = nothing          # any number of steps
n_steps(o::VaryingObs)  = length(o.H)

# ---------------------------------------------------------------------------
# Rank truncation of a full-L model / observation operator
# ---------------------------------------------------------------------------

"""
    truncate_model(μ0, Σ0, Q, r; qscale=1) -> KFModel

Leading r×r blocks of the full-L prior/process-noise models. Valid because POD coefficients
are defined against fixed orthonormal modes. `qscale` multiplies Q (the strided-cadence
random-walk accumulation; see the drivers' OBS_STRIDE note).
"""
function truncate_model(μ0::AbstractVector, Σ0::AbstractMatrix, Q::AbstractMatrix, r::Int;
                        qscale::Real=1.0)
    1 <= r <= length(μ0) || error("truncate_model: r=$r outside 1:$(length(μ0))")
    return KFModel(μ0[1:r], Σ0[1:r, 1:r], qscale .* Q[1:r, 1:r])
end

"""
    tail_clutter(Hfull, Σ0full, r) -> (m × m) matrix

Discarded-tail clutter H_tail Σ0[tail,tail] H_tailᵀ, tail = r+1:L. For rank r < L the
unestimated modes' prior variance is observation-space noise; folding it into R keeps the
truncated filter consistent ("honest truncation"). Zero matrix when r == L.
"""
function tail_clutter(Hfull::AbstractMatrix, Σ0full::AbstractMatrix, r::Int)
    L = size(Hfull, 2)
    m = size(Hfull, 1)
    r == L && return zeros(Float64, m, m)
    tail = (r+1):L
    Ht = Float64.(Hfull[:, tail])
    return _sym(Ht * Float64.(Σ0full[tail, tail]) * Ht')
end

"""
    truncated_obs(Hfull, R_base, Σ0full, r) -> StaticObs

Rank-r static observation model with honest truncation: H = Hfull[:, 1:r],
R = R_base + tail_clutter(Hfull, Σ0full, r).
"""
truncated_obs(Hfull::AbstractMatrix, R_base::AbstractMatrix, Σ0full::AbstractMatrix, r::Int) =
    StaticObs(Hfull[:, 1:r], Float64.(R_base) + tail_clutter(Hfull, Σ0full, r))

"Per-step version of `truncated_obs` for moving sensors (one Hfull_k, R_k per step)."
truncated_obs(Hfull::AbstractVector{<:AbstractMatrix}, R_base::AbstractVector{<:AbstractMatrix},
              Σ0full::AbstractMatrix, r::Int) =
    VaryingObs([H[:, 1:r] for H in Hfull],
               [Float64.(R) + tail_clutter(H, Σ0full, r) for (H, R) in zip(Hfull, R_base)])

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

"""
Full filter history. Indexing is by assimilation step k = 1..K (columns / third dim).
`chi2[k] = ν_kᵀ S_k⁻¹ ν_k` is the RAW innovation statistic, χ²(m) distributed when the
filter is consistent; `nis(h) = chi2 ./ m` is the per-dimension NIS (≈1 when consistent).
"""
struct KFHistory
    α_pred :: Matrix{Float64}      # (r, K)    prior (forecast) mean α⁻
    P_pred :: Array{Float64,3}     # (r, r, K) prior covariance P⁻
    α_post :: Matrix{Float64}      # (r, K)    posterior mean α⁺
    P_post :: Array{Float64,3}     # (r, r, K) posterior covariance P⁺
    innov  :: Matrix{Float64}      # (m, K)    innovation ν = y − H α⁻
    chi2   :: Vector{Float64}      # (K,)      νᵀ S⁻¹ ν
    m      :: Int                  # obs dimension
    form   :: Symbol               # update form actually used (:gain | :information)
end
nis(h::KFHistory)      = h.chi2 ./ h.m
mean_nis(h::KFHistory) = mean(h.chi2) / h.m
state_rank(h::KFHistory) = size(h.α_post, 1)
n_steps(h::KFHistory)    = size(h.α_post, 2)

"`:auto` picks the information form for a static model with more obs rows than modes."
function _resolve_form(form::Symbol, obs::ObsModel)
    form in (:auto, :gain, :information) || error("form must be :auto, :gain or :information")
    if form === :information && !(obs isa StaticObs)
        error("the information form needs a StaticObs (R factorized once)")
    end
    form === :auto || return form
    return (obs isa StaticObs && obs_dim(obs) > obs_rank(obs)) ? :information : :gain
end

_H(o::StaticObs, k)  = o.H
_H(o::VaryingObs, k) = o.H[k]
_R(o::StaticObs, k)  = o.R
_R(o::VaryingObs, k) = o.R[k]

"""
    run_filter(model, Y, obs; form=:auto) -> KFHistory

Random-walk-in-subspace Kalman filter over the columns of Y (m × K observations, one column
per assimilation step). `obs` is a StaticObs (fixed array) or VaryingObs (moving sensor,
length K). `form` selects the update algebra (see header); `:auto` → information form when
the model is static and m > r, else gain form.
"""
function run_filter(model::KFModel, Y::AbstractMatrix, obs::ObsModel; form::Symbol=:auto)
    r = state_rank(model)
    m, K = size(Y)
    obs_rank(obs) == r || error("run_filter: obs model has rank $(obs_rank(obs)), model has $r")
    obs_dim(obs)  == m || error("run_filter: Y has $m rows, obs model has $(obs_dim(obs))")
    ns = n_steps(obs)
    (ns === nothing || ns == K) || error("run_filter: VaryingObs has $ns steps, Y has $K")
    f = _resolve_form(form, obs)

    α_pred = Matrix{Float64}(undef, r, K);  α_post = similar(α_pred)
    P_pred = Array{Float64,3}(undef, r, r, K);  P_post = similar(P_pred)
    innov  = Matrix{Float64}(undef, m, K)
    chi2   = Vector{Float64}(undef, K)

    α = copy(model.μ0)
    P = copy(model.Σ0)
    for k in 1:K
        # --- predict (F = I) ---
        P = P + model.Q
        α_pred[:, k] = α
        P_pred[:, :, k] = P
        # --- update ---
        y = @view Y[:, k]
        α, P, ν, c2 = f === :information ? _update_information(α, P, y, obs) :
                                           _update_gain(α, P, y, _H(obs, k), _R(obs, k))
        α_post[:, k] = α
        P_post[:, :, k] = P
        innov[:, k] = ν
        chi2[k] = c2
    end
    return KFHistory(α_pred, P_pred, α_post, P_post, innov, chi2, m, f)
end

function _update_gain(α, P, y, H, R)
    ν = y - H * α
    S = cholesky(Symmetric(H * P * H' + R))
    G = (S \ (H * P))'                       # P Hᵀ S⁻¹  (P, S symmetric)
    α⁺ = α + G * ν
    IKH = I - G * H
    P⁺ = _sym(IKH * P * IKH' + G * R * G')   # Joseph form: PSD-preserving
    return α⁺, P⁺, ν, dot(ν, S \ ν)
end

function _update_information(α, P, y, o::StaticObs)
    ν = y - o.H * α
    M = cholesky(Symmetric(inv(cholesky(Symmetric(P))) + o.HtRiH))
    b = o.RiH' * ν                           # HᵀR⁻¹ν
    Mib = M \ b
    α⁺ = α + Mib
    P⁺ = _sym(inv(M))
    return α⁺, P⁺, ν, dot(ν, o.Rchol \ ν) - dot(b, Mib)
end

# ---------------------------------------------------------------------------
# Covariance-only recursions and observability diagnostics
# ---------------------------------------------------------------------------

"""
    riccati_steady_state(model, obs::StaticObs; itmax=100, rtol=1e-12) -> (P∞⁺, iterations)

Steady-state POSTERIOR covariance of the random-walk filter under a static array: the
stabilizing solution of the DARE for the predicted covariance,
    P⁻ = P⁻ (I + W P⁻)⁻¹ + Q,     W = HᵀR⁻¹H,
returned as P∞⁺ = P∞⁻ − Q. Solved by the structure-preserving doubling algorithm (SDA; Chu,
Fan & Lin 2005, for X = AᵀX(I+GX)⁻¹A + H with A = I, G = W, H = Q), which converges
QUADRATICALLY. Independent of Σ0 and of the data.

Why not fixed-point iteration (the pre-consolidation riccati_Pinf): for the forcing-head
modes Q ≪ 1/(HᵀR⁻¹H), so the recursion contracts by only ~1 − √(Q·W) per step — ~1e4 steps
on the real bases. riccati_Pinf stopped at 5000 and silently returned the unconverged
iterate. See also `closed_loop_decay`, which quantifies exactly that slowness.
Errors if any direction is unobservable (`n_unobservable(obs) > 0`): no steady state exists.
"""
function riccati_steady_state(model::KFModel, obs::StaticObs; itmax::Int=100, rtol::Real=1e-12)
    obs_rank(obs) == state_rank(model) || error("riccati_steady_state: rank mismatch")
    nu = n_unobservable(obs)
    nu == 0 || error("riccati_steady_state: no steady state — $nu of $(state_rank(model)) " *
        "coefficient directions are unobservable (HᵀR⁻¹H rank-deficient, e.g. fewer obs rows " *
        "than modes). With F = I those directions random-walk and their variance grows " *
        "without bound. Check n_unobservable(obs) first.")
    r  = state_rank(model)
    Ir = Matrix{Float64}(I, r, r)
    A, G, X = copy(Ir), copy(obs.HtRiH), copy(model.Q)
    for it in 1:itmax
        F  = lu(Ir + G * X)                  # (I + G_k H_k)
        FA = F \ A
        Xn = _sym(X + A' * X * FA)           # H_{k+1} = H_k + A_kᵀ H_k (I+G_kH_k)⁻¹ A_k
        G  = _sym(G + A * (F \ G) * A')      # G_{k+1} = G_k + A_k (I+G_kH_k)⁻¹ G_k A_kᵀ
        A  = A * FA                          # A_{k+1} = A_k (I+G_kH_k)⁻¹ A_k
        if maximum(abs, Xn - X) <= rtol * maximum(abs, Xn)
            return _sym(Xn - model.Q), it
        end
        X = Xn
    end
    error("riccati_steady_state: SDA did not converge in $itmax doublings (rtol=$rtol)")
end

"""
    n_unobservable(obs::StaticObs; rtol=1e-10) -> Int

Number of coefficient directions the static array never observes: eigenvalues of HᵀR⁻¹H at
or below rtol·max eigenvalue. Nonzero ⇒ (F = I, H) is not detectable ⇒ no steady-state
filter covariance exists (posterior variance in those directions grows like k·Q). Always
nonzero when obs_dim < rank.
"""
function n_unobservable(obs::StaticObs; rtol::Real=1e-10)
    λ = eigvals(Symmetric(obs.HtRiH))
    return count(<=(rtol * maximum(λ)), λ)
end

"""
    closed_loop_decay(model, obs::StaticObs, P∞⁺) -> (ρ, τ_steps)

Spectral radius ρ of the steady-state error-propagation matrix (I − G H) = P∞⁺ (P∞⁻)⁻¹ and the
corresponding e-folding memory τ = −1/ln ρ in ASSIMILATION STEPS: how many updates the filter
needs to forget an initial error in its slowest direction. τ comparable to or longer than the
run length K means the filter never reaches its steady-state covariance within the run.
"""
function closed_loop_decay(model::KFModel, obs::StaticObs, P∞⁺::AbstractMatrix)
    Pm = Symmetric(P∞⁺ + model.Q)
    ρ  = maximum(abs, eigvals(Matrix(P∞⁺) / cholesky(Pm)))
    return ρ, (ρ < 1 ? -1 / log(ρ) : Inf)
end

"""
    whitened_condition(H, Σ0) -> cond(H Σ0^{1/2})

Gappy-POD conditioning of a static array against the rank-r prior (Manohar et al. 2018).
"""
function whitened_condition(H::AbstractMatrix, Σ0::AbstractMatrix)
    E = eigen(Symmetric(Float64.(Matrix(Σ0))))
    return cond(Float64.(H) * (E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))))
end

"""
    gramian_condition(obs::VaryingObs, Σ0) -> cond(Σ0^{1/2} (Σ_k H_kᵀR_k⁻¹H_k) Σ0^{1/2})

Frozen-field observability of a whole moving-sensor track (a single 3×r H_k is rank-
deficient, so `whitened_condition` is meaningless per step). Eigenvalue ratio max/min.
"""
function gramian_condition(obs::VaryingObs, Σ0::AbstractMatrix)
    E   = eigen(Symmetric(Float64.(Matrix(Σ0))))
    S0h = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0))) * E.vectors'
    r   = obs_rank(obs)
    Gram = zeros(Float64, r, r)
    for (H, R) in zip(obs.H, obs.R)
        Gram .+= H' * (cholesky(Symmetric(R)) \ H)
    end
    eg = eigvals(Symmetric(S0h * Gram * S0h))
    return maximum(eg) / max(minimum(eg), eps(Float64))
end

_sym(A::AbstractMatrix) = (A + A') / 2
