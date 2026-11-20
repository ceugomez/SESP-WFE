# statmodel.jl
# cgf cego6160@colorado.edu 5.19.26
#
# The offline statistical models of the POD coefficients, regressed from the ensemble:
# prior mean/covariance (manuscript Eqs. 14–15) and random-walk process noise (Eq. 13).
# The representativeness-error model is in reperror.jl.
include("pod.jl");

# Prior + process noise for a full-rank (L) basis. kalman.jl truncates to rank r.
struct models
    Qα     :: Matrix{Float32}   # (L, L) process-noise covariance of α per snapshot interval (6 s)
    μ_α0   :: Vector{Float32}   # (L,)   prior mean of α
    Σ_α0   :: Matrix{Float32}   # (L, L) prior covariance of α
end


function get_prior_and_process_noise(basis::PODBasis)
    μ_α0, Σ_α0 = get_prior_from_ensemble(basis)
    Qα         = get_process_noise(basis)
    return models(Qα, μ_α0, Σ_α0)
end

# Companion-file convention for the prior/process-noise model, saved alongside the basis
# (truth_<id>_basis.jld2) so a downstream run can load it instead of recomputing.
function models_filename(output_dir::String, truth_id::String) :: String
    return joinpath(output_dir, "truth_$(truth_id)_models.jld2")
end

function save_models(mdl::models, filepath::String)
    jldsave(filepath; Qα=mdl.Qα, μ_α0=mdl.μ_α0, Σ_α0=mdl.Σ_α0)
    @info "Saved prior/process-noise model to $filepath"
end

function load_models(filepath::String) :: models
    d = load(filepath)
    return models(d["Qα"], d["μ_α0"], d["Σ_α0"])
end

# Prior mean/covariance of the POD coefficients α, pooled over all members and times.
# Mean over coefficient set in ensemble + variance over coefficient set in ensemble
#
#   α_0 = (1 / (M·N_t)) Σ_{i,n} α^(i)(t_n)
#   P_0 = (1 / (M·N_t - 1)) Σ_{i,n} (α^(i)(t_n) - α_0)(α^(i)(t_n) - α_0)'
#
# basis.coeffs is (L, N) with N = M·N_t columns (member-blocks of post-spinup snapshots),
# so this is just the column-wise mean and unbiased sample covariance.
function get_prior_from_ensemble(basis::PODBasis)
    a    = basis.coeffs                              # (L, N), N = M·N_t
    μ_α0 = vec(mean(a; dims=2))                       # (L,)
    Σ_α0 = cov(a; dims=2, corrected=true)             # (L, L), divides by N-1
    return Float32.(μ_α0), Float32.(Σ_α0)
end


# Process-noise covariance Q_α for the random-walk-in-subspace SSM, built from the
# ensemble's temporal increments of the POD coefficients:
#
#   Δα_n^(i) = α^(i)(t_{n+1}) - α^(i)(t_n)
#   Q_α = (1/M) Σ_{i=1}^{M} [ 1/(N_t-1) Σ_n (Δα_n^(i) - mean_n Δα^(i))(·)' ]
#
# i.e. for each member compute the sample covariance of its own increments (about that
# member's mean increment), then average those per-member covariances over the M members.
# basis.coeffs is (L, N) in equal member-blocks of T = N/M columns ordered by time.
#
# The lag is ONE snapshot interval (6 s). Drivers assimilating every s-th snapshot scale this
# by s (`qscale`), which is exact only for uncorrelated increments; measured on the highvar
# truth basis the true lag-s increment variance differs from s·Qα by 0.14–9× per mode at
# s = 5–20. Compute Q at the actual lag before trusting strided runs.
function get_process_noise(basis::PODBasis)
    a = basis.coeffs                                  # (L, N)
    L = basis.L
    M = length(basis.member_ids)
    N = size(a, 2)
    @assert N % M == 0 "coeffs columns ($N) not divisible by member count ($M)"
    T = N ÷ M                                          # N_t snapshots per member
    @assert T >= 2 "need >=2 snapshots per member to form increments"

    Qα = zeros(Float64, L, L)
    for mi in 1:M
        cols  = (mi-1)*T+1 : mi*T
        Δa    = Float64.(a[:, cols][:, 2:end] .- a[:, cols][:, 1:end-1])   # (L, T-1)
        Qα  .+= cov(Δa; dims=2, corrected=true)                            # per-member, /(N_t-1)
    end
    Qα ./= M
    return Float32.(Qα)
end
