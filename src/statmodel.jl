# statmodel.jl
# cgf cego6160@colorado.edu 5.19.26
# regress statistical evolution, prior, measurement error model from ensemble set
include("pod.jl");

struct models
    Qα     :: Matrix{Float32}   # LxL matrix of process noise covariances for evolution of coeff α
    μ_α0   :: Vector{Float32}   # Mx1 vector of prior means for coeff α (maybe just a normal dist)
    Σ_α0   :: Matrix{Float32}   # MxM prior covariance matrix for coeff α  
end


function get_prior_and_process_noise(basis::PODBasis)
    μ_α0, Σ_α0 = get_prior_from_ensemble(basis)
    Qα         = get_process_noise(basis)
    return models(Qα, μ_α0, Σ_α0)
end

# Prior mean/covariance of the POD coefficients α, pooled over all members and times.
#
#   α_0 = (1 / (M·N_t)) Σ_{i,n} α^(i)(t_n)
#   P_0 = (1 / (M·N_t - 1)) Σ_{i,n} (α^(i)(t_n) - α_0)(α^(i)(t_n) - α_0)'
#
# basis.coeffs is (L, N) with N = M·N_t columns (member-blocks of post-spinup snapshots),
# so this is just the column-wise mean and unbiased sample covariance
function get_prior_from_ensemble(basis::PODBasis)
    a    = basis.coeffs                              # (L, N), N = M·N_t
    μ_α0 = vec(mean(a; dims=2))                       # (L,)
    Σ_α0 = cov(a; dims=2, corrected=true)             # (L, L), divides by N-1
    return Float32.(μ_α0), Float32.(Σ_α0)
end

function get_truncation_measurment_error(meas::measSet, )
    # gets truncation error for each location
    # stub


    return nothing
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
