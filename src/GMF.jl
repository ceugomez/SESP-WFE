# cgf cego6160@colorado.edu
# Gaussian Mixture Filter for POD coefficient space
using Distributions
include(joinpath(WORKDIR, "pod.jl"))            # POD utilities
# structure definition
struct GMFilter
    K       :: Int              # number of mixture components (one per ensemble member)
    L       :: Int              # number of POD modes
    weights :: Vector{Float32}  # length K, sum to 1
    means   :: Matrix{Float32}  # K × L
    vars    :: Matrix{Float32}  # K × L (diagonal covariance assumption)
end

# Initialize equal-weight GMF prior from POD basis.
# Each ensemble member becomes one mixture component; its mean and variance
# are the temporal statistics of that member's POD coefficients.
#
# basis.coeffs : (L, N) — rows are POD modes, columns are snapshots.
#                Snapshots are stacked member-major: columns 1:T are member 1,
#                columns T+1:2T are member 2, etc.
# basis.modes  : (state_dim, L) — rows are spatial DOFs ([u;v;w] flattened),
#                columns are orthonormal mode shapes ψ_k.
#
# M = number of ensemble members, T = timesteps per member, N = M*T.
function init_prior(basis::PODBasis, M::Int) :: GMFilter
    L, N = size(basis.coeffs)
    T = N ÷ M
    @assert T * M == N "N=$N is not evenly divisible by M=$M"

    means = Matrix{Float32}(undef, M, L)
    vars  = Matrix{Float32}(undef, M, L)

    for m in 1:M
        cols = (m-1)*T+1 : m*T                     # this member's timestep columns
        a_m  = basis.coeffs[:, cols]                # (L, T)
        means[m, :] = vec(mean(a_m, dims=2))        # temporal mean per mode
        vars[m, :]  = vec(var(a_m, dims=2))         # temporal variance per mode
    end

    weights = fill(Float32(1/M), M)     # equal weight given to each ensemble member for now

    return GMFilter(M, L, weights, means, vars) # sum of gaussian with mean and variance for each 
end

