# cgf cego6160@colorado.edu
# Gaussian Mixture Filter for POD coefficient space
using Distributions
using BlockDiagonals
include(joinpath(WORKDIR, "pod.jl"))            # POD utilities
include(joinpath(WORKDIR, "measurement.jl"))    # measurement utilities
include(joinpath(WORKDIR, "utils.jl"))          # shared filter code and evaluation metrics
# structure definition
struct GMFilter
    K       :: Int              # number of mixture components (one per ensemble member)
    L       :: Int              # number of POD modes
    weights :: Vector{Float64}  # length K, sum to 1
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
function init_prior(basis::PODBasis, M::Int; β::Float64=0.1) :: GMFilter    # Equation 4
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

    # Inflate component variances by a fraction of the inter-member spread.
    # For leading modes, temporal variance is tiny (turbulent wobble) while
    # inter-member spread is large (different wind directions). Without
    # inflation the components are delta-like spikes with near-zero density
    # at the truth, causing weight collapse onto the wrong component.
    σ²_across = vec(var(means, dims=1))             # (L,) variance of component means per mode
    for m in 1:M
        vars[m, :] .+= Float32.(β .* σ²_across)
    end

    weights = fill(1.0/M, M)            # equal weight given to each ensemble member for now

    return GMFilter(M, L, weights, means, vars)
end

# runtime loop for gaussian mixture filter
# takes POD basis (also used in the above), the prior, and measurements + measurement characteristics
function runtime_loop(basis::PODBasis, prior::GMFilter, Y::Vector{measSet}, mvar::Matrix{Float32}, Q=nothing)::Vector{GMFilter}
    # basis - POD basis from the ensemble model order reduction 
    # prior - Gaussian mixture prior over coefficients
    # Y     - vector of measurement sets for each timestep
    # let's get into it...
    gm = [prior]    # make a vector of GMFilters
    # Run the loop as many times as there are measurements
    for i in 1:length(Y)
        Yk = vcat([m.val for m in Y[i].measurements]...)     # stack measurements at this time index
        H = make_H_matrix(basis, Y[i])                       # get H matrix for measurement set
        nmeas = length(Y[i].measurements)
        R_meas = kron(I(nmeas), mvar)                # measurement noise covariance (3n_meas × 3n_meas)
        # preallocate
        mean_update = Matrix{Float32}(undef, gm[i].K, gm[i].L)                                                                                                                                                                                                                            
        var_update  = Matrix{Float32}(undef, gm[i].K, gm[i].L)
        raw_weights = Vector{Float64}(undef, gm[i].K) 
        # Kalman update loop
        for j in 1:gm[i].K      # for all components
            predicted_measurements = H*gm[i].means[j,:]                 # ŷ_k = H * μ_k             (measurements predicted from mean component values for coefficient j)
            innov = Yk - predicted_measurements                         # ν_k = Y - ŷ_k             (innovation, difference btwn predicted and observed)
            Sigma_j = diagm(gm[i].vars[j, :])                           # diagonal covariance for component
            innov_cov = H * Sigma_j * H' + R_meas                       # S_k = H * Σ_k * H' + R    (get innovation covariance)
            innov_cov = (innov_cov + innov_cov') / 2                    # symmetrize to             (correct Float32 rounding)
            Kgain = Sigma_j * H' * inv(innov_cov)                       # K_k = Σ_k * H' * inv(S_k) (compute kalman gain)
            mean_update[j, :] = gm[i].means[j,:] + Kgain * innov        # μ_k⁺ = μ_k + K_k * ν_k    (update means w/ kgain)
            var_update[j,:] =  diag((I - Kgain * H) * Sigma_j)            # Σ_k⁺ = (I - K_k * H) * Σ_k (update variance, keep only diagonal)
            log_likelihood = logpdf(MvNormal(predicted_measurements, innov_cov), Yk)    # update weights (log space to avoid underflow)
            raw_weights[j] =  log(gm[i].weights[j]) + log_likelihood
            if !isnothing(Q)                                                                                                                                                                                   
            # inflate variances by process noise
            var_update[j, :] .+= diag(Q)                                                                                                                                                                   
            end  
        end

        log_w = raw_weights .- maximum(raw_weights)          # log-sum-exp: shift for numerical stability
        new_weights = exp.(log_w) ./ sum(exp.(log_w))        # normalize weights
        # w_min = 1.0 / (3.0 * gm[i].K)                      # weight floor: prevent component collapse by enforcing a minimum weight per component
        # new_weights = max.(new_weights, w_min)             # compute new weights
        new_weights ./= sum(new_weights)                     # renormalize
        push!(gm, GMFilter(gm[i].K, gm[i].L, new_weights, mean_update, var_update))
    end
    return gm
end
