# cgf cego6160@colorado.edu
# Gaussian Mixture Filter for POD coefficient space
using Distributions
using BlockDiagonals
include(joinpath(WORKDIR, "pod.jl"))            # POD utilities
include(joinpath(WORKDIR, "measurement.jl"))    # measurement utilities
include(joinpath(WORKDIR, "utils.jl"))          # shared filter code and evaluation metrics
# structure definition
struct GMFilter
    K       :: Int              # number of mixture components (one per ensemble member OR some arbitrary number)
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
function init_prior(basis::PODBasis, M::Int; β::Float64=0.05) :: GMFilter    # Equation 4
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

# Initialize a uniform Gaussian mixture prior over coefficient space.
# Component means are placed on a uniform grid spanning [min, max] of the
# ensemble member means for each coefficient independently. Within-component
# variance is set to ε · σ²_across so components are narrow relative to the
# total ensemble spread, making them discriminable by measurements.
#
# K        : number of mixture components (independent of ensemble size)
# ε        : within-component variance as a fraction of inter-member spread;
#            ~1/K keeps components approximately non-overlapping
function init_prior_uniform(basis::PODBasis, M::Int, K::Int; ε::Float64=0.1) :: GMFilter
    L, N = size(basis.coeffs)
    T = N ÷ M
    @assert T * M == N "N=$N is not evenly divisible by M=$M"

    # compute ensemble member means (same layout as init_prior)
    member_means = Matrix{Float32}(undef, M, L)
    for m in 1:M
        cols = (m-1)*T+1 : m*T
        member_means[m, :] = vec(mean(basis.coeffs[:, cols], dims=2))
    end

    # per-mode range and spread from ensemble member means
    ζ_min     = vec(minimum(member_means, dims=1))   # (L,)
    ζ_max     = vec(maximum(member_means, dims=1))   # (L,)
    σ²_across = vec(var(member_means, dims=1))       # (L,)

    # uniform grid: K points equally spaced across [ζ_min, ζ_max] per mode
    means = Matrix{Float32}(undef, K, L)
    for l in 1:L
        means[:, l] = Float32.(LinRange(ζ_min[l], ζ_max[l], K))
    end

    # within-component variance: ε · σ²_across, same for all components
    vars = repeat(Float32.(ε .* σ²_across)', K, 1)   # (K, L)

    return GMFilter(K, L, fill(1.0/K, K), means, vars)
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
            var_update[j,:] =  diag((I - Kgain * H) * Sigma_j)          # Σ_k⁺ = (I - K_k * H) * Σ_k (update variance, keep only diagonal)
            log_likelihood = logpdf(MvNormal(predicted_measurements, innov_cov), Yk)    # update weights (log space to avoid underflow)
            raw_weights[j] =  log(gm[i].weights[j]) + log_likelihood
            if !isnothing(Q)                                                                                                                                                                                   
            # inflate variances by process noise
            var_update[j, :] .+= diag(Q)                                                                                                                                                                   
            end  
        end

        log_w = raw_weights .- maximum(raw_weights)          # log-sum-exp: shift for numerical stability
        new_weights = exp.(log_w) ./ sum(exp.(log_w))        # normalize weights
        new_weights ./= sum(new_weights)                     # renormalize
        push!(gm, GMFilter(gm[i].K, gm[i].L, new_weights, mean_update, var_update))
    end
    return gm
end

# ══════════════════════════════════════════════════════════════════════════════
# Psiaki resampling: runtime loop with mixand narrowing
# After each Kalman update, components whose variances have dropped below
# P_max are split into daughter mixands with bounded covariances.
# Similar components are then merged (Runnalls KL) to control count growth.
# ══════════════════════════════════════════════════════════════════════════════

# Resample a single component into n_daughters daughters.
# Each daughter has covariance clamped to P_max (per-mode floor).
# Means are sampled from the parent Gaussian; weights are split equally.
function resample_component(μ::Vector{Float32}, σ²::Vector{Float32},
                            weight::Float64, P_max::Vector{Float32};
                            n_daughters::Int=3)
    L = length(μ)
    # Clamp daughter variances to the floor
    σ²_daughter = max.(σ², P_max)
    # Sample daughter means from the parent distribution
    daughter_means = Matrix{Float32}(undef, n_daughters, L)
    daughter_vars  = Matrix{Float32}(undef, n_daughters, L)
    daughter_weights = fill(weight / n_daughters, n_daughters)
    for d in 1:n_daughters
        # Sample from N(μ, σ²_parent - σ²_daughter) to spread daughters out,
        # but only along modes where we actually clamped
        Δσ² = max.(σ² .- σ²_daughter, 0f0)  # spreading variance (zero if not clamped)
        if d == 1
            daughter_means[d, :] = μ  # first daughter keeps the parent mean
        else
            daughter_means[d, :] = μ .+ Float32.(randn(L)) .* sqrt.(Δσ²)
        end
        daughter_vars[d, :] = σ²_daughter
    end
    return daughter_weights, daughter_means, daughter_vars
end

# KL divergence upper bound for merging two diagonal Gaussians (Runnalls 2007).
# For diagonal covariances this simplifies considerably.
function runnalls_kl_cost(w_i::Float64, μ_i::Vector{Float32}, σ²_i::Vector{Float32},
                          w_j::Float64, μ_j::Vector{Float32}, σ²_j::Vector{Float32})
    w_ij = w_i + w_j
    # Moment-preserving merged covariance (diagonal)
    α_i = w_i / w_ij
    α_j = w_j / w_ij
    μ_ij = α_i .* μ_i .+ α_j .* μ_j
    σ²_ij = α_i .* σ²_i .+ α_j .* σ²_j .+ α_i * α_j .* (μ_i .- μ_j).^2
    # KL cost: 0.5 * w_ij * [ sum(log(σ²_ij / σ²_k) + σ²_k/σ²_ij - 1) for weighted mix ]
    # Simplified: use the trace + log-det form for each component against the merged
    cost = 0.0
    for l in 1:length(μ_i)
        cost += w_i * (log(σ²_ij[l] / σ²_i[l]) + σ²_i[l]/σ²_ij[l] - 1 +
                       (μ_i[l] - μ_ij[l])^2 / σ²_ij[l])
        cost += w_j * (log(σ²_ij[l] / σ²_j[l]) + σ²_j[l]/σ²_ij[l] - 1 +
                       (μ_j[l] - μ_ij[l])^2 / σ²_ij[l])
    end
    return 0.5 * cost
end

# Merge the two most similar components until count <= K_max.
function merge_to_limit(weights::Vector{Float64}, means::Matrix{Float32},
                        vars::Matrix{Float32}, K_max::Int)
    K = length(weights)
    while K > K_max
        best_cost = Inf
        best_i, best_j = 1, 2
        for i in 1:K, j in (i+1):K
            c = runnalls_kl_cost(weights[i], vec(means[i,:]), vec(vars[i,:]),
                                weights[j], vec(means[j,:]), vec(vars[j,:]))
            if c < best_cost
                best_cost = c
                best_i, best_j = i, j
            end
        end
        # Moment-preserving merge
        w_new = weights[best_i] + weights[best_j]
        α_i = weights[best_i] / w_new
        α_j = weights[best_j] / w_new
        μ_new = α_i .* means[best_i,:] .+ α_j .* means[best_j,:]
        σ²_new = α_i .* vars[best_i,:] .+ α_j .* vars[best_j,:] .+
                 α_i * α_j .* (means[best_i,:] .- means[best_j,:]).^2
        # Replace i with merged, remove j
        weights[best_i] = w_new
        means[best_i,:] = μ_new
        vars[best_i,:]  = σ²_new
        keep = setdiff(1:K, best_j)
        weights = weights[keep]
        means   = means[keep, :]
        vars    = vars[keep, :]
        K -= 1
    end
    return weights, means, vars
end

# Runtime loop with Psiaki-style resampling for mixand narrowing and
# likelihood tempering to control weight concentration rate.
#
# τ (tau): likelihood tempering exponent in (0, 1]. The log-likelihood
#          contribution to the weight update is scaled by τ, so that
#          raw_weight_j = log(w_j) + τ * logpdf(...). At τ=1 this recovers
#          the standard Bayesian update; smaller τ slows weight collapse,
#          keeping the mixture alive longer with sparse observations.
# P_max_frac: fraction of prior variance to use as the per-mode variance floor.
# K_max: maximum number of components after merge step.
function runtime_loop_resample(basis::PODBasis, prior::GMFilter, Y::Vector{measSet},
                               mvar::Matrix{Float32}, Q=nothing;
                               τ::Float64=0.0, P_max_frac::Float64=0.1,
                               K_max::Int=20)::Vector{GMFilter}
    # Compute per-mode variance floor from the prior
    P_max = Float32.(P_max_frac .* vec(mean(prior.vars, dims=1)))

    gm = [prior]
    for i in 1:length(Y)
        Yk = vcat([m.val for m in Y[i].measurements]...)
        H = make_H_matrix(basis, Y[i])
        nmeas = length(Y[i].measurements)
        R_meas = kron(I(nmeas), mvar)

        cur = gm[i]
        mean_update = Matrix{Float32}(undef, cur.K, cur.L)
        var_update  = Matrix{Float32}(undef, cur.K, cur.L)
        raw_weights = Vector{Float64}(undef, cur.K)

        for j in 1:cur.K
            predicted_measurements = H * cur.means[j,:]
            innov = Yk - predicted_measurements
            Sigma_j = diagm(cur.vars[j, :])
            innov_cov = H * Sigma_j * H' + R_meas
            innov_cov = (innov_cov + innov_cov') / 2
            Kgain = Sigma_j * H' * inv(innov_cov)
            mean_update[j, :] = cur.means[j,:] + Kgain * innov
            var_update[j,:]  = diag((I - Kgain * H) * Sigma_j)
            log_likelihood = logpdf(MvNormal(predicted_measurements, innov_cov), Yk)
            raw_weights[j] = log(cur.weights[j]) + τ * log_likelihood
            if !isnothing(Q)
                var_update[j, :] .+= diag(Q)
            end
        end

        log_w = raw_weights .- maximum(raw_weights)
        new_weights = exp.(log_w) ./ sum(exp.(log_w))
        new_weights ./= sum(new_weights)

        # ── Psiaki resampling: split components that are too narrow ──────
        all_weights = Float64[]
        all_means   = Matrix{Float32}(undef, 0, cur.L)
        all_vars    = Matrix{Float32}(undef, 0, cur.L)

        for j in 1:cur.K
            if any(var_update[j, :] .< P_max)
                dw, dm, dv = resample_component(
                    vec(mean_update[j,:]), vec(var_update[j,:]),
                    new_weights[j], P_max)
                append!(all_weights, dw)
                all_means = vcat(all_means, dm)
                all_vars  = vcat(all_vars, dv)
            else
                push!(all_weights, new_weights[j])
                all_means = vcat(all_means, mean_update[j:j,:])
                all_vars  = vcat(all_vars, var_update[j:j,:])
            end
        end

        # ── Merge back down to K_max if needed ───────────────────────────
        if length(all_weights) > K_max
            all_weights, all_means, all_vars = merge_to_limit(
                all_weights, all_means, all_vars, K_max)
        end
        all_weights ./= sum(all_weights)

        K_new = length(all_weights)
        push!(gm, GMFilter(K_new, cur.L, all_weights,
                           Matrix{Float32}(all_means),
                           Matrix{Float32}(all_vars)))
    end
    return gm
end
