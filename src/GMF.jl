# cgf cego6160@colorado.edu
# Gaussian Mixture Filter for POD coefficient space
using Distributions
using BlockDiagonals
include(joinpath(WORKDIR, "pod.jl"))            # POD utilities
include(joinpath(WORKDIR, "measurement.jl"))

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
function init_prior(basis::PODBasis, M::Int) :: GMFilter    # Equation 4
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

    weights = fill(1.0/M, M)            # equal weight given to each ensemble member for now

    return GMFilter(M, L, weights, means, vars)
end


# runtime loop -  
function runtime_loop(truth_field::NCfield, basis::PODBasis, prior::GMFilter, max_iter::Int)::Vector{GMFilter}
    # U - true field, to sample from
    # basis - POD basis from the ensemble model order reduction 
    # pr - Gaussian mixture prior over coefficients
    #   - Predicted obs: ŷ_k = H * μ_k
    #   - Innovation: ν_k = Y - ŷ_k
    #   - Innovation covariance: S_k = H * Σ_k * H' + R (Σ_k is full L×L per-component covariance)
    #   - Kalman gain: K_k = Σ_k * H' * inv(S_k)
    #   - Updated mean: μ_k⁺ = μ_k + K_k * ν_k
    #   - Updated covariance: Σ_k⁺ = (I - K_k * H) * Σ_k
    #   - Updated weight: w_k⁺ ∝ w_k * 𝒩(Y; ŷ_k, S_k) — this is the likelihood of the observation under component k
    println("Initializing GM Filter...")
    truth_tidx = 25         # Always keep the truth field at the same timestep (truth field has 53 snapshots)
    n_measurements = 30;    # 3 observations
    measurement_variance = diagm(Float32[0.01, 0.01, 0.001])  # uncorrelated noise in each component of the wind direction
    n_components = prior.K  # number of components in each gaussian mixture (same as no. of ensemble members)

    # project truth snapshot into coefficient space for RMSE diagnostics
    u_snap = vec(truth_field.u[:,:,:,truth_tidx])
    v_snap = vec(truth_field.v[:,:,:,truth_tidx])
    w_snap = vec(truth_field.w[:,:,:,truth_tidx])
    truth_state = Float32[u_snap; v_snap; w_snap]                   # (state_dim,) ground truth state vector
    truth_coeffs = basis.modes' * truth_state                       # (L,) — kept for reference, not used in diagnostics

    gm = [prior]    # make a vector of loops
    # Run the loop N times, assimilating new observations each time
    for i in 1:max_iter
        # get randomly sampled location set
        measurement_locations = randLocsInBounds(grid, n_measurements)               # sample n_meas random locations from the grid
        measurements = get_measurement_set(truth_field, grid, measurement_locations, measurement_variance, truth_tidx)  # sample n_measurements from truth_field with noise 
        Y = vcat([m.val for m in measurements.measurements]...)                     # stack 'em
        # get H matrix for measurement set
        H = make_H_matrix(basis, measurements)
        R_meas = kron(I(n_measurements), measurement_variance)   # measurement noise covariance (3n_meas × 3n_meas)
        # preallocate
        mean_update = Vector{Vector{Float32}}()
        var_update  = Vector{Vector{Float32}}()
        raw_weights = Vector{Float64}()

        # Kalman update loop, for all components
        for j in 1:n_components
            # get predicted measurements
            # ŷ_k = H * μ_k
            predicted_measurements = H*gm[i].means[j,:]                     # measurements predicted from mean component values for coefficient j
            # compute innovation
            # ν_k = Y - ŷ_k
            innov = Y - predicted_measurements   # difference btwn predicted and observed
            # diagonal covariance for this component
            Sigma_j = diagm(gm[i].vars[j, :])
            # get innovation covariance
            # S_k = H * Σ_k * H' + R
            innov_cov = H * Sigma_j * H' + R_meas
            innov_cov = (innov_cov + innov_cov') / 2   # symmetrize to correct Float32 rounding
            # compute kalman gain
            # K_k = Σ_k * H' * inv(S_k)
            Kgain = Sigma_j * H' * inv(innov_cov)
            # update the means
            # μ_k⁺ = μ_k + K_k * ν_k
            push!(mean_update, gm[i].means[j,:] + Kgain * innov)
            # update the variance (keep diagonal only)
            # Σ_k⁺ = (I - K_k * H) * Σ_k
            push!(var_update, diag((I - Kgain * H) * Sigma_j))
            # update the weights for each component (log-space to avoid underflow in high dimensions)
            log_likelihood = logpdf(MvNormal(predicted_measurements, innov_cov), Y)
            push!(raw_weights, log(gm[i].weights[j]) + log_likelihood)
        end
        mean_update = reduce(hcat, mean_update)'                            # K × L
        var_update  = reduce(hcat, var_update)'                             # K × L
        log_w = raw_weights .- maximum(raw_weights)          # log-sum-exp: shift for numerical stability
        new_weights = exp.(log_w) ./ sum(exp.(log_w))        # normalize weights
        # weight floor: prevent component collapse by enforcing a minimum weight per component
        w_min = 1.0 / (3.0 * n_components)
        new_weights = max.(new_weights, w_min)
        new_weights ./= sum(new_weights)                     # renormalize
        push!(gm, GMFilter(gm[i].K, gm[i].L, new_weights, mean_update, var_update))

        # diagnostics
        post_mean   = vec(new_weights' * mean_update)                       # weighted mean over components (L,)
        mean_spread = sum(new_weights[j] * sum((mean_update[j, :] .- post_mean).^2) for j in 1:n_components)
        total_var   = sum(new_weights[j] * sum(var_update[j, :]) for j in 1:n_components) + mean_spread
        reconstructed    = basis.modes * post_mean                      # (state_dim,) reconstructed wind field
        field_rmse       = sqrt(mean((reconstructed .- truth_state).^2))  # RMSE in m/s
        field_max_abs_error = maximum(abs.(reconstructed .- truth_state))
        @info "iter $i / $max_iter  |  total var: $(round(total_var, digits=4))  |  field RMSE: $(round(field_rmse, digits=4)) m/s | max abs error: $(round(field_max_abs_error,digits=4)) m/s"
    end
    return gm
end



function make_H_matrix(basis::PODBasis, mset::measSet)::Matrix{Float32}
	n = basis.grid.n_grid
	n_meas = length(mset.measurements)
	H = Matrix{Float32}(undef, 3 * n_meas, basis.L)          # preallocate, should be 3n_meas x # of POD modes 
	for (i, m) in enumerate(mset.measurements)
		ix, iy, iz = m.locidx
		flat_u = LinearIndices((basis.grid.Nx, basis.grid.Ny, basis.grid.Nz))[ix, iy, iz]
		flat_v = flat_u + n
		flat_w = flat_u + 2n
		H[3(i-1)+1, :] = basis.modes[flat_u, :]   # u row                                                                                                                                              
		H[3(i-1)+2, :] = basis.modes[flat_v, :]   # v row                                                                                                                                              
		H[3(i-1)+3, :] = basis.modes[flat_w, :]   # w row                                                                                                                                              
	end
	return H
end
