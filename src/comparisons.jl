# cgf cego6160@colorado.edu 
# function set to run comparative metrics for GMF
# 4.9.26

# 3 comparisons:
#   Unregularized LS; does direct fit to the data (ignores prior)
#   Single-gaussian KF; uses a big gigantic prior (mean and var of ensemble)
#   (stretch) windseer
include(joinpath(WORKDIR, "measurement.jl"))

# get least squares solution
function leastsquares(basis::PODBasis, Y::Vector{measSet})::Vector{Float64}
	# given a set of measurements, find best coefficient vector ζ with least-squares
	Yk = vcat([vcat([m.val for m in yset.measurements]...) for yset in Y]...)  # stack all the measurements; this is a time-invariant solution anyway
	H = vcat([make_H_matrix(basis, yset) for yset in Y]...)
	return H \ Yk
end

# Single-component Kalman filter for comparison against GMF.
# Initialises from the ensemble-wide mean and variance (collapsed single Gaussian prior).
# Accepts optional process noise Q (diagonal Matrix{Float64}) for dynamic tracking.
function kf(basis::PODBasis, prior::GMFilter, Y::Vector{measSet},
            mvar::Matrix{Float32}, Q=nothing)::Vector{Tuple{Vector{Float64}, Vector{Float64}}}
    L = basis.L
    # Collapse GMF prior to a single Gaussian via the law of total variance:
    #   σ²_total = E[σ²_k] + Var[μ_k]
    # The first term is the mean within-component variance; the second is the
    # between-component spread. Omitting the second term would make the KF
    # artificially overconfident relative to the GMF.
    μ  = vec(Float64.(prior.weights' * prior.means))                       # (L,)
    mean_var = vec(Float64.(prior.weights' * prior.vars))                  # E[σ²_k]
    sq_means = vec(Float64.(prior.weights' * (prior.means .^ 2)))         # E[μ_k²]
    σ² = mean_var .+ (sq_means .- μ.^2)                                   # law of total variance

    history = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    push!(history, (copy(μ), copy(σ²)))

    for i in 1:length(Y)
        Yk   = Float64.(vcat([m.val for m in Y[i].measurements]...))
        H    = Float64.(make_H_matrix(basis, Y[i]))
        nmeas = length(Y[i].measurements)
        R    = kron(I(nmeas), Float64.(mvar))

        Sigma = diagm(σ²)
        S     = H * Sigma * H' + R
        S     = (S + S') / 2
        K     = Sigma * H' * inv(S)
        μ     = μ + K * (Yk - H * μ)
        σ²    = diag((I - K * H) * Sigma)

        if !isnothing(Q)
            σ² .+= diag(Q)
        end

        push!(history, (copy(μ), copy(σ²)))
    end
    return history
end

# Compare GMF, KF, and LS estimators over a sequence of timesteps.
# tidx_seq: truth snapshot index for each assimilation step (length == number of steps).
#           Pass fill(tidx, n) for a static field or collect(1:n) for a dynamic one.
function compare_estimators(filter_history::Vector{GMFilter},
                            kf_history::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                            α_ls::Vector{Float64},
                            basis::PODBasis, truth_field::NCfield,
                            tidx_seq::Vector{Int})
    n = length(tidx_seq)
    rmse_gmf = Vector{Float64}(undef, n)
    rmse_kf  = Vector{Float64}(undef, n)
    rmse_ls  = Vector{Float64}(undef, n)

    for (i, tidx) in enumerate(tidx_seq)
        truth = get_truth_state(truth_field, tidx)

        # GMF posterior mean at step i (filter_history[1] is prior, so step i → index i+1)
        gm = filter_history[i+1]
        α_gmf = vec(Float64.(gm.weights' * gm.means))
        rmse_gmf[i], _ = field_errors(basis.modes * Float32.(α_gmf), truth)

        # KF posterior mean at step i
        α_kf_i = kf_history[i+1][1]
        rmse_kf[i], _  = field_errors(basis.modes * Float32.(α_kf_i), truth)

        # LS is time-invariant: same estimate at every step
        rmse_ls[i], _  = field_errors(basis.modes * Float32.(α_ls), truth)
    end

    @info "Final RMSE — GMF: $(round(rmse_gmf[end], digits=4)) | KF: $(round(rmse_kf[end], digits=4)) | LS: $(round(rmse_ls[end], digits=4)) m/s"
    @info "GMF weight distribution (top 5): $(sort(filter_history[end].weights, rev=true)[1:min(5,length(filter_history[end].weights))])"

    truth_coeffs = basis.modes' * get_truth_state(truth_field, tidx_seq[1])
    dists = [norm(filter_history[1].means[j,:] .- truth_coeffs) for j in 1:filter_history[1].K]
    @info "Prior distances to truth: min=$(minimum(dists)), winning component=$(argmin(dists))"

    return rmse_gmf, rmse_kf, rmse_ls
end
