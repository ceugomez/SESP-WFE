# kf.jl
# cgf cego6160@colorado.edu 6.19.26
include("fasteddy_io.jl") # include IO
include("measurement.jl") # include measurement
include("pod.jl")         # include POD structs
include("statmodel.jl")   # include prior/process-noise models

using LinearAlgebra

# Result of an assimilation run: the filtered coefficient trajectory and its covariances.
struct KFResult
	α      :: Matrix{Float32}          # (L, K) posterior mean coeffs at each assimilation step
	P      :: Vector{Matrix{Float32}}  # (K,) posterior covariances
	α_pred :: Matrix{Float32}          # (L, K) prior (forecast) mean before update
end

# Random-walk-in-subspace Kalman filter (F = I), assimilating a sequence of measurement
# sets. Prior, process noise Qα, and the subspace come from the offline ensemble.
#   predict:  α⁻ = α₊,           P⁻ = P₊ + Qα
#   update:   S = H P⁻ Hᵀ + R,   K = P⁻ Hᵀ S⁻¹
#             α₊ = α⁻ + K(y - H α⁻),  P₊ = (I - K H) P⁻
function runtime_loop(basis::PODBasis, meassq::Vector{measSet}, mdl::models,
                      domain::AnalysisDomain, mvar::Matrix{Float32})::KFResult
	L = basis.L         # no. of POD coefficients kept
	K = length(meassq)  # no. of measurement locations per observation time 
	α      = Matrix{Float32}(undef, L, K)   # coefficient vector
	α_pred = Matrix{Float32}(undef, L, K)   # predicted coefficient vector
	Ps     = Vector{Matrix{Float32}}(undef, K)  # covariance

	# initialise from the ensemble prior
	α_post = copy(mdl.μ_α0) # get initial mean coefficients
	P_post = copy(mdl.Σ_α0) # get initial coefficient covariance
	Iα     = Matrix{Float32}(I, L, L)  

	for k in 1:K
		# --- predict (F = I) ---
		α_minus = α_post
		P_minus = P_post + mdl.Qα

		# --- update ---
		ms = meassq[k]		# measurement at time k
		nm = length(ms.measurements)
		H  = build_measurement_operator(ms, basis, domain)   # (3nm × L)
		y  = stack_measurements(ms)                          # (3nm,)
		R  = build_measurement_covariance(nm, mvar)          # (3nm × 3nm)

		S    = Symmetric(H * P_minus * H' + R)
		Kgain = (P_minus * H') / S                           # (L × 3nm)
		α_post = α_minus + Kgain * (y - H * α_minus)
		P_post = (Iα - Kgain * H) * P_minus
		P_post = 0.5f0 * (P_post + P_post')                  # symmetrise

		α_pred[:, k] = α_minus
		α[:, k]      = α_post
		Ps[k]        = P_post
	end

	return KFResult(α, Ps, α_pred)
end
