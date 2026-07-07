# kf.jl
# cgf cego6160@colorado.edu 6.19.26
include("fasteddy_io.jl") # include IO
include("pod.jl")         # include POD structs (before measurement: it annotates ::PODBasis)
include("measurement.jl") # include measurement
include("statmodel.jl")   # include prior/process-noise models
include("reperror.jl")    # include representativeness (reconstruction-error) model

using LinearAlgebra

# Result of an assimilation run: the filtered coefficient trajectory, its covariances, and
# the per-step innovation diagnostics used to check filter consistency (χ²/whiteness).
struct KFResult
	α      :: Matrix{Float32}          # (L, K) posterior mean coeffs at each assimilation step
	P      :: Vector{Matrix{Float32}}  # (K,) posterior covariances
	α_pred :: Matrix{Float32}          # (L, K) prior (forecast) mean before update
	P_pred :: Vector{Matrix{Float32}}  # (K,) prior (forecast) covariances P⁻
	innov  :: Vector{Vector{Float32}}  # (K,) innovation ν = y − Hα⁻ (length 3·nm_k, may vary by step)
	S      :: Vector{Matrix{Float32}}  # (K,) innovation covariance S = H P⁻ Hᵀ + R
	nis    :: Vector{Float32}          # (K,) normalized innovation squared νᵀS⁻¹ν / dim(ν)
end

# Random-walk-in-subspace Kalman filter (F = I), assimilating a sequence of measurement
# sets. Prior, process noise Qα, and the subspace come from the offline ensemble.
#   predict:  α⁻ = α₊,           P⁻ = P₊ + Qα
#   update:   S = H P⁻ Hᵀ + R,   K = P⁻ Hᵀ S⁻¹
#             α₊ = α⁻ + K(y - H α⁻),  P₊ = (I - K H) P⁻
# `rep`: optional representativeness model. When supplied the observation-error covariance
# is inflated, R_total = R_sensor + R_rep, so the filter stays consistent despite the rank-r
# truncation (the unrepresented subspace is folded into R). Pass nothing for sensor-only R.
function runtime_loop(basis::PODBasis, meassq::Vector{measSet}, mdl::models,
                      domain::AnalysisDomain, mvar::Matrix{Float32};
                      rep::Union{RepError,Nothing}=nothing)::KFResult
	L = basis.L         # no. of POD coefficients kept
	K = length(meassq)  # no. of measurement locations per observation time 
	α      = Matrix{Float32}(undef, L, K)   # coefficient vector
	α_pred = Matrix{Float32}(undef, L, K)   # predicted coefficient vector
	Ps     = Vector{Matrix{Float32}}(undef, K)  # posterior covariance
	Pp     = Vector{Matrix{Float32}}(undef, K)  # prior (forecast) covariance
	innov  = Vector{Vector{Float32}}(undef, K)  # innovations ν
	Ss     = Vector{Matrix{Float32}}(undef, K)  # innovation covariances S
	nis    = Vector{Float32}(undef, K)          # normalized innovation squared

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
		R  = build_measurement_covariance(nm, mvar)          # (3nm × 3nm) sensor noise
		isnothing(rep) || (R += build_rep_covariance(ms, rep, domain))   # + representativeness

		ν    = y - H * α_minus                               # innovation
		S    = Symmetric(H * P_minus * H' + R)
		Kgain = (P_minus * H') / S                           # (L × 3nm)
		α_post = α_minus + Kgain * ν
		P_post = (Iα - Kgain * H) * P_minus
		P_post = 0.5f0 * (P_post + P_post')                  # symmetrise

		α_pred[:, k] = α_minus
		α[:, k]      = α_post
		Ps[k]        = P_post
		Pp[k]        = Matrix(P_minus)
		innov[k]     = ν
		Ss[k]        = Matrix(S)
		nis[k]       = Float32((ν' * (S \ ν)) / length(ν))   # NIS: ≈1 when consistent
	end

	return KFResult(α, Ps, α_pred, Pp, innov, Ss, nis)
end

# ---------------------------------------------------------------------------
# Persistence + performance diagnostics
# ---------------------------------------------------------------------------

# Standard filter-history filename, tagged by which member was treated as truth.
kf_result_filename(output_dir::String, truth_id::String) =
	joinpath(output_dir, "kf_result_truth_$(truth_id).jld2")

# Save the full filter history and the quantities needed to evaluate performance.
# truth_U is the truth member's snapshot matrix over the analysis domain (state_dim × K),
# column k aligned with assimilation step k; truth_coeffs = project(basis, truth_U) is the
# oracle (best in-subspace) coefficient trajectory. Computes, per step:
#   coeff_rmse   — ‖α_k − a_k^truth‖ / √L   (subspace estimation error)
#   field_rmse   — ‖Prα_k − x_k^truth‖ / √d  (physical reconstruction error of the filter)
#   oracle_rmse  — ‖Pra_k^truth − x_k^truth‖/√d (projection floor: irreducible truncation error)
# field_rmse can't beat oracle_rmse; the gap above the floor is the online filter's own error.
# Writes a JLD2 (full arrays incl. truth identity) and, if csvpath given, a per-step CSV.
function save_kf_result(filepath::String, result::KFResult, basis::PODBasis,
                        truth_U::AbstractMatrix{Float32}, truth_coeffs::AbstractMatrix{Float32},
                        times::Vector{Float32}, truth_id::String,
                        sensor_locs::Vector{Matrix{Float32}};
                        csvpath::Union{String,Nothing}=nothing,
                        coeff_csvpath::Union{String,Nothing}=nothing,
                        n_coeff_modes::Int=6)
	L, K = size(result.α)
	Pr = basis.modes
	Pstd        = Matrix{Float32}(undef, L, K)          # per-mode posterior std = √diag(P)
	coeff_rmse  = Vector{Float64}(undef, K)
	field_rmse  = Vector{Float64}(undef, K)
	oracle_rmse = Vector{Float64}(undef, K)
	for k in 1:K
		Pstd[:, k]     = sqrt.(max.(diag(result.P[k]), 0f0))
		xt             = @view truth_U[:, k]
		coeff_rmse[k]  = sqrt(mean(abs2, result.α[:, k] .- truth_coeffs[:, k]))
		field_rmse[k]  = sqrt(mean(abs2, Pr * result.α[:, k]      .- xt))
		oracle_rmse[k] = sqrt(mean(abs2, Pr * truth_coeffs[:, k]  .- xt))
	end

	jldsave(filepath;
		α=result.α, α_pred=result.α_pred, Pstd=Pstd,
		truth_coeffs=Matrix(truth_coeffs), nis=result.nis, innov=result.innov,
		coeff_rmse=coeff_rmse, field_rmse=field_rmse, oracle_rmse=oracle_rmse,
		times=times, truth_id=truth_id, member_ids=basis.member_ids,
		eigenvalues=basis.eigenvalues, sensor_locs=sensor_locs, L=L, K=K)
	@info "Saved KF history to $filepath  (truth=$truth_id, L=$L, K=$K, " *
	      "mean field-RMSE=$(round(mean(field_rmse),digits=4)), " *
	      "oracle floor=$(round(mean(oracle_rmse),digits=4)), " *
	      "mean NIS=$(round(mean(result.nis),digits=3)))"

	if !isnothing(csvpath)
		open(csvpath, "w") do io
			println(io, "step,time,coeff_rmse,field_rmse,oracle_rmse,nis")
			for k in 1:K
				println(io, "$k,$(times[k]),$(coeff_rmse[k]),$(field_rmse[k]),$(oracle_rmse[k]),$(result.nis[k])")
			end
		end
		@info "Saved KF per-step diagnostics CSV to $csvpath"
	end

	# Coefficient trajectories of the leading modes (long format) for the α ± σ vs truth figure.
	if !isnothing(coeff_csvpath)
		nm = min(n_coeff_modes, L)
		open(coeff_csvpath, "w") do io
			println(io, "step,time,mode,alpha,sigma,truth")
			for k in 1:K, i in 1:nm
				println(io, "$k,$(times[k]),$i,$(result.α[i,k]),$(Pstd[i,k]),$(truth_coeffs[i,k])")
			end
		end
		@info "Saved KF coefficient trajectories CSV to $coeff_csvpath ($(nm) modes)"
	end
	return nothing
end

# ---------------------------------------------------------------------------
# Spatial (per-cell) reconstruction-error map
# ---------------------------------------------------------------------------

# Per-cell RMSE-over-time of the TOTAL reconstruction error e_k = Pr·α_k − x_k^truth.
# This is the field_rmse of save_kf_result resolved in space instead of collapsed over it,
# so mean-over-cells of speed² equals the time-mean field_rmse² (cross-check). It carries the
# truncation floor because it uses the true field x_truth (truth_U), NOT its projection.
#   α        — filter posterior coefficients (L × K)
#   truth_U  — truth snapshots over the analysis domain (state_dim × K), columns aligned with α
# Returns 3D arrays (Nx,Ny,Nz) in the masked layout (0 outside the domain): per-component RMSE
# u/v/w and the total velocity-error RMSE `speed` = √(rmse_u²+rmse_v²+rmse_w²). Streams in column
# chunks so the dense Pr·α (state_dim × K) is never fully resident.
function field_rmse_map(basis::PODBasis, α::AbstractMatrix{Float32},
                        truth_U::AbstractMatrix{Float32},
                        grid::GridInfo, domain::AnalysisDomain; chunk::Int=32)
	Pr = basis.modes
	state_dim, K = size(Pr, 1), size(α, 2)
	size(Pr, 1) == domain.state_dim ||
		error("basis state_dim=$(size(Pr,1)) ≠ domain.state_dim=$(domain.state_dim)")
	size(truth_U, 2) == K ||
		error("α has $K cols but truth_U has $(size(truth_U,2)) — column misalignment")
	ss = zeros(Float64, state_dim)                       # per-cell Σ_k e_k²
	for c0 in 1:chunk:K
		cc = c0:min(c0 + chunk - 1, K)
		E  = Pr * α[:, cc] .- @view truth_U[:, cc]        # (state_dim, |cc|)
		ss .+= vec(sum(abs2, E; dims=2))
	end
	rmse = Float32.(sqrt.(ss ./ K))                      # per-cell, [u;v;w] masked layout
	u3, v3, w3 = unpack_snapshot(rmse, grid, domain)     # 3D, 0 outside the domain
	speed = sqrt.(u3 .^ 2 .+ v3 .^ 2 .+ w3 .^ 2)
	return (u=u3, v=v3, w=w3, speed=speed)
end
