# kf_adapters.jl
# cgf cego6160@colorado.edu 6.19.26
#
# Glue between the SESP-WFE structs (PODBasis, models, measSet, RepError) and the pure
# filter in kalman.jl, plus saving filter histories and their error diagnostics.
# Including this file loads the whole library except experiment.jl / regions.jl.
include("fasteddy_io.jl")  # IO, grid, analysis domain
include("pod.jl")          # POD structs (before measurement: it annotates ::PODBasis)
include("measurement.jl")  # synthetic observations, H
include("statmodel.jl")    # prior / process-noise models
include("reperror.jl")     # representativeness (reconstruction-error) model
include("kalman.jl")       # THE filter: KFModel, StaticObs/VaryingObs, run_filter, KFHistory, ...

using LinearAlgebra

# ---------------------------------------------------------------------------
# Glue: SESP-WFE structs (models, measSet, PODBasis) -> kalman.jl types.
# All filtering goes through run_filter (src/kalman.jl); this file only adapts inputs and
# persists outputs. Do not add a predict/update loop here or in a driver.
# ---------------------------------------------------------------------------

"Rank-r KFModel from the full-L offline `models` (leading blocks; Q scaled by `qscale`)."
KFModel(mdl::models, r::Int; qscale::Real=1.0) =
	truncate_model(mdl.μ_α0, mdl.Σ_α0, mdl.Qα, r; qscale)

"Stack a measurement sequence into the (3nm × K) Float64 observation matrix run_filter takes."
function observation_matrix(meassq::Vector{measSet})::Matrix{Float64}
	m = 3 * length(meassq[1].measurements)
	Y = Matrix{Float64}(undef, m, length(meassq))
	for (k, ms) in enumerate(meassq)
		y = stack_measurements(ms)
		length(y) == m || error("observation_matrix: step $k has $(length(y)) rows, expected $m")
		Y[:, k] = y
	end
	return Y
end

"""
    obs_operator_and_base_R(ms, basis, domain, mvar; rep=nothing) -> (Hfull, R_base)

Full-rank (3nm × L) measurement operator for sensor set `ms` and the rank-independent part
of the observation-error covariance, R_base = R_sensor (+ R_rep if `rep` given). Rank-r
truncation (with tail clutter) is then `truncated_obs(Hfull, R_base, mdl.Σ_α0, r)`.
"""
function obs_operator_and_base_R(ms::measSet, basis::PODBasis, domain::AnalysisDomain,
                                 mvar::Matrix{Float32};
                                 rep::Union{RepError,RepErrorLR,Nothing}=nothing)
	nm = length(ms.measurements)
	H  = Float64.(build_measurement_operator(ms, basis, domain))
	R  = Float64.(build_measurement_covariance(nm, mvar))
	isnothing(rep) || (R += Float64.(build_rep_covariance(ms, rep, domain)))
	return H, R
end

"""
    rank_r_obs(Hfull, R_base, Σ0full, r; rep=nothing) -> StaticObs

Rank-r static observation model. With a RepErrorLR, R_base already holds the error of the
discarded modes r+1:L (its residual is taken against the leading r modes), so NO tail clutter
is added and rep.rank must equal r. Otherwise (kernel RepError, or no rep model) this is
`truncated_obs` — honest truncation with the Σ0 tail folded into R.
"""
function rank_r_obs(Hfull::AbstractMatrix, R_base::AbstractMatrix, Σ0full::AbstractMatrix, r::Int;
                    rep::Union{RepError,RepErrorLR,Nothing}=nothing)
	if rep isa RepErrorLR
		rep.rank == r || error("rank_r_obs: RepErrorLR built for rank $(rep.rank), filter rank $r")
		return StaticObs(Hfull[:, 1:r], R_base)
	end
	return truncated_obs(Hfull, R_base, Σ0full, r)
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
function save_kf_result(filepath::String, result::KFHistory, basis::PODBasis,
                        truth_U::AbstractMatrix{Float32}, truth_coeffs::AbstractMatrix{Float32},
                        times::Vector{Float32}, truth_id::String,
                        sensor_locs::Vector{Matrix{Float32}};
                        csvpath::Union{String,Nothing}=nothing,
                        coeff_csvpath::Union{String,Nothing}=nothing,
                        n_coeff_modes::Int=6)
	L, K = size(result.α_post)
	size(basis.modes, 2) == L || error("save_kf_result: basis has $(size(basis.modes,2)) modes, filter rank $L")
	Pr = basis.modes
	α  = Float32.(result.α_post)
	nisv = Float32.(nis(result))
	Pstd        = Matrix{Float32}(undef, L, K)          # per-mode posterior std = √diag(P)
	coeff_rmse  = Vector{Float64}(undef, K)
	field_rmse  = Vector{Float64}(undef, K)
	oracle_rmse = Vector{Float64}(undef, K)
	for k in 1:K
		Pstd[:, k]     = Float32.(sqrt.(max.(diag(@view result.P_post[:, :, k]), 0.0)))
		xt             = @view truth_U[:, k]
		coeff_rmse[k]  = sqrt(mean(abs2, α[:, k] .- truth_coeffs[:, k]))
		field_rmse[k]  = sqrt(mean(abs2, Pr * α[:, k]      .- xt))
		oracle_rmse[k] = sqrt(mean(abs2, Pr * truth_coeffs[:, k]  .- xt))
	end

	jldsave(filepath;
		α=α, α_pred=Float32.(result.α_pred), Pstd=Pstd,
		truth_coeffs=Matrix(truth_coeffs), nis=nisv, innov=Float32.(result.innov),
		chi2=result.chi2, update_form=String(result.form),
		coeff_rmse=coeff_rmse, field_rmse=field_rmse, oracle_rmse=oracle_rmse,
		times=times, truth_id=truth_id, member_ids=basis.member_ids,
		eigenvalues=basis.eigenvalues, sensor_locs=sensor_locs, L=L, K=K)
	@info "Saved KF history to $filepath  (truth=$truth_id, L=$L, K=$K, " *
	      "mean field-RMSE=$(round(mean(field_rmse),digits=4)), " *
	      "oracle floor=$(round(mean(oracle_rmse),digits=4)), " *
	      "mean NIS=$(round(mean_nis(result),digits=3)))"

	if !isnothing(csvpath)
		open(csvpath, "w") do io
			println(io, "step,time,coeff_rmse,field_rmse,oracle_rmse,nis")
			for k in 1:K
				println(io, "$k,$(times[k]),$(coeff_rmse[k]),$(field_rmse[k]),$(oracle_rmse[k]),$(nisv[k])")
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
				println(io, "$k,$(times[k]),$i,$(α[i,k]),$(Pstd[i,k]),$(truth_coeffs[i,k])")
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
