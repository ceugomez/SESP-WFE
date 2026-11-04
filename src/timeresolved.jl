# timeresolved.jl — optional per-timestep / per-mode diagnostics for the dense KF sweeps.
# cgf cego6160@colorado.edu 7.17.26
#
# Shared VERBATIM by run_kf_twin_dense.jl (rung 1) and run_kf_rank_sweep_dense.jl (rung 2)
# so the twin and LOO time-resolved products are computed by identical code and can be
# differenced without a definition drifting between the two.
#
# WHY THIS EXISTS: the summary CSV reports one wake RMSE per rank. That number is dominated
# by the static forcing head (~99% of snapshot energy) and CANNOT distinguish "tracks the
# time-varying structure" from "nails the time-mean and ignores the meander". These
# diagnostics isolate the time-varying content of the wake estimate:
#   - per-timestep wake RMSE (KF and oracle)         -> temporal box-and-whisker over the run
#   - bias/fluctuation split of the wake error:
#       bias  = ||time-mean of the error field||     (static miscalibration)
#       fluct = RMS of the mean-REMOVED error        (the transient-tracking number)
#   - the truth's OWN wake fluctuation RMS = the persistence-of-mean baseline: output the
#     time mean and your error IS the full fluctuation. Tracking skill := 1 - fluct(KF)/fluct(truth);
#     ~0 means the transient is not tracked even though aggregate RMSE looks good.
#   - KF and oracle modal coefficient trajectories   -> the mode-13 (~50-60 s meander) time
#     series, the most direct picture of whether the estimate follows the oscillation and of
#     the known random-walk-Q over-swing at the transient modes.
#
# Everything is accumulated incrementally (nw-length and K-length vectors); the (nw × K) wake
# error field is never materialized — the same memory discipline as resid2_full.

using JLD2

"Parse a comma list like \"15,20,50\" into sorted unique Ints; empty string -> no dump."
parse_dump_ranks(s::AbstractString) =
    isempty(strip(s)) ? Int[] : sort(unique(parse.(Int, strip.(split(s, ',')))))

mutable struct RankDump
    err_sum_kf :: Vector{Float64}   # Σ_k (est_k - truth_k) over wake cells (for the bias field)
    err_sum_or :: Vector{Float64}
    kf_mse_t   :: Vector{Float64}   # per-step Σ_cells (est-truth)²   (unnormalized)
    or_mse_t   :: Vector{Float64}
end
RankDump(nw::Int, K::Int) = RankDump(zeros(nw), zeros(nw), zeros(K), zeros(K))

function accumulate_step!(d::RankDump, k::Int, e_kf::AbstractVector, e_or::AbstractVector)
    d.err_sum_kf .+= e_kf
    d.err_sum_or .+= e_or
    d.kf_mse_t[k] = sum(abs2, e_kf)
    d.or_mse_t[k] = sum(abs2, e_or)
    return nothing
end

# Finalize to RMSE-scaled diagnostics (per wake element). Uses the identity
#   mean_k ||e_k - ē||² = mean_k ||e_k||² - ||ē||²,  ē = (1/K) Σ_k e_k
# so the fluctuation RMS needs only the running sum and the per-step totals, no second pass.
function finalize_dump(d::RankDump, K::Int, nw::Int)
    kf_bias2 = sum(abs2, d.err_sum_kf ./ K) / nw
    or_bias2 = sum(abs2, d.err_sum_or ./ K) / nw
    kf_tot   = (sum(d.kf_mse_t) / K) / nw          # mean_k ||e_k||² per element
    or_tot   = (sum(d.or_mse_t) / K) / nw
    return (
        kf_wake_rmse_t = sqrt.(d.kf_mse_t ./ nw),
        or_wake_rmse_t = sqrt.(d.or_mse_t ./ nw),
        kf_bias_rmse   = sqrt(kf_bias2),
        or_bias_rmse   = sqrt(or_bias2),
        kf_fluct_rmse  = sqrt(max(kf_tot - kf_bias2, 0.0)),
        or_fluct_rmse  = sqrt(max(or_tot - or_bias2, 0.0)),
    )
end

# Truth wake baseline (rank-independent, computed once). `truthW` columns are the FULL time
# axis; `TIDX` picks the assimilated snapshots, matching `xtw = @view truthW[:, tk]` upstream.
function truth_wake_baseline(truthW::AbstractMatrix, nw::Int, TIDX::Vector{Int})
    K    = length(TIDX)
    xsum = zeros(Float64, nw)
    tot  = 0.0
    for k in 1:K
        xt = @view truthW[:, TIDX[k]]
        xsum .+= xt
        tot  += sum(abs2, xt)
    end
    mean_energy = sum(abs2, xsum ./ K) / nw
    fluct       = max((tot / K) / nw - mean_energy, 0.0)
    return (truth_wake_mean_rmse = sqrt(mean_energy), truth_wake_fluct_rmse = sqrt(fluct))
end

# Write the time-resolved product. `dumps`/`Acoef`/`astar_coef` are keyed by rank. Coefficient
# matrices are (r × K): rows are POD modes (row 13 = the meander), columns the assimilated times.
function save_timeresolved(path::String, t::Vector{Float64}, dump_ranks::Vector{Int},
                           base, dumps::Dict, Acoef::Dict, astar_coef::Dict)
    jldopen(path, "w") do f
        f["t"]                    = t
        f["dump_ranks"]           = dump_ranks
        f["truth_wake_mean_rmse"] = base.truth_wake_mean_rmse
        f["truth_wake_fluct_rmse"] = base.truth_wake_fluct_rmse
        for r in dump_ranks
            fd = dumps[r]
            f["r$r/A"]              = Acoef[r]           # (r × K) KF coefficients
            f["r$r/astar"]          = astar_coef[r]      # (r × K) oracle coefficients
            f["r$r/kf_wake_rmse_t"] = fd.kf_wake_rmse_t
            f["r$r/or_wake_rmse_t"] = fd.or_wake_rmse_t
            f["r$r/kf_bias_rmse"]   = fd.kf_bias_rmse
            f["r$r/or_bias_rmse"]   = fd.or_bias_rmse
            f["r$r/kf_fluct_rmse"]  = fd.kf_fluct_rmse
            f["r$r/or_fluct_rmse"]  = fd.or_fluct_rmse
        end
    end
    return path
end
