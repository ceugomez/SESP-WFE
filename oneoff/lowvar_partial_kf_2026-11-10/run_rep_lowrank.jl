# run_rep_lowrank.jl — ONE-OFF (2026-11-20): low-rank + diagonal representativeness error,
# tested against the parametric kernel R_rep (src/reperror.jl) in the r=10 POD-KF.
#
# Formulation ("option 2"). For each member k of the all-11 pool, fold basis Φ_{-k} (rank r,
# built from the other 10 members via the pool Gram) and held-out residual
#     r_{k,t} = x_{k,t} − Φ_{-k} Φ_{-k}ᵀ x_{k,t}          (rank-r residual: NO tail clutter needed)
# Stack all N = 11·67 residuals as columns of ℛ (d × N). Uncentered second moment (as σ² in
# reperror.jl): C = ℛℛᵀ/N. Its POD via the N×N Gram G = ℛᵀℛ = VΛVᵀ gives F_q = ℛV_q/√N, and
#     R_rep(q) = H F_q F_qᵀ Hᵀ + diag(H (diag C − rowsum F_q²))      ⪰ 0 by construction
# The diagonal ("nugget") restores the exact per-cell variance the q modes miss, so diag R_rep is
# the same for every q; q = N is the exact empirical covariance at the sensors ("option 1").
# Only the sensor rows of ℛ and V are needed for the rake, but nothing here is rake-specific:
# F_q for any geometry is ℛV_q/√N (not saved by this one-off; see TODO at the end).
#
# Memory: ℛ is never held whole (11 × 9.8 GB). The Gram and diag C are accumulated over row
# chunks from the cache, the fold bases (d × r each) and the fold coefficients a_k = Φ_{-k}ᵀX_k.
#
# Filter test (same as run_r10_compare.jl): truth = nature run, rake, POD-KF r=10, σ ∈ SIGMAS,
# same NOISE_SEED draws. Variants:
#   param      R_sensor + R_rep(param, L=50 residual) + tail clutter  (= production r10_rep)
#   lr_q<q>    R_sensor + R_rep(q),  q ∈ QS (q = 0: diagonal only; q = N: exact empirical)
# Writes out_n67_all11/rep_lowrank/: spectrum.csv, rdiag.csv, summary.csv, lowrank_sensor.jld2,
# s<σ×100>/coeffs_<variant>.csv, steps.csv.
#   SIGMAS=0.5,0.1 sbatch run_rep_lowrank.sh
using LinearAlgebra, Random, Statistics, JLD2, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kf_adapters.jl"))
include(joinpath(SESP_ROOT, "src", "experiment.jl"))

const CONFIG_PATH = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar/ensemble_config.json"
const POOL_IDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "10", "11"]
const N_SNAP   = 67
const OUT      = joinpath(@__DIR__, "out_n67_all11")
const BASIS_IN = joinpath(OUT, "bases")
const LR_OUT   = joinpath(OUT, "rep_lowrank"); mkpath(LR_OUT)
const SIGMAS   = parse.(Float64, strip.(split(get(ENV, "SIGMAS", "0.5,0.1"), ",")))
const RANK     = 10
const QS_REQ   = [0, 10, 25, 50, 100, 200, typemax(Int)]   # typemax → N (exact empirical)
const ROWCHUNK = 1 << 18

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))
@info "SIGMAS=$SIGMAS rank=$RANK threads=$(Threads.nthreads())"; flush(stderr)

"Per-step SSE of est = G·D against truth (chunked over rows)."
function sse_linear(G::AbstractMatrix{Float32}, D::AbstractMatrix{Float32}, truth_U)
    d, K = size(truth_U)
    s    = zeros(K)
    buf  = Matrix{Float32}(undef, 1 << 20, K)
    for ch in Iterators.partition(1:d, 1 << 20)
        b = view(buf, 1:length(ch), :)
        mul!(b, view(G, ch, :), D)
        @inbounds for k in 1:K, (ii, r) in enumerate(ch)
            e = Float64(b[ii, k]) - Float64(truth_U[r, k]); s[k] += e * e
        end
    end
    return s
end

# ---------------------------------------------------------------------------
# 1. Pool, cache, pool Gram, fold bases (rank r)
# ---------------------------------------------------------------------------
config = load_ensemble_config(CONFIG_PATH; include_ids=POOL_IDS)
cut(m) = (keep = findall(t -> t >= T_SPINUP, m.sim_times)[1:N_SNAP];
          MemberInfo(m.id, m.output_dir, m.met_dir_deg, m.U_g_ms, m.V_g_ms,
                     m.files[keep], m.timesteps[keep], m.sim_times[keep]))
config  = EnsembleConfig(config.n_members, cut.(config.prior_members), cut(config.truth_member))
members = config.prior_members
grid    = load_grid(members[1].files[1])
domain  = AnalysisDomain(grid; margin_m=MARGIN_M)
n, dstate = domain.n_grid, domain.state_dim
# REUSE=1: skip the folds / residual Gram (sections 1b–3a) and load lowrank_sensor.jld2 instead
const LR_JLD = joinpath(LR_OUT, "lowrank_sensor.jld2")
const REUSE  = get(ENV, "REUSE", "0") == "1" && isfile(LR_JLD)
@info "REUSE=$REUSE"
if !REUSE
cache   = load_member_cache(members, domain; t_spinup=T_SPINUP)
T       = size(first(values(cache)), 2)
_, pool_gram = compute_pod_streaming(members, grid, domain; L=RANK, t_spinup=T_SPINUP,
                                     truth_id="truth", cache)
M  = length(POOL_IDS)
N  = M * T
QS = unique(min.(QS_REQ, N))

Φf = Dict{String,Matrix{Float32}}()     # fold bases (d × r)
Af = Dict{String,Matrix{Float32}}()     # fold coefficients of the held-out member (r × T)
for id in POOL_IDS
    b = loo_basis_from_gram(pool_gram, members, grid, domain; held_id=id, L=RANK,
                            t_spinup=T_SPINUP, cache)
    Φf[id] = b.modes
    Af[id] = b.modes' * cache[id]
    @info "fold $id: basis on $(b.member_ids)"; flush(stderr)
    b = nothing; GC.gc()
end
pool_gram = nothing; GC.gc()

# ---------------------------------------------------------------------------
# 2. Residual Gram G = ℛᵀℛ and diag C, streamed over row chunks
# ---------------------------------------------------------------------------
G     = zeros(Float64, N, N)
diagC = zeros(Float64, dstate)
Rc    = Matrix{Float32}(undef, ROWCHUNK, N)
Gc    = Matrix{Float32}(undef, N, N)
for ch in Iterators.partition(1:dstate, ROWCHUNK)
    R = view(Rc, 1:length(ch), :)
    for (i, id) in enumerate(POOL_IDS)
        blk = (i-1)*T+1:i*T
        R[:, blk] .= view(cache[id], ch, :)
        mul!(view(R, :, blk), view(Φf[id], ch, :), Af[id], -1f0, 1f0)
    end
    mul!(Gc, R', R); G .+= Gc
    diagC[ch] .= vec(sum(abs2, Float64.(R); dims=2)) ./ N
end
Rc = nothing; Gc = nothing; GC.gc()
G = Symmetric((G + G') ./ 2)
eg = eigen(G; sortby=-)
λ, V = max.(eg.values, 0.0), eg.vectors
@info @sprintf("residual: mean var %.4f m²/s², total energy check |tr G/N/d − mean diagC| = %.2e",
               mean(diagC), abs(tr(G) / N / dstate - mean(diagC)))
open(joinpath(LR_OUT, "spectrum.csv"), "w") do io
    println(io, "q,eig_over_N,cum_frac")
    cs = cumsum(λ) ./ sum(λ)
    for q in 1:N; @printf(io, "%d,%.6e,%.6f\n", q, λ[q] / N, cs[q]); end
end
for q in QS
    @info @sprintf("  q=%4d captures %.1f%% of residual energy", q, q == 0 ? 0.0 : 100sum(λ[1:q]) / sum(λ))
end
end  # !REUSE

# ---------------------------------------------------------------------------
# 3. Truth, sensors, sensor rows of ℛ, R_rep(q) at the rake
# ---------------------------------------------------------------------------
basis = load_pod_basis(basis_filename(BASIS_IN, "truth"))          # L=50, for H and "param"
mdl   = load_models(models_filename(BASIS_IN, "truth"))
rep   = load_rep_error(rep_error_filename(BASIS_IN, "truth"))
Φ     = Matrix{Float32}(basis.modes[:, 1:RANK])
tm      = resolve_member(config, "truth")
truth_U = load_member_snapshots(tm; t_spinup=T_SPINUP, domain)
truth_t = tm.sim_times[_spinup_mask(tm, T_SPINUP)]
K       = length(truth_t)
truth_c = Φ' * truth_U
U4   = unpack_to_NCfield(truth_U, truth_t, grid, domain)
locs = rake_locations(grid, domain)
ns, m = length(locs), 3length(locs)
lin  = _masked_linear_index(domain, grid)
Random.seed!(NOISE_SEED)
seq0 = get_static_measurement_sequence(U4, grid, locs, domain, ns, sensor_mvar(), [1])
sp   = [lin[mm.locidx...] for mm in seq0[1].measurements]
obsrows = vec(permutedims(hcat(sp, sp .+ n, sp .+ 2n)))

if !REUSE
Rs = reduce(hcat, [Float64.(cache[id][obsrows, :]) .- Float64.(Φf[id][obsrows, :]) * Float64.(Af[id])
                   for id in POOL_IDS])                              # (m × N) residuals at sensors
dC = diagC[obsrows]
@assert maximum(abs.(vec(sum(abs2, Rs; dims=2)) ./ N .- dC)) < 1e-3 * maximum(dC)
function R_lowrank(q)
    Fq  = Rs * V[:, 1:q] ./ sqrt(N)                                   # sensor rows of F_q
    nug = dC .- vec(sum(abs2, Fq; dims=2))
    Symmetric(Fq * Fq' + Diagonal(max.(nug, 0.0)))
end
Rlr = Dict(q => R_lowrank(q) for q in QS)
R_param = Float64.(build_rep_covariance(seq0[1], rep, domain))
tailC   = tail_clutter(Float64.(build_measurement_operator(seq0[1], basis, domain)), mdl.Σ_α0, RANK)

# what the truth's actual rank-r rep error looks like at the sensors (no noise)
truth_rep = Float64.(truth_U[obsrows, :]) .- Float64.(Φ[obsrows, :]) * Float64.(truth_c)
truth_ms  = vec(mean(abs2, truth_rep; dims=2))
open(joinpath(LR_OUT, "rdiag.csv"), "w") do io
    println(io, "variant,min_eig,mean_diag,mean_diag_over_truth_ms,n_neg_eig")
    rows = [("param(L50)", Symmetric(R_param)), ("param+tail", Symmetric(R_param + tailC))]
    append!(rows, [("lr_q$q", Rlr[q]) for q in QS])
    for (nm, A) in rows
        e = eigvals(A)
        @printf(io, "%s,%.5f,%.5f,%.3f,%d\n", nm, minimum(e), mean(diag(A)), mean(diag(A)) / mean(truth_ms), count(<(0), e))
        @info @sprintf("R_rep %-12s min eig %+.4f  mean diag %.4f  (truth rank-%d rep ms %.4f → ratio %.2f)  neg eigs %d",
                       nm, minimum(e), mean(diag(A)), RANK, mean(truth_ms), mean(diag(A)) / mean(truth_ms), count(<(0), e))
    end
end
jldsave(joinpath(LR_OUT, "lowrank_sensor.jld2"); Rs, V, λ, dC, obsrows, R_param, tailC,
        Rlr=Dict(string(q) => Matrix(Rlr[q]) for q in QS), truth_ms, N, QS)
Φf = nothing; Af = nothing; GC.gc()
else
    jl = load(LR_JLD)
    jl["obsrows"] == obsrows || error("saved obsrows differ from this run's rake")
    N, QS   = jl["N"], jl["QS"]
    Rlr     = Dict(q => Symmetric(jl["Rlr"][string(q)]) for q in QS)
    R_param, tailC, truth_ms = jl["R_param"], jl["tailC"], jl["truth_ms"]
    @info "loaded $(LR_JLD): N=$N, QS=$QS"
end

# ---------------------------------------------------------------------------
# 4. POD-KF r=10 with each R_rep, per σ
# ---------------------------------------------------------------------------
summary = []
for σ in SIGMAS
    tag = @sprintf("s%03d", round(Int, 100σ)); dir = joinpath(LR_OUT, tag); mkpath(dir)
    @info "================ σ = $σ ($tag) ================"; flush(stderr)
    MVAR = Matrix{Float32}(Float32(σ^2) * I, 3, 3)
    Random.seed!(NOISE_SEED)
    seq  = get_static_measurement_sequence(U4, grid, locs, domain, ns, MVAR, collect(1:K))
    Y    = observation_matrix(seq)
    Rsen = Float64.(build_measurement_covariance(ns, MVAR))
    Hfull, R_base = obs_operator_and_base_R(seq[1], basis, domain, MVAR; rep)
    variants = Tuple{String,Any}[("param", () -> truncated_obs(Hfull, R_base, mdl.Σ_α0, RANK))]
    append!(variants, [("lr_q$q", (() -> StaticObs(Hfull[:, 1:RANK], Rsen + Matrix(Rlr[q])))) for q in QS])
    steps_io = open(joinpath(dir, "steps.csv"), "w")
    println(steps_io, "variant,step,time,field_rmse,field_spread,nis,oracle_rmse")
    for (name, mk) in variants
        obs = try mk() catch e
            @warn "$name σ=$σ: obs model failed" exception=e; continue
        end
        res = run_filter(KFModel(mdl, RANK), Y, obs)
        α   = Float32.(res.α_post)
        σα  = reduce(hcat, [sqrt.(max.(diag(res.P_post[:, :, k]), 0)) for k in 1:K])
        rm  = sqrt.(sse_linear(Φ, α, truth_U) ./ dstate)
        orc = sqrt.(sse_linear(Φ, truth_c, truth_U) ./ dstate)
        spr = [sqrt(tr(res.P_post[:, :, k]) / dstate) for k in 1:K]
        nv  = nis(res)
        z2  = vec(mean(((α .- truth_c) ./ σα) .^ 2; dims=2))
        for k in 1:K
            @printf(steps_io, "%s,%d,%g,%.6f,%.6f,%.6f,%.6f\n", name, k, truth_t[k], rm[k], spr[k], nv[k], orc[k])
        end
        open(joinpath(dir, "coeffs_$(name).csv"), "w") do io
            println(io, "step,time,mode,alpha,sigma,truth")
            for k in 1:K, i in 1:RANK
                @printf(io, "%d,%g,%d,%.6f,%.6f,%.6f\n", k, truth_t[k], i, α[i, k], σα[i, k], truth_c[i, k])
            end
        end
        push!(summary, (sigma=σ, variant=name, min_eig_R=eigmin(Symmetric(obs.R)), field_rmse=mean(rm),
                        field_rmse_2ndhalf=mean(rm[K÷2+1:end]), oracle=mean(orc), field_spread=mean(spr),
                        mean_nis=mean(nv), coeff_z2_mean=mean(z2), coeff_z2_max=maximum(z2)))
        @info @sprintf("%-10s σ=%.2f  RMSE %.4f (2nd %.4f) | spread %.4f | NIS %.2f | z² mean %.2f max %.1f | min eig R %.4f",
                       name, σ, mean(rm), mean(rm[K÷2+1:end]), mean(spr), mean(nv), mean(z2), maximum(z2),
                       eigmin(Symmetric(obs.R))); flush(stderr)
    end
    close(steps_io)
end
open(joinpath(LR_OUT, "summary.csv"), "w") do io
    ks = keys(summary[1]); println(io, join(ks, ","))
    for s in summary; println(io, join([s[k] for k in ks], ",")); end
end
@info "=== done → $LR_OUT ==="
# TODO (production, if adopted): build F_q = ℛV_q/√N over the full field once per fold basis in
# src/reperror.jl and store it (d × q, ~1.5 GB per 10 modes) + nugget, so any sensor geometry
# (transect, lattice) gets R_rep by row indexing.
