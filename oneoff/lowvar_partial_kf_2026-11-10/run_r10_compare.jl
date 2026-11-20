# run_r10_compare.jl — ONE-OFF (2026-11-20): POD-KF (r=10) vs UNLOCALIZED EnOI vs UNLOCALIZED
# cycled EnKF on the 11-member lowvar pool, truth = nature run, rake sensors, at several sensor
# noise levels (SIGMAS, per-component σ in m/s; default 0.5 = the shared SENSOR_VAR, and 0.1).
#
# Why: the localized EnOI in run_enoi.jl sat on the prior (localization discards the real
# long-range covariances of a 737-snapshot pool) and every method over-trusted σ=0.5 obs near
# the rake. This compares the three methods head-to-head in field RMSE, calibration (spread vs
# RMSE, NIS) and in the r=10 POD coefficients:
#   POD-KF  α, √diag P_post                          (its own state)
#   EnOI    Φᵀx_a and √diag(ΦᵀP_aΦ), P_a = B − BHᵀS⁻¹HB (exact, from pool moments)
#   EnKF    mean / std of the projected ensemble ΦᵀE
# Φ = leading 10 modes of the saved all-11 truth basis (out_n67_all11/bases, reused: the basis,
# models and rep-error do not depend on sensor noise). Noise draws: Random.seed!(NOISE_SEED) then
# randn·σ, so every σ sees the SAME standard normals (paired).
#
# R: EnOI/EnKF use R_sensor = σ²I (H picks cells exactly). POD-KF uses R_sensor + R_rep + tail
# clutter (as truth_rake_r10_rep). R_rep is indefinite (~−0.03 m²/s² min eig); if the final R is
# not PD at small σ, R_rep is eigen-clipped to PSD and the run says so (clip flag in summary).
#
# Sanity at σ=0.5: POD-KF α must reproduce kf/kf_result_truth_rake_r10_rep.jld2, EnOI RMSE must
# match enoi/summary.csv enoi_cinf (0.1659).
#   SIGMAS=0.5,0.1 sbatch run_r10_compare.sh     → out_n67_all11/r10_compare/s<σ×100>/
using LinearAlgebra, Random, Statistics, JLD2, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kf_adapters.jl"))
include(joinpath(SESP_ROOT, "src", "experiment.jl"))

const CONFIG_PATH = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar/ensemble_config.json"
const POOL_IDS  = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "10", "11"]
const N_SNAP    = 67
# OUT_TAG (2026-11-20): output folder suffix, default _all11; REP_MODEL lowrank (default) loads
# bases/rep_error_lr_truth_truth_r<RANK>.jld2 written by run_partial_kf.jl (rank-matched, no tail
# clutter); kernel = original parametric R_rep + tail clutter (reproduces job 24403).
const OUT       = joinpath(@__DIR__, "out_n67" * get(ENV, "OUT_TAG", "_all11"))
const REP_MODEL = get(ENV, "REP_MODEL", "lowrank")
const BASIS_IN  = joinpath(OUT, "bases")
const CMP_OUT   = joinpath(OUT, "r10_compare")
const SIGMAS    = parse.(Float64, strip.(split(get(ENV, "SIGMAS", "0.5,0.1"), ",")))
const RANK      = 10
const NE        = 50
const ENS_SEED  = 20261118          # same as run_enoi.jl
const ROWCHUNK  = 1 << 20

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))
@info "SIGMAS=$SIGMAS  rank=$RANK  Ne=$NE  threads=$(Threads.nthreads())"; flush(stderr)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
"Per-step SSE of est = x0 + G·D against truth (chunked over rows). x0 = nothing → 0."
function sse_linear(x0, G::AbstractMatrix{Float32}, D::AbstractMatrix{Float32}, truth_U)
    d, K = size(truth_U)
    s    = zeros(K)
    buf  = Matrix{Float32}(undef, ROWCHUNK, K)
    for ch in Iterators.partition(1:d, ROWCHUNK)
        b = view(buf, 1:length(ch), :)
        mul!(b, view(G, ch, :), D)
        isnothing(x0) || (b .+= view(x0, ch))
        @inbounds for k in 1:K, (ii, r) in enumerate(ch)
            e = Float64(b[ii, k]) - Float64(truth_U[r, k]); s[k] += e * e
        end
    end
    return s
end

function accum_sqdev!(acc::Vector{Float64}, X::AbstractMatrix{Float32}, μ::AbstractVector{Float32})
    d, T = size(X)
    Threads.@threads for ch in collect(Iterators.partition(1:d, 1 << 16))
        @inbounds for t in 1:T, r in ch
            v = Float64(X[r, t]) - Float64(μ[r]); acc[r] += v * v
        end
    end
    return acc
end

function ens_moments(E::Matrix{Float32})
    d, Ne = size(E)
    m = Vector{Float32}(undef, d); v = Vector{Float32}(undef, d)
    Threads.@threads for ch in collect(Iterators.partition(1:d, 1 << 16))
        @inbounds for r in ch
            s = 0.0
            for j in 1:Ne; s += E[r, j]; end
            μ = s / Ne; q = 0.0
            for j in 1:Ne; q += (E[r, j] - μ)^2; end
            m[r] = μ; v[r] = q / (Ne - 1)
        end
    end
    return m, v
end

"PSD projection (eigenvalues clipped at 0)."
psd_clip(A) = (F = eigen(Symmetric(Matrix{Float64}(A))); F.vectors * Diagonal(max.(F.values, 0)) * F.vectors')

# ---------------------------------------------------------------------------
# 1. Pool, grid, cache, basis, truth
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
cache   = load_member_cache(members, domain; t_spinup=T_SPINUP)
T       = size(first(values(cache)), 2)

basis = load_pod_basis(basis_filename(BASIS_IN, "truth"))
mdl   = load_models(models_filename(BASIS_IN, "truth"))
rep   = REP_MODEL == "lowrank" ? load_rep_error_lr(rep_error_lr_filename(BASIS_IN, "truth", RANK)) :
                                 load_rep_error(rep_error_filename(BASIS_IN, "truth"))
@info "REP_MODEL=$REP_MODEL, OUT=$OUT"
sort(basis.member_ids) == sort(POOL_IDS) || error("basis pool $(basis.member_ids) ≠ $POOL_IDS")
Φ = Matrix{Float32}(basis.modes[:, 1:RANK])                       # (d × r), orthonormal
@info "basis L=$(basis.L), using r=$RANK"; flush(stderr)

tm      = resolve_member(config, "truth")
truth_U = load_member_snapshots(tm; t_spinup=T_SPINUP, domain)
truth_t = tm.sim_times[_spinup_mask(tm, T_SPINUP)]
K       = length(truth_t)
truth_c = Φ' * truth_U                                              # (r × K) oracle coefficients
sse_orc = sse_linear(nothing, Φ, truth_c, truth_U)

U4   = unpack_to_NCfield(truth_U, truth_t, grid, domain)            # full-grid copy for sampler
locs = rake_locations(grid, domain)
ns, m = length(locs), 3length(locs)
lin  = _masked_linear_index(domain, grid)

# sensor cells (geometry only; noise level irrelevant here)
Random.seed!(NOISE_SEED)
seq0    = get_static_measurement_sequence(U4, grid, locs, domain, ns, sensor_mvar(), [1])
sp      = [lin[mm.locidx...] for mm in seq0[1].measurements]
any(==(0), sp) && error("sensor outside analysis domain")
obsrows = vec(permutedims(hcat(sp, sp .+ n, sp .+ 2n)))             # [u;v;w] per sensor

# ---------------------------------------------------------------------------
# 2. Pool moments (σ-independent): x̄, BHᵀ, HBHᵀ, diag B, and their r=10 projections
# ---------------------------------------------------------------------------
pool = POOL_IDS
Nf   = length(pool) * T
Yens = Float64.(reduce(hcat, [cache[id][obsrows, :] for id in pool]))   # (m × Nf)
Hxb  = vec(mean(Yens; dims=2))
HA   = Yens .- Hxb
HBH  = HA * HA' ./ (Nf - 1)
W    = Float32.(hcat(fill(1 / Nf, Nf), HA' ./ (Nf - 1)))
acc  = zeros(Float32, dstate, 1 + m)
for (i, id) in enumerate(pool)
    mul!(acc, cache[id], view(W, (i-1)*T+1:i*T, :), true, true)
end
xbar = acc[:, 1]; BH = acc[:, 2:end]; acc = nothing; GC.gc()
varB = zeros(Float64, dstate)
for id in pool; accum_sqdev!(varB, cache[id], xbar); end
varB ./= (Nf - 1)
Cpool = reduce(hcat, [Φ' * cache[id] for id in pool])              # (r × Nf)
cbar  = vec(mean(Float64.(Cpool); dims=2))
CA    = Float64.(Cpool) .- cbar
ΦBΦ   = CA * CA' ./ (Nf - 1)                                         # (r × r)
ΦBH   = Float64.(Φ' * BH)                                            # (r × m)
sse_prior = sse_linear(xbar, zeros(Float32, dstate, 1), zeros(Float32, 1, K), truth_U)
@info @sprintf("pool N=%d  mean var(B)=%.4f  prior RMSE=%.4f  oracle(r=%d)=%.4f", Nf, mean(varB),
               mean(sqrt.(sse_prior ./ dstate)), RANK, mean(sqrt.(sse_orc ./ dstate))); flush(stderr)

# EnKF forecast increments (σ-independent)
μinc = zeros(Float32, dstate)
for id in pool; μinc .+= (view(cache[id], :, T) .- view(cache[id], :, 1)); end
μinc ./= length(pool) * (T - 1)
incr_pool = [(id, t) for id in pool for t in 1:T-1]

# ---------------------------------------------------------------------------
# 3. Per noise level
# ---------------------------------------------------------------------------
summary = []
rmse_of(s) = sqrt.(s ./ dstate)

for σ in SIGMAS
    tag = @sprintf("s%03d", round(Int, 100σ))
    dir = joinpath(CMP_OUT, tag); mkpath(dir)
    @info "================ σ = $σ m/s ($tag) ================"; flush(stderr)
    MVAR = Matrix{Float32}(Float32(σ^2) * I, 3, 3)
    Random.seed!(NOISE_SEED)
    seq = get_static_measurement_sequence(U4, grid, locs, domain, ns, MVAR, collect(1:K))
    Y   = observation_matrix(seq)
    Rs  = Float64.(build_measurement_covariance(ns, MVAR))

    steps_io = open(joinpath(dir, "steps.csv"), "w")
    println(steps_io, "method,step,time,field_rmse,field_spread,nis,prior_rmse,oracle_rmse")
    coef_io  = open(joinpath(dir, "coeffs.csv"), "w")
    println(coef_io, "method,step,time,mode,alpha,sigma,truth")
    function emit(method, rm, spread, nisv, α, σα)
        for k in 1:K
            @printf(steps_io, "%s,%d,%g,%.6f,%.6f,%.6f,%.6f,%.6f\n", method, k, truth_t[k], rm[k],
                    spread[k], nisv[k], sqrt(sse_prior[k] / dstate), sqrt(sse_orc[k] / dstate))
            for i in 1:RANK
                @printf(coef_io, "%s,%d,%g,%d,%.6f,%.6f,%.6f\n", method, k, truth_t[k], i,
                        α[i, k], σα[i, k], truth_c[i, k])
            end
        end
        flush(steps_io); flush(coef_io)
    end
    function record!(method, rm, spread, nisv, α, σα; clipped=false)
        z2 = vec(mean(((α .- truth_c) ./ σα) .^ 2; dims=2))         # per-mode normalized err²
        push!(summary, (sigma=σ, method=method, field_rmse=mean(rm),
                        field_rmse_2ndhalf=mean(rm[K÷2+1:end]), field_spread=mean(spread),
                        mean_nis=mean(nisv), coeff_z2_mean=mean(z2),
                        coeff_z2_modes=join(round.(z2; sigdigits=3), " "), rep_clipped=clipped,
                        dstate=dstate))
        @info @sprintf("%-6s σ=%.2f  RMSE %.4f (2nd half %.4f) | spread %.4f | NIS %.2f | mean z² %.2f",
                       method, σ, mean(rm), mean(rm[K÷2+1:end]), mean(spread), mean(nisv), mean(z2))
        flush(stderr)
    end

    # ---- POD-KF, r=10, R_sensor + R_rep + tail clutter (= truth_rake_r10_rep) ----
    Hfull, R_base = obs_operator_and_base_R(seq[1], basis, domain, MVAR; rep)
    @info @sprintf("POD-KF: min eig R_base = %.4f (σ² = %.4f)", eigmin(Symmetric(R_base)), σ^2)
    clipped = false
    obs = try
        rank_r_obs(Hfull, R_base, mdl.Σ_α0, RANK; rep)
    catch e
        @warn "R not PD at σ=$σ — eigen-clipping R_rep to PSD" exception=e
        clipped = true
        rank_r_obs(Hfull, Rs + psd_clip(R_base - Rs), mdl.Σ_α0, RANK; rep)
    end
    @info @sprintf("POD-KF: min eig R_total (incl. tail clutter) = %.4f", eigmin(Symmetric(obs.R)))
    res  = run_filter(KFModel(mdl, RANK), Y, obs)
    αp   = Float32.(res.α_post)
    σp   = reduce(hcat, [sqrt.(max.(diag(res.P_post[:, :, k]), 0)) for k in 1:K])
    rm   = rmse_of(sse_linear(nothing, Φ, αp, truth_U))
    spr  = [sqrt(tr(res.P_post[:, :, k]) / dstate) for k in 1:K]
    emit("podkf", rm, spr, nis(res), αp, σp)
    record!("podkf", rm, spr, nis(res), αp, σp; clipped)
    if σ == 0.5
        ref = joinpath(OUT, "kf", "kf_result_truth_rake_r10_rep.jld2")
        if isfile(ref)
            αr = load(ref, "α")
            rel = norm(αr .- αp) / norm(αr)
            @info "sanity: POD-KF vs existing truth_rake_r10_rep rel‖Δα‖ = $rel"
            rel < 1e-4 || @warn "POD-KF does NOT reproduce truth_rake_r10_rep (rel $rel)"
        end
    end

    # ---- static EnOI, unlocalized, R = σ²I ----
    S    = Symmetric(HBH + Rs)
    Sinv = inv(S)
    Kg   = BH * Float32.(Sinv)                                       # (d × m)
    Dinn = Y .- Hxb
    nisv = [dot(Dinn[:, k], Sinv * Dinn[:, k]) / m for k in 1:K]
    rm   = rmse_of(sse_linear(xbar, Kg, Float32.(Dinn), truth_U))
    diagPa = varB .- vec(sum(Float64.(Kg) .* Float64.(BH); dims=2))
    spr  = fill(sqrt(mean(max.(diagPa, 0))), K)
    αe   = Float32.(Float64.(Φ' * xbar) .+ ΦBH * Sinv * Dinn)        # Φᵀx_a
    Pαe  = Symmetric(ΦBΦ - ΦBH * Sinv * ΦBH')
    σe   = repeat(sqrt.(max.(diag(Pαe), 0)), 1, K)
    emit("enoi", rm, spr, nisv, αe, σe)
    record!("enoi", rm, spr, nisv, αe, σe)
    Kg = nothing; GC.gc()

    # ---- cycled perturbed-obs EnKF, unlocalized, Ne=50, R = σ²I ----
    rng  = MersenneTwister(ENS_SEED)
    allc = [(id, t) for id in pool for t in 1:T]
    init = allc[randperm(rng, Nf)[1:NE]]
    E    = Matrix{Float32}(undef, dstate, NE)
    for (j, (id, t)) in enumerate(init); E[:, j] = view(cache[id], :, t); end
    BHe  = Matrix{Float32}(undef, dstate, m)
    Lr   = cholesky(Symmetric(Rs)).L
    sse_k = zeros(K); spr = zeros(K); nisv = zeros(K)
    αk = zeros(Float32, RANK, K); σk = zeros(Float32, RANK, K)
    for k in 1:K
        if k > 1
            for j in 1:NE
                id, t = incr_pool[rand(rng, eachindex(incr_pool))]
                X = cache[id]
                @views E[:, j] .+= X[:, t+1] .- X[:, t] .- μinc
            end
        end
        HE  = Float64.(E[obsrows, :])
        hm  = vec(mean(HE; dims=2))
        HAe = HE .- hm
        mul!(BHe, E, Float32.(HAe' ./ (NE - 1)))
        Sf  = cholesky(Symmetric(HAe * HAe' ./ (NE - 1) + Rs))
        dk  = Y[:, k] .- hm
        nisv[k] = dot(dk, Sf \ dk) / m
        ε   = Lr * randn(rng, m, NE); ε .-= mean(ε; dims=2)
        mul!(E, BHe, Float32.(Sf \ (Y[:, k] .+ ε .- HE)), true, true)
        xm, xv = ens_moments(E)
        sse_k[k] = sum(abs2, Float64.(xm) .- Float64.(view(truth_U, :, k)))
        spr[k]   = sqrt(mean(Float64, xv))
        Ce = Φ' * E                                                   # (r × Ne)
        αk[:, k] = vec(mean(Ce; dims=2)); σk[:, k] = vec(std(Ce; dims=2))
        k % 10 == 1 && (@info @sprintf("  enkf step %d/%d  RMSE %.4f spread %.4f NIS %.2f", k, K,
                                       sqrt(sse_k[k] / dstate), spr[k], nisv[k]); flush(stderr))
    end
    E = nothing; BHe = nothing; GC.gc()
    rm = rmse_of(sse_k)
    emit("enkf", rm, spr, nisv, αk, σk)
    record!("enkf", rm, spr, nisv, αk, σk)
    close(steps_io); close(coef_io)
end

open(joinpath(CMP_OUT, "summary.csv"), "w") do io
    ks = keys(summary[1])
    println(io, join(ks, ","))
    for s in summary; println(io, join([s[k] for k in ks], ",")); end
end
@info "=== done → $CMP_OUT ==="
