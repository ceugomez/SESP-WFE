# run_enoi.jl — ONE-OFF (2026-11-18): ensemble-OI / EnKF baselines against the POD-KF, on the
# same member pool, truth, sensors, noise draws and metrics as run_partial_kf.jl.
#
# Why (see discussion 2026-11-18): the POD-KF's first update with Σ_α0 = sample covariance is
# EnOI restricted to the leading L POD directions, and its forecast is a linear random walk in
# coefficient space. So the informative baselines are full-state, ensemble-covariance methods
# that drop the subspace:
#   enoi_c<c>   STATIC EnOI. B = sample covariance of the whole snapshot pool (members × time),
#               analysis every step from the climatological mean: x_a = x̄ + K (y_k − Hx̄),
#               K = (ρ∘BHᵀ)(ρ∘HBHᵀ + R)⁻¹. c = Gaspari–Cohn half-width [m] (zero beyond 2c);
#               c = Inf is unlocalized EnOI (the ablation of truncation + dynamics + R_rep).
#   enkf_c<c>   CYCLED stochastic (perturbed-obs) EnKF, Ne members, batch update with the same
#               localization. Forecast = persistence + additive noise drawn from the pool's
#               6 s increments (centered): the full-state analogue of the POD-KF random walk
#               (Q_α is exactly the projection of this increment covariance). Initial ensemble
#               = Ne random pool snapshots (the analogue of μ_α0, Σ_α0).
# R = R_sensor only for every EnOI/EnKF run: H picks grid cells exactly, there is no subspace
# truncation to represent, so the matched POD-KF experiment is *_norep (rep variant also listed).
#
# KEY DIAGNOSTIC — bins.csv: RMSE by distance from the nearest sensor (3-D, m). Localization
# should win near the rake and decay to the prior far away; the subspace claim is that the
# POD-KF keeps skill in the far field through coherent modes. POD-KF and its projection floor
# (pod_oracle) are binned identically from the run_partial_kf outputs in the same OUT dir.
#
# Reads OUT/bases + OUT/kf of the matching run_partial_kf run (POD rows are skipped with a
# warning if absent). Writes OUT/enoi/: <exp>_steps.csv, bins.csv, summary.csv.
#   POOL=... OUT_TAG=_all11 N_SNAP=67 sbatch run_enoi.sh
using LinearAlgebra, Random, Statistics, JLD2, NCDatasets, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
const SRC_DIR   = joinpath(SESP_ROOT, "src")
include(joinpath(SRC_DIR, "kf_adapters.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))

# ---------------------------------------------------------------------------
# Configuration (pool / N_SNAP / OUT naming identical to run_partial_kf.jl)
# ---------------------------------------------------------------------------
const CONFIG_PATH  = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar/ensemble_config.json"
const DEFAULT_POOL = ["03", "10", "11"]
const POOL_IDS     = haskey(ENV, "POOL") ? String.(strip.(split(ENV["POOL"], ","))) : DEFAULT_POOL
const N_SNAP       = parse(Int, get(ENV, "N_SNAP", "0"))
const POOL_TAG     = haskey(ENV, "OUT_TAG") ? ENV["OUT_TAG"] :
                     POOL_IDS == DEFAULT_POOL ? "" : "_pool" * join(POOL_IDS)
const OUT       = joinpath(@__DIR__, (N_SNAP > 0 ? "out_n$(N_SNAP)" : "out") * POOL_TAG)
const BASIS_OUT = joinpath(OUT, "bases")
const KF_OUT    = joinpath(OUT, "kf")
const ENOI_OUT  = joinpath(OUT, "enoi")
mkpath(ENOI_OUT)

# truths: nature run vs the full pool; LOO members vs the pool without them (as run_partial_kf)
const TRUTHS   = haskey(ENV, "TRUTHS") ? String.(strip.(split(ENV["TRUTHS"], ","))) :
                 filter(t -> t == "truth" || t in POOL_IDS, ["truth", "11", "04"])
# ENS_FROM (2026-11-20): path to an earlier enoi/ dir on the SAME pool / N_SNAP / truths. EnOI and
# EnKF do not depend on the POD rep-error model, so their rows (bins, steps, summary) are copied
# from there and only the POD-KF rows are recomputed (skips ~6 h of EnKF).
const ENS_FROM = get(ENV, "ENS_FROM", "")
const STATIC_C = isempty(ENS_FROM) ? Float64[Inf, 250, 500, 1000, 2000] : Float64[]  # GC half-width [m]; support = 2c
const CYCLE_C  = isempty(ENS_FROM) ? Float64[500, 1000, 2000] : Float64[]
const NE       = parse(Int, get(ENV, "ENOI_NE", "50"))  # cycled EnKF ensemble size
const ENS_SEED = 20261118
const DIST_BINS = Float64[0, 100, 250, 500, 1000, 2000, Inf]   # m from nearest sensor
const Z_AGL_SLICES = Float32[10, 50, 100]
const ROWCHUNK = 1 << 20
# POD-KF experiments to bin alongside, per truth id: (name, basis fold)
const POD_EXPS = Dict("truth" => [("truth_rake_full_norep", "truth"), ("truth_rake_full_rep", "truth")],
                      "11"    => [("loo11_rake_full_rep", "11")],
                      "04"    => [("loo04_rake_full_rep", "04")])

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))
@info "BLAS threads $(BLAS.get_num_threads()), Julia threads $(Threads.nthreads()), OUT=$OUT, " *
      "truths=$TRUTHS, Ne=$NE"; flush(stderr)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
"Gaspari & Cohn (1999, eq. 4.10) compactly supported 5th-order correlation; c = half-width."
function gaspari_cohn(r::Real, c::Real)
    isinf(c) && return 1.0
    z = r / c
    z >= 2 && return 0.0
    z <= 1 && return -0.25z^5 + 0.5z^4 + 0.625z^3 - (5 / 3)z^2 + 1
    return z^5 / 12 - 0.5z^4 + 0.625z^3 + (5 / 3)z^2 - 5z + 4 - (2 / 3) / z
end

# Top-down slice rows (copied from run_partial_kf.jl make_slices; horizontal u,v rows only).
function slice_rows(grid::GridInfo, domain::AnalysisDomain, zs)
    lin  = _masked_linear_index(domain, grid)
    n    = domain.n_grid
    topo = grid.zPos[:, :, 1]
    map(zs) do z
        p  = [lin[i, j, argmin(abs.(grid.zPos[i, j, :] .- (topo[i, j] + z)))]
              for i in 1:grid.Nx, j in 1:grid.Ny]
        pv = filter(>(0), vec(p))
        (z=z, rows=vcat(pv, pv .+ n))
    end
end

"Σ_t (X[:,t] − μ)² accumulated into acc (Float64), threaded over rows."
function accum_sqdev!(acc::Vector{Float64}, X::AbstractMatrix{Float32}, μ::AbstractVector{Float32})
    d, T = size(X)
    Threads.@threads for ch in collect(Iterators.partition(1:d, 1 << 16))
        @inbounds for t in 1:T, r in ch
            v = Float64(X[r, t]) - Float64(μ[r]); acc[r] += v * v
        end
    end
    return acc
end

"Row mean and row variance (unbiased) of an ensemble matrix, threaded."
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

"Multiply the obs columns of sensor j (3 per sensor) of a (d × 3ns) matrix by ρ[:, j] on every component block."
function localize!(M::AbstractMatrix{Float32}, ρ::Matrix{Float32}, n::Int)
    ns = size(ρ, 2)
    Threads.@threads for j in 1:ns
        @inbounds for oc in 3j-2:3j, comp in 0:2, p in 1:n
            M[comp*n+p, oc] *= ρ[p, j]
        end
    end
    return M
end

# Error accumulators: per-step full-field SSE and per-bin SSE (summed over steps).
mutable struct ErrAcc
    sse_k   :: Vector{Float64}      # (K,)
    sse_bin :: Matrix{Float64}      # (nbin, K)
end
ErrAcc(nbin, K) = ErrAcc(zeros(K), zeros(nbin, K))

"Accumulate (est − truth)² for rows `ch` into acc, est given as a (|ch| × K) matrix."
function accum_err!(acc::ErrAcc, E::AbstractMatrix{Float32}, Xt::AbstractMatrix{Float32},
                    ch::UnitRange{Int}, rowbin::Vector{Int8}; kcols=axes(Xt, 2))
    @inbounds for (kk, k) in enumerate(kcols), (ii, r) in enumerate(ch)
        e = Float64(E[ii, kk]) - Float64(Xt[r, k]); e2 = e * e
        acc.sse_k[k] += e2; acc.sse_bin[rowbin[r], k] += e2
    end
end

"Static-gain estimate x = x0 + G*D over all rows (chunked), errors vs truth. x0 = nothing → 0."
function eval_linear(x0, G::AbstractMatrix{Float32}, D::AbstractMatrix, truth_U, rowbin, nbin)
    d, K = size(truth_U)
    acc = ErrAcc(nbin, K)
    Df  = Float32.(D)
    buf = Matrix{Float32}(undef, ROWCHUNK, K)
    for ch in Iterators.partition(1:d, ROWCHUNK)
        b = view(buf, 1:length(ch), :)
        mul!(b, view(G, ch, :), Df)
        isnothing(x0) || (b .+= view(x0, ch))
        accum_err!(acc, b, truth_U, ch, rowbin)
    end
    return acc
end

slice_rmse(x0, G, D, truth_U, rows) =
    (E = Float32.(G[rows, :]) * Float32.(D); isnothing(x0) || (E .+= x0[rows]);
     vec(sqrt.(mean(abs2, E .- truth_U[rows, :]; dims=1))))

"Mean of a per-row quantity over each distance bin (binrows[b] = row indices of bin b)."
bin_mean(v::AbstractVector, binrows) =
    [isempty(idx) ? NaN : mean(Float64, view(v, idx)) for idx in binrows]

"diag((I − KH)(ρ∘B)) = varB − Σ_j K[:,j]·(ρ∘BHᵀ)[:,j], row-wise without a d×m temporary."
function posterior_diag(varB::Vector{Float64}, Kg::Matrix{Float32}, BHl::Matrix{Float32})
    d, mm = size(Kg)
    out = similar(varB)
    Threads.@threads for ch in collect(Iterators.partition(1:d, 1 << 16))
        @inbounds for r in ch
            s = 0.0
            for j in 1:mm; s += Float64(Kg[r, j]) * Float64(BHl[r, j]); end
            out[r] = varB[r] - s
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# 1. Pool, grid, cache (N_SNAP cut identical to run_partial_kf.jl)
# ---------------------------------------------------------------------------
config = load_ensemble_config(CONFIG_PATH; include_ids=POOL_IDS)
if N_SNAP > 0
    cut(m) = (keep = findall(t -> t >= T_SPINUP, m.sim_times)[1:N_SNAP];
              MemberInfo(m.id, m.output_dir, m.met_dir_deg, m.U_g_ms, m.V_g_ms,
                         m.files[keep], m.timesteps[keep], m.sim_times[keep]))
    config = EnsembleConfig(config.n_members, cut.(config.prior_members), cut(config.truth_member))
    @warn "N_SNAP mode: first $N_SNAP post-spinup snapshots per member"
end
members = config.prior_members
grid    = load_grid(members[1].files[1])
domain  = AnalysisDomain(grid; margin_m=MARGIN_M)
n, dstate = domain.n_grid, domain.state_dim
@info "Grid $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$dstate, pool=$POOL_IDS"; flush(stderr)
cache = load_member_cache(members, domain; t_spinup=T_SPINUP)
T     = size(first(values(cache)), 2)

# geometry: cell coordinates (masked order), sensors, distances, bins
cx, cy, cz = grid.xPos[domain.mask], grid.yPos[domain.mask], grid.zPos[domain.mask]
lin   = _masked_linear_index(domain, grid)
slices = slice_rows(grid, domain, Z_AGL_SLICES)
MVAR  = sensor_mvar()
locs  = rake_locations(grid, domain)
ns, m = length(locs), 3length(locs)
nbin  = length(DIST_BINS) - 1

# ---------------------------------------------------------------------------
# 2. Per truth: fold pool, measurements, EnOI sweep, EnKF sweep, POD binning
# ---------------------------------------------------------------------------
bins_io = open(joinpath(ENOI_OUT, "bins.csv"), "w")
println(bins_io, "method,exp,truth,c_m,bin_lo_m,bin_hi_m,n_rows,rmse_est,rmse_prior,spread")
if !isempty(ENS_FROM)
    ENS_FROM == ENOI_OUT && error("ENS_FROM must differ from this run's enoi dir")
    for ln in Iterators.drop(eachline(joinpath(ENS_FROM, "bins.csv")), 1)
        startswith(ln, "pod_") || println(bins_io, ln)
    end
    for f in readdir(ENS_FROM)
        (endswith(f, "_steps.csv") || f == "summary.csv") && cp(joinpath(ENS_FROM, f), joinpath(ENOI_OUT, f); force=true)
    end
    @info "copied EnOI/EnKF rows + steps + summary from $ENS_FROM"
end
results = []

function write_bins(method, name, tid, c, acc::ErrAcc, prior::ErrAcc, spread_bin, nrow_bin, K)
    for b in 1:nbin
        nr = nrow_bin[b]
        @printf(bins_io, "%s,%s,%s,%g,%g,%g,%d,%.6f,%.6f,%.6f\n", method, name, tid, c,
                DIST_BINS[b], DIST_BINS[b+1], nr,
                sqrt(sum(acc.sse_bin[b, :]) / (nr * K)), sqrt(sum(prior.sse_bin[b, :]) / (nr * K)),
                sqrt(spread_bin[b]))
    end
    flush(bins_io)
end

function write_steps(name, truth_t, rm_est, rm_prior, spread, nisv, sl)
    open(joinpath(ENOI_OUT, "$(name)_steps.csv"), "w") do io
        println(io, join(vcat(["step", "time", "field_est", "field_prior", "field_spread", "nis"],
                              [@sprintf("z%03d_est", round(Int, s.z)) for s in slices]), ","))
        for k in eachindex(truth_t)
            println(io, join(vcat(Any[k, truth_t[k], rm_est[k], rm_prior[k], spread[k], nisv[k]],
                                  [s[k] for s in sl]), ","))
        end
    end
end

for tid in TRUTHS
    @info "================ truth = $tid ================"; flush(stderr)
    pool = filter(!=(tid), POOL_IDS)
    Nf   = length(pool) * T
    tm      = resolve_member(config, tid)
    truth_U = tid in POOL_IDS ? cache[tid] : load_member_snapshots(tm; t_spinup=T_SPINUP, domain)
    truth_t = tm.sim_times[_spinup_mask(tm, T_SPINUP)]
    K       = length(truth_t)

    # measurements: same seed / order as run_partial_kf (rake is sampled first for every truth)
    U4 = unpack_to_NCfield(truth_U, truth_t, grid, domain)
    Random.seed!(NOISE_SEED)
    seq = get_static_measurement_sequence(U4, grid, locs, domain, ns, MVAR, collect(1:K))
    U4 = nothing; GC.gc()
    Y = observation_matrix(seq)                                  # (m × K)
    R = Float64.(build_measurement_covariance(ns, MVAR))
    sp = [lin[mm.locidx...] for mm in seq[1].measurements]       # sensor cells (masked index)
    any(==(0), sp) && error("sensor outside analysis domain")
    obsrows = vec(permutedims(hcat(sp, sp .+ n, sp .+ 2n)))      # [u;v;w] per sensor
    # sanity: noiseless H applied to the truth equals the cell values
    @assert maximum(abs.(truth_U[obsrows, 1] .- Y[:, 1])) < 6 * sqrt(SENSOR_VAR)

    # distances cell→sensor, sensor→sensor; per-row distance bin
    Dist = Matrix{Float32}(undef, n, ns)
    Threads.@threads for j in 1:ns
        sx, sy, sz = cx[sp[j]], cy[sp[j]], cz[sp[j]]
        @inbounds for p in 1:n
            Dist[p, j] = sqrt((cx[p] - sx)^2 + (cy[p] - sy)^2 + (cz[p] - sz)^2)
        end
    end
    cellbin = Int8[searchsortedlast(DIST_BINS, minimum(view(Dist, p, :))) for p in 1:n]
    rowbin  = vcat(cellbin, cellbin, cellbin)
    binrows  = [findall(==(Int8(b)), rowbin) for b in 1:nbin]
    nrow_bin = length.(binrows)
    Dss = [sqrt((cx[a] - cx[b])^2 + (cy[a] - cy[b])^2 + (cz[a] - cz[b])^2) for a in sp, b in sp]
    @info "rows per distance bin: " * join(["[$(DIST_BINS[b]),$(DIST_BINS[b+1])): $(nrow_bin[b])" for b in 1:nbin], "  ")

    # ---- pool moments: x̄, BHᵀ (one GEMM pass), HBHᵀ, diag B ----
    Yens = Float64.(reduce(hcat, [cache[id][obsrows, :] for id in pool]))   # (m × Nf)
    Hxb  = vec(mean(Yens; dims=2))
    HA   = Yens .- Hxb
    HBH  = HA * HA' ./ (Nf - 1)
    W    = Float32.(hcat(fill(1 / Nf, Nf), HA' ./ (Nf - 1)))               # (Nf × (1+m))
    acc  = zeros(Float32, dstate, 1 + m)
    for (i, id) in enumerate(pool)
        mul!(acc, cache[id], view(W, (i-1)*T+1:i*T, :), true, true)
    end
    xbar = acc[:, 1]; BH = acc[:, 2:end]; acc = nothing; GC.gc()
    varB = zeros(Float64, dstate)
    for id in pool; accum_sqdev!(varB, cache[id], xbar); end
    varB ./= (Nf - 1)
    @info @sprintf("pool %s: N=%d, mean var(B)=%.4f m²/s²", join(pool, " "), Nf, mean(varB)); flush(stderr)

    # prior (climatological mean) errors — shared baseline for every method on this truth
    prior_acc = eval_linear(xbar, zeros(Float32, dstate, 1), zeros(1, K), truth_U, rowbin, nbin)
    rm_prior  = sqrt.(prior_acc.sse_k ./ dstate)

    # ---- static EnOI sweep ----
    for c in STATIC_C
        name = "enoi_c$(isinf(c) ? "inf" : Int(c))" * (tid == "truth" ? "" : "_loo$tid")
        ρ    = Float32.(gaspari_cohn.(Dist, c))
        ρyy  = kron(gaspari_cohn.(Dss, c), ones(3, 3))
        S    = Symmetric(ρyy .* HBH + R)
        Sinv = inv(S)
        BHl  = localize!(copy(BH), ρ, n)
        Kg   = BHl * Float32.(Sinv)                                       # (d × m)
        diagPa = posterior_diag(varB, Kg, BHl)
        BHl = nothing; GC.gc()
        Dinn = Y .- Hxb
        nisv = [dot(Dinn[:, k], Sinv * Dinn[:, k]) / m for k in 1:K]
        a    = eval_linear(xbar, Kg, Dinn, truth_U, rowbin, nbin)
        rm   = sqrt.(a.sse_k ./ dstate)
        sl   = [slice_rmse(xbar, Kg, Dinn, truth_U, s.rows) for s in slices]
        spread = fill(sqrt(mean(max.(diagPa, 0))), K)
        write_steps(name, truth_t, rm, rm_prior, spread, nisv, sl)
        write_bins("enoi", name, tid, c, a, prior_acc, bin_mean(max.(diagPa, 0), binrows), nrow_bin, K)
        neg = count(<(0), diagPa)
        push!(results, (name=name, method="enoi", truth=tid, c_m=c, Ne=Nf, field_est=mean(rm),
                        field_est_2ndhalf=mean(rm[K÷2+1:end]), field_prior=mean(rm_prior),
                        field_spread=spread[1], mean_nis=mean(nisv),
                        z050_est=mean(sl[findfirst(s -> s.z == 50f0, slices)]), neg_var_rows=neg))
        @info @sprintf("%-22s RMSE %.4f | prior %.4f | spread %.4f | NIS %.2f | neg-var rows %d",
                       name, mean(rm), mean(rm_prior), spread[1], mean(nisv), neg); flush(stderr)
        Kg = nothing; GC.gc()
    end

    # ---- cycled perturbed-obs EnKF sweep (persistence + increment noise) ----
    μinc = zeros(Float32, dstate)                       # mean 6 s increment (telescopes)
    for id in pool; μinc .+= (view(cache[id], :, T) .- view(cache[id], :, 1)); end
    μinc ./= length(pool) * (T - 1)
    incr_pool = [(id, t) for id in pool for t in 1:T-1]
    Lr = cholesky(Symmetric(R)).L
    for c in CYCLE_C
        name = "enkf_ne$(NE)_c$(Int(c))" * (tid == "truth" ? "" : "_loo$tid")
        rng  = MersenneTwister(ENS_SEED)
        ρ    = Float32.(gaspari_cohn.(Dist, c))
        ρyy  = kron(gaspari_cohn.(Dss, c), ones(3, 3))
        allc = [(id, t) for id in pool for t in 1:T]
        init = NE <= Nf ? allc[randperm(rng, Nf)[1:NE]] : allc[rand(rng, 1:Nf, NE)]
        E    = Matrix{Float32}(undef, dstate, NE)
        for (j, (id, t)) in enumerate(init); E[:, j] = view(cache[id], :, t); end
        a    = ErrAcc(nbin, K)
        sl   = [zeros(K) for _ in slices]
        spread = zeros(K); nisv = zeros(K); var_bin = zeros(nbin)
        BHe  = Matrix{Float32}(undef, dstate, m)
        for k in 1:K
            if k > 1                                    # forecast: random walk with pool Q
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
            localize!(BHe, ρ, n)
            S   = Symmetric(ρyy .* (HAe * HAe' ./ (NE - 1)) + R)
            Sf  = cholesky(S)
            dk  = Y[:, k] .- hm
            nisv[k] = dot(dk, Sf \ dk) / m
            ε   = Lr * randn(rng, m, NE); ε .-= mean(ε; dims=2)      # centered obs perturbations
            mul!(E, BHe, Float32.(Sf \ (Y[:, k] .+ ε .- HE)), true, true)
            xm, xv = ens_moments(E)
            accum_err!(a, reshape(xm, :, 1), truth_U, 1:dstate, rowbin; kcols=k:k)
            for (is, s) in enumerate(slices)
                sl[is][k] = sqrt(mean(abs2, xm[s.rows] .- truth_U[s.rows, k]))
            end
            spread[k] = sqrt(mean(Float64, xv))
            var_bin .+= bin_mean(xv, binrows) ./ K
            k % 10 == 1 && (@info @sprintf("  %s step %d/%d  RMSE %.4f spread %.4f NIS %.2f", name, k, K,
                                           sqrt(a.sse_k[k] / dstate), spread[k], nisv[k]); flush(stderr))
        end
        E = nothing; BHe = nothing; GC.gc()
        rm = sqrt.(a.sse_k ./ dstate)
        write_steps(name, truth_t, rm, rm_prior, spread, nisv, sl)
        write_bins("enkf", name, tid, c, a, prior_acc, var_bin, nrow_bin, K)
        push!(results, (name=name, method="enkf", truth=tid, c_m=c, Ne=NE, field_est=mean(rm),
                        field_est_2ndhalf=mean(rm[K÷2+1:end]), field_prior=mean(rm_prior),
                        field_spread=mean(spread), mean_nis=mean(nisv),
                        z050_est=mean(sl[findfirst(s -> s.z == 50f0, slices)]), neg_var_rows=0))
        @info @sprintf("%-22s RMSE %.4f (2nd half %.4f) | spread %.4f | NIS %.2f",
                       name, mean(rm), mean(rm[K÷2+1:end]), mean(spread), mean(nisv)); flush(stderr)
    end
    BH = nothing; GC.gc()

    # ---- POD-KF (and its projection floor) binned the same way ----
    for (pname, fold) in get(POD_EXPS, tid, [])
        kfp, bp = joinpath(KF_OUT, "kf_result_$(pname).jld2"), basis_filename(BASIS_OUT, fold)
        pcp = joinpath(KF_OUT, "posterior_cov_$(pname).jld2")
        if !(isfile(kfp) && isfile(bp) && isfile(pcp))
            @warn "POD outputs for $pname not found in $OUT — skipping POD binning"; continue
        end
        r  = load(kfp)
        b  = load_pod_basis(bp)
        size(b.modes, 2) == r["L"] || (@warn "$pname: rank-truncated run, skipping"; continue)
        if r["K"] != K || sort(r["member_ids"]) != sort(pool)
            @warn "$pname: K/pool mismatch with this run ($(r["K"]), $(r["member_ids"])) — skipping"; continue
        end
        Pbar = Float32.(dropdims(mean(load(pcp, "P_post"); dims=3); dims=3))
        pvar = vec(sum((b.modes * Pbar) .* b.modes; dims=2))          # time-mean diag(Pr P Prᵀ)
        a_est = eval_linear(nothing, b.modes, r["α"], truth_U, rowbin, nbin)
        a_orc = eval_linear(nothing, b.modes, r["truth_coeffs"], truth_U, rowbin, nbin)
        write_bins("pod_kf", pname, tid, NaN, a_est, prior_acc, bin_mean(pvar, binrows), nrow_bin, K)
        write_bins("pod_oracle", pname, tid, NaN, a_orc, prior_acc, zeros(nbin), nrow_bin, K)
        @info @sprintf("%-22s (POD) RMSE %.4f | oracle %.4f  [binned]", pname,
                       mean(sqrt.(a_est.sse_k ./ dstate)), mean(sqrt.(a_orc.sse_k ./ dstate))); flush(stderr)
        b = nothing; GC.gc()
    end
    truth_U = nothing; GC.gc()
end
close(bins_io)

isempty(results) || open(joinpath(ENOI_OUT, "summary.csv"), "w") do io
    isempty(results) && return
    ks = keys(results[1])
    println(io, join(ks, ","))
    for s in results; println(io, join([s[k] for k in ks], ",")); end
end
@info "=== done: $(length(results)) EnOI/EnKF runs → $ENOI_OUT ==="
