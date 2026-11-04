# run_kf_rank_sweep_dense.jl
# cgf cego6160@colorado.edu 7.15.26
#
# Rank sweep of the reduced-order KF with a DENSE sensor lattice filling the
# downwind (wake) region, instead of the 15-sensor tower rake used by
# run_kf_rank_sweep.jl / run_theta_baseline.jl.
#
# Purpose: the rake result (job 5051) showed the filter losing to the
# Θ-baseline in the wake at r=50 while its own subspace oracle sat at 0.230 —
# i.e. representation was fine and the ONLINE problem was under-determined
# (NIS 1.34, cond(H·Pr) large). This script removes the under-determination as
# a variable: an array spanning the whole downwind half of the analysis domain
# (crest to the interior x edge, full interior y), every SENSOR_STRIDE_XY grid
# cells in x and y, on every SENSOR_STRIDE_Z-th vertical level below
# WAKE_AGL_M. At the default 50-cell (500 m) spacing that is 175 sensors / 525
# observation rows vs the rake's 15 / 45 — ~10x over-determined against rank
# <= 50. The KF-vs-oracle gap that survives at this sensor count is
# attributable to the dynamics model (random walk + Q), not to the rake
# geometry. cond(H_r Σ0^{1/2}) is reported per rank for the same reason as in
# the rake sweep (gappy-POD conditioning, Manohar et al. 2018).
#
# SENSOR SPACING IS NOT FREE — READ BEFORE CHANGING IT (job 5069):
# R_rep from build_rep_covariance is INDEFINITE at every sensor density. The
# kernel in reperror.jl evaluates θ, L∥, L⊥ at each PAIR's midpoint cell (as
# specified in paper-drafts/reconstruction_error.md), and a stationary kernel
# form evaluated with spatially-varying parameters is not positive definite
# (Paciorek & Schervish 2006). Freezing the parameters at one cell makes the
# same matrix PSD to roundoff — nonstationarity is the whole difference.
# R_total stays PD only while R_rep's negative eigenvalues sit below the
# R_sensor floor (0.25). Measured min eig(R_sensor + R_rep), downwind span:
#   300 m: -0.247 (Cholesky FAILS)   400 m: +0.038   500 m: +0.204
#   600 m: +0.231                    800 m: +0.235
# 500 m is the default because it clears the floor with margin. Tightening the
# spacing walks back toward an invalid R; the PD guard below fails loudly
# rather than letting a bad R through. Making R_rep diagonal is NOT a fix — it
# would treat correlated observations as independent and flatter the filter.
const RTOT_MIN_EIG = 0.05   # PD guard: require min eig(R_base) above this (20% of the 0.25 floor)
#
# Everything else is held identical to run_kf_rank_sweep.jl: same basis, models,
# rep-error model, honest tail-clutter inflation of R, same wake scoring mask,
# same RNG seed.
#
# NUMERICS: with ~1.5e3 sensors the innovation covariance S is (3NM)² and
# refactorizing it every step would dominate the run. run_filter (src/kalman.jl)
# therefore uses the equivalent information/Woodbury form for a StaticObs with
# 3NM > r: R_total is factorized ONCE per rank and each step does only r×r work.
# All in Float64 (the dense rep-error correlations make R_total poorly
# conditioned in Float32). test/test_kalman.jl checks it against the gain form.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "paths.jl"))
const WORKDIR  = SRC_DIR
include(joinpath(WORKDIR, "timeresolved.jl"))   # defines parse_dump_ranks used by DUMP_RANKS below
const T_SPINUP = 900.0
const MARGIN_M = 600f0
const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const RANKS    = [2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]
# Ranks for which the time-resolved product (per-step wake RMSE, bias/fluct split, modal
# trajectories) is written. Must be a subset of RANKS and >= 13 to carry the meander mode.
# "" disables the dump. Kept VERBATIM in sync with run_kf_twin_dense.jl. See src/timeresolved.jl.
const DUMP_RANKS = parse_dump_ranks(get(ENV, "DUMP_RANKS", "15,20,50"))
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))
# Sensor lattice spacing, in GRID CELLS (dx = dy = 10 m). See the PD note in the header
# before reducing SENSOR_STRIDE_XY: below ~40 cells R_total loses positive definiteness.
const SENSOR_STRIDE_XY = parse(Int, get(ENV, "SENSOR_STRIDE_XY", "50"))
const SENSOR_STRIDE_Z  = parse(Int, get(ENV, "SENSOR_STRIDE_Z",  "10"))
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

Random.seed!(20260713)   # SAME seed as run_theta_baseline.jl -> paired noise

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
mdl    = load_models(models_filename(BASES_DIR, TRUTH_ID))
rep    = load_rep_error(rep_error_filename(BASES_DIR, TRUTH_ID))
Lfull  = basis.L
all(r -> r <= Lfull, RANKS) || error("RANKS exceed basis L=$Lfull")
let bad = setdiff(DUMP_RANKS, RANKS)
    isempty(bad) || error("DUMP_RANKS $(DUMP_RANKS) must be a subset of RANKS; offenders: $bad")
end

# --- Truth ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = Matrix{Float32}(0.25f0 * I, 3, 3)
TIDX = collect(1:OBS_STRIDE:length(truth_t))
K    = length(TIDX)
QSCALE = Float32(OBS_STRIDE)

crest = argmax(grid.zPos[:, :, 1])
crest_x, crest_y = grid.xPos[crest[1], crest[2], 1], grid.yPos[crest[1], crest[2], 1]

# --- Wake box (scoring mask AND sensor-lattice extent; same numbers as the rake sweep) ---
# Full analysis y-width: constant scoring area across LOO folds (the old
# ±750 m strip let off-axis truths sweep their wake out of the box).
WAKE_DX_M, WAKE_AGL_M = 2500f0, 250f0
n   = domain.n_grid
lin = _masked_linear_index(domain, grid)
terrain3 = repeat(grid.zPos[:, :, 1:1], 1, 1, grid.Nz)
wake3 = (grid.xPos .>= crest_x) .& (grid.xPos .<= crest_x + WAKE_DX_M) .&
        ((grid.zPos .- terrain3) .<= WAKE_AGL_M) .& domain.mask
ps   = lin[wake3]
widx = vcat(ps, ps .+ n, ps .+ 2n)

# --- Sensor array spanning the downwind half of the analysis domain: from the crest to
#     the interior x edge, across the full interior y, every SENSOR_STRIDE_XY cells in
#     x,y and every SENSOR_STRIDE_Z-th level below WAKE_AGL_M. The array is deliberately
#     WIDER than the wake scoring box — the filter estimates global POD coefficients, so
#     observations outside the box still constrain the modes that carry the wake.
#     A column's vertical levels are selected per-column because the sigma coordinate
#     rides the terrain, so a fixed k-list would sit at different AGL over the hill than
#     in the flat wake. Level 1 is skipped (surface-adjacent cell).
ix_dom_lo = findfirst(i -> any(domain.mask[i, :, :]), 1:grid.Nx)
ix_hi     = findlast( i -> any(domain.mask[i, :, :]), 1:grid.Nx)
iy_lo     = findfirst(j -> any(domain.mask[:, j, :]), 1:grid.Ny)
iy_hi     = findlast( j -> any(domain.mask[:, j, :]), 1:grid.Ny)
ix_lo     = max(ix_dom_lo, findfirst(i -> grid.xPos[i, 1, 1] >= crest_x, 1:grid.Nx))
locs = Vector{Vector{Float32}}()
for i in ix_lo:SENSOR_STRIDE_XY:ix_hi, j in iy_lo:SENSOR_STRIDE_XY:iy_hi
    terr = grid.zPos[i, j, 1]
    for k in 2:SENSOR_STRIDE_Z:grid.Nz
        (grid.zPos[i, j, k] - terr) <= WAKE_AGL_M || break
        domain.mask[i, j, k] || continue
        push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
    end
end
NM = length(locs)
NM > 0 || error("sensor array is empty — check the downwind span against the domain margin")
@info "Sensor array: NM=$NM sensors ($(3NM) obs rows); xy stride $(SENSOR_STRIDE_XY) cells " *
      "($(SENSOR_STRIDE_XY * grid.dx) m), z stride $(SENSOR_STRIDE_Z) levels; " *
      "x $(grid.xPos[ix_lo,1,1])–$(grid.xPos[ix_hi,1,1]) m, y $(grid.yPos[1,iy_lo,1])–$(grid.yPos[1,iy_hi,1]) m"

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; GC.gc()

# --- Static observation pieces (full rank, sliced per r) ---
Hfull  = build_measurement_operator(meas_seq[1], basis, domain)   # (3NM × Lfull)
R_base = Float64.(build_measurement_covariance(NM, MVAR) +
                  build_rep_covariance(meas_seq[1], rep, domain)) # sensor + stored rep

# PD guard (see the header note). R_total = R_base + H_tail Σ H_tailᵀ and the tail term is
# PSD, so R_base PD ⇒ R_total PD: checking R_base here is sufficient and catches a bad
# spacing before the rank loop rather than inside it.
let e = eigvals(Symmetric(R_base))
    λmin = minimum(e)
    @info "min eig(R_sensor + R_rep) = $(round(λmin, digits=4))  cond = $(round(maximum(e)/max(λmin,1e-12), digits=1))"
    λmin >= RTOT_MIN_EIG || error(
        "R_base min eigenvalue $(round(λmin, digits=4)) < $(RTOT_MIN_EIG): the rep-error " *
        "kernel is not PSD and at $(SENSOR_STRIDE_XY * grid.dx) m spacing its negative " *
        "eigenvalues are no longer masked by the $(MVAR[1,1]) sensor-noise floor. " *
        "Increase SENSOR_STRIDE_XY (500 m / stride 50 is the calibrated default), or fix " *
        "the midpoint-parameter kernel in src/reperror.jl. See the header note.")
end
Y = Matrix{Float64}(undef, 3NM, K)                                # pre-stacked observations
for k in 1:K
    Y[:, k] = stack_measurements(meas_seq[k])
end
meas_seq = nothing; GC.gc()

# --- Truth reductions (identical to the rake sweep) ---
astar   = basis.modes' * truth_U                 # (Lfull × Ktot) oracle coefficients
truthW  = truth_U[widx, :]                       # wake-region truth (nw × Ktot)
PrW     = basis.modes[widx, :]                   # wake rows of the modes (nw × Lfull)
state_d = domain.state_dim

# --- Time-resolved dump setup (rank-independent pieces computed once) ---
tr_base    = truth_wake_baseline(truthW, length(widx), TIDX)   # persistence-of-mean baseline
dump_t     = Float64.(truth_t[TIDX])                           # assimilated snapshot times
dumps      = Dict{Int,Any}()
Acoef      = Dict{Int,Matrix{Float32}}()
astar_coef = Dict{Int,Matrix{Float32}}()

const RESID_CHUNK = 32
function resid2_full(modes_r, C::AbstractMatrix{Float32}, cols::Vector{Int})
    s = 0.0
    for c0 in 1:RESID_CHUNK:length(cols)
        cc = c0:min(c0 + RESID_CHUNK - 1, length(cols))
        E  = modes_r * C[:, cc] .- truth_U[:, cols[cc]]
        s += mapreduce(x -> abs2(Float64(x)), +, E)
    end
    return s
end

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
csvpath = joinpath(KF_DIR, "kf_rank_sweep_dense_truth_$(TRUTH_ID)$(stride_tag).csv")
open(csvpath, "w") do io
    println(io, "r,n_sensors,kf_rmse_full,kf_rmse_wake,oracle_rmse_full,oracle_rmse_wake,mean_nis,cond_HPr")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation: R = R_base + H_tail Σ0_tail H_tailᵀ.
        # With 3NM ≫ r, run_filter's :auto picks the information/Woodbury form (R factorized
        # once per rank, r×r work per step) — see the NUMERICS note in the header.
        obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
        model = KFModel(mdl, r; qscale=QSCALE)
        cnd   = whitened_condition(obs.H, model.Σ0)
        h     = run_filter(model, Y, obs)
        A     = Float32.(h.α_post)
        PrWr  = @view PrW[:, 1:r]
        kf_w = 0.0; or_w = 0.0
        dumping  = r in DUMP_RANKS
        rankdump = dumping ? RankDump(length(widx), K) : nothing
        for k in 1:K
            tk = TIDX[k]
            a_r  = @view astar[1:r, tk]
            xtw  = @view truthW[:, tk]
            e_kf = PrWr * A[:, k] .- xtw
            e_or = PrWr * a_r     .- xtw
            kf_w += sum(abs2, e_kf)
            or_w += sum(abs2, e_or)
            dumping && accumulate_step!(rankdump, k, e_kf, e_or)
        end
        if dumping
            dumps[r]      = finalize_dump(rankdump, K, length(widx))
            Acoef[r]      = copy(A)                          # (r × K) KF modal trajectory
            astar_coef[r] = Matrix{Float32}(astar[1:r, TIDX])  # (r × K) oracle trajectory
        end
        modes_r = @view basis.modes[:, 1:r]
        kf_f = resid2_full(modes_r, A,                TIDX)
        or_f = resid2_full(modes_r, astar[1:r, TIDX], TIDX)
        row = (sqrt(kf_f / (K * state_d)), sqrt(kf_w / (K * length(widx))),
               sqrt(or_f / (K * state_d)), sqrt(or_w / (K * length(widx))),
               mean_nis(h), cnd)
        println(io, join((r, NM, row...), ","))
        flush(io)
        @info "r=$r  KF full=$(round(row[1],digits=4)) wake=$(round(row[2],digits=4))  " *
              "oracle full=$(round(row[3],digits=4)) wake=$(round(row[4],digits=4))  " *
              "NIS=$(round(row[5],digits=3))  cond(HPr)=$(round(row[6],digits=1))"
    end
end
@info "Saved $csvpath"

if !isempty(DUMP_RANKS)
    tspath = replace(csvpath, ".csv" => "_timeseries.jld2")
    save_timeresolved(tspath, dump_t, DUMP_RANKS, tr_base, dumps, Acoef, astar_coef)
    @info "Saved $tspath  (time-resolved ranks $(DUMP_RANKS))"
end
