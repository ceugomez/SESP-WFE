# run_kf_transect_sweep.jl
# cgf cego6160@colorado.edu 7.15.26
#
# Rank sweep of the reduced-order KF observed by a TRANSITING POINT OBSERVER: one sensor
# at fixed x and fixed height AGL, translating up and down along y across the wake as a
# triangle wave. Companion to run_kf_rank_sweep.jl (15-sensor static rake) and
# run_kf_rank_sweep_dense.jl (175-sensor downwind array).
#
# WHAT THIS ASKS, vs the array sweep. The array removes under-determination to isolate the
# dynamics model: 525 obs rows against rank <= 50, so a surviving KF-vs-oracle gap is Q's
# fault, not the geometry's. This script asks the opposite, deployment-relevant question:
# what can a SINGLE moving point pin down? Each update sees 3 rows against rank r — far
# MORE under-determined per update than the rake. It can only work by accumulating
# information across updates, and how much accumulates is set by the balance between Q's
# inflation and the measurement noise (effective window ~ sqrt(r_obs/q) per mode). Fast
# modes decorrelate before the observer can revisit them; the forcing head does not.
# Expect the estimable object to be the conditional envelope (deficit, meander band), not
# the turbulence — which is the paper's affirmative claim, tested directly.
#
# WHY A MOVING SENSOR SIDESTEPS THE PSD PROBLEM (see run_kf_rank_sweep_dense.jl header).
# With one sensor per update, build_rep_covariance collapses to diag(σ²_u,σ²_v,σ²_w) and
# the non-PSD midpoint-parameter kernel is never evaluated. R_total is 3x3 and trivially PD — no spacing floor, no guard needed.
#
# THE BIAS THIS BUYS INSTEAD — READ BEFORE TRUSTING NIS. Sidestepping the kernel does not
# make the rep error white; it relocates the correlation from space into TIME, where no
# matrix's eigenvalues betray it. Successive samples along the track are rep-correlated and
# the filter treats them as independent — the SAME over-counting bias a diagonal R_rep would
# buy, and it makes the filter overconfident (NIS < 1, posterior variance a lower bound).
# SPEED IS THE LEVER: the along-track step is v·Δt, so the correlation between consecutive
# samples is ~exp(-v·Δt / L⊥), L⊥ ≈ 466 m fitted.
#   v=15  m/s -> 90 m  -> rho ~ 0.82   (badly correlated; the original default)
#   v=78  m/s -> 468 m -> rho ~ 0.37   (decorrelated; current default)
#   v=128 m/s -> 768 m -> rho ~ 0.19
# At 78 m/s the bias is much reduced but NOT zero — read NIS with that in mind.
#
# CADENCE INTERACTS WITH SPEED. The old three-way tension (short cadence => correlated rep,
# 30 s => dt ~ T_meander/2 where the meander's oscillatory autocorrelation makes F = I WORSE
# than climatology, 60 s => too few updates) was a trap only for a SLOW sensor, where the
# sole way to decorrelate the rep error was to wait. A fast observer decorrelates by moving
# instead, so 6 s cadence keeps persistence solidly valid (dt << T ~ 60 s) AND gets rho~0.37
# — the corner the static rake cannot reach. Default is 6 s; sweep OBS_STRIDE to map it, but
# note stride 5 still lands on the T/2 persistence trap.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, RANKS, sensors, seed

const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))
# Transect geometry. x and z match the static rake (x = 3 km, downstream of the hill) so the
# comparison against run_kf_rank_sweep.jl differs only in the sensor's motion.
const TRANSECT_X_M     = parse(Float32, get(ENV, "TRANSECT_X_M", "3000"))
const TRANSECT_Z_AGL_M = parse(Float32, get(ENV, "TRANSECT_Z_AGL_M", "100"))
# Default spans the full interior y (crest_y ± 1900 m = 635–4435 m, just inside the 605–4455
# sponge margin), matching the scoring mask's full analysis y-width.
const TRANSECT_HALFY_M = parse(Float32, get(ENV, "TRANSECT_HALFY_M", "1900"))
# SPEED IS NOT A FREE KNOB — IT ALIASES. The triangle wave sampled at a fixed cadence
# revisits the same cells whenever its period resonates with Δt: the observer then sweeps
# a sparse comb forever while looking busy. Distinct y cells visited ≈
# 2·span / gcd(v·Δt, 2·span), so at 3800 m span / 6 s cadence:
#   v=50 -> 39 cells | v=100 -> 20 cells (!) | v=150 -> 39 cells
#   v=78 -> 256 cells | v=128 -> 238 cells
# 78 m/s is the default for three independent reasons that happen to coincide:
#   (1) gcd(468, 7600) = 4 -> near-complete coverage (256 distinct of 268 samples);
#   (2) the 468 m along-track step ≈ fitted L⊥ (466 m), so consecutive samples DECORRELATE
#       (ρ 0.82 -> 0.37) — this is what retires most of the over-counting bias below;
#   (3) the 48.7 s pass time < T_meander (~60 s), so the observer crosses the wake faster
#       than the wake moves, instead of smearing across two meander cycles.
# The coverage guard below fails loudly if a chosen speed aliases. This is an ABSTRACT point
# sensor (78 m/s is a scan rate, not an aircraft) — see the run log for the framing.
const TRANSECT_SPEED   = parse(Float32, get(ENV, "TRANSECT_SPEED", "78"))
# Guard: fraction of the traversed line's cells the observer must actually visit.
const MIN_COVERAGE_FRAC = 0.25

Random.seed!(NOISE_SEED)   # same seed in every driver -> paired noise

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = resolve_member(config, TRUTH_ID)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
mdl    = load_models(models_filename(BASES_DIR, TRUTH_ID))
rep    = load_rep_error(rep_error_filename(BASES_DIR, TRUTH_ID))
Lfull  = basis.L
all(r -> r <= Lfull, RANKS) || error("RANKS exceed basis L=$Lfull")

truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = sensor_mvar()
TIDX = collect(1:OBS_STRIDE:length(truth_t))
K    = length(TIDX)
QSCALE = Float32(OBS_STRIDE)   # random-walk variance-adds; see the caveat in statmodel.jl

_, crest_y = hill_crest(grid)

# --- Transiting observer ---
meas_seq, locs = get_transect_sequence(truth, grid, domain, MVAR, TIDX, truth_t;
    x_m=TRANSECT_X_M, z_agl_m=TRANSECT_Z_AGL_M,
    y_lo=crest_y - TRANSECT_HALFY_M, y_hi=crest_y + TRANSECT_HALFY_M,
    speed_ms=TRANSECT_SPEED)
truth = nothing; GC.gc()

dt_obs   = Float64(truth_t[TIDX[min(2,end)]] - truth_t[TIDX[1]])
span_m   = 2 * TRANSECT_HALFY_M
n_trav   = TRANSECT_SPEED * (truth_t[TIDX[end]] - truth_t[TIDX[1]]) / span_m
@info "Transect: x=$(TRANSECT_X_M) m, $(TRANSECT_Z_AGL_M) m AGL, y ∈ " *
      "$(crest_y - TRANSECT_HALFY_M)–$(crest_y + TRANSECT_HALFY_M) m, speed=$(TRANSECT_SPEED) m/s"
@info "  K=$K updates at Δt=$(dt_obs) s (3 obs rows each); " *
      "$(round(TRANSECT_SPEED*dt_obs, digits=1)) m along-track per update; " *
      "$(round(n_trav, digits=1)) traverses over the record"

# Aliasing guard (see the SPEED note above). If the triangle wave resonates with the
# sampling cadence the observer revisits a sparse comb of cells forever — the run still
# "works" and the coverage is silently a fraction of the line, so fail here instead.
let ys = unique([l[2] for l in locs])
    n_line = round(Int, 2 * TRANSECT_HALFY_M / grid.dy) + 1
    frac   = length(ys) / min(n_line, K)
    @info "  coverage: $(length(ys)) distinct y cells visited of $(min(n_line, K)) reachable " *
          "($(round(100frac, digits=1))%); along-track step vs L⊥ ≈ 466 m -> " *
          "ρ_rep ≈ $(round(exp(-TRANSECT_SPEED*dt_obs/466), digits=2))"
    frac >= MIN_COVERAGE_FRAC || error(
        "transect visits only $(length(ys)) distinct y cells ($(round(100frac,digits=1))% of " *
        "reachable): speed $(TRANSECT_SPEED) m/s aliases against the $(dt_obs) s cadence " *
        "(gcd($(TRANSECT_SPEED*dt_obs), $(2*TRANSECT_HALFY_M)) is large). Pick a speed whose " *
        "along-track step shares little with the span — 78 m/s is the calibrated default.")
end

# --- Per-step observation pieces. H and R CHANGE each step (the sensor moves), unlike the
#     static-rake scripts which build them once. Each is tiny (3 rows), so this is cheap.
Hk = [build_measurement_operator(ms, basis, domain) for ms in meas_seq]      # K x (3 × Lfull)
Rk = [Float64.(build_measurement_covariance(1, MVAR) +
               build_rep_covariance(ms, rep, domain)) for ms in meas_seq]    # K x (3 × 3)
Yk_mat = observation_matrix(meas_seq)                                      # (3 × K)
meas_seq = nothing; GC.gc()

astar   = basis.modes' * truth_U
state_d = domain.state_dim

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
csvpath = joinpath(KF_DIR, "kf_transect_sweep_truth_$(TRUTH_ID)$(stride_tag).csv")
open(csvpath, "w") do io
    println(io, "r,n_updates,kf_rmse_full,oracle_rmse_full," *
                "mean_nis,cond_gram,var_frac")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation per step: the discarded tail is clutter
        # at THIS sensor position, R_k = R_base_k + H_tail,k Σ0_tail H_tail,kᵀ (3×3, PD).
        # VaryingObs -> gain form (S is 3×3, no Woodbury needed).
        obs   = truncated_obs(Hk, Rk, mdl.Σ_α0, r)
        model = KFModel(mdl, r; qscale=QSCALE)
        # Frozen-field observability of the WHOLE transect, the moving-sensor analogue of
        # the rake's cond(H_r Σ0^{1/2}): a single 3×r H has rank 3 < r and no meaningful
        # condition number, so accumulate the information Gramian over every update,
        # whitened by the prior. G = Σ0^{1/2} (Σ_k H_kᵀ R_k⁻¹ H_k) Σ0^{1/2}. This is what
        # the transect could determine IF the field held still; comparing it to the
        # achieved skill isolates how much Q's inflation costs. Uses R_k WITHOUT the tail
        # clutter (as the pre-consolidation code did), unlike the filter's own R.
        cnd   = gramian_condition(VaryingObs([H[:, 1:r] for H in Hk], Rk), model.Σ0)
        h     = run_filter(model, Yk_mat, obs)
        A     = Float32.(h.α_post)
        tr0   = tr(model.Σ0)
        vfrac = sum(tr(@view h.P_post[:, :, k]) for k in 1:K) / tr0
        modes_r = @view basis.modes[:, 1:r]
        kf_f = resid2_full(modes_r, A,                truth_U, TIDX)
        or_f = resid2_full(modes_r, astar[1:r, TIDX], truth_U, TIDX)
        row = (sqrt(kf_f / (K * state_d)), sqrt(or_f / (K * state_d)),
               mean_nis(h), cnd, vfrac / K)
        println(io, join((r, K, row...), ","))
        flush(io)
        @info "r=$r  KF full=$(round(row[1],digits=4))  oracle full=$(round(row[2],digits=4))  " *
              "NIS=$(round(row[3],digits=3))  cond(Gram)=$(round(row[4],digits=1))  " *
              "P/P0=$(round(row[5],digits=3))"
    end
end
@info "Saved $csvpath"
