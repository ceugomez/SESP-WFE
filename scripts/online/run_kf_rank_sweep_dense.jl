# run_kf_rank_sweep_dense.jl
# cgf cego6160@colorado.edu 7.15.26
#
# Rank sweep of the reduced-order KF with a DENSE sensor lattice over the
# downwind half of the domain, instead of the 15-sensor tower rake used by
# run_kf_rank_sweep.jl / run_theta_baseline.jl.
#
# Purpose: with the rake the full-rank online problem is under-determined
# (45 obs rows, cond(H·Pr) large). This script removes the under-determination as
# a variable: an array spanning the whole downwind half of the analysis domain
# (crest to the interior x edge, full interior y), every SENSOR_STRIDE_XY grid
# cells in x and y, on every SENSOR_STRIDE_Z-th vertical level below
# LATTICE_AGL_M. At the default 50-cell (500 m) spacing that is 175 sensors / 525
# observation rows vs the rake's 15 / 45 — ~10x over-determined against rank
# <= 50. The KF-vs-oracle gap that survives at this sensor count is
# attributable to the dynamics model (random walk + Q), not to the rake
# geometry. cond(H_r Σ0^{1/2}) is reported per rank for the same reason as in
# the rake sweep (gappy-POD conditioning, Manohar et al. 2018).
#
# SENSOR SPACING IS NOT FREE: R_rep is indefinite (midpoint-parameter kernel), and below
# ~400 m spacing its negative eigenvalues are no longer masked by the sensor floor (job 5069
# died at 100 m). check_R_pd in src/experiment.jl documents the measured eigenvalues and
# fails loudly; 500 m is the calibrated default.
#
# Everything else is held identical to run_kf_rank_sweep.jl: same basis, models,
# rep-error model, honest tail-clutter inflation of R, same RNG seed (all from
# src/experiment.jl). Scored on the full analysis domain (wake-region score removed
# 2026-11-05 pending a redesign).
#
# NUMERICS: with ~1.5e3 sensors the innovation covariance S is (3NM)² and
# refactorizing it every step would dominate the run. run_filter (src/kalman.jl)
# therefore uses the equivalent information/Woodbury form for a StaticObs with
# 3NM > r: R_total is factorized ONCE per rank and each step does only r×r work.
# All in Float64 (the dense rep-error correlations make R_total poorly
# conditioned in Float32). test/test_kalman.jl checks it against the gain form.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, RANKS, sensors, seed

const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))
# Sensor lattice spacing, in GRID CELLS (dx = dy = 10 m). See check_R_pd before reducing.
const SENSOR_STRIDE_XY = parse(Int, get(ENV, "SENSOR_STRIDE_XY", "50"))
const SENSOR_STRIDE_Z  = parse(Int, get(ENV, "SENSOR_STRIDE_Z",  "10"))

Random.seed!(NOISE_SEED)   # same seed in every driver -> paired noise across experiments

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = resolve_member(config, TRUTH_ID)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
mdl    = load_models(models_filename(BASES_DIR, TRUTH_ID))
rep    = load_rep_error(rep_error_filename(BASES_DIR, TRUTH_ID))
Lfull  = basis.L
all(r -> r <= Lfull, RANKS) || error("RANKS exceed basis L=$Lfull")

# --- Truth ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = sensor_mvar()
TIDX = collect(1:OBS_STRIDE:length(truth_t))
K    = length(TIDX)
QSCALE = Float32(OBS_STRIDE)   # random-walk variance-adds; see the caveat in statmodel.jl

# --- Sensor lattice spanning the downwind half of the analysis domain ---
locs = dense_lattice_locations(grid, domain; stride_xy=SENSOR_STRIDE_XY, stride_z=SENSOR_STRIDE_Z)
NM   = length(locs)
@info "Sensor array: NM=$NM sensors ($(3NM) obs rows); xy stride $(SENSOR_STRIDE_XY) cells " *
      "($(SENSOR_STRIDE_XY * grid.dx) m), z stride $(SENSOR_STRIDE_Z) levels"

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; GC.gc()

# --- Static observation pieces (full rank, sliced per r) ---
Hfull  = build_measurement_operator(meas_seq[1], basis, domain)   # (3NM × Lfull)
R_base = Float64.(build_measurement_covariance(NM, MVAR) +
                  build_rep_covariance(meas_seq[1], rep, domain)) # sensor + stored rep

# R_total = R_base + H_tail Σ H_tailᵀ and the tail term is PSD, so R_base PD ⇒ R_total PD:
# checking R_base here catches a bad spacing before the rank loop.
check_R_pd(R_base; context="Sensor spacing $(SENSOR_STRIDE_XY * grid.dx) m.")
Y = Matrix{Float64}(undef, 3NM, K)                                # pre-stacked observations
for k in 1:K
    Y[:, k] = stack_measurements(meas_seq[k])
end
meas_seq = nothing; GC.gc()

# --- Truth reductions (identical to the rake sweep) ---
astar   = basis.modes' * truth_U                 # (Lfull × Ktot) oracle coefficients
state_d = domain.state_dim

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
csvpath = joinpath(KF_DIR, "kf_rank_sweep_dense_truth_$(TRUTH_ID)$(stride_tag).csv")
open(csvpath, "w") do io
    println(io, "r,n_sensors,kf_rmse_full,oracle_rmse_full,mean_nis,cond_HPr")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation: R = R_base + H_tail Σ0_tail H_tailᵀ.
        # With 3NM ≫ r, run_filter's :auto picks the information/Woodbury form (R factorized
        # once per rank, r×r work per step) — see the NUMERICS note in the header.
        obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
        model = KFModel(mdl, r; qscale=QSCALE)
        cnd   = whitened_condition(obs.H, model.Σ0)
        h     = run_filter(model, Y, obs)
        A     = Float32.(h.α_post)
        modes_r = @view basis.modes[:, 1:r]
        kf_f = resid2_full(modes_r, A,                truth_U, TIDX)
        or_f = resid2_full(modes_r, astar[1:r, TIDX], truth_U, TIDX)
        row = (sqrt(kf_f / (K * state_d)), sqrt(or_f / (K * state_d)),
               mean_nis(h), cnd)
        println(io, join((r, NM, row...), ","))
        flush(io)
        @info "r=$r  KF full=$(round(row[1],digits=4))  oracle full=$(round(row[2],digits=4))  " *
              "NIS=$(round(row[3],digits=3))  cond(HPr)=$(round(row[4],digits=1))"
    end
end
@info "Saved $csvpath"

