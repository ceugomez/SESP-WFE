# run_kf_rank_sweep.jl
# cgf cego6160@colorado.edu 7.13.26
#
# Rank sweep of the reduced-order KF against the Θ-baseline (design doc R1
# follow-up + the rank-sensitivity result). With the 15-sensor rake the
# full-rank filter is under-determined (45 obs rows vs r ≤ 50); this sweeps the
# truncation rank r with everything else fixed to find where (whether) the
# filter beats the baseline. Scored on the full analysis domain (the wake-region
# score was removed 2026-11-05 pending a redesign; add it back via src/regions.jl).
#
# Honest truncation: for rank r the modes r+1..L are no longer estimated, so
# their variance is observation-space clutter. R_total(r) =
#   R_sensor + R_rep(file, built at full L) + H_tail Σ_α0[tail,tail] H_tailᵀ
# i.e. the stored representativeness model (residual beyond L) plus the
# discarded tail's empirical prior covariance sampled at the sensors. Prior,
# Q are the leading r×r blocks of the full-basis models (valid because POD
# coefficients are defined against fixed orthonormal modes).
#
# Same rake, same MVAR, same RNG seed as run_theta_baseline.jl -> identical
# measurement noise draws (paired comparison).
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, RANKS, sensors, seed

const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
# Observation cadence: assimilate every OBS_STRIDE-th snapshot (outputs are 6 s
# apart, so stride 5 = 30 s cadence — the design-rule value). The 6 s cadence
# violates whiteness of the representativeness error (meander τ ≈ 60 s), a
# suspected driver of the slow forcing-mode drift seen in job 5051.
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))

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

# --- Truth + rake (identical to run_theta_baseline.jl) ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = sensor_mvar()
TIDX = collect(1:OBS_STRIDE:length(truth_t))
K    = length(TIDX)
# Q is calibrated on 6 s increments; for a strided cadence the random walk
# accumulates OBS_STRIDE increments between updates (variance adds — see the caveat in
# src/statmodel.jl: exact only for uncorrelated increments).
QSCALE = Float32(OBS_STRIDE)

locs = rake_locations(grid, domain)
NM   = length(locs)
meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; GC.gc()

# --- Static observation pieces (full rank, sliced per r) ---
Hfull, R_base = obs_operator_and_base_R(meas_seq[1], basis, domain, MVAR; rep=rep)  # sensor + stored rep
Y = observation_matrix(meas_seq)

# --- Truth reductions ---
# truth_U stays RESIDENT: full-domain RMSE is the DIRECT residual (resid2_full in
# src/experiment.jl explains why the Parseval shortcut is unusable here).
astar   = basis.modes' * truth_U                 # (Lfull × Ktot) oracle coefficients
state_d = domain.state_dim

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
csvpath = joinpath(KF_DIR, "kf_rank_sweep_truth_$(TRUTH_ID)$(stride_tag).csv")
open(csvpath, "w") do io
    println(io, "r,kf_rmse_full,oracle_rmse_full,mean_nis,cond_HPr")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation: R = R_base + H_tail Σ0_tail H_tailᵀ.
        obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
        model = KFModel(mdl, r; qscale=QSCALE)
        # Whitened observability conditioning cond(H_r Σ0^{1/2}) — the gappy-POD
        # diagnostic (Manohar 2018): how well this rake geometry determines the
        # rank-r coefficient prior. Predicts fold-to-fold skill differences when
        # the truth's wind direction carries the flow structure off the fixed rake.
        cnd   = whitened_condition(obs.H, model.Σ0)
        h     = run_filter(model, Y, obs)
        A     = Float32.(h.α_post)              # posterior coeffs (r × K), for scoring
        modes_r = @view basis.modes[:, 1:r]     # contiguous leading columns
        kf_f = resid2_full(modes_r, A,                truth_U, TIDX)
        or_f = resid2_full(modes_r, astar[1:r, TIDX], truth_U, TIDX)
        row = (sqrt(kf_f / (K * state_d)), sqrt(or_f / (K * state_d)),
               mean_nis(h), cnd)
        println(io, join((r, row...), ","))
        @info "r=$r  KF full=$(round(row[1],digits=4))  oracle full=$(round(row[2],digits=4))  " *
              "NIS=$(round(row[3],digits=3))  cond(HPr)=$(round(row[4],digits=1))"
    end
end
@info "Saved $csvpath"
