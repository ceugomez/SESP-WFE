# run_kf_twin_dense.jl
# cgf cego6160@colorado.edu 7.17.26
#
# IDENTICAL-TWIN (deliberate inverse crime) dense-array rank sweep — rung 1 of the
# experiment ladder. Assimilates a truth member that IS a column block of the basis
# it is being estimated in, so the projection residual of the truth onto the subspace
# is (up to the L=50 truncation) zero BY CONSTRUCTION.
#
# Purpose: rung 2 (run_kf_rank_sweep_dense.jl, LOO folds) confounds two error sources —
# tracking error (the random-walk dynamics model + Q) and representation error (a
# held-out turbulent realization is not a linear combination of the other eleven).
# Removing the second isolates the first. This is what licenses the claim "time-varying
# flow CAN be tracked when the ensemble spans the truth"; the difference between this
# rung and the matched LOO fold is the measured closure price, which is the ONLY way
# these numbers may be quoted (see the reporting note below).
#
# THE ONLY DIFFERENCE FROM run_kf_rank_sweep_dense.jl IS WHICH BASIS IS LOADED:
#   BASIS_ID=truth   -> Bases-<tag>/truth_truth_{basis,models}.jld2, built from all 12 prior
#                       members (run_pod_all_bases.jl, fold_id=="truth").
#   TRUTH_ID=<00..11> -> the assimilated member, which is one of those 12.
# Same sensor lattice, same Q, same RNG seed, same rank list as the LOO
# dense sweep, so the two CSVs are row-comparable at fixed r.
#
# REPORTING — NOT A PERFORMANCE NUMBER. The truth member contributes to the basis, to
# μ_α0/Σ_α0, and to Q. Every score here is optimistically biased and is meaningless as a
# standalone skill claim. Quote it only as a difference against the matched LOO fold, and
# label it in the paper as a deliberate inverse crime.
#
# R_rep — WHY THE STORED MODEL IS NOT USED BY DEFAULT (REP_MODE=none):
# rep_error_truth_truth.jld2 is an OUT-OF-SAMPLE quantity: build_rep_error re-derives a
# leave-one-out basis per member and accumulates that member's residual against a basis it
# is absent from. That is exactly the error this rung is designed to remove. Feeding it in
# would inflate R by a representativeness error the twin does not incur, flattening the
# filter and destroying the rung1-vs-rung2 difference the experiment exists to measure.
# With REP_MODE=none, R_total = R_sensor + H_tail Σ_tail H_tailᵀ: the 0.25 sensor floor plus
# the honest tail clutter from unestimated modes r+1..L. The one error this leaves unmodeled
# is the truth's own L=50 truncation residual, which is in-sample and small — REP_MODE=loo is
# kept only as a diagnostic to show how much the LOO R_rep costs, not as a defensible twin.
#
# NOTE ON SENSOR SPACING: the 500 m / stride-50 floor (check_R_pd in src/experiment.jl)
# is a constraint of the R_rep kernel's indefiniteness (Paciorek & Schervish 2006), NOT of the
# sensor geometry. With REP_MODE=none there is no R_rep and R_base = 0.25·I is PD at any
# density, so SENSOR_STRIDE_XY may be tightened here. Keep it at 50 for the headline
# rung1-vs-rung2 comparison — the two rungs must share a lattice to be differenced.
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, RANKS, sensors, seed

const BASIS_ID = get(ENV, "BASIS_ID", "truth")     # basis/models fold to estimate in
const TRUTH_ID = get(ENV, "TRUTH_ID", "00")        # assimilated member (must be IN the basis)
const REP_MODE = get(ENV, "REP_MODE", "none")      # "none" (default) | "loo" (diagnostic only)
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))
const SENSOR_STRIDE_XY = parse(Int, get(ENV, "SENSOR_STRIDE_XY", "50"))
const SENSOR_STRIDE_Z  = parse(Int, get(ENV, "SENSOR_STRIDE_Z",  "10"))
REP_MODE in ("none", "loo") || error("REP_MODE must be \"none\" or \"loo\", got \"$REP_MODE\"")

Random.seed!(NOISE_SEED)   # same seed as the LOO dense sweep -> paired observation noise

basis  = load_pod_basis(basis_filename(BASES_DIR, BASIS_ID))
config = load_ensemble_config(CONFIG_PATH)
mdl    = load_models(models_filename(BASES_DIR, BASIS_ID))

# The twin condition, enforced rather than assumed: if TRUTH_ID is not a column block of
# this basis then this is a LOO fold wearing a twin's filename, and every conclusion drawn
# from the output would be wrong in the direction that flatters us.
let members_str = join(basis.member_ids, ", ")
    TRUTH_ID in basis.member_ids || error(
        "NOT A TWIN: truth member '$TRUTH_ID' is not in basis '$BASIS_ID' " *
        "(members: $members_str). For an in-basis truth use BASIS_ID=truth with TRUTH_ID " *
        "in 00..11; for a held-out truth use run_kf_rank_sweep_dense.jl instead.")
end
@info "TWIN (inverse crime): truth member $TRUTH_ID IS in basis $BASIS_ID " *
      "($(length(basis.member_ids)) members, L=$(basis.L)); REP_MODE=$REP_MODE"

truth_member = resolve_member(config, TRUTH_ID)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
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

# --- Sensor lattice: the same as the LOO dense sweep (the two rungs must observe the
#     same points to be differenced) ---
locs = dense_lattice_locations(grid, domain; stride_xy=SENSOR_STRIDE_XY, stride_z=SENSOR_STRIDE_Z)
NM   = length(locs)
@info "Sensor array: NM=$NM sensors ($(3NM) obs rows); xy stride $(SENSOR_STRIDE_XY) cells " *
      "($(SENSOR_STRIDE_XY * grid.dx) m), z stride $(SENSOR_STRIDE_Z) levels"

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; GC.gc()

# --- Static observation pieces (full rank, sliced per r) ---
Hfull  = build_measurement_operator(meas_seq[1], basis, domain)   # (3NM × Lfull)
R_base = if REP_MODE == "loo"
    rep = load_rep_error(rep_error_filename(BASES_DIR, BASIS_ID))
    @warn "REP_MODE=loo: adding the OUT-OF-SAMPLE LOO rep-error model to a twin whose truth " *
          "is in the basis. Diagnostic only — this R over-states the twin's error budget."
    Float64.(build_measurement_covariance(NM, MVAR) +
             build_rep_covariance(meas_seq[1], rep, domain))
else
    Float64.(build_measurement_covariance(NM, MVAR))              # sensor floor only
end

check_R_pd(R_base; context=REP_MODE == "none" ?
    "With REP_MODE=none R_base is the sensor floor alone, so the measurement covariance itself is wrong." :
    "Sensor spacing $(SENSOR_STRIDE_XY * grid.dx) m.")
Y = Matrix{Float64}(undef, 3NM, K)
for k in 1:K
    Y[:, k] = stack_measurements(meas_seq[k])
end
meas_seq = nothing; GC.gc()

# --- Truth reductions (identical to the LOO dense sweep) ---
astar   = basis.modes' * truth_U                 # (Lfull × Ktot) oracle coefficients
state_d = domain.state_dim

# In-sample projection residual of the truth onto the FULL L=50 basis. For a held-out member
# this is the representation error; for a twin it is only the L-truncation, and reporting it
# is how the "zero by construction" claim is checked rather than asserted. Chunked through
# resid2_full — forming basis.modes*astar for all timesteps at once is a 3n×Ktot Float64
# array that OOM-killed job 5098 at 128 GB.
let allcols     = collect(1:size(truth_U, 2)),
    truth_sq    = mapreduce(x -> abs2(Float64(x)), +, truth_U),
    proj_resid  = sqrt(resid2_full(basis.modes, astar, truth_U, allcols) / length(truth_U)),
    truth_rms   = sqrt(truth_sq / length(truth_U))
    @info "In-basis truth projection residual at L=$Lfull: $(round(proj_resid, digits=4)) m/s RMS " *
          "($(round(100 * proj_resid / truth_rms, digits=2))% of truth RMS) — the twin's " *
          "irreducible representation floor."
end

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
rep_tag    = REP_MODE == "none" ? "" : "_rep$(REP_MODE)"
csvpath = joinpath(KF_DIR, "kf_twin_dense_basis$(BASIS_ID)_truth_$(TRUTH_ID)$(stride_tag)$(rep_tag).csv")
open(csvpath, "w") do io
    println(io, "r,n_sensors,kf_rmse_full,oracle_rmse_full,mean_nis,cond_HPr")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation: R = R_base + H_tail Σ0_tail H_tailᵀ.
        # With 3NM ≫ r, run_filter's :auto picks the information/Woodbury form (R factorized
        # once per rank, r×r work per step) — see the NUMERICS note in run_kf_rank_sweep_dense.jl.
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

