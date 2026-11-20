# run_simple_KF_assimilation.jl
# cgf cego6160@colorado.edu 6.22.26
#
# One KF run at a single rank on the 15-sensor rake, saving the FULL filter history
# (kf_result_truth_<id>.jld2 + per-step and coefficient CSVs). The rank sweeps only save
# summaries; this is the run the field products (scripts/products/make_kf_field_*.jl) and
# the rake Θ-baseline's kf_* columns read.
#   TRUTH_ID=03 RANK=15 julia scripts/online/run_simple_KF_assimilation.jl
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, sensors, seed

const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")          # held-out/truth member id (picks the basis)
const RANK     = parse(Int, get(ENV, "RANK", "0"))      # 0 = full basis rank L; else truncate to r<L

Random.seed!(NOISE_SEED)   # same seed in every driver -> paired noise with run_theta_baseline.jl

# load basis set from file
basis = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))

config = load_ensemble_config(CONFIG_PATH)
truth_member = resolve_member(config, TRUTH_ID)

grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

# --- Offline models: run_pod_all_bases.jl saves these as companions to each basis in
# BASES_DIR; load them if present rather than recomputing. ---
models_file = models_filename(BASES_DIR, TRUTH_ID)
mdl = isfile(models_file) ? load_models(models_file) : get_prior_and_process_noise(basis)

# Representativeness model, R_total = R_sensor + R_rep. If it is missing it is rebuilt here
# WITHOUT basis_dir: inner folds must not reuse the outer fold bases in BASES_DIR, which
# contain this fold's truth (see NESTING in src/reperror.jl).
basis_members = filter(m -> m.id in basis.member_ids, config.prior_members)
rep_file = rep_error_filename(BASES_DIR, TRUTH_ID)
rep = isfile(rep_file) ? load_rep_error(rep_file) :
    let r = build_rep_error(basis, basis_members, domain;
                             config_path=CONFIG_PATH, t_spinup=T_SPINUP, margin_m=MARGIN_M)
        save_rep_error(r, rep_file); r
    end
GC.gc()

# --- Now load the truth field over the analysis domain and sample measurements from it. ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

# Static rake of towers downstream of the hill (src/experiment.jl).
MVAR = sensor_mvar()
TIDX = collect(1:length(truth_t))          # every post-spinup truth timestep
locs = rake_locations(grid, domain)
NM   = length(locs)

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)

# project truth onto the subspace (L × T), a = Prᵀx — the oracle (best in-subspace) trajectory,
# computed against the FULL basis before any rank truncation below.
truth_coeffs = project(basis, truth_U)

# Rank truncation (RANK=0 -> full L). The discarded tail r+1..L is folded back into R as static
# clutter (honest truncation, same treatment as run_kf_rank_sweep.jl). The rake is static, so one
# StaticObs serves every step.
r = (RANK > 0 && RANK < basis.L) ? RANK : basis.L
Hfull, R_base = obs_operator_and_base_R(meas_seq[1], basis, domain, MVAR; rep=rep)
obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
model = KFModel(mdl, r)
if r < basis.L
    basis = PODBasis(basis.coeffs[1:r, :], basis.modes[:, 1:r], basis.eigenvalues[1:r], r,
                      basis.n_snapshots, basis.grid, basis.member_ids, basis.truth_id)
    truth_coeffs = truth_coeffs[1:r, :]   # match the filter's truncated width for scoring
    @info "Truncated basis+models to RANK=$r (of L_full=$(size(Hfull, 2)))"
end

# run assimilation (src/kalman.jl)
result = run_filter(model, observation_matrix(meas_seq), obs)

# per-step sensor locations [x,y,z] (3 × nm), for the saved history
sensor_locs = [reduce(hcat, [m.loc for m in ms.measurements]) for ms in meas_seq]

# save the full filter history + truth identity + performance diagnostics
save_kf_result(kf_result_filename(KF_DIR, TRUTH_ID), result, basis,
               truth_U, truth_coeffs[:, TIDX], truth_t[TIDX], TRUTH_ID, sensor_locs;
               csvpath=joinpath(KF_DIR, "kf_diagnostics_truth_$(TRUTH_ID).csv"),
               coeff_csvpath=joinpath(KF_DIR, "kf_coeffs_truth_$(TRUTH_ID).csv"))

err = result.α_post .- truth_coeffs[:, TIDX]
rmse_coeff = vec(sqrt.(mean(err.^2; dims=1)))
@info "KF done: $(size(result.α_post,2)) steps, mean coeff-RMSE = $(round(mean(rmse_coeff), digits=4))"
