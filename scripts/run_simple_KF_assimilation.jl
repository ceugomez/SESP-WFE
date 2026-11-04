# run_simple_KF_assimilation.jl
# cgf cego6160@colorado.edu 6.22.26
# run a single assimilation loop in POD basis to exercise the codebase (like taking a dog for a walk to stretch its legs)
using LinearAlgebra, Random, Distributions
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, SRC_DIR, ...
const WORKDIR     = SRC_DIR
const T_SPINUP    = 900.0   # seconds — must match what build_pod_basis used for this basis
const MARGIN_M    = 600f0   # lateral exclusion margin — must match build_pod_basis
const TRUTH_ID    = get(ENV, "TRUTH_ID", "truth") # held-out/truth member id; pick the basis. Override via env to sweep folds.
const RANK        = parse(Int, get(ENV, "RANK", "0")) # 0 = full basis rank L (default); else truncate to r<L
include(joinpath(WORKDIR, "fasteddy_io.jl")) # include r/w functions and structs
include(joinpath(WORKDIR,"pod.jl"))          # POD structs (before measurement: it annotates ::PODBasis)
include(joinpath(WORKDIR, "measurement.jl")) # include measurement functions and structs
include(joinpath(WORKDIR,"statmodel.jl"))    # include prior/process-noise models
include(joinpath(WORKDIR,"reperror.jl"))     # representativeness (reconstruction-error) model
include(joinpath(WORKDIR,"kf_adapters.jl"))           # Kalman filter glue (filter itself: src/kalman.jl)



# load basis set from file
basis = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))

# get which member is treated as truth 
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]

grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

# --- Build offline models first. run_pod.jl saves these as companions to the basis (in
# BASES_DIR) by default; load them if present rather than recomputing. ---
models_file = models_filename(BASES_DIR, TRUTH_ID)
mdl = isfile(models_file) ? load_models(models_file) : get_prior_and_process_noise(basis)

# Representativeness model from ensemble. R_total = R_sensor + R_rep keeps the filter consistent truncation.
basis_members = filter(m -> m.id in basis.member_ids, config.prior_members)
rep_file = rep_error_filename(BASES_DIR, TRUTH_ID)
rep = isfile(rep_file) ? load_rep_error(rep_file) :
    let r = build_rep_error(basis, basis_members, domain;
                             config_path=CONFIG_PATH, t_spinup=T_SPINUP, margin_m=MARGIN_M, basis_dir=BASES_DIR)
        save_rep_error(r, rep_file); r
    end
GC.gc()

# --- Now load the truth field over the analysis domain and sample measurements from it. ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

# set up measurement sequence: a static rake of towers downstream of the hill,
# each tower carrying sensors at several AGL heights.
MVAR  = Matrix{Float32}(0.25f0 * I, 3, 3)   # real measurement noise variance, diag(sigma_u,v,w)^2
TIDX  = collect(1:length(truth_t))          # every post-spinup truth timestep

# hill crest = location of max terrain height in the grid
crest = argmax(grid.zPos[:, :, 1])
crest_x, crest_y = grid.xPos[crest[1], crest[2], 1], grid.yPos[crest[1], crest[2], 1]

# a single line of towers aligned with the x-axis at x=3 km (fixed geometry)
TOWER_X_M       = 3000f0                        # fixed absolute downstream position, m
TOWER_Y_OFFSETS_M = Float32[-500, -250, 0, 250, 500]  # towers spread across y at that x, m
TOWER_HEIGHTS_M = Float32[10, 50, 100]                # sensor heights AGL per tower, m

locs = Vector{Vector{Float32}}()
for dy in TOWER_Y_OFFSETS_M, z in TOWER_HEIGHTS_M
    y = crest_y + dy
    i, j, k = nearest_cell(grid, domain, TOWER_X_M, y, z)   # snap to grid; errors if outside the analysis domain
    push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
end
NM = length(locs)

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)

# project truth onto the subspace (L × T), a = Prᵀx — the oracle (best in-subspace) trajectory,
# computed against the FULL basis before any rank truncation below.
truth_coeffs = project(basis, truth_U)

# Rank truncation (RANK=0 -> full L). The discarded tail r+1..L is folded back into R as static
# clutter (honest truncation, same treatment as run_kf_rank_sweep.jl). Rake is static, so one
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

# run assimilation (src/kalman.jl). rep=nothing above disables representativeness inflation.
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
