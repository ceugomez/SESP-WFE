# run_simple_KF_assimilation.jl
# cgf cego6160@colorado.edu 6.22.26
# run a single assimilation loop in POD basis to exercise the codebase (like taking a dog for a walk to stretch its legs)
using LinearAlgebra, Random, Distributions
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, SRC_DIR, ...
const WORKDIR     = SRC_DIR
const T_SPINUP    = 900.0   # seconds — must match what build_pod_basis used for this basis
const MARGIN_M    = 600f0   # lateral exclusion margin — must match build_pod_basis
const TRUTH_ID    = get(ENV, "TRUTH_ID", "truth") # held-out/truth member id; pick the basis. Override via env to sweep folds.
include(joinpath(WORKDIR, "fasteddy_io.jl")) # include r/w functions and structs
include(joinpath(WORKDIR,"pod.jl"))          # POD structs (before measurement: it annotates ::PODBasis)
include(joinpath(WORKDIR, "measurement.jl")) # include measurement functions and structs
include(joinpath(WORKDIR,"statmodel.jl"))    # include prior/process-noise models
include(joinpath(WORKDIR,"reperror.jl"))     # representativeness (reconstruction-error) model
include(joinpath(WORKDIR,"KF.jl"))           # include the Kalman filter loop



# load basis set from file
basis = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))

# get which member is treated as truth 
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]

grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

# --- Build offline models first ---
mdl = get_prior_and_process_noise(basis)

# Representativeness model from ensemble. R_total = R_sensor + R_rep keeps the filter consistent truncation.
basis_members = filter(m -> m.id in basis.member_ids, config.prior_members)
rep_file = rep_error_filename(KF_DIR, TRUTH_ID)
rep = isfile(rep_file) ? load_rep_error(rep_file) :
    let r = build_rep_error(basis, basis_members, domain; t_spinup=T_SPINUP)
        save_rep_error(r, rep_file); r
    end
GC.gc()

# --- Now load the truth field over the analysis domain and sample measurements from it. ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

NM    = 2                                   # number of sensors
MVAR  = Matrix{Float32}(0.25f0 * I, 3, 3)   # real measurement noise variance, diag(sigma_u,v,w)^2
TIDX  = collect(1:length(truth_t))          # every post-spinup truth timestep
meas_seq = get_measurement_sequence(truth, grid, domain, NM, MVAR, TIDX)

# run assimilation (pass rep=nothing to disable representativeness inflation)
result = runtime_loop(basis, meas_seq, mdl, domain, MVAR; rep=rep)

# project truth onto the subspace (L × T), a = PrᵀWx — the oracle (best in-subspace) trajectory
truth_coeffs = project(basis, truth_U)

# per-step sensor locations [x,y,z] (3 × nm), for the saved history
sensor_locs = [reduce(hcat, [m.loc for m in ms.measurements]) for ms in meas_seq]

# save the full filter history + truth identity + performance diagnostics
save_kf_result(kf_result_filename(KF_DIR, TRUTH_ID), result, basis,
               truth_U, truth_coeffs[:, TIDX], truth_t[TIDX], TRUTH_ID, sensor_locs;
               csvpath=joinpath(KF_DIR, "kf_diagnostics_truth_$(TRUTH_ID).csv"),
               coeff_csvpath=joinpath(KF_DIR, "kf_coeffs_truth_$(TRUTH_ID).csv"))

err = result.α .- truth_coeffs[:, TIDX]
rmse_coeff = vec(sqrt.(mean(err.^2; dims=1)))
@info "KF done: $(size(result.α,2)) steps, mean coeff-RMSE = $(round(mean(rmse_coeff), digits=4))"
