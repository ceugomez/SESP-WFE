# run_simple_KF_assimilation.jl
# cgf cego6160@colorado.edu 6.22.26
# run a single assimilation loop in POD basis to exercise the codebase (like taking a dog for a walk to stretch its legs)
using LinearAlgebra, Random, Distributions
const WORKDIR     = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/src"                 # source code directory
const OUTPUT_DIR  = "/home/cego6160/workspace/ensemble_runs_paper/output"                       # basis directory
const CONFIG_PATH = "/home/cego6160/workspace/ensemble_runs_paper/ensemble/ensemble_config.json"
const T_SPINUP    = 900.0   # seconds — must match what build_pod_basis used for this basis
const MARGIN_M    = 600f0   # lateral exclusion margin — must match build_pod_basis
const TRUTH_ID    = "truth" # which member this basis was built holding out as truth
include(joinpath(WORKDIR, "fasteddy_io.jl")) # include r/w functions and structs
include(joinpath(WORKDIR, "measurement.jl")) # include measurement functions and structs
include(joinpath(WORKDIR,"pod.jl"))          # include POD functions and structs
include(joinpath(WORKDIR,"statmodel.jl"))    # include prior/process-noise models
include(joinpath(WORKDIR,"KF.jl"))           # include the Kalman filter loop



# load basis set from file
basis = load_pod_basis(basis_filename(OUTPUT_DIR, TRUTH_ID))

# get which member is treated as truth 
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]

# load truth field over the analysis domain only 
grid       = load_grid(truth_member.files[1])
domain     = AnalysisDomain(grid; margin_m=MARGIN_M)
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

# Sample measurement set from truth field
NM    = 2                                   # number of sensors
MVAR  = Matrix{Float32}(0.25f0 * I, 3, 3)   # real measurement noise variance, diag(sigma_u,v,w)^2
TIDX  = collect(1:length(truth_t))          # every post-spinup truth timestep
meas_seq = get_measurement_sequence(truth, grid, domain, NM, MVAR, TIDX)

# build offline statistical models (prior + process noise) from the ensemble basis
mdl = get_prior_and_process_noise(basis)

# run assimilation
result = runtime_loop(basis, meas_seq, mdl, domain, MVAR)

# diagnostic: reconstruction RMSE of the filtered field vs. truth, per assimilation step
truth_coeffs = basis.modes' * truth_U          # project truth onto subspace (L × T)
err = result.α .- truth_coeffs[:, TIDX]
rmse_coeff = vec(sqrt.(mean(err.^2; dims=1)))
@info "KF done: $(size(result.α,2)) steps, mean coeff-RMSE = $(round(mean(rmse_coeff), digits=4))"
