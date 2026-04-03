# cgf cego6160@colorado.edu
# test-drive Gaussian mixture filter codebase
const WORKDIR = "/home/cego6160/workspace/prediction/src"
include(joinpath(WORKDIR, "fasteddy_io.jl"))    # code and syntax for reading priors data
include(joinpath(WORKDIR, "GMF.jl"))            # code and syntax for gaussian mixture filter
include(joinpath(WORKDIR, "plotting.jl"))
include(joinpath(WORKDIR, "measurement.jl"))    # measurement functions


# ------------------------------------------ Setup -----------------------------------------------------------
# load POD basis from file
basis = load_pod_basis("/home/cego6160/workspace/prediction/output/pod_basis.jld2")
grid = basis.grid       # grid definition for LES run
@info("Loaded basis")
@info("Basis : $(grid.Nx) x $(grid.Ny) x $(grid.Nz)")
@info("Modes: $(basis.L)")          #  

encfg = load_ensemble_config("/home/cego6160/workspace/runs_lad/ensemble_config.json")
@info("Ensemble config loaded")
@info("Members: $(encfg.n_members)")

prior = init_prior(basis,encfg.n_members)                                                       # uniform weight Gaussian prior over coefficient space
@info("Initialized prior over coefficient space")
@info "Prior: K=$(prior.K) components, L=$(prior.L) modes"
# plot_gaussian_mixture.(Ref(prior), 1:10) # call on coefficients 1-10

# get truth datasets
truth_field_squish = load_member_snapshots(encfg.truth_member)
truth_field = reconstruct_NCfield(truth_field_squish, encfg.truth_member, grid)


# construct measurement relation
loc1 = Float32[3.5, 4.2, 0.05]   # is location in m or km? trying for 3.5 km, 4.2 km, 50 m.
loc2 = Float32[0.1, 0.1, 0.05]
#meas_set = get_measurement_set()
meas_obj = get_scalar_measurement(truth_field, grid, loc1, diagm(Float32[0.01, 0.01, 0.0]), 2)
H = make_H_matrix(basis, measSet([meas_obj]))






#  The Bayesian update (core of the filter) — one pass per component k:
#   - Predicted obs: ŷ_k = H * μ_k
#   - Innovation: ν_k = Y - ŷ_k
#   - Innovation covariance: S_k = H * Σ_k * H' + R (Σ_k is diagonal from vars[k,:])
#   - Kalman gain: K_k = Σ_k * H' * inv(S_k)
#   - Updated mean: μ_k⁺ = μ_k + K_k * ν_k
#   - Updated covariance: Σ_k⁺ = (I - K_k * H) * Σ_k
#   - Updated weight: w_k⁺ ∝ w_k * 𝒩(Y; ŷ_k, S_k) — this is the likelihood of the observation under component k