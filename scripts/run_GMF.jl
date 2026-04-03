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



# Deliverable Level 1- Estimation of a static field from sparse measurements, say 10 agents
max_iter = 100
filter_history = runtime_loop(truth_field, basis, prior, max_iter)
