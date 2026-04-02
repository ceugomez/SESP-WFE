# cgf cego6160@colorado.edu 4.1.26
# unpack and visualize modal basis constructed by POD decomposition of LES ensemble
using Plots, Printf
const WORKDIR = @__DIR__
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod_ensemble.jl"))
include("plotting_fn.jl")
@info "Loading basis"
basis = load_pod_basis("/home/cego6160/workspace/prediction/src/pod_basis.jld2")    # get pod basis from file 





