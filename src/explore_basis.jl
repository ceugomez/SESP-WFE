# cgf cego6160@colorado.edu 4.1.26
# unpack and visualize modal basis constructed by POD decomposition of LES ensemble
using Plots, Printf
include("run_pod.jl") # include the top-level pod runtime file for constants and functions. Also tags fasteddy_io.jl and 
include("plotting_fn.jl")
@info "Loading basis"
basis = load_pod_basis("~/workspace/")





