# run_pod.jl — SLURM driver script for POD basis construction 
# cgf cego6160@colorado.edu 4.1.26
using LinearAlgebra

# ---- Configuration ----
const CONFIG_PATH  = "/home/cego6160/workspace/runs/ensemble_config.json"
const WORKDIR      = "/home/cego6160/workspace/prediction/src"
const OUTPUT_DIR   = "/home/cego6160/workspace/prediction/output"
const L            = 50          # number of POD modes to retain
const T_SPINUP     = 900.0       # seconds — exclude spin-up transient behavior (default 15 min)
const BASIS_FILE   = joinpath(OUTPUT_DIR, "pod_basis.jld2")
include(joinpath(WORKDIR, "fasteddy_io.jl"))    # utilities to load and manipulate fasteddy data
include(joinpath(WORKDIR, "pod.jl"))   # this is where all the POD functions live


# allocate threads for decomposition
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"


# build basis
@info "=== Building POD basis (L=$L, t_spinup=$(T_SPINUP) s) ==="
basis = build_pod_basis(CONFIG_PATH; L=L, t_spinup=T_SPINUP)

# save constructed basis
using JLD2
save_pod_basis(basis, BASIS_FILE)

@info "Saved POD basis to file"
@info "=== Done ==="


