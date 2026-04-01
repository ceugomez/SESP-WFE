# run_pod.jl — Driver script for POD basis construction 
# cgf cego6160@colorado.edu 4.1.26
using LinearAlgebra

# ---- Configuration ----
const CONFIG_PATH  = "/home/cego6160/workspace/runs/ensemble_fp/ensemble_config.json"   # ensemble config file that we generated when we generated ensemble namelists
const WORKDIR      = "/home/cego6160/workspace/prediction/src"                          # where stuff should go (except simulation files)
const L            = 50          # number of POD modes to retain
const T_SPINUP     = 900.0       # seconds — exclude spin-up transient behavior (default 15 min)
const BASIS_FILE   = joinpath(WORKDIR, "pod_basis.jld2")
include(joinpath(WORKDIR, "pod_ensemble.jl"))   # this is where all the POD functions live

# ---- BLAS threads ----
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"


# ---- Build basis ----
@info "=== Building POD basis (L=$L, t_spinup=$(T_SPINUP) s) ==="
basis = build_pod_basis(CONFIG_PATH; L=L, t_spinup=T_SPINUP)



# ---- Save projected coefficients ----
using JLD2
save_pod_basis(basis, BASIS_FILE)

@info "Saved POD basis to file"
@info "=== Done ==="
