# run_pod.jl — SLURM driver script for POD basis construction 
# cgf cego6160@colorado.edu 4.1.26
using LinearAlgebra

# ---- Configuration ----
const CONFIG_PATH      = "/home/cego6160/workspace/ensemble_runs_paper/ensemble/ensemble_config.json"
const WORKDIR          = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE/src"
const OUTPUT_DIR       = "/home/cego6160/workspace/ensemble_runs_paper/output"
const L                = 50          # number of POD modes to retain
const T_SPINUP         = 900.0       # seconds — exclude spin-up transient behavior (default 15 min)
# Member selection: set to nothing to use all prior members from config.
# INCLUDE_MEMBERS: only use these member ids (nothing = all)
# EXCLUDE_MEMBERS: drop these member ids from the prior (nothing = none);
# TRUTH_MEMBER_ID: override which member is treated as truth (nothing = use config default)
const MARGIN_M         = 600f0                  # lateral exclusion margin in metres (sponge + buffer)
const INCLUDE_MEMBERS  = ["00", "01", "02", "03", "04", "05", "06", "07", "08"]     # only completed members
const EXCLUDE_MEMBERS  = nothing                # supplemental to INCLUDE_MEMBERS 
const TRUTH_MEMBER_ID  = nothing                # use config default (member_truth)
include(joinpath(WORKDIR, "fasteddy_io.jl"))    # utilities to load and manipulate fasteddy data
include(joinpath(WORKDIR, "pod.jl"))            # this is where all the POD functions live


# allocate threads for decomposition
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"


# build basis
@info "=== Building POD basis (L=$L, t_spinup=$(T_SPINUP) s) ==="
basis = build_pod_basis(CONFIG_PATH;
    L             = L,
    t_spinup      = T_SPINUP,
    include_ids   = INCLUDE_MEMBERS,
    exclude_ids   = EXCLUDE_MEMBERS,
    truth_id      = TRUTH_MEMBER_ID,
    margin_m      = MARGIN_M,
)

# save constructed basis, filename tagged by the resolved truth member id
using JLD2
mkpath(OUTPUT_DIR)
basis_file = basis_filename(OUTPUT_DIR, basis.truth_id)
save_pod_basis(basis, basis_file)

@info "Saved POD basis to $basis_file"
@info "=== Done ==="


