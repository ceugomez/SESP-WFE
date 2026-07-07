# run_pod.jl — SLURM driver script for POD basis construction 
# cgf cego6160@colorado.edu 4.1.26
using LinearAlgebra

# ---- Configuration ----
include(joinpath(@__DIR__, "paths.jl"))         # CONFIG_PATH, BASES_DIR, HELDOUT_DIR, SRC_DIR, ...
const WORKDIR          = SRC_DIR
const L                = 50          # number of POD modes to retain
const T_SPINUP         = 900.0       # seconds — exclude spin-up transient behavior (default 15 min)
# Member selection: set to nothing to use all prior members from config.
# INCLUDE_MEMBERS: only use these member ids (nothing = all)
# EXCLUDE_MEMBERS: drop these member ids from the prior (nothing = none);
# TRUTH_MEMBER_ID: override which member is treated as truth (nothing = use config default)
const MARGIN_M         = 600f0                  # lateral exclusion margin in metres (sponge + buffer)
const INCLUDE_MEMBERS  = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]     # all 12 completed members
const EXCLUDE_MEMBERS  = nothing                # supplemental to INCLUDE_MEMBERS
# TRUTH_MEMBER_ID: member held out as truth (excluded from the basis). Env TRUTH_ID lets the
# all-bases loop build every LOO fold; "" / unset → config default (member_truth).
const TRUTH_MEMBER_ID  = get(ENV, "TRUTH_ID", "") == "" ? nothing : ENV["TRUTH_ID"]
include(joinpath(WORKDIR, "fasteddy_io.jl"))    # utilities to load and manipulate fasteddy data
include(joinpath(WORKDIR, "pod.jl"))            # this is where all the POD functions live


# allocate threads for decomposition
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"


# build basis
@info "=== Building POD basis (L=$L, t_spinup=$(T_SPINUP) s) ==="
spectrum_csv = joinpath(HELDOUT_DIR, "eigenvalue_spectrum.csv")
basis = build_pod_basis(CONFIG_PATH;
    L             = L,
    t_spinup      = T_SPINUP,
    include_ids   = INCLUDE_MEMBERS,
    exclude_ids   = EXCLUDE_MEMBERS,
    truth_id      = TRUTH_MEMBER_ID,
    margin_m      = MARGIN_M,
    spectrum_csv  = spectrum_csv,
)

# save constructed basis, filename tagged by the resolved truth member id
using JLD2
basis_file = basis_filename(BASES_DIR, basis.truth_id)
save_pod_basis(basis, basis_file)

@info "Saved POD basis to $basis_file"
@info "=== Done ==="


