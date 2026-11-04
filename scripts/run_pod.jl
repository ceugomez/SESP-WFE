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
include(joinpath(WORKDIR, "measurement.jl"))    # measurement structs (build_rep_error annotates ::measSet)
include(joinpath(WORKDIR, "statmodel.jl"))      # prior / process-noise model
include(joinpath(WORKDIR, "reperror.jl"))       # representativeness (reconstruction-error) model


# allocate threads for decomposition
n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"


# build basis
@info "=== Building POD basis (L=$L, t_spinup=$(T_SPINUP) s) ==="
spectrum_csv = joinpath(HELDOUT_DIR, "eigenvalue_spectrum.csv")
basis, gram = build_pod_basis(CONFIG_PATH;
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

# --- Companion products, saved by default alongside the basis: the prior/process-noise ---
# --- model (exact, from basis.coeffs — see statmodel.jl) and the leave-one-out spatial ---
# --- representativeness-error model (see reperror.jl). Both are keyed by the same truth_id. ---
# --- `gram` (the full-pool Gram matrix just computed above, in-memory only) is passed into ---
# --- build_rep_error so its leave-one-out folds are sliced from it instead of rebuilt from ---
# --- scratch — the whole point of computing basis, models, and rep-error together here. ---
# --- basis_dir is deliberately NOT passed here: rep_error's inner LOO folds hold out a ---
# --- member from THIS basis's own (already-reduced) pool, so they are a pool smaller by one ---
# --- than the outer per-fold bases that share BASES_DIR/truth_<id>_basis.jld2 filenames (the ---
# --- convention run_pod_all_bases.sh and heldout_residual_curve use for the FULL-pool folds). ---
# --- Saving these inner folds to the same directory would silently overwrite those files with ---
# --- a wrong (smaller) pool. The Gram-matrix slice already makes rebuilding them cheap enough ---
# --- that there is no need to persist them. ---
@info "=== Building companion products (prior, process noise, representativeness) ==="
mdl = get_prior_and_process_noise(basis)
save_models(mdl, models_filename(BASES_DIR, basis.truth_id))

config        = load_ensemble_config(CONFIG_PATH; include_ids=INCLUDE_MEMBERS, exclude_ids=EXCLUDE_MEMBERS)
basis_members = filter(m -> m.id in basis.member_ids, config.prior_members)
grid          = load_grid(basis_members[1].files[1])
domain        = AnalysisDomain(grid; margin_m=MARGIN_M)

rep = build_rep_error(basis, basis_members, domain;
                       gram=gram, t_spinup=T_SPINUP, margin_m=MARGIN_M)
save_rep_error(rep, rep_error_filename(BASES_DIR, basis.truth_id))

@info "=== Done ==="


