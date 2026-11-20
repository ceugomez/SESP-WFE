# run_pod_all_bases.jl — build every POD basis (truth + all LOO folds) in ONE Julia process.
# cgf cego6160@colorado.edu 7.13.26
#
# Replaces the old per-fold run_pod.jl loop, which re-launched Julia fresh per fold and
# re-streamed each member's ~600 GB snapshot matrix from disk independently for every fold
# (O(M) disk reads per fold, O(M) folds => O(M^2) total reads). Here every member is read from
# disk exactly ONCE into a RAM cache, the full 12-member Gram matrix is built exactly ONCE, and
# every fold (the "truth" pool basis + each single-member-held-out LOO basis) is derived from
# that one Gram matrix via gram_slice/loo_basis_from_gram — the same pattern already used by
# heldout_residual_curve. Mathematically identical to the old per-fold rebuild: a principal
# submatrix of C = X'X is exactly the Gram matrix you'd get building it from that member subset
# from scratch, so results (bases, models, rep-error) match to floating-point round-off.
using LinearAlgebra

include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, shared setup
const L         = 50
const INCLUDE_MEMBERS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
const EXCLUDE_MEMBERS = nothing

include(joinpath(SRC_DIR, "fasteddy_io.jl"))
include(joinpath(SRC_DIR, "pod.jl"))
include(joinpath(SRC_DIR, "measurement.jl"))
include(joinpath(SRC_DIR, "statmodel.jl"))
include(joinpath(SRC_DIR, "reperror.jl"))

using JLD2

n_blas = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
BLAS.set_num_threads(n_blas)
@info "BLAS threads: $(BLAS.get_num_threads())"
flush(stderr)

# Which folds to build: "truth" (all 12 prior members, held out = config truth member) plus
# each prior member id (LOO fold holding that member out of the basis). Optional ARGS restrict
# to a subset:  sbatch slurm/offline/run_pod_all_bases.sh truth 09 10 11
const FOLD_IDS = isempty(ARGS) ? vcat(["truth"], INCLUDE_MEMBERS) : ARGS

config = load_ensemble_config(CONFIG_PATH; include_ids=INCLUDE_MEMBERS, exclude_ids=EXCLUDE_MEMBERS)
grid   = load_grid(config.prior_members[1].files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
@info "Grid: $(grid.Nx)×$(grid.Ny)×$(grid.Nz), analysis domain state_dim=$(domain.state_dim)"

@info "Caching all $(length(config.prior_members)) members in RAM (one disk read each)..."
cache = load_member_cache(config.prior_members, domain; t_spinup=T_SPINUP)

@info "Building full-pool ($(length(config.prior_members))-member) Gram matrix once..."
spectrum_csv = joinpath(HELDOUT_DIR, "eigenvalue_spectrum.csv")
truth_basis, pool_gram = compute_pod_streaming(config.prior_members, grid, domain;
    L, t_spinup=T_SPINUP, truth_id=config.truth_member.id, spectrum_csv, cache)

id2m = Dict(m.id => m for m in config.prior_members)

fail = 0
for fold_id in FOLD_IDS
    @info "=============================================================="
    @info ">>> POD basis: held-out truth=$fold_id"
    @info "=============================================================="
    flush(stderr)   # progress must survive an OOM SIGKILL (buffered stderr is lost)
    try
        if fold_id == "truth"
            basis = truth_basis
            fold_gram = pool_gram
            basis_members = config.prior_members
        else
            basis = loo_basis_from_gram(pool_gram, config.prior_members, grid, domain;
                                         held_id=fold_id, L, t_spinup=T_SPINUP, cache)
            fold_gram = gram_slice(pool_gram, fold_id)
            basis_members = [id2m[id] for id in basis.member_ids]
        end

        save_pod_basis(basis, basis_filename(BASES_DIR, basis.truth_id))

        @info "  Building companion products (prior, process noise, representativeness)..."
        mdl = get_prior_and_process_noise(basis)
        save_models(mdl, models_filename(BASES_DIR, basis.truth_id))

        rep = build_rep_error(basis, basis_members, domain;
                               gram=fold_gram, cache=cache, t_spinup=T_SPINUP, margin_m=MARGIN_M)
        save_rep_error(rep, rep_error_filename(BASES_DIR, basis.truth_id))

        @info ">>> done: $fold_id"
    catch e
        @error ">>> FAILED: $fold_id" exception=(e, catch_backtrace())
        global fail += 1
    end
end

@info "=== Done (fail=$fail) ==="
fail == 0 || exit(1)
