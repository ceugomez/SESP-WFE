# run_partial_kf.jl — ONE-OFF (2026-11-10): POD basis + a few KF experiments from the part of
# the lowvar ensemble that has finished running (members 03, 10, 11 + the truth run).
#
# Self-contained on purpose: it does NOT include scripts/paths.jl and writes nothing to
# Bases-lowvar/ or results/lowvar/ (the real 12-member pod_all_bases job will write those).
# Everything goes to ./out/ next to this file:
#   out/bases/   truth_<id>_{basis,models}.jld2, rep_error_truth_<id>.jld2, eigenvalue spectrum
#   out/kf/      kf_result_<exp>.jld2 (full history), <exp>_steps.csv (per-step performance),
#                <exp>_coeffs.csv (leading-mode α±σ vs truth), summary.csv
#   out/fields/  <exp>_snap.nc — top-down slices (truth / estimate / oracle / prior / posterior
#                std) at Z_AGL_SLICES and SNAP_STEPS, for plot_partial_kf.py
#
# Library calls are the production ones (compute_pod_streaming, loo_basis_from_gram,
# build_rep_error, run_filter, save_kf_result, rake/dense sensors, NOISE_SEED), so results are
# directly comparable to the pipeline — only the member pool is different.
#
#   sbatch run_partial_kf.sh            (≈ 250 GB peak; see the launcher)
using LinearAlgebra, Random, Statistics, JLD2, NCDatasets, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
const SRC_DIR   = joinpath(SESP_ROOT, "src")
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, sensors, NOISE_SEED, check_R_pd

# ---------------------------------------------------------------------------
# Hardcoded one-off configuration
# ---------------------------------------------------------------------------
const CONFIG_PATH = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar/ensemble_config.json"
# POOL env (comma-separated) overrides the pool; default = finished members as of 2026-11-10.
# 2026-11-11: member 04 finished -> POOL=03,04,10,11 (outputs get a _pool<ids> suffix).
const DEFAULT_POOL = ["03", "10", "11"]
const POOL_IDS     = haskey(ENV, "POOL") ? String.(strip.(split(ENV["POOL"], ","))) : DEFAULT_POOL
const L_MAX       = 50                          # same as run_pod_all_bases.jl
# REP_MODEL (2026-11-20): "lowrank" (default) = RepErrorLR, low-rank + diagonal LOO residual
# covariance, rank-matched per experiment (no tail clutter; see src/reperror.jl); "kernel" = the
# original parametric RepError (+ tail clutter at r < L) — use it to reproduce runs before
# 2026-11-20. REP_Q = residual modes kept by the low-rank model.
const REP_MODEL   = get(ENV, "REP_MODEL", "lowrank")
const REP_Q       = parse(Int, get(ENV, "REP_Q", "50"))
REP_MODEL in ("lowrank", "kernel") || error("REP_MODEL must be lowrank or kernel, got $REP_MODEL")

const N_SNAP    = parse(Int, get(ENV, "N_SNAP", "0"))     # >0: use only the first N post-spinup
                                                          # snapshots of every member (quick run)
# OUT_TAG env overrides the folder suffix (2026-11-18: 11 finished members -> OUT_TAG=_all11)
const POOL_TAG  = haskey(ENV, "OUT_TAG") ? ENV["OUT_TAG"] :
                  POOL_IDS == DEFAULT_POOL ? "" : "_pool" * join(POOL_IDS)
const OUT       = joinpath(@__DIR__, (N_SNAP > 0 ? "out_n$(N_SNAP)" : "out") * POOL_TAG)
const BASIS_OUT = joinpath(OUT, "bases")
const KF_OUT    = joinpath(OUT, "kf")
const FIELD_OUT = joinpath(OUT, "fields")
foreach(mkpath, (BASIS_OUT, KF_OUT, FIELD_OUT))

const Z_AGL_SLICES = Float32[10, 50, 100]                 # m AGL, top-down slice heights
const SNAP_STEPS   = [1, 2, 5, 10, 30, 90, 180, 268]      # assimilation steps to dump (clipped to K)

# truth = "truth" -> nature run vs the 3-member basis; truth = "11" -> LOO: member 11 vs a
# basis of {03, 10} (wind speeds bracket 11's, so this is also an interpolation test).
# rank 0 = full L; rep = include R_rep in R (else R_sensor only).
const EXPERIMENTS = [
    (name="truth_rake_full_rep",   truth="truth", sensors=:rake,  rank=0,  rep=true),
    (name="truth_rake_full_norep", truth="truth", sensors=:rake,  rank=0,  rep=false),
    (name="truth_rake_r10_rep",    truth="truth", sensors=:rake,  rank=10, rep=true),
    (name="truth_dense_full_rep",  truth="truth", sensors=:dense, rank=0,  rep=true),
    (name="loo11_rake_full_rep",   truth="11",    sensors=:rake,  rank=0,  rep=true),
    # LOO on 04 (between 11 and 10 in forcing) — only when 04 is in the pool
    ("04" in POOL_IDS ? [(name="loo04_rake_full_rep", truth="04", sensors=:rake, rank=0, rep=true)] : [])...,
]

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))
@info "BLAS threads: $(BLAS.get_num_threads())"; flush(stderr)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
truncate_basis(b::PODBasis, r::Int) =
    r >= b.L ? b : PODBasis(b.coeffs[1:r, :], b.modes[:, 1:r], b.eigenvalues[1:r], r,
                            b.n_snapshots, b.grid, b.member_ids, b.truth_id)

# Top-down slices: per (i,j) the level nearest z_agl above the first-level height (same
# convention as scripts/products/make_kf_field_snapshot.jl). `valid` are linear (i,j) indices
# of in-domain columns; rows_* index the stacked [u;v;w] state.
function make_slices(grid::GridInfo, domain::AnalysisDomain, zs)
    lin  = _masked_linear_index(domain, grid)
    n    = domain.n_grid
    topo = grid.zPos[:, :, 1]
    map(zs) do z
        p = [lin[i, j, argmin(abs.(grid.zPos[i, j, :] .- (topo[i, j] + z)))]
             for i in 1:grid.Nx, j in 1:grid.Ny]
        valid = findall(>(0), vec(p))
        pv    = vec(p)[valid]
        (z=z, valid=valid, rows_u=pv, rows_v=pv .+ n, rows_w=pv .+ 2n)
    end
end

function to2d(vals::AbstractVector, valid, grid::GridInfo)
    a = fill(NaN32, grid.Nx * grid.Ny)
    a[valid] = vals
    return reshape(a, grid.Nx, grid.Ny)
end

rmse(a, b) = sqrt(mean(abs2, a .- b))

# One fold basis + its companion models, built from the pool Gram matrix and saved.
function build_fold(fold::String, pool_basis, pool_gram, members, grid, domain, cache)
    if fold == "truth"
        basis, gram = pool_basis, pool_gram
    else
        basis = loo_basis_from_gram(pool_gram, members, grid, domain;
                                    held_id=fold, L=L_MAX, t_spinup=T_SPINUP, cache)
        gram  = gram_slice(pool_gram, fold)
    end
    bmembers = filter(m -> m.id in basis.member_ids, members)
    @info "Fold $fold: basis on members $(basis.member_ids), L=$(basis.L)"; flush(stderr)
    save_pod_basis(basis, basis_filename(BASIS_OUT, fold))
    mdl = get_prior_and_process_noise(basis)
    save_models(mdl, models_filename(BASIS_OUT, fold))
    # LOO over the basis's own pool; gram is the fold's own Gram (never basis_dir — see NESTING)
    if REP_MODEL == "lowrank"
        # one model per filter rank this fold's rep experiments use (rank 0 → full L)
        rks = sort(unique([(ex.rank > 0 && ex.rank < basis.L) ? ex.rank : basis.L
                           for ex in EXPERIMENTS if ex.truth == fold && ex.rep]))
        rep = isempty(rks) ? Dict{Int,RepErrorLR}() :
              build_rep_error_lowrank(basis, bmembers, domain; gram, cache, ranks=rks, q=REP_Q,
                                      t_spinup=T_SPINUP)
        for (r, rp) in rep
            save_rep_error_lr(rp, rep_error_lr_filename(BASIS_OUT, fold, r))
        end
    else
        rep = build_rep_error(basis, bmembers, domain; gram, cache, t_spinup=T_SPINUP, margin_m=MARGIN_M)
        save_rep_error(rep, rep_error_filename(BASIS_OUT, fold))
    end
    return (basis=basis, mdl=mdl, rep=rep)
end

# ---------------------------------------------------------------------------
# 1. Pool, grid, cache, pool basis
# ---------------------------------------------------------------------------
config = load_ensemble_config(CONFIG_PATH; include_ids=POOL_IDS)

# N_SNAP=N: keep only the first N post-spinup snapshots of every member, a CONTIGUOUS window
# (not a stride, so the 6 s process noise Q stays the right lag). Writes to out_n<N>/.
if N_SNAP > 0
    cut(m) = (keep = findall(t -> t >= T_SPINUP, m.sim_times)[1:N_SNAP];
              MemberInfo(m.id, m.output_dir, m.met_dir_deg, m.U_g_ms, m.V_g_ms,
                         m.files[keep], m.timesteps[keep], m.sim_times[keep]))
    config = EnsembleConfig(config.n_members, cut.(config.prior_members), cut(config.truth_member))
    @warn "N_SNAP mode: first $N_SNAP post-spinup snapshots per member"
end
members = config.prior_members
grid   = load_grid(members[1].files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
@info "Grid $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$(domain.state_dim), pool=$(POOL_IDS)"

cache = load_member_cache(members, domain; t_spinup=T_SPINUP)
pool_basis, pool_gram = compute_pod_streaming(members, grid, domain;
    L=L_MAX, t_spinup=T_SPINUP, truth_id="truth",
    spectrum_csv=joinpath(BASIS_OUT, "eigenvalue_spectrum_truth.csv"), cache)

folds = Dict{String,Any}()
for f in unique(ex.truth for ex in EXPERIMENTS)
    folds[f] = build_fold(f, pool_basis, pool_gram, members, grid, domain, cache)
    GC.gc()
end

slices = make_slices(grid, domain, Z_AGL_SLICES)
MVAR   = sensor_mvar()
sensor_locs_for(s) = s === :rake  ? rake_locations(grid, domain) :
                     s === :dense ? dense_lattice_locations(grid, domain) :
                     error("unknown sensor set $s")

# ---------------------------------------------------------------------------
# 2. One experiment
# ---------------------------------------------------------------------------
function run_experiment(ex, fb, truth_U, truth_t, meas_seq)
    @info "======== experiment $(ex.name) ========"; flush(stderr)
    r = (ex.rank > 0 && ex.rank < fb.basis.L) ? ex.rank : fb.basis.L
    repm = !ex.rep ? nothing : REP_MODEL == "lowrank" ? fb.rep[r] : fb.rep
    Hfull, R_base = obs_operator_and_base_R(meas_seq[1], fb.basis, domain, MVAR; rep=repm)
    λmin = check_R_pd(R_base; context="experiment $(ex.name).")
    # low-rank R_rep is rank-matched (no tail clutter); kernel/none: tail clutter folded in if r < L
    obs   = rank_r_obs(Hfull, R_base, fb.mdl.Σ_α0, r; rep=repm)
    # cross-check vs the validated one-off (run_rep_lowrank.jl, q=50, r=10, nature run, rake)
    lrref = joinpath(@__DIR__, "out_n67_all11", "rep_lowrank", "lowrank_sensor.jld2")
    if repm isa RepErrorLR && ex.name == "truth_rake_r10_rep" && REP_Q == 50 && isfile(lrref)
        Rref = load(lrref, "Rlr")["50"]
        Rnew = R_base - Float64.(build_measurement_covariance(length(meas_seq[1].measurements), MVAR))
        rel  = norm(Rnew - Rref) / norm(Rref)
        @info "cross-check R_rep vs run_rep_lowrank.jl (q=50, r=10): rel diff = $rel"
        rel < 1e-2 || @warn "R_rep differs from the validated one-off by $rel"
    end
    model = KFModel(fb.mdl, r)
    b     = truncate_basis(fb.basis, r)
    K     = length(meas_seq)
    truth_coeffs = project(b, truth_U)                     # oracle (best in-subspace) trajectory

    result = run_filter(model, observation_matrix(meas_seq), obs)
    locs   = [reduce(hcat, [m.loc for m in ms.measurements]) for ms in meas_seq]

    save_kf_result(joinpath(KF_OUT, "kf_result_$(ex.name).jld2"), result, b,
                   truth_U, truth_coeffs, truth_t, ex.truth, locs;
                   csvpath=joinpath(KF_OUT, "$(ex.name)_kf_diag.csv"),
                   coeff_csvpath=joinpath(KF_OUT, "$(ex.name)_coeffs.csv"))

    # full covariance history (save_kf_result keeps only √diag P), for posterior-σ fields at any step
    jldsave(joinpath(KF_OUT, "posterior_cov_$(ex.name).jld2");
            P_post=Float32.(result.P_post), P_pred=Float32.(result.P_pred), times=truth_t)

    # ---- extra per-step metrics: prior baseline, posterior spread, per-slice errors ----
    Pr, α, d = b.modes, Float32.(result.α_post), size(b.modes, 1)
    xprior = Pr * Float32.(model.μ0)
    field_est    = [rmse(Pr * α[:, k], view(truth_U, :, k)) for k in 1:K]
    field_oracle = [rmse(Pr * truth_coeffs[:, k], view(truth_U, :, k)) for k in 1:K]
    field_prior  = [rmse(xprior, view(truth_U, :, k)) for k in 1:K]
    field_spread = [sqrt(tr(view(result.P_post, :, :, k)) / d) for k in 1:K]   # Pr orthonormal
    nisv = nis(result)

    sl = map(slices) do s
        rows = vcat(s.rows_u, s.rows_v)                    # horizontal wind at this height
        Ps   = Pr[rows, :]
        G    = Float64.(Ps' * Ps)                          # tr(Ps P Psᵀ) = ⟨G, P⟩
        Xt   = truth_U[rows, :]
        (z=s.z, est=[rmse(Ps * α[:, k], Xt[:, k]) for k in 1:K],
         oracle=[rmse(Ps * truth_coeffs[:, k], Xt[:, k]) for k in 1:K],
         prior=[rmse(Ps * Float32.(model.μ0), Xt[:, k]) for k in 1:K],
         spread=[sqrt(max(sum(G .* view(result.P_post, :, :, k)), 0.0) / length(rows)) for k in 1:K])
    end

    open(joinpath(KF_OUT, "$(ex.name)_steps.csv"), "w") do io
        hdr = ["step", "time", "field_est", "field_oracle", "field_prior", "field_spread", "nis"]
        for s in sl, q in ("est", "oracle", "prior", "spread")
            push!(hdr, @sprintf("z%03d_%s", round(Int, s.z), q))
        end
        println(io, join(hdr, ","))
        for k in 1:K
            row = Any[k, truth_t[k], field_est[k], field_oracle[k], field_prior[k], field_spread[k], nisv[k]]
            for s in sl
                push!(row, s.est[k], s.oracle[k], s.prior[k], s.spread[k])
            end
            println(io, join(row, ","))
        end
    end

    # ---- top-down snapshots ----
    ks = sort(unique(vcat(filter(<=(K), SNAP_STEPS), K)))   # always include the last step
    snap = joinpath(FIELD_OUT, "$(ex.name)_snap.nc")
    isfile(snap) && rm(snap)
    NCDataset(snap, "c") do ds
        defDim(ds, "x", grid.Nx); defDim(ds, "y", grid.Ny)
        defDim(ds, "zlev", length(slices)); defDim(ds, "snap", length(ks))
        defVar(ds, "xPos", Float32, ("x", "y"))[:, :] = grid.xPos[:, :, 1]
        defVar(ds, "yPos", Float32, ("x", "y"))[:, :] = grid.yPos[:, :, 1]
        defVar(ds, "topo", Float32, ("x", "y"))[:, :] = grid.zPos[:, :, 1]
        defVar(ds, "z_agl", Float32, ("zlev",))[:] = Z_AGL_SLICES
        defVar(ds, "step",  Int32,   ("snap",))[:] = ks
        defVar(ds, "time",  Float32, ("snap",))[:] = truth_t[ks]
        d4 = ("x", "y", "zlev", "snap")
        V = Dict(nm => defVar(ds, nm, Float32, d4; fillvalue=NaN32) for nm in
                 [string(f, "_", c) for f in ("truth", "est", "oracle", "prior") for c in ("u", "v", "w")]
                 ∪ ["std_u", "std_v"])
        for (iz, s) in enumerate(slices)
            Pu, Pv, Pw = Pr[s.rows_u, :], Pr[s.rows_v, :], Pr[s.rows_w, :]
            for (is, k) in enumerate(ks)
                Pk = Float32.(result.P_post[:, :, k])
                for (c, Pc, rows) in (("u", Pu, s.rows_u), ("v", Pv, s.rows_v), ("w", Pw, s.rows_w))
                    V["truth_$c"][:, :, iz, is]  = to2d(truth_U[rows, k], s.valid, grid)
                    V["est_$c"][:, :, iz, is]    = to2d(Pc * α[:, k], s.valid, grid)
                    V["oracle_$c"][:, :, iz, is] = to2d(Pc * truth_coeffs[:, k], s.valid, grid)
                    V["prior_$c"][:, :, iz, is]  = to2d(Pc * Float32.(model.μ0), s.valid, grid)
                end
                V["std_u"][:, :, iz, is] = to2d(sqrt.(max.(vec(sum((Pu * Pk) .* Pu; dims=2)), 0f0)), s.valid, grid)
                V["std_v"][:, :, iz, is] = to2d(sqrt.(max.(vec(sum((Pv * Pk) .* Pv; dims=2)), 0f0)), s.valid, grid)
            end
        end
        sx = locs[1]
        defDim(ds, "sensor", size(sx, 2)); defDim(ds, "xyz", 3)
        defVar(ds, "sensor_xyz", Float32, ("sensor", "xyz"))[:, :] = permutedims(sx)
        ds.attrib["experiment"] = ex.name;  ds.attrib["truth_id"] = ex.truth
        ds.attrib["rank"] = r;  ds.attrib["rep"] = Int(ex.rep);  ds.attrib["sensors"] = String(ex.sensors)
        ds.attrib["basis_members"] = join(b.member_ids, " ")
    end
    @info "wrote $snap"

    half = (K ÷ 2 + 1):K
    return (name=ex.name, truth=ex.truth, sensors=String(ex.sensors), rank=r, rep=ex.rep,
            rep_model=ex.rep ? REP_MODEL : "none",
            basis_members=join(b.member_ids, " "), n_obs=size(Hfull, 1), min_eig_R=λmin,
            field_est=mean(field_est), field_est_2ndhalf=mean(field_est[half]),
            field_oracle=mean(field_oracle), field_prior=mean(field_prior),
            field_spread=mean(field_spread), mean_nis=mean_nis(result),
            z050_est=mean(sl[findfirst(s -> s.z == 50f0, sl)].est))
end

# ---------------------------------------------------------------------------
# 3. Loop over truths; sample measurements once per (truth, sensor set) with NOISE_SEED
# ---------------------------------------------------------------------------
exp_summary = []
for tid in unique(ex.truth for ex in EXPERIMENTS)
    tm      = resolve_member(config, tid)
    truth_U = tid in POOL_IDS ? cache[tid] : load_member_snapshots(tm; t_spinup=T_SPINUP, domain)
    truth_t = tm.sim_times[_spinup_mask(tm, T_SPINUP)]
    TIDX    = collect(1:length(truth_t))
    exs     = filter(ex -> ex.truth == tid, EXPERIMENTS)

    U4 = unpack_to_NCfield(truth_U, truth_t, grid, domain)   # full-grid copy for the sampler
    seqs = Dict{Symbol,Vector{measSet}}()
    for s in unique(ex.sensors for ex in exs)
        locs = sensor_locs_for(s)
        Random.seed!(NOISE_SEED)      # same draws for every experiment on this (truth, sensors)
        seqs[s] = get_static_measurement_sequence(U4, grid, locs, domain, length(locs), MVAR, TIDX)
        @info "truth=$tid sensors=$s: $(length(locs)) sensors × $(length(TIDX)) steps"
    end
    U4 = nothing; GC.gc()

    for ex in exs
        try
            push!(exp_summary, run_experiment(ex, folds[tid], truth_U, truth_t, seqs[ex.sensors]))
        catch e
            @error "experiment $(ex.name) FAILED" exception=(e, catch_backtrace())
        end
        GC.gc(); flush(stderr)
    end
    truth_U = nothing; GC.gc()
end

open(joinpath(KF_OUT, "summary.csv"), "w") do io
    isempty(exp_summary) && return
    ks = keys(exp_summary[1])
    println(io, join(ks, ","))
    for s in exp_summary
        println(io, join([s[k] for k in ks], ","))
    end
end
@info "=== done: $(length(exp_summary))/$(length(EXPERIMENTS)) experiments ==="
for s in exp_summary
    @info @sprintf("%-24s field RMSE %.3f (2nd half %.3f) | oracle %.3f | prior %.3f | spread %.3f | NIS %.2f",
                   s.name, s.field_est, s.field_est_2ndhalf, s.field_oracle, s.field_prior, s.field_spread, s.mean_nis)
end
length(exp_summary) == length(EXPERIMENTS) || exit(1)
