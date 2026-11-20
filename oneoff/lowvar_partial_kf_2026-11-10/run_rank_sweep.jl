# run_rank_sweep.jl — ONE-OFF (2026-11-10): full-field RMSE vs truncation rank r, reusing the
# bases/models/rep-error that run_partial_kf.jl saved in out_n<N_SNAP>/bases/.
#
# Paired with run_partial_kf.jl: same truth window (first N_SNAP post-spinup snapshots), same
# sensors, same NOISE_SEED draws, same honest truncation (discarded tail folded into R). For
# each rank the error is the RMS over ALL steps and cells, Σ_k‖Pr_r α_k − x_k‖² / (d·K), via the
# direct residual (resid2_full) — NOT the mean of the per-step RMSEs in <exp>_steps.csv.
#   N_SNAP=67 julia run_rank_sweep.jl      (or via run_rank_sweep.sh)
# Output: out_n<N>/kf/rank_sweep.csv
using LinearAlgebra, Random, Statistics, JLD2, Printf

const SESP_ROOT = "/home/cego6160/workspace/ensemble_runs_paper/SESP-WFE"
include(joinpath(SESP_ROOT, "src", "kf_adapters.jl"))
include(joinpath(SESP_ROOT, "src", "experiment.jl"))   # RANKS, NOISE_SEED, sensors, resid2_full

const CONFIG_PATH = "/home/cego6160/workspace/ensemble_runs_paper/ensemble-lowvar/ensemble_config.json"
const POOL_IDS    = ["03", "10", "11"]
const N_SNAP      = parse(Int, get(ENV, "N_SNAP", "67"))
const OUT         = joinpath(@__DIR__, N_SNAP > 0 ? "out_n$(N_SNAP)" : "out")
const BASIS_IN    = joinpath(OUT, "bases")

const SWEEPS = [
    (name="truth_rake_rep",   truth="truth", sensors=:rake,  rep=true),
    (name="truth_rake_norep", truth="truth", sensors=:rake,  rep=false),
    (name="truth_dense_rep",  truth="truth", sensors=:dense, rep=true),
    (name="loo11_rake_rep",   truth="11",    sensors=:rake,  rep=true),
]

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))

# kf/posterior_cov_sweep_<sweep>.jld2: groups r<rank>/{P_post,P_pred,alpha,alpha_pred} + times
pcov_file(name) = joinpath(OUT, "kf", "posterior_cov_sweep_$(name).jld2")
foreach(s -> rm(pcov_file(s.name); force=true), SWEEPS)

# The main-run experiment (run_partial_kf.jl) that each (sweep, rank) reproduces exactly — same
# basis, models, R, NOISE_SEED draws — so its saved α must match to round-off. Verifies that
# this sweep's covariances stand in for the main run's (which did not save P before 2026-11-10).
const MAIN_EQUIV = Dict(("truth_rake_rep", 50) => "truth_rake_full_rep",
                        ("truth_rake_norep", 50) => "truth_rake_full_norep",
                        ("truth_rake_rep", 10) => "truth_rake_r10_rep",
                        ("truth_dense_rep", 50) => "truth_dense_full_rep",
                        ("loo11_rake_rep", 50) => "loo11_rake_full_rep")
function check_against_main(s, r, α)
    ex = get(MAIN_EQUIV, (s.name, r), nothing)
    f  = isnothing(ex) ? "" : joinpath(OUT, "kf", "kf_result_$(ex).jld2")
    isfile(f) || return
    αm  = load(f, "α")
    rel = norm(αm .- α) / norm(αm)
    @info "  check vs main run $ex: rel‖Δα‖ = $(rel)"
    rel < 1e-4 || @warn "  sweep $(s.name) r=$r does NOT reproduce main experiment $ex (rel $rel)"
end

config = load_ensemble_config(CONFIG_PATH; include_ids=POOL_IDS)
if N_SNAP > 0      # identical windowing to run_partial_kf.jl
    cut(m) = (keep = findall(t -> t >= T_SPINUP, m.sim_times)[1:N_SNAP];
              MemberInfo(m.id, m.output_dir, m.met_dir_deg, m.U_g_ms, m.V_g_ms,
                         m.files[keep], m.timesteps[keep], m.sim_times[keep]))
    config = EnsembleConfig(config.n_members, cut.(config.prior_members), cut(config.truth_member))
end
grid   = load_grid(config.prior_members[1].files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
MVAR   = sensor_mvar()
d      = domain.state_dim

rows = []
for tid in unique(s.truth for s in SWEEPS)
    basis = load_pod_basis(basis_filename(BASIS_IN, tid))
    mdl   = load_models(models_filename(BASIS_IN, tid))
    rep   = load_rep_error(rep_error_filename(BASIS_IN, tid))
    tm      = resolve_member(config, tid)
    truth_U = load_member_snapshots(tm; t_spinup=T_SPINUP, domain)
    truth_t = tm.sim_times[_spinup_mask(tm, T_SPINUP)]
    K       = length(truth_t)
    half    = (K ÷ 2 + 1):K
    a       = project(basis, truth_U)                       # (L, K) oracle coefficients
    sw      = filter(s -> s.truth == tid, SWEEPS)

    U4 = unpack_to_NCfield(truth_U, truth_t, grid, domain)
    seqs = Dict{Symbol,Vector{measSet}}()
    for s in unique(x.sensors for x in sw)
        locs = s === :rake ? rake_locations(grid, domain) : dense_lattice_locations(grid, domain)
        Random.seed!(NOISE_SEED)
        seqs[s] = get_static_measurement_sequence(U4, grid, locs, domain, length(locs), MVAR, collect(1:K))
    end
    U4 = nothing; GC.gc()

    # oracle and prior depend only on (truth, r): compute once per rank
    ranks = filter(<=(basis.L), unique(vcat(RANKS, basis.L)))
    fl = Dict(r => (oracle = sqrt(resid2_full(view(basis.modes, :, 1:r), a[1:r, :], truth_U, 1:K) / (d * K)),
                       prior  = sqrt(resid2_full(view(basis.modes, :, 1:r),
                                                 repeat(mdl.μ_α0[1:r], 1, K), truth_U, 1:K) / (d * K)))
                 for r in ranks)

    for s in sw
        @info "==== sweep $(s.name) (basis $(basis.member_ids), L=$(basis.L)) ===="; flush(stderr)
        Hfull, R_base = obs_operator_and_base_R(seqs[s.sensors][1], basis, domain, MVAR;
                                                rep = s.rep ? rep : nothing)
        try
            check_R_pd(R_base; context="sweep $(s.name).")
        catch e
            @error "sweep $(s.name) skipped" exception=e; continue
        end
        Y = observation_matrix(seqs[s.sensors])
        for r in ranks
            try
                res = run_filter(KFModel(mdl, r), Y, truncated_obs(Hfull, R_base, mdl.Σ_α0, r))
                α   = Float32.(res.α_post)
                Pr  = view(basis.modes, :, 1:r)
                est      = sqrt(resid2_full(Pr, α, truth_U, 1:K) / (d * K))
                est_half = sqrt(resid2_full(Pr, α[:, half], truth_U, half) / (d * length(half)))
                spread   = sqrt(mean(tr(view(res.P_post, :, :, k)) for k in 1:K) / d)
                # full posterior/prior covariance history for every (sweep, rank), for
                # re-plotting posterior-σ fields at any step without refiltering
                jldopen(pcov_file(s.name), "a+") do f
                    f["r$r/P_post"] = Float32.(res.P_post); f["r$r/P_pred"] = Float32.(res.P_pred)
                    f["r$r/alpha"]  = α;                    f["r$r/alpha_pred"] = Float32.(res.α_pred)
                    haskey(f, "times") || (f["times"] = truth_t)
                end
                check_against_main(s, r, α)
                push!(rows, (sweep=s.name, truth=tid, sensors=String(s.sensors), rep=s.rep, rank=r,
                             n_obs=size(Hfull, 1), field_est=est, field_est_2ndhalf=est_half,
                             field_oracle=fl[r].oracle, field_prior=fl[r].prior,
                             field_spread=spread, mean_nis=mean_nis(res)))
                @info @sprintf("  r=%2d  est %.4f (2nd half %.4f)  oracle %.4f  prior %.4f  NIS %.2f",
                               r, est, est_half, fl[r].oracle, fl[r].prior, mean_nis(res))
            catch e
                @error "  r=$r failed" exception=e
            end
            flush(stderr)
        end
    end
    basis = nothing; truth_U = nothing; GC.gc()
end

out = joinpath(OUT, "kf", "rank_sweep.csv")
open(out, "w") do io
    println(io, join(keys(rows[1]), ","))
    foreach(r -> println(io, join(values(r), ",")), rows)
end
@info "wrote $out ($(length(rows)) rows)"
