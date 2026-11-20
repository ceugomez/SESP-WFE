# scratch_lowvar_truth_highvar_basis.jl  (UNPRINCIPLED SCRATCHPAD — not a result)
# Truth/measurements: the ensemble-lowvar nature run. Basis/prior/Q/R_rep: built from the N_NEAR
# ensemble-highvar members whose forcing direction is closest to the lowvar nature run's.
# Runs the rank-r rake KF (as run_simple_KF_assimilation.jl), writes xy-speed slices of truth and
# estimate at NFRAMES steps plus total RMSE (full masked [u;v;w] state, all post-spinup steps) for
# estimate, oracle (truth projected on the subspace) and the basis-mean prior.
# Run with the DEFAULT ensemble (highvar) -- the lowvar config is read by explicit path here.
# Env: N_NEAR=3 RANK=15 NFRAMES=6 Z_AGL=50   Output: results/scratch/lowvar_truth_highvar_basis.{nc,csv}
using LinearAlgebra, Random, Statistics, JLD2, NCDatasets
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))

const N_NEAR  = parse(Int,     get(ENV, "N_NEAR", "3"))
const RANK    = parse(Int,     get(ENV, "RANK", "15"))
const NFRAMES = parse(Int,     get(ENV, "NFRAMES", "6"))
const Z_AGL   = parse(Float32, get(ENV, "Z_AGL", "50"))
ENSEMBLE_NAME == "ensemble-highvar" || error("run with the default (highvar) ensemble; lowvar truth is loaded explicitly")
const OUT = joinpath(PKG_ROOT, "results", "scratch"); mkpath(OUT)

# only the lowvar TRUTH member has output; load_ensemble_config would try (and fail) on its prior members
truth_member = _make_member_info(JSON3.read(read(joinpath(REPO_ROOT, "ensemble-lowvar", "ensemble_config.json"), String)).truth_member)
θ_t = truth_member.met_dir_deg
allhi = load_ensemble_config(CONFIG_PATH)
angdiff(a, b) = abs(mod(a - b + 180, 360) - 180)
near = sort(allhi.prior_members, by = m -> angdiff(m.met_dir_deg, θ_t))[1:N_NEAR]
ids  = [m.id for m in near]
@info "lowvar truth dir=$θ_t°; highvar basis members: " * join(["$(m.id)=$(round(m.met_dir_deg, digits=2))°" for m in near], ", ")
config = load_ensemble_config(CONFIG_PATH; include_ids=ids)

grid   = load_grid(config.prior_members[1].files[1]);  domain = AnalysisDomain(grid; margin_m=MARGIN_M)
gT     = load_grid(truth_member.files[1])
@assert size(gT.xPos) == size(grid.xPos) && gT.xPos ≈ grid.xPos && gT.yPos ≈ grid.yPos && gT.zPos ≈ grid.zPos "lowvar/highvar grids differ"

cache = load_member_cache(config.prior_members, domain; t_spinup=T_SPINUP)
basis, gram = compute_pod_streaming(config.prior_members, grid, domain; L=max(RANK, 15), t_spinup=T_SPINUP,
                                    truth_id=truth_member.id, cache)
mdl = get_prior_and_process_noise(basis)
rep = build_rep_error(basis, config.prior_members, domain; gram, cache, t_spinup=T_SPINUP, margin_m=MARGIN_M)
cache = nothing; GC.gc()

Random.seed!(NOISE_SEED)
tmask = _spinup_mask(truth_member, T_SPINUP);  t = truth_member.sim_times[tmask]
U = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth = unpack_to_NCfield(U, t, grid, domain)
MVAR = sensor_mvar();  locs = rake_locations(grid, domain)
meas = get_static_measurement_sequence(truth, grid, locs, domain, length(locs), MVAR, collect(1:length(t)))

r = min(RANK, basis.L)
Hfull, R_base = obs_operator_and_base_R(meas[1], basis, domain, MVAR; rep=rep)
obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
res   = run_filter(KFModel(mdl, r), observation_matrix(meas), obs)
α = res.α_post;  K = size(α, 2)
modes = basis.modes[:, 1:r]
α0 = vec(mean(basis.coeffs[1:r, :], dims=2));  xprior = modes * α0

sse = Dict("est"=>0.0, "oracle"=>0.0, "prior"=>0.0);  nel = 0
for c0 in 1:16:K
    cs = c0:min(c0 + 15, K);  Uc = U[:, cs]
    Ee = modes * α[:, cs] .- Uc
    Eo = modes * (modes' * Uc) .- Uc
    sse["est"] += sum(abs2, Float64.(Ee));  sse["oracle"] += sum(abs2, Float64.(Eo))
    sse["prior"] += sum(abs2, Float64.(xprior .- Uc));  nel += length(Ee)
end
rm_ = Dict(k => sqrt(v / nel) for (k, v) in sse)
@info "TOTAL RMSE (m/s): estimate=$(round(rm_["est"], digits=4))  oracle(in-subspace floor)=$(round(rm_["oracle"], digits=4))  prior-mean=$(round(rm_["prior"], digits=4))"

frames = round.(Int, range(1, K, length=NFRAMES))
topo = grid.zPos[:, :, 1]
klev = [argmin(abs.(grid.zPos[i, j, :] .- (topo[i, j] + Z_AGL))) for i in 1:grid.Nx, j in 1:grid.Ny]
function spd(x)
    u, v, _ = unpack_snapshot(Float32.(x), grid, domain)
    return [sqrt(u[i, j, klev[i, j]]^2 + v[i, j, klev[i, j]]^2) for i in 1:grid.Nx, j in 1:grid.Ny]
end
TR = cat([spd(U[:, k]) for k in frames]...; dims=3)
ES = cat([spd(modes * α[:, k]) for k in frames]...; dims=3)

open(joinpath(OUT, "lowvar_truth_highvar_basis.csv"), "w") do io
    println(io, "quantity,rmse_ms\nestimate,$(rm_["est"])\noracle,$(rm_["oracle"])\nprior_mean,$(rm_["prior"])")
end
sl = reduce(hcat, [m.loc for m in meas[1].measurements])
NCDataset(joinpath(OUT, "lowvar_truth_highvar_basis.nc"), "c") do ds
    defDim(ds, "xIndex", grid.Nx); defDim(ds, "yIndex", grid.Ny); defDim(ds, "frame", NFRAMES)
    defDim(ds, "sensor", size(sl, 2)); defDim(ds, "xyz3", 3)
    defVar(ds, "xPos", Float32, ("xIndex", "yIndex"))[:, :] = grid.xPos[:, :, 1]
    defVar(ds, "yPos", Float32, ("xIndex", "yIndex"))[:, :] = grid.yPos[:, :, 1]
    defVar(ds, "mask", Int8, ("xIndex", "yIndex"))[:, :] = Int8.(domain.mask[:, :, 1])
    defVar(ds, "sensor_xyz", Float32, ("sensor", "xyz3"))[:, :] = permutedims(sl)
    defVar(ds, "truth", Float32, ("xIndex", "yIndex", "frame"))[:, :, :] = TR
    defVar(ds, "est",   Float32, ("xIndex", "yIndex", "frame"))[:, :, :] = ES
    defVar(ds, "t", Float64, ("frame",))[:] = Float64.(t[frames])
    ds.attrib["basis_members"] = join(ids, " ");  ds.attrib["truth_dir_deg"] = θ_t
    ds.attrib["rank"] = r;  ds.attrib["z_agl_m"] = Z_AGL
    ds.attrib["rmse_est"] = rm_["est"];  ds.attrib["rmse_oracle"] = rm_["oracle"];  ds.attrib["rmse_prior"] = rm_["prior"]
end
@info "done"
