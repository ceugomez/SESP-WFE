# analyze_filter_covariance.jl
# cgf 2026-07-20
#
# What does the filter's covariance actually look like — does the posterior reduce
# uncertainty from the offline prior Σ_α0, and for WHICH modes? In this SSM (static H, R,
# Q; F=I) the posterior covariance P is DATA-INDEPENDENT: it follows a Riccati recursion
#   predict:  P⁻ = P₊ + Q
#   update:   P₊ = (P⁻⁻¹ + HᵀR⁻¹H)⁻¹        (information form, as in the drivers)
# so the steady state P∞ can be computed WITHOUT re-running assimilation. This script
# builds {H_r, R_tot, Q, Σ_α0} exactly as run_kf_*_dense.jl / run_kf_rank_sweep.jl and
# iterates to P∞, then reports per-mode and trace reduction P∞/Σ_α0.
#
# Config via ENV:
#   FOLD=03              member id (the assimilated truth; only its grid/geometry is used)
#   SENSOR=dense|rake    175-sensor lattice (dense) or 15-sensor tower rake (sparse)
#   BASIS=twin|loo       twin -> truth_truth pool basis (member in-basis); loo -> LOO basis
#   REP=auto|none|loo|diag|full
#                        auto = none for twin, stored rep for loo/rake; matches the drivers.
#                        Explicit arms for the rep-calibration study (fixed basis/sensor):
#                          none = R_sensor only (rep error ignored)
#                          diag = R_sensor + diag(R_rep) (variance inflated, correlations dropped)
#                          full = R_sensor + R_rep (the proper model; == loo)
#                        Explicit arms tag the output file cov_fold<F>_<S>_<B>_rep<arm>.jld2
#                        and record eigmin(R_sensor + R_rep_full) as the PD check.
# Ranks: RANKS from src/experiment.jl (same as the drivers).
using LinearAlgebra, Statistics, JLD2, Printf
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))    # the library
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, RANKS, sensors
const FOLD   = get(ENV, "FOLD", "03")
const SENSOR = get(ENV, "SENSOR", "dense")
const BASIS  = get(ENV, "BASIS", "twin")
const REP    = get(ENV, "REP", "auto")
basis_id = BASIS == "twin" ? "truth" : FOLD
# twin uses the pool basis truth_truth (basis_filename(dir,"truth") -> truth_truth_basis.jld2);
# loo uses the leave-one-out basis truth_<FOLD> (basis_filename(dir,FOLD) -> truth_<FOLD>_basis.jld2)
basis  = load_pod_basis(basis_filename(BASES_DIR, basis_id))
config = load_ensemble_config(CONFIG_PATH)
mdl    = load_models(models_filename(BASES_DIR, basis_id))
truth_member = resolve_member(config, FOLD)
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
Lfull  = basis.L
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)
MVAR = sensor_mvar()
TIDX = collect(1:length(truth_t)); K = length(TIDX)
QSCALE = 1.0f0   # OBS_STRIDE=1
# Only H and R are used (the Riccati recursion is data-independent), so the noise in the
# sampled measurements below is irrelevant and is not seeded.
locs = SENSOR == "dense" ? dense_lattice_locations(grid, domain) : rake_locations(grid, domain)
NM   = length(locs)

meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; truth_U = nothing; GC.gc()
Hfull  = build_measurement_operator(meas_seq[1], basis, domain)
# rep_mode: "none" | "diag" | "full". Legacy REP=loo/auto map onto none/full.
rep_mode = REP in ("none", "diag", "full") ? REP :
    (REP == "loo" || (REP == "auto" && (BASIS == "loo" || SENSOR == "rake"))) ? "full" : "none"
R_sensor = Float64.(build_measurement_covariance(NM, MVAR))
eigmin_Rtot = NaN
R_base = if rep_mode == "none"
    R_sensor
else
    rep = load_rep_error(rep_error_filename(BASES_DIR, basis_id))
    Rrep = Float64.(build_rep_covariance(meas_seq[1], rep, domain))
    eigmin_Rtot = eigmin(Symmetric(R_sensor + Rrep))  # PD check on the FULL rep model
    rep_mode == "diag" ? R_sensor + Diagonal(diag(Rrep)) : R_sensor + Rrep
end
meas_seq = nothing; GC.gc()

@info "CONFIG  fold=$FOLD  sensor=$SENSOR (NM=$NM)  basis=$BASIS ($basis_id)  rep=$rep_mode  eigmin(R_sensor+R_rep)=$eigmin_Rtot  L=$Lfull"

# Riccati fixed point per rank: riccati_steady_state (src/kalman.jl).

# n_obs := # modes whose posterior variance is reduced below half the prior = observed subspace dim.
# transient-band ratio := trace reduction over modes 11..r (where the meander/transient lives),
#   isolated from the energetic head (modes 1..10) that dominates the full trace.
println("\n r | trace(P∞)/Σ0 | band11+ tr ratio | n_obs(<0.5) | n_inflated(>1) | mode13 P∞/Σ0")
results = Dict{Int,Any}()
for r in RANKS
    model = KFModel(mdl, r; qscale=QSCALE)
    Σ0    = model.Σ0
    obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
    nu    = n_unobservable(obs)
    if nu > 0   # e.g. rake (45 obs rows) at r ≥ 46: no steady state exists
        @warn "r=$r: $nu unobservable directions — no steady-state covariance; recording NaN"
        results[r] = (trace_ratio=NaN, band_ratio=NaN, ratio=fill(NaN, r), dP=fill(NaN, r),
                      d0=diag(Σ0), its=0, tau_steps=Inf)
        @printf(" %2d |   no steady state (%d unobservable directions)\n", r, nu)
        continue
    end
    Pinf, its = riccati_steady_state(model, obs)
    _, τ = closed_loop_decay(model, obs, Pinf)   # filter memory, in assimilation steps
    dP = diag(Pinf); d0 = diag(Σ0)
    ratio = dP ./ d0
    trace_ratio = tr(Pinf) / tr(Σ0)
    band = 11:r
    band_ratio = r >= 11 ? sum(dP[band]) / sum(d0[band]) : NaN
    m13 = r >= 13 ? ratio[13] : NaN
    results[r] = (trace_ratio=trace_ratio, band_ratio=band_ratio, ratio=ratio, dP=dP, d0=d0, its=its,
                  tau_steps=τ)
    @printf(" %2d |   %8.2e  |   %s   |    %2d      |     %2d         |  %s\n",
            r, trace_ratio,
            r>=11 ? @sprintf("%6.3f", band_ratio) : "  n/a ",
            count(<(0.5), ratio), count(>(1.0), ratio),
            r>=13 ? @sprintf("%.3f %s", m13, m13>1 ? "INFLATED" : (m13<0.7 ? "obs" : "≈prior")) : "n/a")
end

outdir = joinpath(KF_DIR, "cov_analysis"); mkpath(outdir)
# Explicit rep arms get their own files; legacy auto/loo/none keep the old names.
suffix = REP in ("none", "diag", "full") ? "_rep$(REP)" : ""
outpath = joinpath(outdir, "cov_fold$(FOLD)_$(SENSOR)_$(BASIS)$(suffix).jld2")
jldopen(outpath, "w") do f
    f["ranks"] = RANKS; f["NM"] = NM; f["config"] = "fold$(FOLD)_$(SENSOR)_$(BASIS)$(suffix)"
    f["rep_mode"] = rep_mode; f["eigmin_Rtot"] = eigmin_Rtot
    for r in RANKS
        f["r$r/trace_ratio"] = results[r].trace_ratio
        f["r$r/band_ratio"]  = results[r].band_ratio
        f["r$r/ratio"]       = results[r].ratio
        f["r$r/dP"]          = results[r].dP
        f["r$r/d0"]          = results[r].d0
        f["r$r/tau_steps"]   = results[r].tau_steps   # steady-state filter memory (Inf: none)
    end
end
@info "Saved $outpath"
