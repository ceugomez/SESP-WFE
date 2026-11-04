# run_kf_rank_sweep.jl
# cgf cego6160@colorado.edu 7.13.26
#
# Rank sweep of the reduced-order KF against the Θ-baseline (design doc R1
# follow-up + the rank-sensitivity result). Motivated by theta_baseline job
# 5051: at r=50 with the 15-sensor rake the KF LOSES to the Θ-baseline in the
# wake (0.432 vs 0.397 m/s) while its own subspace oracle sits at 0.230 —
# representation is fine, the online filter at full rank is under-determined
# (NIS 1.34). This sweeps truncation rank r with everything else fixed to find
# where (whether) the filter beats the baseline.
#
# Honest truncation: for rank r the modes r+1..L are no longer estimated, so
# their variance is observation-space clutter. R_total(r) =
#   R_sensor + R_rep(file, built at full L) + H_tail Σ_α0[tail,tail] H_tailᵀ
# i.e. the stored representativeness model (residual beyond L) plus the
# discarded tail's empirical prior covariance sampled at the sensors. Prior,
# Q are the leading r×r blocks of the full-basis models (valid because POD
# coefficients are defined against fixed orthonormal modes).
#
# Same rake, same MVAR, same RNG seed as run_theta_baseline.jl -> identical
# measurement noise draws (paired comparison).
using LinearAlgebra, Random, Statistics, JLD2
include(joinpath(@__DIR__, "paths.jl"))
const WORKDIR  = SRC_DIR
const T_SPINUP = 900.0
const MARGIN_M = 600f0
const TRUTH_ID = get(ENV, "TRUTH_ID", "truth")
const RANKS    = [2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]
# Observation cadence: assimilate every OBS_STRIDE-th snapshot (outputs are 6 s
# apart, so stride 5 = 30 s cadence — the design-rule value). The 6 s cadence
# violates whiteness of the representativeness error (meander τ ≈ 60 s), a
# suspected driver of the slow forcing-mode drift seen in job 5051.
const OBS_STRIDE = parse(Int, get(ENV, "OBS_STRIDE", "1"))
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

Random.seed!(20260713)   # SAME seed as run_theta_baseline.jl -> paired noise

basis  = load_pod_basis(basis_filename(BASES_DIR, TRUTH_ID))
config = load_ensemble_config(CONFIG_PATH)
truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
    config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
grid   = load_grid(truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)
mdl    = load_models(models_filename(BASES_DIR, TRUTH_ID))
rep    = load_rep_error(rep_error_filename(BASES_DIR, TRUTH_ID))
Lfull  = basis.L
all(r -> r <= Lfull, RANKS) || error("RANKS exceed basis L=$Lfull")

# --- Truth + rake (identical to run_theta_baseline.jl) ---
truth_mask = _spinup_mask(truth_member, T_SPINUP)
truth_t    = truth_member.sim_times[truth_mask]
truth_U    = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)
truth      = unpack_to_NCfield(truth_U, truth_t, grid, domain)

MVAR = Matrix{Float32}(0.25f0 * I, 3, 3)
TIDX = collect(1:OBS_STRIDE:length(truth_t))
K    = length(TIDX)
# Q is calibrated on 6 s increments; for a strided cadence the random walk
# accumulates OBS_STRIDE increments between updates (variance adds).
QSCALE = Float32(OBS_STRIDE)

crest = argmax(grid.zPos[:, :, 1])
crest_x, crest_y = grid.xPos[crest[1], crest[2], 1], grid.yPos[crest[1], crest[2], 1]
TOWER_X_M         = 3000f0
TOWER_Y_OFFSETS_M = Float32[-500, -250, 0, 250, 500]
TOWER_HEIGHTS_M   = Float32[10, 50, 100]
locs = Vector{Vector{Float32}}()
for dy in TOWER_Y_OFFSETS_M, z in TOWER_HEIGHTS_M
    i, j, k = nearest_cell(grid, domain, TOWER_X_M, crest_y + dy, z)
    push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
end
NM = length(locs)
meas_seq = get_static_measurement_sequence(truth, grid, locs, domain, NM, MVAR, TIDX)
truth = nothing; GC.gc()

# --- Wake mask (identical to run_theta_baseline.jl) ---
n   = domain.n_grid
lin = _masked_linear_index(domain, grid)
# Full analysis y-width: constant scoring area across LOO folds (the old
# ±750 m strip let off-axis truths sweep their wake out of the box).
WAKE_DX_M, WAKE_AGL_M = 2500f0, 250f0
terrain3 = repeat(grid.zPos[:, :, 1:1], 1, 1, grid.Nz)
wake3 = (grid.xPos .>= crest_x) .& (grid.xPos .<= crest_x + WAKE_DX_M) .&
        ((grid.zPos .- terrain3) .<= WAKE_AGL_M) .& domain.mask
ps   = lin[wake3]
widx = vcat(ps, ps .+ n, ps .+ 2n)

# --- Static observation pieces (full rank, sliced per r) ---
Hfull, R_base = obs_operator_and_base_R(meas_seq[1], basis, domain, MVAR; rep=rep)  # sensor + stored rep
Y = observation_matrix(meas_seq)

# --- Truth reductions ---
# truth_U stays RESIDENT: full-domain RMSE is computed by DIRECT residual
# reconstruction. (The Parseval shortcut ‖x‖²−2αᵀa*+‖α‖² cancels ~1e9-magnitude
# Float32 terms to extract a ~1e5 residual — job 5053 printed 0.0 — so it is
# numerically unusable here. The direct residual has O(0.1 m/s) entries and is
# well-conditioned.)
astar   = basis.modes' * truth_U                 # (Lfull × Ktot) oracle coefficients
truthW  = truth_U[widx, :]                       # wake-region truth (nw × Ktot)
PrW     = basis.modes[widx, :]                   # wake rows of the modes (nw × Lfull)
state_d = domain.state_dim

# Σ‖Pr_r C[:,k] − truth_U[:,cols[k]]‖² over all k, chunked GEMM, Float64 accumulation.
const RESID_CHUNK = 32
function resid2_full(modes_r, C::AbstractMatrix{Float32}, cols::Vector{Int})
    s = 0.0
    for c0 in 1:RESID_CHUNK:length(cols)
        cc = c0:min(c0 + RESID_CHUNK - 1, length(cols))
        E  = modes_r * C[:, cc] .- truth_U[:, cols[cc]]
        s += mapreduce(x -> abs2(Float64(x)), +, E)
    end
    return s
end

stride_tag = OBS_STRIDE > 1 ? "_stride$(OBS_STRIDE)" : ""
csvpath = joinpath(KF_DIR, "kf_rank_sweep_truth_$(TRUTH_ID)$(stride_tag).csv")
open(csvpath, "w") do io
    println(io, "r,kf_rmse_full,kf_rmse_wake,oracle_rmse_full,oracle_rmse_wake,mean_nis,cond_HPr")
    for r in RANKS
        # Filter (src/kalman.jl). Honest truncation: R = R_base + H_tail Σ0_tail H_tailᵀ.
        obs   = truncated_obs(Hfull, R_base, mdl.Σ_α0, r)
        model = KFModel(mdl, r; qscale=QSCALE)
        # Whitened observability conditioning cond(H_r Σ0^{1/2}) — the gappy-POD
        # diagnostic (Manohar 2018): how well this rake geometry determines the
        # rank-r coefficient prior. Predicts fold-to-fold skill differences when
        # the truth's wind direction carries the wake off the fixed rake.
        cnd   = whitened_condition(obs.H, model.Σ0)
        h     = run_filter(model, Y, obs)
        A     = Float32.(h.α_post)              # posterior coeffs (r × K), for scoring
        PrWr  = @view PrW[:, 1:r]               # contiguous (leading columns, column-major)
        kf_w = 0.0; or_w = 0.0
        for k in 1:K
            tk   = TIDX[k]                                      # truth snapshot column
            a_r  = @view astar[1:r, tk]
            xtw  = @view truthW[:, tk]
            kf_w += sum(abs2, PrWr * A[:, k] .- xtw)
            or_w += sum(abs2, PrWr * a_r     .- xtw)
        end
        modes_r = @view basis.modes[:, 1:r]     # contiguous leading columns
        kf_f = resid2_full(modes_r, A,                TIDX)
        or_f = resid2_full(modes_r, astar[1:r, TIDX], TIDX)
        row = (sqrt(kf_f / (K * state_d)), sqrt(kf_w / (K * length(widx))),
               sqrt(or_f / (K * state_d)), sqrt(or_w / (K * length(widx))),
               mean_nis(h), cnd)
        println(io, join((r, row...), ","))
        @info "r=$r  KF full=$(round(row[1],digits=4)) wake=$(round(row[2],digits=4))  " *
              "oracle full=$(round(row[3],digits=4)) wake=$(round(row[4],digits=4))  " *
              "NIS=$(round(row[5],digits=3))  cond(HPr)=$(round(row[6],digits=1))"
    end
end
@info "Saved $csvpath"
