# reperror.jl
# cgf cego6160@colorado.edu 6.25.26
#
# Representativeness (reconstruction-error) model for the reduced-order filter.
# See paper-drafts/reconstruction_error.md.
#
# The rank-r subspace cannot represent an UNSEEN turbulent realization exactly; the
# unrepresented part is a state-space error that the filter must account for as an
# inflation of the observation-noise covariance, R_total = R_sensor + R_rep. This
# module distils that inflation into four stored scalar fields built OFFLINE, then
# assembles the N×N R_rep ONLINE with no reconstruction and no tail modes:
#
#   σ²_rep(ζ)            per-cell residual variance         (physical units, [u;v;w])
#   θ(ζ)                 local mean-flow direction          atan2(v̄, ū) from the prior mean
#   L∥(ζ), L⊥(ζ)         streamwise / cross-stream lengths  from the residual two-point falloff
#
#   Ĉ_rep(ζ₁,ζ₂) = σ_rep(ζ₁) σ_rep(ζ₂) ρ(s),  s = ‖rotate(Δ)/L‖,  ρ = e^{-s} (or e^{-s²})
#
# σ²/L∥/L⊥ are estimated by LEAVE-ONE-MEMBER-OUT cross-validation over the basis's
# training pool (the same fold structure as heldout_residual_curve in pod.jl, and the
# same saved truth_<id>_basis.jld2 fold bases are reused when present): for each held-out
# member k, build a basis from the OTHER members and project k onto it — r_t = x_t − Pr_{-k}
# a_t with a_t = Pr_{-k}ᵀ x_t — so the residual is always measured on a realization the
# fold's basis never saw, not on the training members' own (in-sample, lower) truncation
# error. Accumulating over all folds and dividing by total snapshot count averages over the
# M held-out members. The real truth member is never in this pool — no leakage.
# θ (a mean-flow property, not a residual) is read once from the primary (all-prior) basis.

include("pod.jl")          # PODBasis, GridInfo, AnalysisDomain, unpack_snapshot
include("measurement.jl")  # measSet, _masked_linear_index (build_rep_covariance annotates ::measSet)

using LinearAlgebra
using Statistics
using JLD2

struct RepError
    σ²       :: Vector{Float64}    # (state_dim,) per-cell residual variance, [u;v;w] masked layout
    θ        :: Array{Float64,3}   # (Nx,Ny,Nz) local mean-flow direction (rad), 0 outside domain
    L_par    :: Array{Float64,3}   # (Nx,Ny,Nz) streamwise correlation length (m)
    L_perp   :: Array{Float64,3}   # (Nx,Ny,Nz) cross-stream correlation length (m)
    L_z      :: Float64            # vertical correlation length (m); Inf disables the z term
    kernel   :: Symbol             # :exp (default) or :gauss
    grid     :: GridInfo
    n_grid   :: Int                # domain.n_grid — consistency guard against the basis/domain
end

# ---------------------------------------------------------------------------
# OFFLINE construction
# ---------------------------------------------------------------------------

# Build the representativeness model by LEAVE-ONE-MEMBER-OUT cross-validation over the
# basis's training pool (basis.member_ids). `members` must include those ids (the truth
# member is never passed). For each held-out member k, a basis built from the OTHER members
# is used to project k — an UNSEEN realization to that fold's basis — and the residual
# r = X_k − Pr_{-k}·(Pr_{-k}ᵀ X_k) accumulates σ² and, at a coarse set of probe points, the
# flow-aligned two-point covariance used to fit L∥, L⊥. Averaging is implicit: accumulators
# sum over all M folds' snapshots and are divided by the total snapshot count at the end.
#
# If `gram` (the full-pool PODGram from compute_pod_streaming/build_pod_basis) is supplied,
# any fold that must be rebuilt is derived via loo_basis_from_gram — a Gram-matrix slice, not
# a full Pass-1 rebuild — reusing the work already done to build the primary `basis` in the
# same session. `gram` is in-memory only (never persisted), so a caller that only *loaded*
# `basis` from disk won't have one; pass `config_path` in that case and a fold that isn't
# already cached in `basis_dir` falls back to a full build_pod_basis rebuild.
# If `basis_dir` is given, fold bases are saved as truth_<held>_basis.jld2 and reused on
# re-runs — the same files/convention heldout_residual_curve uses, so running that first (or
# this) populates the cache for both. `cache`, if supplied, is a SnapshotCache (see
# load_member_cache) threaded into any fold that must be rebuilt, to avoid re-reading members
# from disk O(M²) times.
function build_rep_error(basis::PODBasis, members::Vector{MemberInfo}, domain::AnalysisDomain;
                          gram::Union{PODGram,Nothing}=nothing,
                          config_path::Union{String,Nothing}=nothing,
                          t_spinup::Float64=900.0,
                          L::Int=basis.L,
                          margin_m::Float32=600f0,
                          basis_dir::Union{String,Nothing}=nothing,
                          reuse::Bool=true,
                          cache::Union{SnapshotCache,Nothing}=nothing,
                          kernel::Symbol=:exp,
                          L_z::Float64=Inf,
                          probe_stride::Int=20,
                          n_z_probes::Int=6,
                          max_lag_m::Float64=1500.0,
                          chunk::Int=32) :: RepError
    kernel in (:exp, :gauss) || error("kernel must be :exp or :gauss, got $kernel")
    grid = basis.grid
    n    = domain.n_grid
    3n == size(basis.modes, 1) ||
        error("domain.state_dim=$(3n) ≠ basis state_dim=$(size(basis.modes,1)); margin_m mismatch")

    # Order `members` to match the basis column blocks exactly.
    id2m = Dict(m.id => m for m in members)
    all(id -> haskey(id2m, id), basis.member_ids) ||
        error("members vector is missing basis members: $(setdiff(basis.member_ids, keys(id2m)))")
    M = length(basis.member_ids)
    isnothing(basis_dir) || mkpath(basis_dir)

    # --- σ²: per-cell residual second moment, accumulated in physical units ---
    σ²_acc = zeros(Float64, 3n)
    n_tot  = 0

    # --- probe set + flow-aligned two-point covariance accumulators ---
    probes        = _probe_cells(grid, domain, probe_stride, n_z_probes)   # Vector{NTuple{3,Int}}
    lags          = collect(Float64, grid.dx : grid.dx : max_lag_m)        # streamwise/cross lags (m)
    nL            = length(lags)
    C_par         = zeros(Float64, length(probes), nL)   # Σ_t r(ζ)·r(ζ+δ∥)
    C_perp        = zeros(Float64, length(probes), nL)
    C0            = zeros(Float64, length(probes))        # Σ_t ‖r(ζ)‖²  (zero-lag, the ρ denominator)
    # θ is a mean-flow property, not a residual: read once from the primary (all-prior) basis.
    θ3 = _mean_flow_direction(basis, domain)              # (Nx,Ny,Nz)

    lin = _masked_linear_index(domain, grid)              # (Nx,Ny,Nz) → position in a component block

    for (mi, held) in enumerate(basis.member_ids)
        @info "  RepError (LOO): fold $mi/$M, held-out member $held..."
        basis_file = isnothing(basis_dir) ? nothing : basis_filename(basis_dir, held)
        loo_basis = if !isnothing(basis_file) && reuse && isfile(basis_file)
            @info "    reusing saved LOO basis ($basis_file)"
            load_pod_basis(basis_file)
        elseif !isnothing(gram)
            @info "    building via Gram-matrix slice (no full rebuild)"
            b = loo_basis_from_gram(gram, members, grid, domain; held_id=held, L, t_spinup, cache)
            isnothing(basis_file) || save_pod_basis(b, basis_file)
            b
        else
            isnothing(config_path) &&
                error("build_rep_error: no cached fold basis for '$held', and neither `gram` " *
                      "nor `config_path` was given to rebuild it")
            b, _ = build_pod_basis(config_path;
                L, t_spinup, include_ids = basis.member_ids,
                truth_id = held, margin_m, cache)
            isnothing(basis_file) || save_pod_basis(b, basis_file)
            b
        end

        member = id2m[held]
        X  = _member_snapshots(member, cache; t_spinup, domain)   # (3n, T) Float32 — unseen to loo_basis
        Pr = loo_basis.modes
        Tm = size(X, 2)
        # Chunk over time columns: forming the full (3n, T) residual at once would peak at
        # ~3 large arrays (X + Pra + Rres ≈ 100+ GB here). Per-chunk GEMM keeps Pr read efficient
        # while bounding peak memory to X + a couple of (3n × chunk) blocks.
        for cs in 1:chunk:Tm
            cc = cs : min(cs+chunk-1, Tm)
            a_cc = Pr' * X[:, cc]                                 # a_t = Pr_{-k}ᵀ x_t
            Rc   = X[:, cc] .- Pr * a_cc                          # (3n, |cc|) out-of-sample residual
            σ²_acc .+= vec(sum(abs2, Rc; dims=2))                # Float32 reduction → Float64 acc
            _accumulate_two_point!(C_par, C_perp, C0, Rc, probes, lags, θ3, grid, lin, n)
        end
        n_tot += Tm
        X = nothing; loo_basis = nothing; GC.gc()                # free the 3n×T block before next fold
    end

    σ² = σ²_acc ./ n_tot                                         # population variance, averaged over folds

    # --- fit L∥, L⊥ per probe, then fill grid fields by nearest probe ---
    Lpar_p  = [_fit_length(view(C_par,  pi, :) ./ C0[pi], lags, kernel) for pi in eachindex(probes)]
    Lperp_p = [_fit_length(view(C_perp, pi, :) ./ C0[pi], lags, kernel) for pi in eachindex(probes)]
    L_par_f  = _fill_from_probes(Lpar_p,  probes, grid, domain)
    L_perp_f = _fill_from_probes(Lperp_p, probes, grid, domain)

    @info "  RepError (LOO) built: median σ_rep = $(round(sqrt(median(σ²[σ² .> 0])), digits=3)) m/s, " *
          "median L∥ = $(round(median(filter(isfinite, Lpar_p)), digits=1)) m, " *
          "L⊥ = $(round(median(filter(isfinite, Lperp_p)), digits=1)) m"
    return RepError(σ², θ3, L_par_f, L_perp_f, L_z, kernel, grid, n)
end

# Local mean-flow direction θ(ζ) = atan2(v̄, ū) from the prior (ensemble) mean state x̄ = Pr ā.
function _mean_flow_direction(basis::PODBasis, domain::AnalysisDomain) :: Array{Float64,3}
    ā    = Float32.(vec(mean(Float64.(basis.coeffs); dims=2)))   # (L,)
    x̄    = basis.modes * ā                                       # (3n,) physical mean state
    ū, v̄, _ = unpack_snapshot(x̄, basis.grid, domain)            # full-grid, 0 outside domain
    θ = zeros(Float64, basis.grid.Nx, basis.grid.Ny, basis.grid.Nz)
    @inbounds for k in axes(θ,3), j in axes(θ,2), i in axes(θ,1)
        domain.mask[i,j,k] && (θ[i,j,k] = atan(v̄[i,j,k], ū[i,j,k]))
    end
    return θ
end

# Coarse probe set: every `stride` interior cells in x,y at `n_z` evenly spaced z levels.
function _probe_cells(grid::GridInfo, domain::AnalysisDomain, stride::Int, n_z::Int)
    ks = unique(round.(Int, range(4, grid.Nz; length=max(n_z,1))))   # skip near-surface levels
    probes = NTuple{3,Int}[]
    for k in ks, j in 1:stride:grid.Ny, i in 1:stride:grid.Nx
        domain.mask[i,j,k] && push!(probes, (i,j,k))
    end
    return probes
end

# Accumulate flow-aligned two-point residual covariance at the probes for one member's
# residual block Rres (3n, T). For each probe and each lag δ we bilinearly sample the
# velocity-vector residual at ζ+δ·ê∥ and ζ+δ·ê⊥ (ê from the local θ) and add the dot
# product r(ζ)·r(ζ+δ) summed over snapshots. The trace (u·u+v·v+w·w) gives a single,
# component-agnostic correlation length per direction.
function _accumulate_two_point!(C_par, C_perp, C0, Rres, probes, lags, θ3, grid, lin, n)
    T  = size(Rres, 2)
    dx = grid.dx
    x0 = grid.xPos[1,1,1]; y0 = grid.yPos[1,1,1]
    for (pi, (i,j,k)) in enumerate(probes)
        p  = lin[i,j,k]
        θ  = θ3[i,j,k]
        c, s = cos(θ), sin(θ)
        # zero-lag energy (ρ denominator), summed over snapshots
        @inbounds for t in 1:T
            ru = Rres[p,t]; rv = Rres[n+p,t]; rw = Rres[2n+p,t]
            C0[pi] += ru*ru + rv*rv + rw*rw
        end
        xc = grid.xPos[i,1,1]; yc = grid.yPos[1,j,1]
        for (li, δ) in enumerate(lags)
            # streamwise ê∥ = (c,s); cross-stream ê⊥ = (−s,c)
            C_par[pi, li]  += _two_point_dot(Rres, p, xc + δ*c, yc + δ*s, k, grid, lin, n, x0, y0, dx)
            C_perp[pi, li] += _two_point_dot(Rres, p, xc - δ*s, yc + δ*c, k, grid, lin, n, x0, y0, dx)
        end
    end
end

# Σ_t r(ζ_probe)·r̂(x,y,k) with r̂ bilinearly interpolated in (x,y) at fixed level k.
# Returns 0 if the sample point falls outside the analysis domain (any corner unmasked).
function _two_point_dot(Rres, p, x, y, k, grid, lin, n, x0, y0, dx) :: Float64
    fi = (x - x0)/dx + 1; fj = (y - y0)/grid.dy + 1
    i0 = floor(Int, fi); j0 = floor(Int, fj)
    (1 <= i0 < grid.Nx && 1 <= j0 < grid.Ny) || return 0.0
    tx = fi - i0; ty = fj - j0
    # bilinear corner state indices; require all four inside the domain
    q00 = lin[i0,   j0,   k]; q10 = lin[i0+1, j0,   k]
    q01 = lin[i0,   j0+1, k]; q11 = lin[i0+1, j0+1, k]
    (q00 != 0 && q10 != 0 && q01 != 0 && q11 != 0) || return 0.0
    w00 = (1-tx)*(1-ty); w10 = tx*(1-ty); w01 = (1-tx)*ty; w11 = tx*ty
    T = size(Rres, 2)
    acc = 0.0
    @inbounds for t in 1:T
        for off in (0, n, 2n)   # u, v, w components share the kernel
            rp  = Rres[off+p, t]
            rhat = w00*Rres[off+q00,t] + w10*Rres[off+q10,t] +
                   w01*Rres[off+q01,t] + w11*Rres[off+q11,t]
            acc += rp * rhat
        end
    end
    return acc
end

# Fit a correlation length to a normalized falloff ρ(δ) (ρ(0)=1) by least squares in log space.
# exp kernel  ρ=e^{-δ/L}   ⇒ ln ρ = −δ/L      ⇒ L = −Σδ² / Σ(δ lnρ)
# gauss kernel ρ=e^{-(δ/L)²} ⇒ ln ρ = −(δ/L)²  ⇒ L = √( Σδ⁴ / Σ(−δ² lnρ) )
# Uses only lags with 0 < ρ < 1 (decaying, positive). Returns Inf if too few usable points.
function _fit_length(ρ::AbstractVector{<:Real}, lags::Vector{Float64}, kernel::Symbol) :: Float64
    num = 0.0; den = 0.0; nuse = 0
    for (δ, r) in zip(lags, ρ)
        (r > 0 && r < 1) || continue
        lr = log(r)
        if kernel === :exp
            num += δ*δ;      den += -δ*lr
        else
            num += δ^4;      den += -(δ*δ)*lr
        end
        nuse += 1
    end
    nuse >= 2 && den > 0 || return Inf
    return kernel === :exp ? num/den : sqrt(num/den)
end

# Fill a (Nx,Ny,Nz) field from per-probe values by nearest probe (Euclidean in i,j,k cells).
# Non-finite probe fits are dropped; cells outside the domain stay 0.
function _fill_from_probes(vals::Vector{Float64}, probes, grid::GridInfo, domain::AnalysisDomain) :: Array{Float64,3}
    keep = [pi for pi in eachindex(probes) if isfinite(vals[pi])]
    isempty(keep) && error("no finite correlation-length fits at any probe; check residuals/lags")
    F = zeros(Float64, grid.Nx, grid.Ny, grid.Nz)
    pc = [probes[pi] for pi in keep]; pv = [vals[pi] for pi in keep]
    @inbounds for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        domain.mask[i,j,k] || continue
        best = Inf; bidx = 1
        for (pi, (ip,jp,kp)) in enumerate(pc)
            d = (i-ip)^2 + (j-jp)^2 + (kp-k)^2
            d < best && (best = d; bidx = pi)
        end
        F[i,j,k] = pv[bidx]
    end
    return F
end

# ---------------------------------------------------------------------------
# ONLINE assembly
# ---------------------------------------------------------------------------

ρ_kernel(s::Float64, kernel::Symbol) = kernel === :exp ? exp(-s) : exp(-s*s)

# Build the representativeness observation-error covariance R_rep (3*nm × 3*nm) for a
# measurement set, in the same [u;v;w]-per-sensor row layout as build_measurement_operator.
# Per-component anisotropic kernel; no cross-component representativeness correlation is
# modeled. Diagonal recovers σ²_rep (s=0 ⇒ ρ=1). nm=1 collapses to diag(σ²_u,σ²_v,σ²_w) and
# the kernel/lengths are never read — matching the single-sensor case in the design doc.
function build_rep_covariance(ms::measSet, rep::RepError, domain::AnalysisDomain) :: Matrix{Float32}
    grid = rep.grid
    n    = rep.n_grid
    3n == 3*domain.n_grid || error("RepError n_grid=$(n) ≠ domain.n_grid=$(domain.n_grid)")
    lin  = _masked_linear_index(domain, grid)
    nm   = length(ms.measurements)

    # per-sensor: state position p, the three σ's, and (i,j,k)/location for the kernel
    P  = Vector{Int}(undef, nm)
    σ  = Matrix{Float64}(undef, 3, nm)         # rows u,v,w
    ijk = Vector{NTuple{3,Int}}(undef, nm)
    loc = Vector{Vector{Float32}}(undef, nm)
    for (k, m) in enumerate(ms.measurements)
        i,j,l = m.locidx
        p = lin[i,j,l]
        p == 0 && error("sensor $k at $(m.locidx) is outside the analysis domain")
        P[k]   = p
        σ[1,k] = sqrt(max(rep.σ²[p],       0.0))
        σ[2,k] = sqrt(max(rep.σ²[n+p],     0.0))
        σ[3,k] = sqrt(max(rep.σ²[2n+p],    0.0))
        ijk[k] = (i,j,l)
        loc[k] = m.loc
    end

    R = zeros(Float32, 3nm, 3nm)
    for a in 1:nm, b in a:nm
        ρ = a == b ? 1.0 : _pair_correlation(loc[a], loc[b], ijk[a], ijk[b], rep)
        for c in 1:3
            val = Float32(σ[c,a] * σ[c,b] * ρ)
            R[3a-3+c, 3b-3+c] = val
            R[3b-3+c, 3a-3+c] = val
        end
    end
    return R
end

# Flow-aligned anisotropic correlation between two sensor locations, using θ, L∥, L⊥
# evaluated at the midpoint cell (the separation vector lives between the two points).
function _pair_correlation(la::Vector{Float32}, lb::Vector{Float32},
                           ia::NTuple{3,Int}, ib::NTuple{3,Int}, rep::RepError) :: Float64
    Δx = Float64(lb[1] - la[1]); Δy = Float64(lb[2] - la[2]); Δz = Float64(lb[3] - la[3])
    mi = (ia[1]+ib[1])÷2; mj = (ia[2]+ib[2])÷2; mk = (ia[3]+ib[3])÷2
    θ  = rep.θ[mi,mj,mk]
    c, s = cos(θ), sin(θ)
    Δpar  =  c*Δx + s*Δy
    Δperp = -s*Δx + c*Δy
    Lpar  = rep.L_par[mi,mj,mk];  Lperp = rep.L_perp[mi,mj,mk]
    s2 = 0.0
    isfinite(Lpar)  && Lpar  > 0 && (s2 += (Δpar /Lpar )^2)
    isfinite(Lperp) && Lperp > 0 && (s2 += (Δperp/Lperp)^2)
    isfinite(rep.L_z) && rep.L_z > 0 && (s2 += (Δz/rep.L_z)^2)
    return ρ_kernel(sqrt(s2), rep.kernel)
end

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

function rep_error_filename(output_dir::String, truth_id::String) :: String
    return joinpath(output_dir, "rep_error_truth_$(truth_id).jld2")
end

function save_rep_error(rep::RepError, filepath::String)
    jldsave(filepath;
        σ²=rep.σ², θ=rep.θ, L_par=rep.L_par, L_perp=rep.L_perp,
        L_z=rep.L_z, kernel=String(rep.kernel),
        grid_Nx=rep.grid.Nx, grid_Ny=rep.grid.Ny, grid_Nz=rep.grid.Nz,
        grid_dx=rep.grid.dx, grid_dy=rep.grid.dy,
        grid_xPos=rep.grid.xPos, grid_yPos=rep.grid.yPos, grid_zPos=rep.grid.zPos,
        n_grid=rep.n_grid)
    @info "Saved RepError to $filepath"
end

function load_rep_error(filepath::String) :: RepError
    d  = load(filepath)
    Nx=d["grid_Nx"]; Ny=d["grid_Ny"]; Nz=d["grid_Nz"]
    grid = GridInfo(Nx, Ny, Nz, d["grid_dx"], d["grid_dy"],
                    d["grid_xPos"], d["grid_yPos"], d["grid_zPos"], Nx*Ny*Nz, 3*Nx*Ny*Nz)
    return RepError(d["σ²"], d["θ"], d["L_par"], d["L_perp"], d["L_z"],
                    Symbol(d["kernel"]), grid, d["n_grid"])
end
