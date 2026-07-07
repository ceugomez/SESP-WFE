# pod_ensemble.jl
# DESCRIPTION
# cgf cego6160@colorado.edu 4.1.26

using NCDatasets
using LinearAlgebra
using Statistics
using JSON3 
using JLD2

# PODBasis struct, includes all the stuff we need to do sequential estimation
struct PODBasis
    coeffs        :: Matrix{Float32}   # (?,?) - POD modal coefficients for all timesteps. as (rows=modes, cols=timesteps)
    modes         :: Matrix{Float32}   # (state_dim, L) — W-orthonormal POD modes Pr_k (PrᵀWPr=I)
    eigenvalues   :: Vector{Float64}   # (L,) descending in magnitude
    L             :: Int               # length of eigenvalues/modes 
    n_snapshots   :: Int               # number of snapshots ()
    grid          :: GridInfo          # grid definition, see fasteddy_io.jl
    member_ids    :: Vector{String}    # ids of members used to build this basis, in column-block order
    truth_id      :: String            # id of the member held out / treated as truth for this basis
    block_weights :: Vector{Float32}   # (3,) equal-energy component weights [w_u,w_v,w_w]; W = blkdiag(w_c I)
end

# --- Equal-energy component weighting -------------------------------------
# The plain Euclidean POD norm ‖x‖² = Σu² + Σv² + Σw² lets the small-magnitude
# vertical velocity be drowned out by the horizontal components, so the leading
# modes barely resolve w. We instead use the W-inner product ⟨x,y⟩_W = xᵀWy with
# a block-constant W = blkdiag(w_u·I, w_v·I, w_w·I). "Equal energy" fixes the
# weights in closed form: w_c = 1/E_c, E_c = total snapshot energy in block c, so
# each weighted block contributes unit energy. (Global scale of W is irrelevant.)
# Modes become W-orthonormal (PrᵀWPr = I) but still reconstruct physical state
# x = Prα; only the projection picks up a W: a = PrᵀWx (see `project`).

# Optional in-RAM cache of member snapshot matrices, keyed by member id. When present,
# `_member_snapshots` returns the cached matrix instead of re-reading from disk — this is
# the ONLY behavioural change for caching; the returned values are identical to a fresh
# load_member_snapshots(member; t_spinup, domain), so all downstream POD arithmetic is
# unchanged. The cache MUST have been built with the same t_spinup and domain.
const SnapshotCache = Dict{String,Matrix{Float32}}

_member_snapshots(member::MemberInfo, cache::Union{SnapshotCache,Nothing};
                  t_spinup::Float64, domain::AnalysisDomain) =
    isnothing(cache) ? load_member_snapshots(member; t_spinup, domain) : cache[member.id]

# Preload the given members' masked snapshot matrices into a cache (one disk read each).
function load_member_cache(members::Vector{MemberInfo}, domain::AnalysisDomain;
                           t_spinup::Float64) :: SnapshotCache
    cache = SnapshotCache()
    for (i, m) in enumerate(members)
        @info "  Caching member $(m.id) ($i/$(length(members)))..."
        cache[m.id] = load_member_snapshots(m; t_spinup, domain)
    end
    gb = sum(length(v) for v in values(cache)) * 4 / 1e9
    @info "  Cached $(length(cache)) members in RAM (≈ $(round(gb, digits=1)) GB)"
    return cache
end

# Per-component block weights w_c = 1/E_c from one cheap pass over the members.
function equal_energy_block_weights(members::Vector{MemberInfo}, domain::AnalysisDomain;
                                    t_spinup::Float64,
                                    cache::Union{SnapshotCache,Nothing}=nothing) :: Vector{Float32}
    n = domain.n_grid
    E = zeros(Float64, 3)
    for member in members
        X = _member_snapshots(member, cache; t_spinup, domain)   # (3n, T) Float32
        for c in 1:3
            E[c] += sum(abs2, Float64.(@view X[(c-1)*n+1 : c*n, :]))
        end
    end
    return Float32.(1.0 ./ E)
end

# Expand 3 block weights into a length-state_dim column vector (u,v,w blocks).
function _state_weight_vector(n::Int, w::Vector{Float32}) :: Vector{Float32}
    vcat(fill(w[1], n), fill(w[2], n), fill(w[3], n))
end
_state_weight_vector(basis::PODBasis) = _state_weight_vector(size(basis.modes,1) ÷ 3, basis.block_weights)

# Project physical state(s) onto the basis in the equal-energy W-inner product: a = PrᵀWx.
# Use this anywhere coefficients are recovered from a state; raw `modes' * x` is only
# correct for the unweighted (w=1) basis.
project(basis::PODBasis, X::AbstractVecOrMat) = basis.modes' * (_state_weight_vector(basis) .* X)

# ---------------------------------------------------------------------------
# Held-out residual-energy curve (leave-one-member-out cross-validation).
#
# This is the closure diagnostic, NOT the eigenvalue spectrum. The eigenvalue
# spectrum is in-sample (how well the basis represents its own snapshots); this
# measures the irreducible finite-rank projection error on an UNSEEN turbulent
# realization. For each held-out member k we build the basis from the other
# M-1 members, project member k's post-spinup snapshots onto it, and report the
# fraction of k's energy left UNcaptured as a function of truncation rank r.
#
# Because the modes Pr are W-orthonormal, the whole curve comes from one projection
# per held-out member (Parseval in the W-norm): with a = PrᵀWx and per-mode captured
# energy eᵢ = Σ_t a[i,t]², the residual fraction at rank r is
#     E_resid(r) = 1 - (Σ_{i≤r} eᵢ) / Σ_t ‖x_t‖²_W ,   ‖x‖²_W = xᵀWx .
# The smallest r where E_resid(r) drops below the observation-noise floor is the
# defensible truncation rank and quantifies the family-closure error.

struct HeldoutResidual
    ranks          :: Vector{Int}              # (L,) truncation ranks 1:L
    resid_mean     :: Vector{Float64}          # (L,) residual energy fraction averaged over held-out members
    resid_per_mbr  :: Matrix{Float64}          # (L, M) residual fraction, column per held-out member
    member_ids     :: Vector{String}           # (M,) which member was held out, in column order
    total_energy   :: Vector{Float64}          # (M,) total snapshot energy Σ_t‖x_t‖² of each held-out member
    L              :: Int                       # max rank evaluated
end


# Standard basis filename, tagged by held-out/truth member id so bases for different truth choices never overwrite each other.
function basis_filename(output_dir::String, truth_id::String) :: String
    return joinpath(output_dir, "truth_$(truth_id)_basis.jld2")
end

# save and load POD basis
function save_pod_basis(basis::PODBasis, filepath::String)
    jldsave(filepath;
        coeffs        = basis.coeffs,
        modes         = basis.modes,
        eigenvalues   = basis.eigenvalues,
        L             = basis.L,
        n_snapshots   = basis.n_snapshots,
        grid_Nx       = basis.grid.Nx,
        grid_Ny       = basis.grid.Ny,
        grid_Nz       = basis.grid.Nz,
        grid_dx       = basis.grid.dx,
        grid_dy       = basis.grid.dy,
        grid_xPos     = basis.grid.xPos,
        grid_yPos     = basis.grid.yPos,
        grid_zPos     = basis.grid.zPos,
        member_ids    = basis.member_ids,
        truth_id      = basis.truth_id,
        block_weights = basis.block_weights,
    )
    sz_gb = (size(basis.modes, 1) * size(basis.modes, 2) * 4) / 1e9
    @info "Saved POD basis to $filepath  (L=$(basis.L), modes ≈ $(round(sz_gb, digits=2)) GB)"
end

# load POD basis from file and populate structure
function load_pod_basis(filepath::String) :: PODBasis
    d  = load(filepath)
    Nx = d["grid_Nx"]; Ny = d["grid_Ny"]; Nz = d["grid_Nz"]
    grid = GridInfo(
        Nx, Ny, Nz,
        d["grid_dx"], d["grid_dy"],
        d["grid_xPos"], d["grid_yPos"], d["grid_zPos"],
        Nx * Ny * Nz,
        3 * Nx * Ny * Nz,
    )
    return PODBasis(
        d["coeffs"],
        d["modes"],
        d["eigenvalues"],
        d["L"],
        d["n_snapshots"],
        grid,
        d["member_ids"],
        d["truth_id"],
        get(d, "block_weights", Float32[1, 1, 1]),   # legacy bases: unweighted
    )
end

# Build POD basis from ensemble and save to disk.
# include_ids: if not nothing, only use these member ids (Vector{String})
# exclude_ids: if not nothing, drop these member ids (Vector{String}); applied after include filter
# truth_id:    if not nothing, override which member is treated as truth (String)
function build_pod_basis(config_path::String;
                          L::Int=10,
                          t_spinup::Float64=900.0,
                          include_ids::Union{Vector{String},Nothing}=nothing,
                          exclude_ids::Union{Vector{String},Nothing}=nothing,
                          truth_id::Union{String,Nothing}=nothing,
                          margin_m::Float32=600f0,
                          spectrum_csv::Union{String,Nothing}=nothing,
                          cache::Union{SnapshotCache,Nothing}=nothing) :: PODBasis

    config = load_ensemble_config(config_path; include_ids, exclude_ids)

    members = config.prior_members

    resolved_truth_id = config.truth_member.id
    if !isnothing(truth_id)
        all_known = [config.truth_member; config.prior_members]
        idx = findfirst(m -> m.id == truth_id, all_known)
        isnothing(idx) && error("truth_id '$truth_id' not found in config (prior or truth members)")
        resolved_truth_id = all_known[idx].id
        @info "Truth member overridden to: $resolved_truth_id"
    end

    # The truth member must never enter the basis. Drop it from the prior set if it was left in.
    members = filter(m -> m.id != resolved_truth_id, members)
    isempty(members) && error("No prior members left after excluding truth '$resolved_truth_id'")
    @info "Using $(length(members)) prior members (truth '$resolved_truth_id' held out): $(join([m.id for m in members], ", "))"

    @info "Loading grid from first member file..."
    grid = load_grid(members[1].files[1])
    domain = AnalysisDomain(grid; margin_m)
    cx = round(Int, margin_m / grid.dx)
    @info "  Grid: $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$(grid.state_dim)"
    @info "  Analysis domain: $(grid.Nx - 2cx)×$(grid.Ny - 2cx)×$(grid.Nz) interior cells ($(margin_m) m margin), state_dim=$(domain.state_dim)"

    @info "Computing POD (L=$L, t_spinup=$(t_spinup) s)..."
    basis = compute_pod_streaming(members, grid, domain; L, t_spinup, truth_id=resolved_truth_id, spectrum_csv, cache)
    return basis
end

# POD via Method of Snapshots (Sirovich 1987)
# Avoids holding the full snapshot matrix in memory by re-reading members' blocks
# from disk instead of caching them across passes. Pass 1 (Gram matrix) needs all
# pairs Xi'Xj, so it re-reads Xj for each j<=i (O(M(M+1)/2) member loads total);
# passes 2 and 3 (mode recovery, modal coefficients) need each member only once
# (O(M) loads each). Trades extra disk I/O for peak memory usage 
#
# Peak memory: one Float32 member block (state_dim × T) + one Float64 copy of same
#              + the (state_dim × L) mode matrix — never the full (state_dim × M*T) matrix,
#              and never more than two members' data resident at once.
#
# If a `cache` (member id → snapshot matrix) is supplied, member blocks are read from RAM
# instead of disk. This is purely an I/O optimization: the cached matrices are identical to
# load_member_snapshots output, so results match the streaming path exactly. Peak memory then
# includes the cache (M·state_dim·T Float32) plus the same transient Float64 copies as before.
function compute_pod_streaming(members::Vector{MemberInfo}, grid::GridInfo,
                                domain::AnalysisDomain;
                                L::Int=10,
                                t_spinup::Float64=900.0,
                                truth_id::String="",
                                spectrum_csv::Union{String,Nothing}=nothing,
                                cache::Union{SnapshotCache,Nothing}=nothing) :: PODBasis
    M     = length(members)
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in members]
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup snapshot counts: $T_vec"
    T = T_vec[1]
    N = M * T
    L = min(L, N)

    # --- Equal-energy component weighting (so vertical velocity isn't starved) ---
    # Work in the transformed coordinate set
    @info "  Computing equal-energy block weights..."
    bw   = equal_energy_block_weights(members, domain; t_spinup, cache)    # [w_u,w_v,w_w]
    sw   = sqrt.(_state_weight_vector(domain.n_grid, bw))                  # (state_dim,) Float32
    sw64 = Float64.(sw)
    @info "    block weights (u,v,w) = $(bw)"

    # --- Pass 1: build Gram matrix C = X̃'X̃ = X'WX, one member at a time ---
    @info "  Pass 1/2: building $(N)×$(N) Gram matrix (streaming over $M members)..."
    C = zeros(Float64, N, N)
    for (mi, member) in enumerate(members)
        @info "    Loading member $(member.id) ($(mi)/$M)..."
        Xi64 = Float64.(_member_snapshots(member, cache; t_spinup, domain)) .* sw64   # x̃ (state_dim, T)
        col_i = (mi-1)*T+1 : mi*T
        for mj in 1:mi
            Xj64 = mj == mi ? Xi64 : Float64.(_member_snapshots(members[mj], cache; t_spinup, domain)) .* sw64
            col_j = (mj-1)*T+1 : mj*T
            block = Xi64' * Xj64                        # (T×T) — small
            C[col_i, col_j] = block
            if mi != mj
                C[col_j, col_i] = block'
            end
        end
    end

    # --- Eigendecompose the small (N×N) Gram matrix ---
    @info "  Eigendecomposing $(N)×$(N) Gram matrix..."
    decomp = eigen(Symmetric(C))
    λ = reverse(decomp.values)
    V = reverse(decomp.vectors, dims=2)

    # Eigenvalue spectrum (full N eigenvalues) — captured energy per mode and the
    # cumulative fraction of TOTAL snapshot variance (Σλ over all N modes, not just
    # the retained L), so the cumulative curve reaches 1 at rank N. Written before
    # truncation so the spectrum reflects the true decay, not the kept slice.
    isnothing(spectrum_csv) || save_eigenvalue_spectrum_csv(λ, spectrum_csv)

    valid = λ .> 1e-10 * λ[1]
    n_valid = sum(valid)
    if n_valid < L
        @warn "Only $n_valid valid eigenvalues; reducing L from $L to $n_valid"
        L = n_valid
    end
    λ_L = λ[1:L]
    V_L = V[:, 1:L]                                    # (N, L)
    scale = Diagonal(1.0 ./ sqrt.(λ_L))               # (L, L)

    # --- Pass 2: recover whitened truncated modes Pr̃ = X̃ V_L scale, then unscale Pr = Pr̃/√W ---
    @info "  Pass 2/2: recovering $L POD modes (state_dim=$(domain.state_dim))..."
    state_dim = domain.state_dim
    Pr̃ = zeros(Float32, state_dim, L)
    for (mi, member) in enumerate(members)
        X̃ = _member_snapshots(member, cache; t_spinup, domain) .* sw   # x̃ (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        Pr̃ .+= Float32.(Float64.(X̃) * (V_L[col_i, :] * scale))
    end
    Pr = Pr̃ ./ sw                                                     # W-orthonormal physical modes

    # --- Modal coefficients a = PrᵀWX = Pr̃ᵀX̃ (streaming) ---
    @info "  Computing modal coefficients..."
    wv = _state_weight_vector(domain.n_grid, bw)                    # W diagonal (state_dim,)
    a = Matrix{Float32}(undef, L, N)
    for (mi, member) in enumerate(members)
        Xi = _member_snapshots(member, cache; t_spinup, domain)   # (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        a[:, col_i] = Pr' * (wv .* Xi)
    end

    @info "  POD complete: L=$L, N=$N snapshots"
    member_ids = [m.id for m in members]
    return PODBasis(a, Pr, λ_L, L, N, grid, member_ids, truth_id, bw)
end

# Compute the held-out residual-energy curve over the chosen prior members.
# member_ids: which member ids to cycle as held-out (nothing = all prior members
#             in the config, after the usual include/exclude filtering).
# L is the maximum truncation rank evaluated; the curve covers ranks 1:L.
# basis_dir: if given, each fold's leave-one-out basis is saved there as
#            truth_<held>_basis.jld2 (the same artifact the KF assimilation loads
#            via basis_filename), and reused on re-runs instead of rebuilt. The fold
#            that holds out member k IS the offline basis for assimilating k as truth.
# reuse:     when true (default), an existing basis file for a fold is loaded instead
#            of recomputed. Set false to force a rebuild.
function heldout_residual_curve(config_path::String;
                                 L::Int=50,
                                 t_spinup::Float64=900.0,
                                 margin_m::Float32=600f0,
                                 member_ids::Union{Vector{String},Nothing}=nothing,
                                 exclude_ids::Union{Vector{String},Nothing}=nothing,
                                 basis_dir::Union{String,Nothing}=nothing,
                                 reuse::Bool=true,
                                 cache_members::Bool=true) :: HeldoutResidual

    # Resolve the full set of members participating in the cross-validation.
    config = load_ensemble_config(config_path; include_ids=member_ids, exclude_ids)
    ids = [m.id for m in config.prior_members]
    M = length(ids)
    M >= 3 || error("Leave-one-out needs >= 3 members; got $M ($(join(ids, ", ")))")
    @info "Held-out residual curve: $M members, L=$L, t_spinup=$(t_spinup) s"
    @info "  Members: $(join(ids, ", "))"
    isnothing(basis_dir) || mkpath(basis_dir)

    grid   = load_grid(config.prior_members[1].files[1])
    domain = AnalysisDomain(grid; margin_m)

    # Preload every member into RAM ONCE so each fold's POD reads from memory instead of
    # re-reading from disk O(M²) times (the original streaming cost). Pure I/O optimization —
    # results are identical (cache holds exactly load_member_snapshots output for this domain).
    cache = cache_members ?
        let
            @info "Caching all $M members in RAM (built once, reused across folds)..."
            load_member_cache(config.prior_members, domain; t_spinup)
        end : nothing

    resid_per_mbr = fill(NaN, L, M)
    total_energy  = zeros(Float64, M)

    for (k, held) in enumerate(ids)
        # Build (or load) the leave-one-out basis. truth_id=held makes build_pod_basis
        # drop the held member and tag the file as its truth basis.
        basis_file = isnothing(basis_dir) ? nothing : basis_filename(basis_dir, held)
        if !isnothing(basis_file) && reuse && isfile(basis_file)
            @info "  Fold $k/$M: reusing saved basis for held-out member $held ($basis_file)"
            basis = load_pod_basis(basis_file)
        else
            @info "  Fold $k/$M: holding out member $held, building basis from the other $(M-1)..."
            basis = build_pod_basis(config_path;
                L           = L,
                t_spinup    = t_spinup,
                include_ids = member_ids,
                exclude_ids = exclude_ids,
                truth_id    = held,
                margin_m    = margin_m,
                cache       = cache,
            )
            isnothing(basis_file) || save_pod_basis(basis, basis_file)
        end

        held_member = config.prior_members[k]
        X = _member_snapshots(held_member, cache; t_spinup, domain)   # (state_dim, T) Float32
        # Energy in the basis's W-norm (Parseval holds for W-orthonormal modes):
        # total = Σ_t xᵀWx, captured = Σ a². Use the fold's own block weights.
        n  = domain.n_grid
        total = sum(basis.block_weights[c] * sum(abs2, Float64.(@view X[(c-1)*n+1 : c*n, :])) for c in 1:3)
        total_energy[k] = total

        # Project in Float32 (as compute_pod_streaming does), accumulate energy in Float64.
        a = project(basis, X)                                      # (L_k, T) Float32, a = PrᵀWx
        e_mode = vec(sum(abs2, Float64.(a); dims=2))               # (L_k,) captured energy per mode
        captured = cumsum(e_mode)                                  # (L_k,)
        L_k = length(e_mode)
        for r in 1:min(L, L_k)
            resid_per_mbr[r, k] = 1.0 - captured[r] / total
        end
        # If this fold yielded fewer than L modes, hold the last value flat.
        if L_k < L
            resid_per_mbr[L_k+1:L, k] .= resid_per_mbr[L_k, k]
        end
        @info "    member $held: residual at r=$(min(L,L_k)) is $(round(resid_per_mbr[min(L,L_k), k]*100, digits=3))% of energy"
    end

    resid_mean = vec(mean(resid_per_mbr; dims=2))
    return HeldoutResidual(collect(1:L), resid_mean, resid_per_mbr, ids, total_energy, L)
end

# Write the held-out residual curve to a CSV the Python plotting code can read.
# Columns: rank, resid_mean, then one column per held-out member id.
function save_heldout_residual_csv(hr::HeldoutResidual, filepath::String)
    open(filepath, "w") do io
        println(io, "rank,resid_mean," * join(hr.member_ids, ","))
        for r in 1:hr.L
            row = join(string.(hr.resid_per_mbr[r, :]), ",")
            println(io, "$(hr.ranks[r]),$(hr.resid_mean[r]),$row")
        end
    end
    @info "Saved held-out residual curve to $filepath"
end

# ---------------------------------------------------------------------------
# Eigenvalue spectrum of the POD (snapshot Gram) decomposition.
#
# Columns: mode (1-based rank), eigenvalue λ_i (= captured energy in mode i, since
# the snapshot scatter is X̃ᵀX̃ = Σ λ_i), energy_frac = λ_i/Σλ, and cumulative_frac
# = (Σ_{j≤i} λ_j)/Σλ. The denominator is the TOTAL variance over all N eigenvalues,
# so cumulative_frac → 1 at the last mode. This is the in-sample spectrum (how well
# the basis represents its own snapshots); the closure diagnostic is the separate
# held-out residual curve. Python plotting reads this CSV.
function save_eigenvalue_spectrum_csv(λ::AbstractVector{<:Real}, filepath::String)
    λpos  = max.(Float64.(λ), 0.0)        # clamp tiny negative round-off from eigen()
    total = sum(λpos)
    total = total > 0 ? total : 1.0
    cum   = cumsum(λpos)
    open(filepath, "w") do io
        println(io, "mode,eigenvalue,energy_frac,cumulative_frac")
        for i in eachindex(λpos)
            println(io, "$i,$(λpos[i]),$(λpos[i]/total),$(cum[i]/total)")
        end
    end
    @info "Saved eigenvalue spectrum ($(length(λpos)) modes) to $filepath"
end

# ---------------------------------------------------------------------------
# Point-referenced spatial covariance map.
#
# The rank-L centered covariance implied by the basis is, exactly,
#     Cov = Pr [ diag(λ)/N − ā āᵀ ] Prᵀ ,
# because the snapshot scatter XXᵀ = Pr diag(λ) Prᵀ and the ensemble mean lies in
# the basis span (x̄ = Pr ā, ā = mean over the saved modal coefficients). We never
# form the (state_dim × state_dim) matrix; the column at a reference cell `ref` is
#     Cov[:,ref] = Pr ( (λ/N).*φ_ref − ā (āᵀ φ_ref) ) ,
# one state_dim × L matvec. φ_ref = Pr[ref,:] are the modes sampled at that cell.
#
# `ref_idx` is a flat state-vector index (see cell_to_state_index). The returned
# column, unpacked into u/v/w fields, is the covariance of the reference DOF with
# u, v and w everywhere — i.e. the velocity cross-covariance structure about ref.
# unbiased=false divides by N (population); true divides by N-1 (sample).
function covariance_column(basis::PODBasis, ref_idx::Int; unbiased::Bool=false) :: Vector{Float32}
    Pr     = basis.modes
    λ     = basis.eigenvalues
    denom = unbiased ? (basis.n_snapshots - 1) : basis.n_snapshots
    ā     = vec(mean(Float64.(basis.coeffs); dims=2))      # (L,)
    φ_ref = Float64.(Pr[ref_idx, :])                        # (L,)
    c     = (λ ./ denom) .* φ_ref .- ā .* (ā' * φ_ref)     # (L,)
    return Float32.(Pr * Float32.(c))                       # (state_dim,)
end

# Per-cell variance field = diagonal of the same rank-L covariance, computed without
# forming Cov. With M = diag(λ/denom) − ā āᵀ, var[i] = Σ_k (λ_k/denom) Pr[i,k]² − (Pr[i,:]·ā)²,
# i.e. an elementwise (Pr.²)(λ/denom) minus the squared ensemble-mean state (Pr ā).².
# Returned in the same flat [u;v;w] state layout as covariance_column; var[ref_idx] is the
# reference DOF's variance (and equals covariance_column(...)[ref_idx]).
function variance_field(basis::PODBasis; unbiased::Bool=false) :: Vector{Float64}
    Pr      = basis.modes                                   # (state_dim, L) Float32
    denom  = unbiased ? (basis.n_snapshots - 1) : basis.n_snapshots
    w      = basis.eigenvalues ./ denom                    # (L,)
    ā      = vec(mean(Float64.(basis.coeffs); dims=2))     # (L,)
    # Stream over modes to avoid a Float64 copy of the (huge) mode matrix:
    #   term1 = Σ_k w_k φ_k².,  meanst = Σ_k ā_k φ_k  (φ_k = column k of Pr).
    sd     = size(Pr, 1)
    term1  = zeros(Float64, sd)
    meanst = zeros(Float64, sd)
    @inbounds for k in 1:basis.L
        wk = w[k]; ak = ā[k]
        for i in 1:sd
            φ = Float64(Pr[i, k])
            term1[i]  += wk * φ * φ
            meanst[i] += ak * φ
        end
    end
    return max.(term1 .- meanst .^ 2, 0.0)                 # var = E[x²] − x̄²; clamp round-off
end

# Two-point CORRELATION column: cov(ref, ·) normalized to [−1, 1] by the per-cell std,
# corr[i] = cov[i] / √(var[ref]·var[i]). Cells with ~zero variance (e.g. the zero-filled
# sponge margin) are set to 0. var defaults to a freshly computed variance_field; pass a
# precomputed one to amortize across multiple reference points.
function correlation_column(basis::PODBasis, ref_idx::Int;
                            unbiased::Bool=false,
                            var::Union{Vector{Float64},Nothing}=nothing) :: Vector{Float32}
    cov     = Float64.(covariance_column(basis, ref_idx; unbiased))
    v       = isnothing(var) ? variance_field(basis; unbiased) : var
    var_ref = v[ref_idx]
    var_ref > 0 || error("reference cell variance is zero; pick a reference inside the domain")
    denom   = sqrt.(var_ref .* v)
    corr    = similar(cov)
    @inbounds for i in eachindex(cov)
        corr[i] = denom[i] > 0 ? clamp(cov[i] / denom[i], -1.0, 1.0) : 0.0
    end
    return Float32.(corr)
end

# Convenience: two-point map about a physical reference point (x,y metres; z metres AGL)
# and component (:u/:v/:w), returned as an NCfield ready for write_fasteddy_nc.
# normalize=false → covariance; normalize=true → correlation in [−1, 1].
function covariance_map(basis::PODBasis, domain::AnalysisDomain;
                        x_m::Real, y_m::Real, z_agl_m::Real,
                        comp::Symbol=:u, unbiased::Bool=false, normalize::Bool=false)
    3 * domain.n_grid == size(basis.modes, 1) ||
        error("domain.state_dim=$(3*domain.n_grid) ≠ basis state_dim=$(size(basis.modes,1)); " *
              "margin_m must match the value used to build the basis")
    i, j, k = nearest_cell(basis.grid, domain, x_m, y_m, z_agl_m)
    ref_idx = cell_to_state_index(domain, i, j, k, comp)
    kind    = normalize ? "Correlation" : "Covariance"
    @info "$kind reference: cell ($i,$j,$k) comp=$comp → state index $ref_idx"
    field      = normalize ? correlation_column(basis, ref_idx; unbiased) :
                             covariance_column(basis, ref_idx; unbiased)
    u3, v3, w3 = unpack_snapshot(field, basis.grid, domain)
    ncf = NCfield(Float32[0f0], basis.grid,
                  reshape(u3, size(u3)..., 1),
                  reshape(v3, size(v3)..., 1),
                  reshape(w3, size(w3)..., 1))
    return (ncf = ncf, cell = (i, j, k), comp = comp)
end

# Correlation map — thin wrapper over covariance_map with normalize=true.
correlation_map(basis::PODBasis, domain::AnalysisDomain; kwargs...) =
    covariance_map(basis, domain; normalize=true, kwargs...)

# Reconstruct NCfield from POD basis and coefficient vector(s).
# a_hat is (L,) for a single timestep or (L, T) for a time series.
# Reconstruction: x = Pr * a_hat  (modes * coefficients → state vector)
function reconstruct_from_pod(basis::PODBasis, a_hat::AbstractVecOrMat{<:Real},
                               t::Vector{Float32}, domain::AnalysisDomain) :: NCfield
    Pr = basis.modes                          # (domain.state_dim, L)
    X = Float32.(Pr * Float64.(a_hat))        # (domain.state_dim, T)
    return unpack_to_NCfield(X, t, basis.grid, domain)
end
