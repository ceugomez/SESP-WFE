# pod.jl
# cgf cego6160@colorado.edu 4.1.26
#
# Snapshot POD (Sirovich 1987) over an ensemble: the basis Pr, its modal coefficients, the
# leave-one-member-out fold bases (sliced from one shared Gram matrix), the held-out residual
# (closure) curve, the eigenvalue spectrum, and point-referenced covariance maps.
#
# Conventions: the POD is UNCENTERED (mean flow kept, so the forcing distribution is in the
# basis) and uses the plain Euclidean inner product over the masked [u;v;w] state — no
# cell-volume or per-component weighting.

using NCDatasets
using LinearAlgebra
using Statistics
using JSON3 
using JLD2

# A truncated POD basis and the modal coefficients of the snapshots it was built from.
struct PODBasis
    coeffs        :: Matrix{Float32}   # (L, N) modal coefficients a = Prᵀx of every training snapshot,
                                       #        member-blocks of T post-spin-up snapshots each
    modes         :: Matrix{Float32}   # (state_dim, L) — orthonormal POD modes Pr_k (PrᵀPr=I)
    eigenvalues   :: Vector{Float64}   # (L,) descending in magnitude
    L             :: Int               # number of retained modes
    n_snapshots   :: Int               # N = M·T training snapshots
    grid          :: GridInfo          # grid definition, see fasteddy_io.jl
    member_ids    :: Vector{String}    # ids of members used to build this basis, in column-block order
    truth_id      :: String            # id of the member held out / treated as truth for this basis
end

# Optional in-RAM cache of member snapshot matrices, keyed by member id. When present,
# `_member_snapshots` returns the cached matrix instead of re-reading from disk — this is
# the ONLY behavioural change for caching; the returned values are identical to a fresh
# load_member_snapshots(member; t_spinup, domain), so all downstream POD arithmetic is
# unchanged. The cache MUST have been built with the same t_spinup and domain.
const SnapshotCache = Dict{String,Matrix{Float32}}

# In-memory-only record of the full-pool snapshot Gram matrix C = X'X computed while
# building a PODBasis (compute_pod_streaming's Pass 1). NEVER persisted to disk — it exists
# so that, within one session, leave-one-out folds (heldout_residual_curve, build_rep_error)
# can slice a member's rows/cols out of C and re-eigendecompose the (smaller) submatrix
# instead of re-streaming O(M²) pairwise snapshot GEMMs per fold. See loo_basis_from_gram.
struct PODGram
    C          :: Matrix{Float64}   # (N,N) snapshot Gram matrix, member-block column order
    member_ids :: Vector{String}    # (M,) block order matching C's column blocks
    T          :: Int               # snapshots per member (columns per block)
end

_member_snapshots(member::MemberInfo, cache::Union{SnapshotCache,Nothing};
                  t_spinup::Float64, domain::AnalysisDomain) =
    isnothing(cache) ? load_member_snapshots(member; t_spinup, domain) : cache[member.id]

# Preload the given members' masked snapshot matrices into a cache (one disk read each).
function load_member_cache(members::Vector{MemberInfo}, domain::AnalysisDomain;
                           t_spinup::Float64) :: SnapshotCache
    cache = SnapshotCache()
    for (i, m) in enumerate(members)
        @info "  Caching member $(m.id) ($i/$(length(members)))..."
        flush(stderr)   # progress must survive an OOM SIGKILL (buffered stderr is lost)
        cache[m.id] = load_member_snapshots(m; t_spinup, domain)
    end
    gb = sum(length(v) for v in values(cache)) * 4 / 1e9
    @info "  Cached $(length(cache)) members in RAM (≈ $(round(gb, digits=1)) GB)"
    return cache
end

# Project physical state(s) onto the basis: a = Prᵀx. Use this anywhere coefficients
# are recovered from a state.
project(basis::PODBasis, X::AbstractVecOrMat) = basis.modes' * X

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
# Because the modes Pr are orthonormal, the whole curve comes from one projection
# per held-out member (Parseval): with a = Prᵀx and per-mode captured
# energy eᵢ = Σ_t a[i,t]², the residual fraction at rank r is
#     E_resid(r) = 1 - (Σ_{i≤r} eᵢ) / Σ_t ‖x_t‖² .
# NOTE the denominator is TOTAL energy (mean flow included), which with a ~10 m/s mean
# makes the fractions look small; normalize by fluctuation energy when quoting closure.
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
                          cache::Union{SnapshotCache,Nothing}=nothing) :: Tuple{PODBasis,PODGram}

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
    return compute_pod_streaming(members, grid, domain; L, t_spinup, truth_id=resolved_truth_id, spectrum_csv, cache)
end

# POD via Method of Snapshots (Sirovich 1987)
# Avoids holding the full snapshot matrix in memory by re-reading members' blocks
# from disk instead of caching them across passes. Pass 1 (Gram matrix) needs all
# pairs Xi'Xj, so it re-reads Xj for each j<=i (O(M(M+1)/2) member loads total);
# Pass 2 (mode recovery) needs each member only once (O(M) loads). The modal
# coefficients a = PrᵀX never need a third read: since Pr = X V_L Λ_L^{-1/2},
# a = Pr'X = Λ_L^{-1/2}V_L'(X'X) = Λ_L^{-1/2}V_L'C = Λ_L^{1/2}V_L' (using CV_L=V_LΛ_L) —
# pure linear algebra on quantities Pass 1 already produced.
#
# Peak memory: two Float32 member blocks (state_dim × T) + the bounded Float64 chunk
#              buffers (2·GRAM_CHUNK_ROWS·T) + the (state_dim × L) mode matrix — never the
#              full (state_dim × M*T) matrix, and never a full Float64 member copy.
#
# If a `cache` (member id → snapshot matrix) is supplied, member blocks are read from RAM
# instead of disk. This is purely an I/O optimization: the cached matrices are identical to
# load_member_snapshots output, so results match the streaming path exactly. Peak memory then
# includes the cache (M·state_dim·T Float32) plus the same transient Float64 copies as before.
#
# Returns (basis, gram): `gram` is the full Gram matrix C = X'X, kept in memory (never
# persisted) so leave-one-out folds can be sliced from it directly — see loo_basis_from_gram.

# Rows converted to Float64 per chunk in _accum_gram_block!. Bounds Pass 1's transient
# memory at 2·GRAM_CHUNK_ROWS·T·8 bytes regardless of state_dim.
const GRAM_CHUNK_ROWS = 4_000_000

# Accumulate the (T×T) Gram block Xiᵀ·Xj into `block` with Float64 arithmetic, WITHOUT
# materializing Float64 copies of the full (state_dim×T) Float32 snapshot matrices:
# convert one bounded row-chunk at a time into the preallocated buffers Abuf/Bbuf and let
# BLAS accumulate. Same Float64 products/sums as `Float64.(Xi)' * Float64.(Xj)` up to
# summation order. When Xi === Xj (diagonal block) the second conversion is skipped.
function _accum_gram_block!(block::Matrix{Float64},
                            Xi::AbstractMatrix{Float32}, Xj::AbstractMatrix{Float32},
                            Abuf::Matrix{Float64}, Bbuf::Matrix{Float64})
    fill!(block, 0.0)
    s     = size(Xi, 1)
    chunk = size(Abuf, 1)
    for r0 in 1:chunk:s
        r1 = min(r0 + chunk - 1, s)
        nr = r1 - r0 + 1
        A  = view(Abuf, 1:nr, :)
        copyto!(A, view(Xi, r0:r1, :))
        B  = if Xi === Xj
            A
        else
            Bv = view(Bbuf, 1:nr, :)
            copyto!(Bv, view(Xj, r0:r1, :))
            Bv
        end
        mul!(block, A', B, 1.0, 1.0)
    end
    return block
end

function compute_pod_streaming(members::Vector{MemberInfo}, grid::GridInfo,
                                domain::AnalysisDomain;
                                L::Int=10,
                                t_spinup::Float64=900.0,
                                truth_id::String="",
                                spectrum_csv::Union{String,Nothing}=nothing,
                                cache::Union{SnapshotCache,Nothing}=nothing) :: Tuple{PODBasis,PODGram}
    M     = length(members)
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in members]
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup snapshot counts: $T_vec"
    T = T_vec[1]
    N = M * T
    L = min(L, N)

    # --- Pass 1: build Gram matrix C = X'X, one member at a time ---
    # The Float64 promotion needed for accurate T×T blocks happens a bounded row-chunk at a
    # time in preallocated buffers: peak transient is 2·chunk·T·8 bytes independent of
    # state_dim, so Pass 1 adds only ~2·chunk·T·8 on top of the Float32 cache instead of
    # two full state_dim×T Float64 member copies per pair.
    @info "  Pass 1/2: building $(N)×$(N) Gram matrix (streaming over $M members)..."
    C = zeros(Float64, N, N)
    chunk = min(GRAM_CHUNK_ROWS, domain.state_dim)
    Abuf  = Matrix{Float64}(undef, chunk, T)
    Bbuf  = Matrix{Float64}(undef, chunk, T)
    block = Matrix{Float64}(undef, T, T)
    for (mi, member) in enumerate(members)
        @info "    Loading member $(member.id) ($(mi)/$M)..."
        flush(stderr)
        Xi = _member_snapshots(member, cache; t_spinup, domain)   # (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        for mj in 1:mi
            Xj = mj == mi ? Xi : _member_snapshots(members[mj], cache; t_spinup, domain)
            col_j = (mj-1)*T+1 : mj*T
            _accum_gram_block!(block, Xi, Xj, Abuf, Bbuf)         # (T×T) — small
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

    # --- Pass 2: recover truncated modes Pr = X V_L scale ---
    @info "  Pass 2/2: recovering $L POD modes (state_dim=$(domain.state_dim))..."
    state_dim = domain.state_dim
    Pr = zeros(Float32, state_dim, L)
    for (mi, member) in enumerate(members)
        Xi = _member_snapshots(member, cache; t_spinup, domain)   # (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        W = Float32.(V_L[col_i, :] * scale)                       # (T, L) — small
        mul!(Pr, Xi, W, true, true)                               # Pr .+= Xi*W, no state-size temporaries
    end

    # --- Modal coefficients a = PrᵀX = Λ_L^{1/2}V_Lᵀ (exact identity — no reread) ---
    a = Float32.(Diagonal(sqrt.(λ_L)) * V_L')                    # (L, N)

    @info "  POD complete: L=$L, N=$N snapshots"
    member_ids = [m.id for m in members]
    basis = PODBasis(a, Pr, λ_L, L, N, grid, member_ids, truth_id)
    gram  = PODGram(C, member_ids, T)
    return basis, gram
end

# Slice a member's rows/cols out of a full-pool Gram matrix, returning a smaller PODGram
# over the remaining members. Reused both by loo_basis_from_gram (to eigendecompose the
# reduced pool) and by nested leave-one-out (e.g. build_rep_error's inner folds), so a
# fold's own Gram matrix never needs to be re-streamed from disk — it's already a
# principal submatrix of the pool-level Gram that was paid for once.
function gram_slice(gram::PODGram, held_id::String) :: PODGram
    held_pos = findfirst(==(held_id), gram.member_ids)
    isnothing(held_pos) && error("held_id '$held_id' not found in gram.member_ids")
    other_ids = [id for (i, id) in enumerate(gram.member_ids) if i != held_pos]

    T  = gram.T
    N  = length(gram.member_ids) * T
    held_cols  = (held_pos-1)*T+1 : held_pos*T
    other_cols = setdiff(1:N, held_cols)

    return PODGram(gram.C[other_cols, other_cols], other_ids, T)
end

# Build a leave-one-out POD basis for `held_id` directly from an already-computed full-pool
# Gram matrix `gram`, by slicing that member's rows/cols out and re-eigendecomposing the
# (smaller) submatrix — skipping Pass 1 of compute_pod_streaming entirely, since its O(M²)
# pairwise state_dim-length GEMMs are exactly the work `gram` already paid for once when the
# full-pool basis was built. Only the "other" members' physical snapshots are touched (mode
# recovery, one read each via `cache` if supplied); their modal coefficients come for free
# from the same Λ^{1/2}Vᵀ identity used in compute_pod_streaming — no reread needed there either.
function loo_basis_from_gram(gram::PODGram, members::Vector{MemberInfo}, grid::GridInfo,
                              domain::AnalysisDomain;
                              held_id::String, L::Int, t_spinup::Float64,
                              cache::Union{SnapshotCache,Nothing}=nothing) :: PODBasis
    id2m = Dict(m.id => m for m in members)
    sub    = gram_slice(gram, held_id)
    other_ids = sub.member_ids
    T = sub.T

    decomp = eigen(Symmetric(sub.C))
    λ = reverse(decomp.values)
    V = reverse(decomp.vectors, dims=2)

    valid   = λ .> 1e-10 * λ[1]
    n_valid = sum(valid)
    L_k = min(L, n_valid)
    λ_L = λ[1:L_k]
    V_L = V[:, 1:L_k]
    scale = Diagonal(1.0 ./ sqrt.(λ_L))

    state_dim = domain.state_dim
    Pr = zeros(Float32, state_dim, L_k)
    for (mi, oid) in enumerate(other_ids)
        Xi = _member_snapshots(id2m[oid], cache; t_spinup, domain)   # (state_dim, T)
        rows = (mi-1)*T+1 : mi*T
        W = Float32.(V_L[rows, :] * scale)                           # (T, L_k) — small
        mul!(Pr, Xi, W, true, true)                                  # Pr .+= Xi*W, no state-size temporaries
    end

    a = Float32.(Diagonal(sqrt.(λ_L)) * V_L')     # (L_k, (M-1)*T) — free, no reread
    return PODBasis(a, Pr, λ_L, L_k, length(other_ids)*T, grid, other_ids, held_id)
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

    # Build the full-pool Gram matrix ONCE; every fold below slices it instead of paying
    # Pass 1's O(M²) pairwise GEMMs again — see loo_basis_from_gram.
    @info "Building full-pool Gram matrix once (shared across all $M leave-one-out folds)..."
    _, gram = compute_pod_streaming(config.prior_members, grid, domain; L, t_spinup, truth_id="", cache)

    resid_per_mbr = fill(NaN, L, M)
    total_energy  = zeros(Float64, M)

    for (k, held) in enumerate(ids)
        # Build (or load) the leave-one-out basis. truth_id=held tags the fold as the
        # basis for assimilating k as truth (matches build_pod_basis's convention).
        basis_file = isnothing(basis_dir) ? nothing : basis_filename(basis_dir, held)
        if !isnothing(basis_file) && reuse && isfile(basis_file)
            @info "  Fold $k/$M: reusing saved basis for held-out member $held ($basis_file)"
            basis = load_pod_basis(basis_file)
        else
            @info "  Fold $k/$M: holding out member $held (Gram-matrix slice, no full rebuild)..."
            basis = loo_basis_from_gram(gram, config.prior_members, grid, domain;
                                         held_id=held, L, t_spinup, cache)
            isnothing(basis_file) || save_pod_basis(basis, basis_file)
        end

        held_member = config.prior_members[k]
        X = _member_snapshots(held_member, cache; t_spinup, domain)   # (state_dim, T) Float32
        # Energy (Parseval holds for orthonormal modes): total = Σ_t xᵀx, captured = Σ a².
        total = sum(abs2 ∘ Float64, X)   # Float64 accumulation, no state-size copy
        total_energy[k] = total

        # Project in Float32 (as compute_pod_streaming does), accumulate energy in Float64.
        a = project(basis, X)                                      # (L_k, T) Float32, a = Prᵀx
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

# Two-point map about a physical reference point (x,y metres; z metres AGL)
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
