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
    modes         :: Matrix{Float32}   # (state_dim, L) — orthonormal POD modes ψ_k
    eigenvalues   :: Vector{Float64}   # (L,) descending in magnitude
    L             :: Int               # length of eigenvalues/modes 
    n_snapshots   :: Int               # number of snapshots ()
    grid          :: GridInfo          # grid definition, see fasteddy_io.jl
    member_ids    :: Vector{String}    # ids of members used to build this basis, in column-block order
    truth_id      :: String            # id of the member held out / treated as truth for this basis
end

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
# Because the modes Φ are orthonormal, the whole curve comes from one projection
# per held-out member (Parseval): with a = Φ'x and per-mode captured energy
# eᵢ = Σ_t a[i,t]², the residual fraction at rank r is
#     E_resid(r) = 1 - (Σ_{i≤r} eᵢ) / Σ_t ‖x_t‖² .
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
    return joinpath(output_dir, "pod_basis_truth_$(truth_id).jld2")
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
                          margin_m::Float32=600f0) :: PODBasis

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

    # The truth member must NEVER enter the basis (no leakage). Drop it from the prior
    # set if the override (or include/exclude filtering) left it in.
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
    basis = compute_pod_streaming(members, grid, domain; L, t_spinup, truth_id=resolved_truth_id)
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
function compute_pod_streaming(members::Vector{MemberInfo}, grid::GridInfo,
                                domain::AnalysisDomain;
                                L::Int=10,
                                t_spinup::Float64=900.0,
                                truth_id::String="") :: PODBasis
    M     = length(members)
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in members]
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup snapshot counts: $T_vec"
    T = T_vec[1]
    N = M * T
    L = min(L, N)

    # --- Pass 1: build Gram matrix C = X'X, one member at a time ---
    @info "  Pass 1/2: building $(N)×$(N) Gram matrix (streaming over $M members)..."
    C = zeros(Float64, N, N)
    for (mi, member) in enumerate(members)
        @info "    Loading member $(member.id) ($(mi)/$M)..."
        Xi64 = Float64.(load_member_snapshots(member; t_spinup, domain))   # (domain.state_dim, T)
        col_i = (mi-1)*T+1 : mi*T
        for mj in 1:mi
            Xj64 = mj == mi ? Xi64 : Float64.(load_member_snapshots(members[mj]; t_spinup, domain))
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

    valid = λ .> 1e-10 * λ[1]
    n_valid = sum(valid)
    if n_valid < L
        @warn "Only $n_valid valid eigenvalues; reducing L from $L to $n_valid"
        L = n_valid
    end
    λ_L = λ[1:L]
    V_L = V[:, 1:L]                                    # (N, L)
    scale = Diagonal(1.0 ./ sqrt.(λ_L))               # (L, L)

    # --- Pass 2: recover modes Φ = X * V_L * scale, streaming members ---
    @info "  Pass 2/2: recovering $L POD modes (state_dim=$(domain.state_dim))..."
    state_dim = domain.state_dim
    Φ = zeros(Float32, state_dim, L)
    for (mi, member) in enumerate(members)
        Xi = load_member_snapshots(member; t_spinup, domain)   # (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        Φ .+= Float32.(Float64.(Xi) * (V_L[col_i, :] * scale))
    end

    # --- Compute modal coefficients a = Φ'X (streaming) ---
    @info "  Computing modal coefficients..."
    a = Matrix{Float32}(undef, L, N)
    for (mi, member) in enumerate(members)
        Xi = load_member_snapshots(member; t_spinup, domain)   # (state_dim, T) Float32
        col_i = (mi-1)*T+1 : mi*T
        a[:, col_i] = Φ' * Xi
    end

    @info "  POD complete: L=$L, N=$N snapshots"
    member_ids = [m.id for m in members]
    return PODBasis(a, Φ, λ_L, L, N, grid, member_ids, truth_id)
end

# Compute the held-out residual-energy curve over the chosen prior members.
# member_ids: which member ids to cycle as held-out (nothing = all prior members
#             in the config, after the usual include/exclude filtering).
# L is the maximum truncation rank evaluated; the curve covers ranks 1:L.
# basis_dir: if given, each fold's leave-one-out basis is saved there as
#            pod_basis_truth_<held>.jld2 (the same artifact the KF assimilation loads
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
                                 reuse::Bool=true) :: HeldoutResidual

    # Resolve the full set of members participating in the cross-validation.
    config = load_ensemble_config(config_path; include_ids=member_ids, exclude_ids)
    ids = [m.id for m in config.prior_members]
    M = length(ids)
    M >= 3 || error("Leave-one-out needs >= 3 members; got $M ($(join(ids, \", \")))")
    @info "Held-out residual curve: $M members, L=$L, t_spinup=$(t_spinup) s"
    @info "  Members: $(join(ids, \", \"))"
    isnothing(basis_dir) || mkpath(basis_dir)

    grid   = load_grid(config.prior_members[1].files[1])
    domain = AnalysisDomain(grid; margin_m)

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
            )
            isnothing(basis_file) || save_pod_basis(basis, basis_file)
        end

        held_member = config.prior_members[k]
        X = load_member_snapshots(held_member; t_spinup, domain)   # (state_dim, T) Float32
        total = sum(abs2, Float64.(X))
        total_energy[k] = total

        # Project in Float32 (as compute_pod_streaming does), accumulate energy in Float64.
        a = basis.modes' * X                                       # (L_k, T) Float32
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

# Reconstruct NCfield from POD basis and coefficient vector(s).
# a_hat is (L,) for a single timestep or (L, T) for a time series.
# Reconstruction: x = Φ * a_hat  (modes * coefficients → state vector)
function reconstruct_from_pod(basis::PODBasis, a_hat::AbstractVecOrMat{<:Real},
                               t::Vector{Float32}, domain::AnalysisDomain) :: NCfield
    Φ = basis.modes                          # (domain.state_dim, L)
    X = Float32.(Φ * Float64.(a_hat))        # (domain.state_dim, T)
    return unpack_to_NCfield(X, t, basis.grid, domain)
end
