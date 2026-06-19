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
    )
    sz_gb = (size(basis.modes, 1) * size(basis.modes, 2) * 4) / 1e9
    @info "Saved POD basis to $filepath  (L=$(basis.L), modes ≈ $(round(sz_gb, digits=2)) GB)"
end

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
    @info "Using $(length(members)) prior members: $(join([m.id for m in members], ", "))"

    if !isnothing(truth_id)
        all_known = [config.truth_member; config.prior_members]
        idx = findfirst(m -> m.id == truth_id, all_known)
        isnothing(idx) && error("truth_id '$truth_id' not found in config (prior or truth members)")
        @info "Truth member overridden to: $(all_known[idx].id)"
    end

    @info "Loading grid from first member file..."
    grid = load_grid(members[1].files[1])
    domain = AnalysisDomain(grid; margin_m)
    cx = round(Int, margin_m / grid.dx)
    @info "  Grid: $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$(grid.state_dim)"
    @info "  Analysis domain: $(grid.Nx - 2cx)×$(grid.Ny - 2cx)×$(grid.Nz) interior cells ($(margin_m) m margin), state_dim=$(domain.state_dim)"

    @info "Computing POD (L=$L, t_spinup=$(t_spinup) s)..."
    basis = compute_pod_streaming(members, grid, domain; L, t_spinup)
    return basis
end

# POD via Method of Snapshots (Sirovich 1987), memory-efficient streaming version.
#
# Avoids holding the full snapshot matrix in memory by re-reading members' blocks
# from disk instead of caching them across passes. Pass 1 (Gram matrix) needs all
# pairs Xi'Xj, so it re-reads Xj for each j<=i (O(M(M+1)/2) member loads total);
# passes 2 and 3 (mode recovery, modal coefficients) need each member only once
# (O(M) loads each). Trades extra disk I/O for peak memory independent of M.
#
# Peak memory: one Float32 member block (state_dim × T) + one Float64 copy of same
#              + the (state_dim × L) mode matrix — never the full (state_dim × M*T) matrix,
#              and never more than two members' data resident at once.
function compute_pod_streaming(members::Vector{MemberInfo}, grid::GridInfo,
                                domain::AnalysisDomain;
                                L::Int=10,
                                t_spinup::Float64=900.0) :: PODBasis
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
    return PODBasis(a, Φ, λ_L, L, N, grid)
end

# Reconstruct NCfield from POD basis and coefficient vector(s).
# a_hat is (L,) for a single timestep or (L, T) for a time series.
# Reconstruction: x = Φ * a_hat  (modes * coefficients → state vector)
function reconstruct_from_pod(basis::PODBasis, a_hat::AbstractVecOrMat{<:Real},
                               t::Vector{Float32}, domain::AnalysisDomain) :: NCfield
    Φ = basis.modes                          # (domain.state_dim, L)
    X = Float32.(Φ * Float64.(a_hat))        # (domain.state_dim, T)
    T_len = size(X, 2)
    grid  = basis.grid

    u4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    v4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    w4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    for i in 1:T_len
        u4[:,:,:,i], v4[:,:,:,i], w4[:,:,:,i] = unpack_snapshot(view(X, :, i), grid, domain)
    end
    return NCfield(t, grid, u4, v4, w4)
end
