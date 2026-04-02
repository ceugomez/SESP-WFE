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
    L             :: Int
    n_snapshots   :: Int
    grid          :: GridInfo
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
function build_pod_basis(config_path::String;
                          L::Int=10,
                          t_spinup::Float64=900.0) :: PODBasis

    config = load_ensemble_config(config_path)

    @info "Loading grid from first member file..."
    grid = load_grid(config.prior_members[1].files[1])
    @info "  Grid: $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$(grid.state_dim)"

    @info "Building snapshot matrix (t_spinup=$(t_spinup) s)..."
    U = build_snapshot_matrix(config; t_spinup)
    
    @info "  Matrix size: $(size(U, 1)) × $(size(U, 2))  ($(round(sizeof(U)/1e9, digits=1)) GB)"
    @info "Computing POD (L=$L)..."
    basis = compute_pod(U, grid; L)
    return basis
end

# Assemble snapshot matrix from all prior ensemble members, excluding spinup.
# Returns (state_dim, M*T) matrix where M=members, T=post-spinup timesteps per member.
function build_snapshot_matrix(config::EnsembleConfig;
                                    t_spinup::Float64=900.0) :: Matrix{Float32}
    M     = config.n_members
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in config.prior_members]
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup counts: $T_vec"
    T = T_vec[1]
    N = M * T

    snaps_1   = load_member_snapshots(config.prior_members[1]; t_spinup)
    state_dim = size(snaps_1, 1)
    U         = Matrix{Float32}(undef, state_dim, N)
    U[:, 1:T] = snaps_1

    for (mi, member) in enumerate(config.prior_members[2:end])
        @info "  Loading member $(member.id) ($(mi+1)/$M)..."
        snaps = load_member_snapshots(member; t_spinup)
        j     = mi * T
        U[:, j+1:j+T] = snaps
    end
    @info "  Snapshot matrix: $(state_dim) × $N  ($(M) members × $T timesteps)"
    return U
end

# POD via Method of Snapshots (Sirovich 1987).
# Instead of eigendecomposing the huge (state_dim × state_dim) covariance,
# eigendecompose the small (N × N) Gram matrix, then recover full modes.
function compute_pod(X::Matrix{Float32}, grid::GridInfo;
                     L::Int=10) :: PODBasis
    state_dim, N = size(X)
    L = min(L, N)

    @info "  Computing $(N)×$(N) Gram matrix..."
    C = Symmetric(Float64.(X)' * Float64.(X))

    @info "  Eigendecomposing $(N)×$(N) matrix..."
    decomp = eigen(C)
    λ = reverse(decomp.values)
    V = reverse(decomp.vectors, dims=2)

    valid = λ .> 1e-10 * λ[1]
    n_valid = sum(valid)
    if n_valid < L
        @warn "Only $n_valid valid eigenvalues; reducing L from $L to $n_valid"
        L = n_valid
    end
    λ_L = λ[1:L]
    V_L = V[:, 1:L]

    @info "  Computing $L POD modes (state_dim=$state_dim)..."
    Φ = Float32.(Float64.(X) * V_L * Diagonal(1.0 ./ sqrt.(λ_L)))
    a = Φ' * X

    @info "  POD complete: L=$L"
    return PODBasis(a, Φ, λ_L, L, N, grid)
end

# Reconstruct NCfield from POD basis and coefficient vector(s).
# a_hat is (L,) for a single timestep or (L, T) for a time series.
# Reconstruction: x = Φ * a_hat  (modes * coefficients → state vector)
function reconstruct_from_pod(basis::PODBasis, a_hat::AbstractVecOrMat{<:Real},
                               t::Vector{Float32}) :: NCfield
    Φ = basis.modes                          # (state_dim, L)
    X = Float32.(Φ * Float64.(a_hat))        # (state_dim, T)
    T_len = size(X, 2)
    grid  = basis.grid

    u4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    v4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    w4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    for i in 1:T_len
        u4[:,:,:,i], v4[:,:,:,i], w4[:,:,:,i] = unpack_snapshot(view(X, :, i), grid)
    end
    return NCfield(t, grid, u4, v4, w4)
end
