# pod_ensemble.jl
# DESCRIPTION
# cgf cego6160@colorado.edu 4.1.26

using NCDatasets
using LinearAlgebra
using Statistics
using JSON3 
using JLD2

# ============================================================
# Structs
# ============================================================

struct MemberInfo
    id          :: String
    output_dir  :: String
    met_dir_deg :: Float64
    U_g_ms      :: Float64
    V_g_ms      :: Float64
    files       :: Vector{String}    # absolute paths, sorted numerically by timestep
    timesteps   :: Vector{Int}       # model-step suffix values (0, 3000, 6000, …)
    sim_times   :: Vector{Float32}   # simulation time in seconds per file
end

struct EnsembleConfig
    n_members     :: Int
    prior_members :: Vector{MemberInfo}   # 10 prior members, NOT including truth (defined above)
    truth_member  :: MemberInfo
end

# Coordinate arrays are full 3D (terrain-following sigma coords)
struct GridInfo
    Nx        :: Int
    Ny        :: Int
    Nz        :: Int
    dx        :: Float32
    dy        :: Float32
    xPos      :: Array{Float32,3}    # (Nx, Ny, Nz)
    yPos      :: Array{Float32,3}
    zPos      :: Array{Float32,3}
    n_grid    :: Int                 # Nx * Ny * Nz
    state_dim :: Int                 # 3 * n_grid
end

struct PODBasis
    coeffs        :: Matrix{Float32}   # (?,?) - POD modal coefficients for all timesteps. as (rows=modes, cols=timesteps)
    modes         :: Matrix{Float32}   # (state_dim, L) — orthonormal POD modes ψ_k, where L is the number of
    eigenvalues   :: Vector{Float64}   # (L,) descending in magnitude
    L             :: Int
    n_snapshots   :: Int
    grid          :: GridInfo
end

# ============================================================
# File Discovery
# ============================================================

# Returns (sorted_paths, timesteps) for all output files in output_dir.
# Sorting is by numeric timestep suffix — never lexicographic.
# All files are returned (including t=0); spin-up filtering happens in snapshot loaders.
function discover_member_files(output_dir::String, member_id::String)
    all_files = readdir(output_dir; join=false)
    pat = Regex("^FE_Member$(member_id)\\.([0-9]+)\$")
    matched = Tuple{Int,String}[]
    for f in all_files
        m = match(pat, f)
        if !isnothing(m)
            push!(matched, (parse(Int, m.captures[1]), joinpath(output_dir, f)))
        end
    end
    isempty(matched) && error("No output files found for member '$member_id' in $output_dir")
    sort!(matched, by=first)
    paths     = [p[2] for p in matched]
    timesteps = [p[1] for p in matched]
    return paths, timesteps
end

# ============================================================
# Config Loading
# ============================================================

function _read_sim_times(files::Vector{String}) :: Vector{Float32}
    times = Vector{Float32}(undef, length(files))
    for (i, f) in enumerate(files)
        NCDataset(f, "r") do ds
            times[i] = Float32(ds["time"][1])
        end
    end
    return times
end

function _make_member_info(d) :: MemberInfo
    id         = String(d.id)
    output_dir = String(d.output_dir)
    files, timesteps = discover_member_files(output_dir, id)
    sim_times = _read_sim_times(files)
    return MemberInfo(
        id,
        output_dir,
        Float64(d.met_dir_deg),
        Float64(d.U_g_ms),
        Float64(d.V_g_ms),
        files,
        timesteps,
        sim_times,
    )
end

function load_ensemble_config(config_path::String) :: EnsembleConfig
    @info "Parsing ensemble config: $config_path"
    cfg = JSON3.read(read(config_path, String))
    prior_members = [_make_member_info(m) for m in cfg.prior_members]
    truth_member  = _make_member_info(cfg.truth_member)
    return EnsembleConfig(length(prior_members), prior_members, truth_member)
end

# ============================================================
# Grid Loading
# ============================================================

# Read grid metadata once from any output file.
# NCDatasets.jl reverses netCDF dimension order: ds["u"][:,:,:,1] → (Nx, Ny, Nz).
# zPos is a full 3D array — terrain-following coordinates vary in all three dimensions.
function load_grid(filepath::String) :: GridInfo
    NCDataset(filepath, "r") do ds
        u    = ds["u"][:, :, :, 1]       # trigger read; shape (Nx, Ny, Nz)
        xPos = Float32.(ds["xPos"][:, :, :, 1])
        yPos = Float32.(ds["yPos"][:, :, :, 1])
        zPos = Float32.(ds["zPos"][:, :, :, 1])
        Nx, Ny, Nz = size(u)
        GridInfo(Nx, Ny, Nz, 20f0, 20f0, xPos, yPos, zPos, Nx*Ny*Nz, 3*Nx*Ny*Nz)
    end
end

# ============================================================
# Snapshot Loading
# ============================================================

# Canonical state-vector loader: returns [u_flat; v_flat; w_flat] as Float32.
# Flattening is Julia column-major: x (xIndex) varies fastest.
function load_snapshot(filepath::String) :: Vector{Float32}
    NCDataset(filepath, "r") do ds
        u = ds["u"][:, :, :, 1]    # (Nx, Ny, Nz)
        v = ds["v"][:, :, :, 1]
        w = ds["w"][:, :, :, 1]
        vcat(vec(u), vec(v), vec(w))    # join as 1x(Nx*Ny*Nz*3) vector
    end
end

# Boolean mask: which of member's files have sim_time >= t_spinup (seconds).
function _spinup_mask(member::MemberInfo, t_spinup::Float64) :: BitVector
    return member.sim_times .>= Float32(t_spinup)
end

# Load post-spinup snapshots for one member as a (state_dim, T) matrix.
function load_member_snapshots(member::MemberInfo;
                                t_spinup::Float64=900.0) :: Matrix{Float32}
    mask  = _spinup_mask(member, t_spinup)
    files = member.files[mask]
    T     = length(files)
    T == 0 && error("No post-spinup snapshots for member $(member.id) " *
                    "(t_spinup=$(t_spinup) s, max sim_time=$(maximum(member.sim_times)) s)")
    x0        = load_snapshot(files[1]) 
    state_dim = length(x0)
    U         = Matrix{Float32}(undef, state_dim, T)
    U[:, 1]   = x0
    for t in 2:T
        U[:, t] = load_snapshot(files[t])
    end
    return U
end

# sirovich, method of snapshots
function build_snapshot_matrix(config::EnsembleConfig;
                                    t_spinup::Float64=900.0) :: Matrix{Float32}
    M     = config.n_members
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in config.prior_members] 
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup counts: $T_vec"
    T = T_vec[1]    # number of timesteps per member
    N = M * T       # total snapshots (M members times T timesteps)

    snaps_1   = load_member_snapshots(config.prior_members[1]; t_spinup)    # get first snapshot
    state_dim = size(snaps_1, 1)                                            # get size of first snapshot, 
    U         = Matrix{Float32}(undef, state_dim, N)                        # state_dimension rows x N columns
    U[:, 1:T] = snaps_1

    for (mi, member) in enumerate(config.prior_members[2:end])
        @info "  Loading member $(member.id) ($(mi+1)/$M)..."
        snaps = load_member_snapshots(member; t_spinup)
        j     = mi * T
        U[:, j+1:j+T] = snaps
    end
    @info "  Snapshot matrix: $(state_dim) × $N  ($(M) members × $T timesteps)"
    return U    # (state_dim, N)
end

# POD via Method of Snapshots (Sirovich 1987)
function compute_pod(X::Matrix{Float32}, grid::GridInfo;
                     L::Int=10) :: PODBasis
    state_dim, N = size(X)
    L = min(L, N)

    # Do not remove the mean for the basis computation
    @info "  Computing $(N)×$(N) Gram matrix..."
    C = Symmetric(Float64.(X)' * Float64.(X))

    # Step 3 — Eigendecompose; eigen() returns ascending order, so reverse
    @info "  Eigendecomposing $(N)×$(N) matrix..."
    decomp = eigen(C)
    λ = reverse(decomp.values)
    V = reverse(decomp.vectors, dims=2)

    # Discard numerical noise (can appear for near-degenerate snapshot sets)
    valid = λ .> 1e-10 * λ[1]
    n_valid = sum(valid)
    if n_valid < L
        @warn "Only $n_valid valid eigenvalues; reducing L from $L to $n_valid"
        L = n_valid
    end
    λ_L = λ[1:L]       # eigenvalues
    V_L = V[:, 1:L]    # eigenvector, (N, L)

    @info "  Computing $L POD modes (state_dim=$state_dim)..."
    Φ = Float32.(Float64.(X) * V_L * Diagonal(1.0 ./ sqrt.(λ_L)))
    a = Φ' * X  # project basis for coefficients (used as GMF prior) 

    
    @info "  POD complete: L=$L)%"
    return PODBasis(a, Φ, λ_L, L, N, grid)
end


# ============================================================
# Save / Load
# ============================================================

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
    basis = compute_pod(U, grid; L)    #
    return basis
end

