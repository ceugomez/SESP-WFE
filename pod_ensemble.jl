# pod_ensemble.jl
#
# POD reduced-order basis for the FastEddy wind-direction ensemble.
# Implements the method of snapshots (Sirovich 1987) as described in the
# GMF project proposal (Eq. 1–4).
#
# State vector: [u_flat; v_flat; w_flat] in Julia column-major order (x varies fastest).
# Grid: Nx=Ny=242, Nz=82 → state_dim = 3 × 242 × 242 × 82 = 14,406,744.
# Ensemble: 10 prior members (member_00–09) + 1 held-out truth member (member_truth).
# Truth is projected onto the basis but NOT used in basis construction.
#
# Usage:
#   basis = build_pod_basis(config_path; mode=:timemean, L=10)
#   Z_prior, Z_truth = initialize_gmf_prior(config_path, "pod_basis.jld2")

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
    prior_members :: Vector{MemberInfo}   # 10 prior members, NOT including truth
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
    modes         :: Matrix{Float32}   # (state_dim, L) — orthonormal POD modes ψ_k
    mean_field    :: Vector{Float32}   # (state_dim,)   — ensemble/temporal mean
    eigenvalues   :: Vector{Float64}   # (L,) descending
    energy_frac   :: Vector{Float64}   # (L,) cumulative energy fraction
    L             :: Int
    snapshot_mode :: Symbol            # :timemean or :all
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
# Every downstream function depends on this ordering being consistent.
function load_snapshot(filepath::String) :: Vector{Float32}
    NCDataset(filepath, "r") do ds
        u = ds["u"][:, :, :, 1]    # (Nx, Ny, Nz)
        v = ds["v"][:, :, :, 1]
        w = ds["w"][:, :, :, 1]
        vcat(vec(u), vec(v), vec(w))
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
    X         = Matrix{Float32}(undef, state_dim, T)
    X[:, 1]   = x0
    for t in 2:T
        X[:, t] = load_snapshot(files[t])
    end
    return X
end

# ============================================================
# Snapshot Matrix Construction
# ============================================================

# Level 2 (all): all post-spinup snapshots from all prior members → (state_dim, M*T).
# With default t_spinup=900 s: T≈53 per member, total N≈530.
function build_snapshot_matrix_all(config::EnsembleConfig;
                                    t_spinup::Float64=900.0) :: Matrix{Float32}
    M     = config.n_members
    T_vec = [sum(_spinup_mask(m, t_spinup)) for m in config.prior_members]
    @assert all(==(T_vec[1]), T_vec) "Members have unequal post-spinup counts: $T_vec"
    T = T_vec[1]
    N = M * T

    snaps_1   = load_member_snapshots(config.prior_members[1]; t_spinup)
    state_dim = size(snaps_1, 1)
    X         = Matrix{Float32}(undef, state_dim, N)
    X[:, 1:T] = snaps_1

    for (mi, member) in enumerate(config.prior_members[2:end])
        @info "  Loading member $(member.id) ($(mi+1)/$M)..."
        snaps = load_member_snapshots(member; t_spinup)
        j     = mi * T
        X[:, j+1:j+T] = snaps
    end
    @info "  Snapshot matrix: $(state_dim) × $N  ($(M) members × $T timesteps)"
    return X    # (state_dim, N)
end

# ============================================================
# POD via Method of Snapshots (Sirovich 1987)
# ============================================================
# Algorithm:
#   1. μ = mean(X);  X̃ = X - μ   [in-place]
#   2. C = Float64(X̃)' Float64(X̃)   [N×N Gram matrix; Float64 for inner-product stability]
#   3. eigen(C) → λ (descending), V
#   4. Φ = X̃ V_L Diag(1/√λ_L)      [normalised modes]
function compute_pod(X::Matrix{Float32}, grid::GridInfo, snapshot_mode::Symbol;
                     L::Int=10) :: PODBasis
    state_dim, N = size(X)
    L = min(L, N)

    # Do not remove the mean for the basis computation
    # # Step 1 — Centre in-place (avoids allocating a copy of a potentially large matrix)
    # μ = vec(mean(X, dims=2))    # (state_dim,)
    # X .-= μ                     # X is now X̃

    # Step 2 — Gram matrix (upcast to Float64: w fluctuations ~0.1 m/s vs mean ~10 m/s,
    # Float32 inner products accumulate noticeable cancellation error)
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
    λ_L = λ[1:L]
    V_L = V[:, 1:L]    # (N, L)

    # Step 4 — POD modes: Φ = X̃ V_L Diag(1/√λ_L); columns are unit-norm (‖Φ_k‖ = 1)
    @info "  Computing $L POD modes (state_dim=$state_dim)..."
    Φ = Float32.(Float64.(X) * V_L * Diagonal(1.0 ./ sqrt.(λ_L)))

    # Step 5 — Cumulative energy fractions
    energy_frac = cumsum(λ_L) ./ sum(λ[valid])

    @info "  POD complete: L=$L, energy captured = $(round(energy_frac[end]*100, digits=1))%"
    return PODBasis(Φ, μ, λ_L, energy_frac, L, snapshot_mode, N, grid)
end


# ============================================================
# Save / Load
# ============================================================

function save_pod_basis(basis::PODBasis, filepath::String)
    jldsave(filepath;
        modes         = basis.modes,
        mean_field    = basis.mean_field,
        eigenvalues   = basis.eigenvalues,
        energy_frac   = basis.energy_frac,
        L             = basis.L,
        snapshot_mode = String(basis.snapshot_mode),
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
    @info "Saved POD basis to $filepath  (L=$(basis.L), mode=$(basis.snapshot_mode), modes ≈ $(round(sz_gb, digits=2)) GB)"
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
        d["modes"],
        d["mean_field"],
        d["eigenvalues"],
        d["energy_frac"],
        d["L"],
        Symbol(d["snapshot_mode"]),
        d["n_snapshots"],
        grid,
    )
end

# ============================================================
# Top-Level Entry Points
# ============================================================

# Build POD basis from ensemble and save to disk.
#   mode     — :timemean (N=10, one time-mean snapshot per member) or
#               :all      (N≈530, all post-spinup snapshots from all members)
#   L        — number of POD modes to retain
#   t_spinup — exclude snapshots with sim_time < t_spinup seconds (default 900 s = 15 min)
function build_pod_basis(config_path::String;
                          mode::Symbol=:timemean,
                          L::Int=10,
                          t_spinup::Float64=900.0,
                          output_path::String="pod_basis.jld2") :: PODBasis

    config = load_ensemble_config(config_path)

    @info "Loading grid from first member file..."
    grid = load_grid(config.prior_members[1].files[1])
    @info "  Grid: $(grid.Nx)×$(grid.Ny)×$(grid.Nz), state_dim=$(grid.state_dim)"

    @info "Building snapshot matrix (mode=$mode, t_spinup=$(t_spinup) s)..."
    X = if mode == :timemean
        build_snapshot_matrix_timemean(config; t_spinup)
    elseif mode == :all
        build_snapshot_matrix_all(config; t_spinup)
    else
        error("Unknown mode: $mode. Use :timemean or :all")
    end
    @info "  Matrix size: $(size(X, 1)) × $(size(X, 2))  ($(round(sizeof(X)/1e9, digits=1)) GB)"

    @info "Computing POD (L=$L)..."
    basis = compute_pod(X, grid, mode; L)    # X is centred in-place inside compute_pod

    save_pod_basis(basis, output_path)
    return basis
end



# Fraction of variance not explained by the L-mode reconstruction (lower = better).
function reconstruction_rmse(x::Vector{Float32}, basis::PODBasis) :: Float64
    zeta = project_snapshot(x, basis)
    x̂    = reconstruct_snapshot(zeta, basis)
    return Float64(norm(x̂ - x)) / Float64(norm(x .- basis.mean_field))
end

# ============================================================
# NetCDF Export for Visualisation
# ============================================================


# ref_file: any FastEddy member output file — used to read topoPos (terrain
# elevation), which is not stored in the POD basis.
function export_mode_to_netcdf(basis::PODBasis, k::Int, outdir::String,
                                ref_file::String;
                                prefix::String="FE_Pod_mode")
    @assert 1 <= k <= basis.L "Mode index k=$k out of range 1..$(basis.L)"

    g  = basis.grid
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

    # Unpack the flat mode vector back into u, v, w spatial fields
    phi   = basis.modes[:, k]                              # (state_dim,)
    u_mode = reshape(phi[1:g.n_grid],                g.Nx, g.Ny, g.Nz)
    v_mode = reshape(phi[g.n_grid+1:2*g.n_grid],     g.Nx, g.Ny, g.Nz)
    w_mode = reshape(phi[2*g.n_grid+1:3*g.n_grid],   g.Nx, g.Ny, g.Nz)

    # Read terrain elevation from a reference member file
    topoPos = NCDataset(ref_file, "r") do ds
        Float32.(ds["topoPos"][:, :, 1])    # (Nx, Ny) in Julia / (Ny, Nx) in netCDF
    end

    outpath = joinpath(outdir, "$(prefix).$(k)")

    NCDataset(outpath, "c") do ds
        # Dimensions — names in Julia (column-major) order; NCDatasets reverses them
        # when writing to the file, so Python reads u(time, zIndex, yIndex, xIndex).
        defDim(ds, "xIndex", Nx)
        defDim(ds, "yIndex", Ny)
        defDim(ds, "zIndex", Nz)
        defDim(ds, "time",   1)

        # time — store mode index as surrogate; animate_output uses it for the title
        v = defVar(ds, "time", Float32, ("time",))
        v.attrib["units"] = "1"
        ds["time"][1] = Float32(k)

        # 3D fields — Julia dim order (xIndex, yIndex, zIndex, time)
        # → netCDF file stores (time, zIndex, yIndex, xIndex)
        # → Python ds["u"][0] returns (Nz, Ny, Nx) as animate_output expects
        for (name, arr) in (("u", u_mode), ("v", v_mode), ("w", w_mode))
            v = defVar(ds, name, Float32, ("xIndex", "yIndex", "zIndex", "time"))
            v.attrib["units"]     = "m s-1"
            v.attrib["long_name"] = "POD mode $k — $name component"
            ds[name][:, :, :, 1] = arr    # arr is (Nx, Ny, Nz)
        end

        # Grid coordinates — same Julia/netCDF layout as above
        for (name, arr) in (("xPos", g.xPos), ("yPos", g.yPos), ("zPos", g.zPos))
            v = defVar(ds, name, Float32, ("xIndex", "yIndex", "zIndex", "time"))
            v.attrib["units"] = "m"
            ds[name][:, :, :, 1] = arr
        end

        # Terrain elevation — Julia (xIndex, yIndex, time) → file (time, yIndex, xIndex)
        v = defVar(ds, "topoPos", Float32, ("xIndex", "yIndex", "time"))
        v.attrib["units"]     = "m"
        v.attrib["long_name"] = "Terrain elevation"
        ds["topoPos"][:, :, 1] = topoPos    # topoPos is (Nx, Ny)
    end

    frac = basis.energy_frac[k]
    incr = k == 1 ? frac : frac - basis.energy_frac[k-1]
    @info "Exported mode $k to $outpath  " *
          "(incremental energy $(round(incr*100, digits=1))%, " *
          "cumulative $(round(frac*100, digits=1))%)"
    return outpath
end

# Export all L modes at once
function export_all_modes(basis::PODBasis, outdir::String, ref_file::String;
                           prefix::String="FE_Pod_mode")
    mkpath(outdir)
    paths = String[]
    for k in 1:basis.L
        push!(paths, export_mode_to_netcdf(basis, k, outdir, ref_file; prefix))
    end
    @info "Exported $(basis.L) modes to $outdir"
    return paths
end
