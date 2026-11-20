# fasteddy_io.jl
# cgf cego6160@colorado.edu 4.2.26
#
# Reading FastEddy LES output and the ensemble config, the analysis-domain mask, and the
# flat state-vector layout every other module uses:
#     x = [u[mask]; v[mask]; w[mask]]   (column-major over the masked cells, length 3·n_grid)
# plus writing FastEddy-format NetCDF for the Python slice plotters.

using NCDatasets, JSON3

# ---------------- structures --------------
# One ensemble member, read from ensemble_config.json (written by
# preprocessing/create_synthetic_ensemble_sim.py).
struct MemberInfo
    id          :: String            # "00".."11" or "truth"
    output_dir  :: String            # directory holding FE_ensemble_<id>.<timestep> files
    met_dir_deg :: Float64           # wind direction in degrees
    U_g_ms      :: Float64           # equivalent zonal wind speed
    V_g_ms      :: Float64           # equivalent meridional wind speed
    files       :: Vector{String}    # absolute paths, sorted numerically by timestep
    timesteps   :: Vector{Int}       # model-step suffix values (0, 3000, 6000, …)
    sim_times   :: Vector{Float32}   # simulation time in seconds per file
end

# The ensemble: prior members (the offline sample of the forcing distribution) plus the
# separately-run nature/truth member.
struct EnsembleConfig
    n_members     :: Int                  # number of prior members after include/exclude filtering
    prior_members :: Vector{MemberInfo}   # members that sample the forcing prior
    truth_member  :: MemberInfo           # nature run (never in any basis)
end

# The LES terrain-following grid, read from one output file. Coordinate arrays are full 3D
# because the sigma coordinate follows the terrain.
struct GridInfo
    Nx        :: Int                 # number of grid cells in x direction
    Ny        :: Int                 # number of grid cells in y direction 
    Nz        :: Int                 # number of vertical levels
    dx        :: Float32
    dy        :: Float32
    xPos      :: Array{Float32,3}    # (Nx, Ny, Nz)
    yPos      :: Array{Float32,3}
    zPos      :: Array{Float32,3}
    n_grid    :: Int                 # Nx * Ny * Nz
    state_dim :: Int                 # 3 * n_grid
end

# Unpacked (full-grid) wind field of one member over time; zeros outside the analysis domain.
struct NCfield
    t    :: Vector{Float32}      # (T,) timestamps in seconds
    grid :: GridInfo              # LES grid
    u    :: Array{Float32,4}     # (Nx, Ny, Nz, T) zonal wind, m/s
    v    :: Array{Float32,4}     # (Nx, Ny, Nz, T) meridional wind, m/s
    w    :: Array{Float32,4}     # (Nx, Ny, Nz, T) vertical wind, m/s
end

# Analysis domain mask: excludes the lateral sponge fringe + margin on all four x/y faces.
# The flat state vector [u;v;w] used by the POD/filter contains only interior (unforced) cells.
# margin_m: exclusion distance in metres from each lateral face (default 600 m = 400 m sponge + buffer).
struct AnalysisDomain
    mask      :: BitArray{3}   # (Nx, Ny, Nz) — true = interior cell included in state
    n_grid    :: Int           # count(mask)
    state_dim :: Int           # 3 * n_grid
end

function AnalysisDomain(grid::GridInfo; margin_m::Float32=600f0) :: AnalysisDomain
    cx = round(Int, margin_m / grid.dx)
    cy = round(Int, margin_m / grid.dy)
    mask = falses(grid.Nx, grid.Ny, grid.Nz)
    mask[cx+1:grid.Nx-cx, cy+1:grid.Ny-cy, :] .= true
    n = count(mask)
    return AnalysisDomain(mask, n, 3n)
end


# ---------- output functions ----------------
# Unpack a flat masked state vector [u;v;w] back into full 3D fields (zeros outside the domain).
function unpack_snapshot(x::AbstractVector{Float32}, grid::GridInfo, domain::AnalysisDomain)
    n   = domain.n_grid
    u3  = zeros(Float32, grid.Nx, grid.Ny, grid.Nz)
    v3  = zeros(Float32, grid.Nx, grid.Ny, grid.Nz)
    w3  = zeros(Float32, grid.Nx, grid.Ny, grid.Nz)
    u3[domain.mask] = x[1:n]
    v3[domain.mask] = x[n+1:2n]
    w3[domain.mask] = x[2n+1:3n]
    return u3, v3, w3
end

# Unpack a domain-masked snapshot matrix (domain.state_dim, T) into an NCfield,
# zero-filling cells outside the domain mask (e.g. the lateral sponge margin).
function unpack_to_NCfield(X::AbstractMatrix{Float32}, t::Vector{Float32},
                            grid::GridInfo, domain::AnalysisDomain) :: NCfield
    T_len = size(X, 2)
    u4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    v4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    w4 = Array{Float32,4}(undef, grid.Nx, grid.Ny, grid.Nz, T_len)
    for i in 1:T_len
        u4[:,:,:,i], v4[:,:,:,i], w4[:,:,:,i] = unpack_snapshot(view(X, :, i), grid, domain)
    end
    return NCfield(t, grid, u4, v4, w4)
end

# Flat state-vector index of grid cell (i,j,k) for velocity component comp (:u/:v/:w),
# in the SAME ordering load_snapshot uses: vcat(vec(u[mask]), vec(v[mask]), vec(w[mask])),
# column-major within each masked block. Errors if (i,j,k) lies outside the analysis domain.
function cell_to_state_index(domain::AnalysisDomain, i::Int, j::Int, k::Int, comp::Symbol) :: Int
    domain.mask[i, j, k] || error("cell ($i,$j,$k) is outside the analysis domain (sponge margin)")
    # position of (i,j,k) among masked cells, column-major (matches vec(u[mask]))
    lin = reshape(cumsum(vec(domain.mask)), size(domain.mask))
    p   = lin[i, j, k]
    off = comp === :u ? 0 : comp === :v ? domain.n_grid : 2 * domain.n_grid
    return off + p
end

# Map a physical target (x,y in metres, z in metres AGL) to the nearest interior grid cell.
# Returns (i,j,k). "AGL" is measured from the lowest cell centre zPos[i,j,1], not the terrain
# surface, so heights are offset by about half the first cell depth.
function nearest_cell(grid::GridInfo, domain::AnalysisDomain, x_m::Real, y_m::Real, z_agl_m::Real)
    i = argmin(abs.(grid.xPos[:, 1, 1] .- Float32(x_m)))
    j = argmin(abs.(grid.yPos[1, :, 1] .- Float32(y_m)))
    terrain = grid.zPos[i, j, 1]
    k = argmin(abs.(grid.zPos[i, j, :] .- (terrain + Float32(z_agl_m))))
    domain.mask[i, j, k] || error("nearest cell ($i,$j,$k) to (x=$x_m, y=$y_m, z_agl=$z_agl_m) " *
                                  "is inside the sponge margin; move the reference point inward")
    return i, j, k
end

# Write u/v/w 3D fields to a FastEddy-format NetCDF the Python slice plotters can read directly.
# Stores xPos/yPos/zPos/topoPos so plot_xy_slices.py / plot_xz_slices.py work unchanged.
# Dimension order is (xIndex,yIndex,zIndex,time): NCDatasets writes it reversed, so Python sees
# (time,zIndex,yIndex,xIndex) — exactly the layout the plotters expect.
function write_fasteddy_nc(filepath::String,
                            u3::Array{Float32,3}, v3::Array{Float32,3}, w3::Array{Float32,3},
                            grid::GridInfo; t::Float32=0f0)
    isfile(filepath) && rm(filepath)
    NCDataset(filepath, "c") do ds
        defDim(ds, "xIndex", grid.Nx); defDim(ds, "yIndex", grid.Ny)
        defDim(ds, "zIndex", grid.Nz); defDim(ds, "time", 1)
        d4 = ("xIndex", "yIndex", "zIndex", "time")
        for (name, arr) in (("u", u3), ("v", v3), ("w", w3),
                            ("xPos", grid.xPos), ("yPos", grid.yPos), ("zPos", grid.zPos))
            v = defVar(ds, name, Float32, d4)
            v[:, :, :, 1] = arr
        end
        tp = defVar(ds, "topoPos", Float32, ("xIndex", "yIndex", "time"))
        tp[:, :, 1] = grid.zPos[:, :, 1]               # surface height = terrain
        tv = defVar(ds, "time", Float32, ("time",))
        tv[1] = t
    end
    @info "Wrote FastEddy-format NetCDF: $filepath"
end


# ---------- input functions ------------------ 
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
        # derive grid spacing from the coordinate arrays (do NOT hardcode)
        dx = Nx > 1 ? xPos[2,1,1] - xPos[1,1,1] : 1f0
        dy = Ny > 1 ? yPos[1,2,1] - yPos[1,1,1] : 1f0
        GridInfo(Nx, Ny, Nz, dx, dy, xPos, yPos, zPos, Nx*Ny*Nz, 3*Nx*Ny*Nz)
    end
end
# get ensemble config from file
function load_ensemble_config(config_path::String;
                              include_ids::Union{Vector{String},Nothing}=nothing,
                              exclude_ids::Union{Vector{String},Nothing}=nothing) :: EnsembleConfig
    @info "Parsing ensemble config: $config_path"
    cfg = JSON3.read(read(config_path, String))
    entries = collect(cfg.prior_members)
    if !isnothing(include_ids)
        entries = filter(m -> String(m.id) in include_ids, entries)
    end
    if !isnothing(exclude_ids)
        entries = filter(m -> !(String(m.id) in exclude_ids), entries)
    end
    prior_members = [_make_member_info(m) for m in entries]
    truth_member  = _make_member_info(cfg.truth_member)
    return EnsembleConfig(length(prior_members), prior_members, truth_member)
end


# Load a single snapshot as a flat state vector [u; v; w] (Float32).
# If domain is provided, only interior cells are included (length = domain.state_dim).
# Flattening is Julia column-major: x varies fastest within each component block.
function load_snapshot(filepath::String,
                       domain::Union{AnalysisDomain,Nothing}=nothing) :: Vector{Float32}
    NCDataset(filepath, "r") do ds
        u = ds["u"][:, :, :, 1]    # (Nx, Ny, Nz)
        v = ds["v"][:, :, :, 1]
        w = ds["w"][:, :, :, 1]
        if isnothing(domain)
            return vcat(vec(u), vec(v), vec(w))
        else
            m = domain.mask
            return vcat(vec(u[m]), vec(v[m]), vec(w[m]))
        end
    end
end

# Load post-spinup snapshots for one member as a (state_dim, T) matrix.
# state_dim = domain.state_dim if domain provided, else grid.state_dim.
function load_member_snapshots(member::MemberInfo;
                                t_spinup::Float64=900.0,
                                domain::Union{AnalysisDomain,Nothing}=nothing) :: Matrix{Float32}
    mask  = _spinup_mask(member, t_spinup)
    files = member.files[mask]
    T     = length(files)
    T == 0 && error("No post-spinup snapshots for member $(member.id) " *
                    "(t_spinup=$(t_spinup) s, max sim_time=$(maximum(member.sim_times)) s)")
    x0        = load_snapshot(files[1], domain)
    state_dim = length(x0)
    U         = Matrix{Float32}(undef, state_dim, T)
    U[:, 1]   = x0
    for t in 2:T
        U[:, t] = load_snapshot(files[t], domain)
    end
    return U
end


# ------data loading utilities helper functions --------------------
# Returns (sorted_paths, timesteps) for all output files in output_dir.
# Matches files named FE_ensemble_<id>.<timestep>, sorted by numeric timestep.
function discover_member_files(output_dir::String, member_id::String)
    all_files = readdir(output_dir; join=false)
    pat = Regex("^FE_ensemble_$(member_id)\\.([0-9]+)\$")
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

# Simulation time (s) stored in each output file.
function _read_sim_times(files::Vector{String}) :: Vector{Float32}
    times = Vector{Float32}(undef, length(files))
    for (i, f) in enumerate(files)
        NCDataset(f, "r") do ds
            times[i] = Float32(ds["time"][1])
        end
    end
    return times
end

# Build a MemberInfo from one config entry (discovers its output files and times).
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

# Boolean mask: which files have sim_time >= t_spinup.
function _spinup_mask(member::MemberInfo, t_spinup::Float64) :: BitVector
    return member.sim_times .>= Float32(t_spinup)
end
