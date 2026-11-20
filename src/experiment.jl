# experiment.jl
# cgf cego6160@colorado.edu 11.4.26
#
# THE shared observing-system-experiment setup. Every online driver (KF sweeps, Θ-baselines,
# covariance analysis) takes its spin-up, domain margin, sensor noise, RNG seed, rank list,
# sensor geometry and truth-member lookup from here, so paired comparisons (same noise draws,
# same sensors, same scoring region) cannot drift apart between scripts. Scoring regions live
# in regions.jl.
#
# Changing anything here changes EVERY experiment: rerun the affected sweeps and regenerate
# the figures that depend on them.
include("regions.jl")   # Region, region_rows, hill_crest (includes fasteddy_io.jl)

using LinearAlgebra

# ---------------------------------------------------------------------------
# Constants shared by every driver
# ---------------------------------------------------------------------------
const T_SPINUP     = 900.0    # s; snapshots before this are spin-up and never used
const MARGIN_M     = 600f0    # m; lateral exclusion margin (400 m sponge + 200 m buffer).
                              # MUST match the value the bases were built with.
const SENSOR_VAR   = 0.25f0   # m²/s²; per-component sensor noise variance (σ = 0.5 m/s)
const NOISE_SEED   = 20260713 # seeded immediately before measurement sampling in every driver,
                              # so all experiments on one fold see identical noise draws
const RANKS        = [2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]   # truncation ranks swept
const RTOT_MIN_EIG = 0.05     # PD guard on R_sensor + R_rep (20% of the sensor floor);
                              # see check_R_pd
const LATTICE_AGL_M = 250f0   # m above local terrain; ceiling of the dense sensor lattice

"Per-sensor 3×3 measurement-noise covariance diag(σ²_u, σ²_v, σ²_w)."
sensor_mvar() = Matrix{Float32}(SENSOR_VAR * I, 3, 3)

# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------
"The member with id `id`: the config's truth member or one of its prior members."
function resolve_member(config::EnsembleConfig, id::AbstractString) :: MemberInfo
    config.truth_member.id == id && return config.truth_member
    i = findfirst(m -> m.id == id, config.prior_members)
    isnothing(i) && error("member '$id' is neither the truth member nor a prior member")
    return config.prior_members[i]
end

# ---------------------------------------------------------------------------
# Sensor geometry
# ---------------------------------------------------------------------------
# Every sensor is snapped to a grid cell and returned as that cell's [x, y, z] (m), so the
# measurement operator samples exactly one cell (no interpolation).

_cell_loc(grid::GridInfo, i, j, k) = Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]]

"""
    rake_locations(grid, domain) -> Vector{Vector{Float32}}

The 15-sensor tower rake: 5 towers at x = 3 km spread ±500 m in y about the hill crest,
each with sensors at 10, 50 and 100 m AGL (45 observation rows).
"""
function rake_locations(grid::GridInfo, domain::AnalysisDomain;
                        x_m::Float32=3000f0,
                        y_offsets_m::Vector{Float32}=Float32[-500, -250, 0, 250, 500],
                        heights_m::Vector{Float32}=Float32[10, 50, 100])
    _, crest_y = hill_crest(grid)
    locs = Vector{Vector{Float32}}()
    for dy in y_offsets_m, z in heights_m
        i, j, k = nearest_cell(grid, domain, x_m, crest_y + dy, z)   # errors if in the margin
        push!(locs, _cell_loc(grid, i, j, k))
    end
    return locs
end

"""
    dense_lattice_locations(grid, domain; stride_xy=50, stride_z=10, agl_m=LATTICE_AGL_M)

Sensor lattice over the downwind half of the analysis domain: from the crest to the interior
x edge, across the full interior y, every `stride_xy` cells in x and y and every `stride_z`-th
level (starting at level 2) up to `agl_m` above local terrain. Levels are chosen per column
because the sigma coordinate rides the terrain. At the defaults (500 m, 10 levels) this is
175 sensors / 525 observation rows.

SPACING IS NOT FREE: R_rep (reperror.jl) is indefinite, and below ~400 m spacing its negative
eigenvalues are no longer masked by the sensor floor — see check_R_pd.
"""
function dense_lattice_locations(grid::GridInfo, domain::AnalysisDomain;
                                 stride_xy::Int=50, stride_z::Int=10,
                                 agl_m::Float32=LATTICE_AGL_M)
    crest_x, _ = hill_crest(grid)
    ix_dom_lo = findfirst(i -> any(domain.mask[i, :, :]), 1:grid.Nx)
    ix_hi     = findlast( i -> any(domain.mask[i, :, :]), 1:grid.Nx)
    iy_lo     = findfirst(j -> any(domain.mask[:, j, :]), 1:grid.Ny)
    iy_hi     = findlast( j -> any(domain.mask[:, j, :]), 1:grid.Ny)
    ix_lo     = max(ix_dom_lo, findfirst(i -> grid.xPos[i, 1, 1] >= crest_x, 1:grid.Nx))
    locs = Vector{Vector{Float32}}()
    for i in ix_lo:stride_xy:ix_hi, j in iy_lo:stride_xy:iy_hi
        terr = grid.zPos[i, j, 1]
        for k in 2:stride_z:grid.Nz
            (grid.zPos[i, j, k] - terr) <= agl_m || break
            domain.mask[i, j, k] || continue
            push!(locs, _cell_loc(grid, i, j, k))
        end
    end
    isempty(locs) && error("sensor lattice is empty — check the downwind span against the domain margin")
    return locs
end

# ---------------------------------------------------------------------------
# Observation-error guard
# ---------------------------------------------------------------------------
"""
    check_R_pd(R; min_eig=RTOT_MIN_EIG, context="")

Error unless min eig(R) ≥ `min_eig`. R_rep from build_rep_covariance evaluates its kernel
parameters at each pair's midpoint, which is NOT a valid covariance for spatially varying
parameters (Paciorek & Schervish 2006): R_sensor + R_rep stays PD only while R_rep's negative
eigenvalues sit below the sensor floor. Measured min eig on the downwind lattice (job 5069):
300 m: −0.247 | 400 m: +0.038 | 500 m: +0.204 | 600 m: +0.231 | 800 m: +0.235.
It varies by fold (fold 03 at 500 m: +0.186), so the guard runs on every fold.
Making R_rep diagonal is NOT a fix — it treats correlated errors as independent.
"""
function check_R_pd(R::AbstractMatrix; min_eig::Real=RTOT_MIN_EIG, context::AbstractString="")
    e = eigvals(Symmetric(Matrix{Float64}(R)))
    λmin = minimum(e)
    @info "min eig(R) = $(round(λmin, digits=4))  cond = $(round(maximum(e)/max(λmin, 1e-12), digits=1))"
    λmin >= min_eig || error(
        "R min eigenvalue $(round(λmin, digits=4)) < $min_eig. $context The rep-error kernel is " *
        "not PSD and its negative eigenvalues are no longer masked by the sensor-noise floor; " *
        "widen the sensor spacing (500 m is the calibrated default) or fix the midpoint-" *
        "parameter kernel in src/reperror.jl.")
    return λmin
end

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
"""
    resid2_full(modes_r, C, truth_U, cols; chunk=32) -> Float64

Σ_k ‖modes_r · C[:, k] − truth_U[:, cols[k]]‖², by chunked GEMM with Float64 accumulation.
Uses the DIRECT residual: the Parseval shortcut ‖x‖² − 2αᵀa + ‖α‖² cancels ~1e9 Float32
terms to recover a ~1e5 residual and returned 0.0 (job 5053).
"""
function resid2_full(modes_r::AbstractMatrix, C::AbstractMatrix{Float32},
                     truth_U::AbstractMatrix{Float32}, cols::AbstractVector{Int}; chunk::Int=32)
    s = 0.0
    for c0 in 1:chunk:length(cols)
        cc = c0:min(c0 + chunk - 1, length(cols))
        E  = modes_r * C[:, cc] .- truth_U[:, cols[cc]]
        s += mapreduce(x -> abs2(Float64(x)), +, E)
    end
    return s
end
