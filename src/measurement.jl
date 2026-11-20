# measurement.jl
# cgf cego6160@colorado.edu 4.2.26
#
# Synthetic observations of the truth field and the linear measurement model in POD space.
# A sensor samples the [u,v,w] of the grid cell nearest its location (no interpolation) plus
# independent Gaussian noise; correspondingly H takes the same cell's rows of the modes:
#     y_k = H α_k + ε_k,   H = rows of Pr at the sensor cells,   ε_k ~ N(0, R).

# One point measurement: location, grid cell, and noisy [u,v,w].
struct meas
	loc    :: Vector{Float32} # x,y,z position in continuous space of sensor
	locidx :: Vector{Int}     # x,y,z position in grid index space of sensor 
	val    :: Vector{Float32} # value of the measurement (with noise), as [u,v,w] (m/s)
end

# All measurements taken at one timestep.
struct measSet
	measurements::Vector{meas}    # set of measurements
end

# Measurements at the same fixed `locations` (length nm) at every timestep in tidx.
function get_static_measurement_sequence(U::NCfield, grid::GridInfo, locations::Vector{Vector{Float32}}, domain::AnalysisDomain, nm::Int, mvar::Matrix{Float32}, tidx::Vector{Int})::Vector{measSet}
	length(locations) == nm || error("expected $nm static locations, got $(length(locations))")
	return [get_measurement_set(U, grid, locations, mvar, t) for t in tidx]
end


# One noisy point measurement at `loc` (nearest grid cell), y = x(cell) + ε, ε ~ N(0, diag(var)).
# Draws from the global RNG: drivers seed it (NOISE_SEED) before sampling.
function get_scalar_measurement(U::NCfield, grid::GridInfo, loc::Vector{Float32}, var::Matrix{Float32}, tidx::Int)::meas
	locidx = loc2idx(loc, grid)
	val = Float32[U.u[locidx[1], locidx[2], locidx[3], tidx],
		U.v[locidx[1], locidx[2], locidx[3], tidx],
		U.w[locidx[1], locidx[2], locidx[3], tidx]]
	val = val + randn(Float32, 3) .* sqrt.(diag(var))     # independent noise per component
	return meas(loc, locidx, val)
end

# Measurements at `locs` at time index tidx.
function get_measurement_set(U::NCfield, grid::GridInfo, locs::Vector{Vector{Float32}}, var::Matrix{Float32}, tidx::Int)::measSet
	m = [get_scalar_measurement(U, grid, loc, var, tidx) for loc in locs]
	return measSet(m)
end

# Transiting observer: ONE point sensor at fixed x and fixed height AGL, translating up
# and down along y as a triangle wave between y_lo and y_hi at constant ground speed.
# Returns one measSet (a single measurement) per entry of tidx, so each filter update sees
# 3 rows. The sensor snaps to the nearest grid cell at each step (cell size 10 m vs ~90 m
# of travel per 6 s step, so the snap is a sub-step detail).
#
# Height is held at z_agl_m ABOVE GROUND, re-resolved per position, because the sigma
# coordinate rides the terrain — a fixed absolute z would change its height above ground
# as it crosses the hill's flank.
#
# `times` must be the simulation times (s) of the FULL post-spinup record, i.e. the same
# vector tidx indexes into; the phase is measured from times[tidx[1]].
#
# NOTE on representativeness: with one sensor per update, build_rep_covariance collapses to
# diag(σ²_u,σ²_v,σ²_w) at the sensor cell and the L∥/L⊥/θ kernel is never read (see
# build_rep_covariance in reperror.jl). This sidesteps the non-PSD kernel entirely, but it
# does NOT make the rep error white — successive samples along the track remain correlated
# in space and time, and the filter treats them as independent. Motion decorrelates faster
# than a static sensor at the same cadence, but the over-counting bias is real: expect NIS
# below 1 at short cadence.
function get_transect_sequence(U::NCfield, grid::GridInfo, domain::AnalysisDomain,
                               mvar::Matrix{Float32}, tidx::Vector{Int}, times::Vector{Float32};
                               x_m::Float32, z_agl_m::Float32,
                               y_lo::Float32, y_hi::Float32,
                               speed_ms::Float32) :: Tuple{Vector{measSet},Vector{Vector{Float32}}}
    y_hi > y_lo || error("transect needs y_hi > y_lo, got y_lo=$y_lo y_hi=$y_hi")
    span = y_hi - y_lo
    t0   = times[tidx[1]]
    seq  = Vector{measSet}(undef, length(tidx))
    locs = Vector{Vector{Float32}}(undef, length(tidx))
    for (m, t) in enumerate(tidx)
        # triangle wave: travel s along a path that folds back at each end
        s = speed_ms * (times[t] - t0)
        u = mod(s, 2span)
        y = u <= span ? y_lo + u : y_hi - (u - span)
        i, j, k = nearest_cell(grid, domain, x_m, y, z_agl_m)
        loc = Float32[grid.xPos[i,j,k], grid.yPos[i,j,k], grid.zPos[i,j,k]]
        locs[m] = loc
        seq[m]  = get_measurement_set(U, grid, [loc], mvar, t)
    end
    return seq, locs
end

# ---------- KF measurement model in POD space ----------
# State vector packing (see fasteddy_io.jl unpack_snapshot): x = [u;v;w], each block of
# length n = domain.n_grid, ordered by column-major linear index over the masked cells.
# This builds the integer offset p of grid cell (ix,iy,iz) within a masked component block.
function _masked_linear_index(domain::AnalysisDomain, grid::GridInfo)::Array{Int,3}
	lin = zeros(Int, grid.Nx, grid.Ny, grid.Nz)
	lin[domain.mask] = 1:domain.n_grid
	return lin
end

# Build the measurement operator H (3*nm × L) mapping POD coefficients α to predicted
# measurements y = H α. Each sensor contributes 3 rows (u,v,w), taken from the Pr (mode) rows at
# that sensor's state indices [p, n+p, 2n+p]. Sensors outside the analysis domain error.
function build_measurement_operator(ms::measSet, basis::PODBasis, domain::AnalysisDomain)::Matrix{Float32}
	Pr   = basis.modes                       # (domain.state_dim, L)
	n   = domain.n_grid
	lin = _masked_linear_index(domain, basis.grid)
	nm  = length(ms.measurements)
	H   = Matrix{Float32}(undef, 3nm, basis.L)
	for (k, m) in enumerate(ms.measurements)
		p = lin[m.locidx[1], m.locidx[2], m.locidx[3]]
		p == 0 && error("sensor $k at grid index $(m.locidx) lies outside the analysis domain")
		H[3k-2, :] = Pr[p,       :]          # u
		H[3k-1, :] = Pr[n+p,     :]          # v
		H[3k,   :] = Pr[2n+p,    :]          # w
	end
	return H
end

# Stack a measSet's noisy values into a single observation vector y (3*nm,), [u;v;w] per sensor.
function stack_measurements(ms::measSet)::Vector{Float32}
	nm = length(ms.measurements)
	y  = Vector{Float32}(undef, 3nm)
	for (k, m) in enumerate(ms.measurements)
		y[3k-2:3k] = m.val
	end
	return y
end

# Block-diagonal measurement-noise covariance R (3*nm × 3*nm) from the per-sensor 3×3 mvar.
function build_measurement_covariance(nm::Int, mvar::Matrix{Float32})::Matrix{Float32}
	R = zeros(Float32, 3nm, 3nm)
	for k in 1:nm
		R[3k-2:3k, 3k-2:3k] = mvar
	end
	return R
end


# Nearest grid cell (i,j,k) to a physical location [x, y, z] (z absolute, matched within
# that column because the sigma coordinate follows the terrain).
function loc2idx(loc::Vector{Float32}, grid::GridInfo)::Vector{Int}
	_, ix = findmin(xi -> abs(xi - loc[1]), grid.xPos[:, 1, 1])
	_, iy = findmin(yi -> abs(yi - loc[2]), grid.yPos[1, :, 1])
	_, iz = findmin(zi -> abs(zi - loc[3]), grid.zPos[ix, iy, :])
	return [ix, iy, iz]
end

