# measurement.jl
# cgf cego6160@colorado.edu 4.2.26
# functions to retrieve noisy measurements from a wind field NCfield

using Distributions

# Measurement structure, contains the measurement, index, and 
struct meas
	loc    :: Vector{Float32} # x,y,z position in continuous space of sensor
	locidx :: Vector{Int}     # x,y,z position in grid index space of sensor 
	val    :: Vector{Float32} # value of the measurement (with noise), as [u,v,w] (m/s)
end

# set of measurements, at one timestep
struct measSet
	measurements::Vector{meas}    # set of measurements
end

# set of measurements for every timestep 
function get_measurement_sequence(U::NCfield, grid::GridInfo, domain::AnalysisDomain, nm::Int, mvar::Matrix{Float32}, tidx::Vector{Int})::Vector{measSet}
	# samples a set of nm measurements across the U truth field
	# for every timestep in tidx [constant for phase 1, sequential for phase 2]
	return [get_measurement_set(U, grid, randLocsInBounds(grid, domain, nm), mvar, t) for t in tidx]
end


# takes a scalar measurement from the field at a continous position 
function get_scalar_measurement(U::NCfield, grid::GridInfo, loc::Vector{Float32}, var::Matrix{Float32}, tidx::Int)::meas
	# as below, 
	# Y = H ζ + ε, (Eq. 5) 
	# where ε ~ N(0, var)
	# var = diag(sigma_u, sigma_v, sigma_w)
	locidx = loc2idx(loc, grid) # get grid index around where measurement was taken
	# get wind value at nearest location to sensor
	val = Float32[U.u[locidx[1], locidx[2], locidx[3], tidx],
		U.v[locidx[1], locidx[2], locidx[3], tidx],
		U.w[locidx[1], locidx[2], locidx[3], tidx]]
	val = val + randn(Float32, 3) .* sqrt.(diag(var))     # measurement + (uncorrelated, but different) noise
	return meas(loc, locidx, val)
end

# get a set of measurements given locations and variance at time index tidx
function get_measurement_set(U::NCfield, grid::GridInfo, locs::Vector{Vector{Float32}}, var::Matrix{Float32}, tidx::Int)::measSet
	# get a set of measurements given locations and variance at time index tidx
	m = [get_scalar_measurement(U, grid, loc, var, tidx) for loc in locs]
	return measSet(m)
end

# get a uniformly random location within the analysis domain (sponge margin excluded), in continuous space
function randLocInBounds(grid::GridInfo, domain::AnalysisDomain)::Vector{Float32}
	# get a random location in bounds. Uniformly distributed across X and Y within the
	# domain's interior bounding box, randomly sampled index from Z (cuz sigma coords)
	ix_lo = findfirst(ix -> any(domain.mask[ix, :, :]), 1:grid.Nx)
	ix_hi = findlast(ix  -> any(domain.mask[ix, :, :]), 1:grid.Nx)
	iy_lo = findfirst(iy -> any(domain.mask[:, iy, :]), 1:grid.Ny)
	iy_hi = findlast(iy  -> any(domain.mask[:, iy, :]), 1:grid.Ny)
	randX = rand(Uniform(grid.xPos[ix_lo, 1, 1], grid.xPos[ix_hi, 1, 1]))
	randY = rand(Uniform(grid.yPos[1, iy_lo, 1], grid.yPos[1, iy_hi, 1]))
	randZidx = rand(4:grid.Nz)
	randZ = grid.zPos[grid.Nx÷2, grid.Ny÷2, randZidx]
	return Float32[randX, randY, randZ]
end
# get a set of uniformly distributed random locations within the analysis domain
function randLocsInBounds(grid::GridInfo, domain::AnalysisDomain, n::Int)::Vector{Vector{Float32}}
	return [randLocInBounds(grid, domain) for _ in 1:n]
end


# Generate a flight-track measurement sequence: n_vehicles vehicles wander
# through the domain as correlated random walks, producing one measurement
# per vehicle per timestep.
#   step_frac : fraction of domain extent to move per timestep 
function get_flighttrack_sequence(U::NCfield, grid::GridInfo, domain::AnalysisDomain, n_vehicles::Int,
                                  mvar::Matrix{Float32}, tidx::Vector{Int};
                                  step_frac::Float64=0.1)::Vector{measSet}
    ix_lo = findfirst(ix -> any(domain.mask[ix, :, :]), 1:grid.Nx)
    ix_hi = findlast(ix  -> any(domain.mask[ix, :, :]), 1:grid.Nx)
    iy_lo = findfirst(iy -> any(domain.mask[:, iy, :]), 1:grid.Ny)
    iy_hi = findlast(iy  -> any(domain.mask[:, iy, :]), 1:grid.Ny)
    x_lo, x_hi = grid.xPos[ix_lo, 1, 1], grid.xPos[ix_hi, 1, 1]
    y_lo, y_hi = grid.yPos[1, iy_lo, 1], grid.yPos[1, iy_hi, 1]
    step_xy = Float32(step_frac * max(x_hi - x_lo, y_hi - y_lo))

    # z extent in physical space at domain centre; step is same fraction
    z_all   = grid.zPos[grid.Nx÷2, grid.Ny÷2, :]
    z_lo    = z_all[4]          # keep vehicles above near-surface levels
    z_hi    = z_all[end]
    step_z  = Float32(step_frac * (z_hi - z_lo))

    positions = [randLocInBounds(grid, domain) for _ in 1:n_vehicles]

    seq = Vector{measSet}(undef, length(tidx))
    for (i, t) in enumerate(tidx)
        seq[i] = get_measurement_set(U, grid, positions, mvar, t)
        for v in 1:n_vehicles
            dx    = randn(Float32) * step_xy
            dy    = randn(Float32) * step_xy
            dz    = randn(Float32) * step_z
            new_x = clamp(positions[v][1] + dx, x_lo, x_hi)
            new_y = clamp(positions[v][2] + dy, y_lo, y_hi)
            new_z = clamp(positions[v][3] + dz, z_lo, z_hi)
            positions[v] = Float32[new_x, new_y, new_z]
        end
    end
    return seq
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
# measurements y = H α. Each sensor contributes 3 rows (u,v,w), taken from the Φ rows at
# that sensor's state indices [p, n+p, 2n+p]. Sensors outside the analysis domain error.
function build_measurement_operator(ms::measSet, basis::PODBasis, domain::AnalysisDomain)::Matrix{Float32}
	Φ   = basis.modes                       # (domain.state_dim, L)
	n   = domain.n_grid
	lin = _masked_linear_index(domain, basis.grid)
	nm  = length(ms.measurements)
	H   = Matrix{Float32}(undef, 3nm, basis.L)
	for (k, m) in enumerate(ms.measurements)
		p = lin[m.locidx[1], m.locidx[2], m.locidx[3]]
		p == 0 && error("sensor $k at grid index $(m.locidx) lies outside the analysis domain")
		H[3k-2, :] = Φ[p,       :]          # u
		H[3k-1, :] = Φ[n+p,     :]          # v
		H[3k,   :] = Φ[2n+p,    :]          # w
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

# Helper functions
function idx2loc(locidx::Vector{Int}, grid::GridInfo)::Vector{Float32}
	# for input indices in discretized space 
	# output that grid cell
	return Vector{Float32}([grid.xPos[locidx[1]], grid.yPos[locidx[2]], grid.zPos[locidx[3]]])
end

function loc2idx(loc::Vector{Float32}, grid::GridInfo)::Vector{Int}
	# for input location in continuous space within the domain
	# find the closest index in x,y,z grid space and return that values
	_, ix = findmin(xi -> abs(xi - loc[1]), grid.xPos[:, 1, 1])      # get x closest to loc                                                                                                                          
	_, iy = findmin(yi -> abs(yi - loc[2]), grid.yPos[1, :, 1])      # get y                                                                                                                                        
	_, iz = findmin(zi -> abs(zi - loc[3]), grid.zPos[ix, iy, :])    # get z                                                                                                                                      
	return [ix, iy, iz]
end

