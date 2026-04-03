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
	val = val + randn(Float32, 3) .* sqrt.(diag(var));     # measurement + (uncorrelated, but different) noise
	return meas(loc, locidx, val)
end

function get_measurement_set(U::NCfield, grid::GridInfo, locs::Vector{Vector{Float32}}, var::Matrix{Float32}, tidx::Int)::measSet
	# get a set of measurements given locations and variance at time index tidx
	m = [get_scalar_measurement(U, grid, loc, var, tidx) for loc in locs]
	return measSet(m)
end


# get a uniformly random location in the grid, in continuous space
function randLocInBounds(grid::GridInfo)::Vector{Float32}
	# get a random location in bounds. Uniformly distributed across X and Y, randomly sampled index from Z (cuz sigma coords)
	Xextent = grid.Nx*grid.dx
	Yextent = grid.Ny*grid.dy
	randX = rand(Uniform(0, Xextent))
	randY = rand(Uniform(0, Yextent))
	randZidx = rand(4:grid.Nz)
	randZ = grid.zPos[randZidx]
	return Float32[randX, randY, randZ]
end
# get a set of uniformly distributed random locations in the grid 
function randLocsInBounds(grid::GridInfo, n::Int)::Vector{Vector{Float32}}
	return [randLocInBounds(grid) for _ in 1:n]
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

