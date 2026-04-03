# cgf cego6160@colorado.edu 4.2.26
# functions to retrieve noisy measurements from a wind field NCfield

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

# takes a scalar measurement from the field at a position index, 
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
	m = get_scalar_measurement(Ref(U), Ref(grid), locs[1:end], var, Ref(tidx))
	return measSet(m)
end


function make_H_matrix(basis::PODBasis, mset::measSet)::Matrix{Float32}
	n = basis.grid.n_grid
	n_meas = length(mset.measurements)
	H = Matrix{Float32}(undef, 3 * n_meas, basis.L)          # preallocate, should be 3n_meas x # of POD modes 
	for (i, m) in enumerate(mset.measurements)
		ix, iy, iz = m.locidx
		flat_u = LinearIndices((basis.grid.Nx, basis.grid.Ny, basis.grid.Nz))[ix, iy, iz]
		flat_v = flat_u + n
		flat_w = flat_u + 2n
		H[3(i-1)+1, :] = basis.modes[flat_u, :]   # u row                                                                                                                                              
		H[3(i-1)+2, :] = basis.modes[flat_v, :]   # v row                                                                                                                                              
		H[3(i-1)+3, :] = basis.modes[flat_w, :]   # w row                                                                                                                                              
	end
	return H
end
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

