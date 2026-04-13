# cgf cego6160@colorado.edu
# shared utility functions for GMF, LS, and KF estimators

# Extract flattened [u;v;w] state vector from truth field at a given time index
function get_truth_state(truth_field::NCfield, tidx::Int)::Vector{Float32}
    return Float32[vec(truth_field.u[:,:,:,tidx]);
                   vec(truth_field.v[:,:,:,tidx]);
                   vec(truth_field.w[:,:,:,tidx])]
end

# RMSE and max absolute error between reconstructed and truth state vectors
function field_errors(reconstructed::Vector{Float32}, truth::Vector{Float32})::Tuple{Float64,Float64}
    diff = reconstructed .- truth
    return sqrt(mean(diff.^2)), maximum(abs.(diff))
end

# Build observation matrix H mapping POD coefficient space to measurement space.
# H is (3*n_meas × L), where each triplet of rows corresponds to [u,v,w] at one sensor location.
function make_H_matrix(basis::PODBasis, mset::measSet)::Matrix{Float32}
    n      = basis.grid.n_grid
    n_meas = length(mset.measurements)
    H      = Matrix{Float32}(undef, 3 * n_meas, basis.L)
    for (i, m) in enumerate(mset.measurements)
        ix, iy, iz = m.locidx
        flat_u = LinearIndices((basis.grid.Nx, basis.grid.Ny, basis.grid.Nz))[ix, iy, iz]
        flat_v = flat_u + n
        flat_w = flat_u + 2n
        H[3(i-1)+1, :] = basis.modes[flat_u, :]
        H[3(i-1)+2, :] = basis.modes[flat_v, :]
        H[3(i-1)+3, :] = basis.modes[flat_w, :]
    end
    return H
end
