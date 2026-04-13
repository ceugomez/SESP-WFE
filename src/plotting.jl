# cgf cego6160@colorado.edu 
# 4.1.26
# Plotting utilities for use on POD modal decomposition and GM 
using Distributions, Plots, StatsPlots

#TBD TBD TBD TBD
# should really have: 1 to plot gaussian mixture, 1 to plot x/z and x/y mode shape 

# 3-d plot of gaussian mixture posterior, with x as coefficient value, y aS 


# ascii_slice display wind field 
function ascii_slice(field::NCfield, t_idx::Int; plane::Symbol=:xy, fixed_idx::Int=1,
                       width::Int=60, height::Int=20)                                  
    u = field.u[:,:,:,t_idx]                                                                                                              
    v = field.v[:,:,:,t_idx]                                                                                                              
    w = field.w[:,:,:,t_idx]                                                                                                              
    spd = sqrt.(u.^2 .+ v.^2 .+ w.^2)   # (Nx, Ny, Nz)                                                                                    
                                                                                                                                            
    # extract 2D slice                                                                                                                    
      if plane == :xy                                                                                                                       
          s = spd[:, :, fixed_idx]          # (Nx, Ny)                                                                                      
          xlabel, ylabel = "x", "y"                                                                                                         
      elseif plane == :xz                                                                                                                   
          s = spd[:, fixed_idx, :]          # (Nx, Nz)                                                                                      
          xlabel, ylabel = "x", "z"                                                                                                         
      else                                                                                                                                  
          error("plane must be :xy or :xz")                                                                                                 
      end                                                                                                                                   
                                                                                                                                            
      # downsample to terminal size                                                                                                         
      nx, ny = size(s)                                                                                                                      
      xi = round.(Int, LinRange(1, nx, width))                                                                                              
      yi = round.(Int, LinRange(1, ny, height))                                                                                             
      s2 = s[xi, yi]                                                                                                                        
                                                                                                                                            
      lo, hi = minimum(s2), maximum(s2)                                                                                                     
      chars = " .:-=+*#%@"                                                                                                                  
      nc = length(chars)                                                                                                                    
                                                                                                                                            
      println("t=$(field.t[t_idx])s  |  $(plane) slice @ $(fixed_idx)  |  speed $(round(lo,digits=1))–$(round(hi,digits=1)) m/s")           
      for j in height:-1:1                                                                                                                  
          row = join(chars[clamp(round(Int, (s2[i,j]-lo)/(hi-lo+eps()) * (nc-1)) + 1, 1, nc)]                                               
                     for i in 1:width)                                                                                                      
          println(row)                                                                                                                      
      end                                                                                                                                   
      println(repeat("─", width))                                                                                                           
      println("$(xlabel)→   ($(xlabel)=$(nx)pts, $(ylabel)=$(ny)pts downsampled to $(width)×$(height))")                                    
end                                                                                                                                                     
  
function plot_gaussian_mixture(g::GMFilter, coeff_idx::Int=1; truth_val::Union{Float32,Nothing}=nothing)
    components = [Normal(g.means[k, coeff_idx], sqrt(g.vars[k, coeff_idx])) for k in 1:g.K]
    mixture = MixtureModel(components, g.weights)
    plot(mixture,
         components=false,
         label="Total Mixture Density",
         linewidth=4,
         color=:black,
         title="GMM: Coeff $(coeff_idx)",
         xlabel="Coefficient value",
         ylabel="Density",
         legend=:topright)
    plot!(mixture,
          components=true,
          label="",
          linestyle=:dash,
          alpha=0.5,
          linewidth=1.5)
    if !isnothing(truth_val)
        vline!([truth_val], color=:red, linewidth=2, linestyle=:dash, label="Truth")
    end
    savefig("/home/cego6160/workspace/prediction/output/combined_mixture_coeff$(coeff_idx).png")
end

function plot_gaussian_mixture_ridgeline(gvec::Vector{GMFilter}, coeff_idx::Int, truth_coeffs::Vector{Float32};
                                         scale=2.5, downsample=2)
    truth_val = truth_coeffs[coeff_idx]
    all_means = [g.means[k, coeff_idx] for g in gvec for k in 1:g.K]
    x_min, x_max = quantile(all_means, [0.005, 0.995])
    # ensure truth value is always visible in the x range
    x_min = min(x_min, truth_val)
    x_max = max(x_max, truth_val)
    buffer = (x_max - x_min) * 0.1
    x_vals = range(x_min - buffer, x_max + buffer, length=400)

    p = plot(legend=:topright,
             title="Ridgeline Evolution: Coeff $(coeff_idx)",
             xlabel="Coefficient Value",
             ylabel="Time (Filter Step)",
             grid=false,
             yticks=0:10:length(gvec))
    for t in 1:downsample:length(gvec)
        g = gvec[t]
        components = [Normal(g.means[k, coeff_idx], sqrt(g.vars[k, coeff_idx])) for k in 1:g.K]
        mixture = MixtureModel(components, g.weights)

        # calc & normalize pdf
        dists = pdf.(Ref(mixture), x_vals)
        if maximum(dists) > 0
            dists = (dists ./ maximum(dists)) .* scale
        end

        plot!(p, x_vals, t .+ dists,
              fillrange = t,
              fillalpha = 0.5,
              color = :viridis,
              linecolor = :white,
              linewidth = 0.5,
              label = "")
    end

    # vertical line at true projected coefficient value
    vline!(p, [truth_val],
           color = :red,
           linewidth = 2,
           linestyle = :dash,
           label = "Truth")

    display(p)
    savefig("/home/cego6160/workspace/prediction/output/ridgeline_coeff$(coeff_idx).png")
    return p
end

function plot_field_reconstruction(gm_final::GMFilter, basis::PODBasis, truth_field::NCfield, truth_tidx::Int;
                                   y_idx::Int=-1)
    grid = basis.grid
    iy = y_idx > 0 ? y_idx : grid.Ny ÷ 2   # default: mid-y xz slice

    # posterior mean coefficients → reconstructed state vector → u field
    post_mean     = vec(gm_final.weights' * gm_final.means)             # (L,)
    recon_state   = basis.modes * post_mean                             # (state_dim,)
    n             = grid.n_grid
    recon_u       = reshape(recon_state[1:n],       grid.Nx, grid.Ny, grid.Nz)
    recon_v       = reshape(recon_state[n+1:2n],    grid.Nx, grid.Ny, grid.Nz)
    recon_w       = reshape(recon_state[2n+1:3n],   grid.Nx, grid.Ny, grid.Nz)

    truth_u = truth_field.u[:, :, :, truth_tidx]
    truth_v = truth_field.v[:, :, :, truth_tidx]
    truth_w = truth_field.w[:, :, :, truth_tidx]

    # xz slices at mid-y
    truth_spd  = sqrt.(truth_u[:, iy, :].^2 .+ truth_v[:, iy, :].^2)
    recon_spd  = sqrt.(recon_u[:, iy, :].^2 .+ recon_v[:, iy, :].^2)
    error_u    = recon_u[:, iy, :] .- truth_u[:, iy, :]    # signed error in u

    # between-component uncertainty: weighted std dev of reconstructed u across components
    comp_u = zeros(Float32, grid.Nx, grid.Nz, gm_final.K)
    for k in 1:gm_final.K
        state_k = basis.modes * gm_final.means[k, :]
        comp_u[:, :, k] = reshape(state_k[1:n], grid.Nx, grid.Ny, grid.Nz)[:, iy, :]
    end
    mean_u_field = sum(gm_final.weights[k] .* comp_u[:, :, k] for k in 1:gm_final.K)
    uncert_u     = sqrt.(sum(gm_final.weights[k] .* (comp_u[:, :, k] .- mean_u_field).^2
                             for k in 1:gm_final.K))

    # x and z axes for plotting (km)
    x_km = grid.xPos[:, iy, 1] ./ 1000
    z_km = grid.zPos[grid.Nx÷2, iy, :] ./ 1000

    spd_lim  = maximum(truth_spd)
    err_lim  = maximum(abs.(error_u))
    unc_lim  = maximum(uncert_u)

    p1 = heatmap(x_km, z_km, truth_spd',  clims=(0, spd_lim), color=:viridis,
                 title="Truth wind speed", xlabel="x (km)", ylabel="z (km)")
    p2 = heatmap(x_km, z_km, recon_spd',  clims=(0, spd_lim), color=:viridis,
                 title="Posterior mean wind speed", xlabel="x (km)", ylabel="z (km)")
    p3 = heatmap(x_km, z_km, error_u',    clims=(-err_lim, err_lim), color=:RdBu,
                 title="Error in u  (recon − truth)", xlabel="x (km)", ylabel="z (km)")
    p4 = heatmap(x_km, z_km, uncert_u',   clims=(0, unc_lim), color=:plasma,
                 title="Uncertainty in u  (between-component σ)", xlabel="x (km)", ylabel="z (km)")

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800), dpi=150)
    savefig(p, "/home/cego6160/workspace/prediction/output/field_reconstruction.png")
    @info "Saved field_reconstruction.png"
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Mode shape visualisation
# Plots U, V, W for a single POD mode in both xz (mid-y) and xy (near-surface)
# cross-sections — two rows, three columns.
# ─────────────────────────────────────────────────────────────────────────────
function plot_mode_shapes(basis::PODBasis, mode_idx::Int; y_idx::Int=-1, z_idx::Int=-1,
                          outdir::String="/home/cego6160/workspace/prediction/output")
    grid = basis.grid
    iy   = y_idx > 0 ? y_idx : grid.Ny ÷ 2
    iz   = z_idx > 0 ? z_idx : max(1, grid.Nz ÷ 8)   # near-surface default

    n   = grid.n_grid
    ψ   = basis.modes[:, mode_idx]
    vol = reshape(ψ[1:n],     grid.Nx, grid.Ny, grid.Nz)
    u_xz = vol[:, iy, :]          # (Nx, Nz) — xz at mid-y
    u_xy = vol[:, :, iz]          # (Nx, Ny) — xy at near-surface z
    vol_v = reshape(ψ[n+1:2n],   grid.Nx, grid.Ny, grid.Nz)
    v_xz = vol_v[:, iy, :]
    v_xy = vol_v[:, :, iz]
    vol_w = reshape(ψ[2n+1:3n],  grid.Nx, grid.Ny, grid.Nz)
    w_xz = vol_w[:, iy, :]
    w_xy = vol_w[:, :, iz]

    x_km = grid.xPos[:, iy, 1] ./ 1000
    y_km = grid.yPos[1, :, iz]  ./ 1000
    z_km = grid.zPos[grid.Nx÷2, iy, :] ./ 1000

    energy_frac = round(100 * basis.eigenvalues[mode_idx] / sum(basis.eigenvalues), digits=1)

    function clim(a) max(maximum(abs.(a)), eps(Float32)) end

    # row 1: xz slices
    p1 = heatmap(x_km, z_km, u_xz', color=:RdBu, clims=(-clim(u_xz), clim(u_xz)),
                 title="u  (xz)", xlabel="x (km)", ylabel="z (km)")
    p2 = heatmap(x_km, z_km, v_xz', color=:RdBu, clims=(-clim(v_xz), clim(v_xz)),
                 title="v  (xz)", xlabel="x (km)", ylabel="z (km)")
    p3 = heatmap(x_km, z_km, w_xz', color=:RdBu, clims=(-clim(w_xz), clim(w_xz)),
                 title="w  (xz)", xlabel="x (km)", ylabel="z (km)")
    # row 2: xy slices
    p4 = heatmap(x_km, y_km, u_xy', color=:RdBu, clims=(-clim(u_xy), clim(u_xy)),
                 title="u  (xy)", xlabel="x (km)", ylabel="y (km)")
    p5 = heatmap(x_km, y_km, v_xy', color=:RdBu, clims=(-clim(v_xy), clim(v_xy)),
                 title="v  (xy)", xlabel="x (km)", ylabel="y (km)")
    p6 = heatmap(x_km, y_km, w_xy', color=:RdBu, clims=(-clim(w_xy), clim(w_xy)),
                 title="w  (xy)", xlabel="x (km)", ylabel="y (km)")

    z_km_val = round(grid.zPos[grid.Nx÷2, iy, iz] / 1000, digits=2)
    p = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1600, 800), dpi=150,
             plot_title="POD mode $(mode_idx)  ($(energy_frac)% energy)  |  top row: xz at mid-y  |  bottom row: xy at z≈$(z_km_val) km")
    fpath = joinpath(outdir, "mode_shape_$(lpad(mode_idx, 3, '0')).png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Export filter history for Python animations.
# Saves per-iteration posterior xz slices (mid-y), truth slices, RMSE, and
# total variance to a JLD2 file.
#
# truth_tidxs: Vector{Int} of length n_iter — which truth timestep to compare
#              against at each iteration. For phase 1 pass fill(25, max_iter);
#              for phase 2 pass the advancing index vector.
# ─────────────────────────────────────────────────────────────────────────────
# Export filter history for Python animations.
# Saves per-iteration posterior slices for xz (mid-y), xy (near-surface z), and yz (mid-x),
# plus RMSE time series for GMF, KF, and LS.
#
# truth_tidxs  : Vector{Int} of length n_iter — truth snapshot index per iteration.
# kf_history   : output of kf(), Vector of (μ, σ²) tuples, length n_iter+1.
# rmse_ls      : Vector{Float64} of length n_iter from compare_estimators.
function save_filter_history(filter_history::Vector{GMFilter},
                             kf_history::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                             rmse_ls::Vector{Float64},
                             basis::PODBasis,
                             truth_field::NCfield,
                             truth_tidxs::Vector{Int};
                             z_idx::Int=-1,
                             sensor_locs::Union{Vector{measSet},Nothing}=nothing,
                             outpath::String="/home/cego6160/workspace/prediction/output/filter_history.jld2")
    n_iter = length(filter_history) - 1   # filter_history[1] is prior
    grid   = basis.grid
    n      = grid.n_grid
    iy     = grid.Ny ÷ 2                            # mid-y for xz slice
    ix     = grid.Nx ÷ 2                            # mid-x for yz slice
    iz     = z_idx > 0 ? z_idx : max(1, grid.Nz ÷ 8)  # near-surface for xy slice

    # (spatial_dim1, spatial_dim2, 3_components, n_iter) — component order: 1=u, 2=v, 3=w
    recon_xz = Array{Float32,4}(undef, grid.Nx, grid.Nz, 3, n_iter)
    truth_xz = Array{Float32,4}(undef, grid.Nx, grid.Nz, 3, n_iter)
    recon_xy = Array{Float32,4}(undef, grid.Nx, grid.Ny, 3, n_iter)
    truth_xy = Array{Float32,4}(undef, grid.Nx, grid.Ny, 3, n_iter)
    recon_yz = Array{Float32,4}(undef, grid.Ny, grid.Nz, 3, n_iter)
    truth_yz = Array{Float32,4}(undef, grid.Ny, grid.Nz, 3, n_iter)

    rmse_gmf_vec = Vector{Float32}(undef, n_iter)
    rmse_kf_vec  = Vector{Float32}(undef, n_iter)
    totvar_vec   = Vector{Float32}(undef, n_iter)

    for i in 1:n_iter
        gm        = filter_history[i+1]
        post_mean = vec(gm.weights' * gm.means)
        state_gmf = basis.modes * post_mean

        u_r = reshape(state_gmf[1:n],     grid.Nx, grid.Ny, grid.Nz)
        v_r = reshape(state_gmf[n+1:2n],  grid.Nx, grid.Ny, grid.Nz)
        w_r = reshape(state_gmf[2n+1:3n], grid.Nx, grid.Ny, grid.Nz)

        recon_xz[:, :, 1, i] = u_r[:, iy, :]
        recon_xz[:, :, 2, i] = v_r[:, iy, :]
        recon_xz[:, :, 3, i] = w_r[:, iy, :]
        recon_xy[:, :, 1, i] = u_r[:, :, iz]
        recon_xy[:, :, 2, i] = v_r[:, :, iz]
        recon_xy[:, :, 3, i] = w_r[:, :, iz]
        recon_yz[:, :, 1, i] = u_r[ix, :, :]
        recon_yz[:, :, 2, i] = v_r[ix, :, :]
        recon_yz[:, :, 3, i] = w_r[ix, :, :]

        tidx = truth_tidxs[i]
        truth_xz[:, :, 1, i] = truth_field.u[:, iy, :, tidx]
        truth_xz[:, :, 2, i] = truth_field.v[:, iy, :, tidx]
        truth_xz[:, :, 3, i] = truth_field.w[:, iy, :, tidx]
        truth_xy[:, :, 1, i] = truth_field.u[:, :, iz, tidx]
        truth_xy[:, :, 2, i] = truth_field.v[:, :, iz, tidx]
        truth_xy[:, :, 3, i] = truth_field.w[:, :, iz, tidx]
        truth_yz[:, :, 1, i] = truth_field.u[ix, :, :, tidx]
        truth_yz[:, :, 2, i] = truth_field.v[ix, :, :, tidx]
        truth_yz[:, :, 3, i] = truth_field.w[ix, :, :, tidx]

        truth_state = Float32[vec(truth_field.u[:,:,:,tidx]);
                              vec(truth_field.v[:,:,:,tidx]);
                              vec(truth_field.w[:,:,:,tidx])]
        rmse_gmf_vec[i] = sqrt(mean((state_gmf .- truth_state).^2))

        state_kf = basis.modes * Float32.(kf_history[i+1][1])
        rmse_kf_vec[i]  = sqrt(mean((state_kf .- truth_state).^2))

        mean_spread = sum(gm.weights[j] * sum((gm.means[j,:] .- post_mean).^2) for j in 1:gm.K)
        totvar_vec[i]   = sum(gm.weights[j] * sum(gm.vars[j,:]) for j in 1:gm.K) + mean_spread
    end

    x_km       = grid.xPos[:, iy, 1]    ./ 1000
    y_km       = grid.yPos[1, :, iz]    ./ 1000
    z_km       = grid.zPos[grid.Nx÷2, iy, :] ./ 1000
    z_slice_km = Float32(grid.zPos[grid.Nx÷2, iy, iz] / 1000)
    x_slice_km = Float32(grid.xPos[ix, iy, 1]         / 1000)

    # Sensor positions: (n_iter, n_sensors, 3) in km — NaN if no sensors provided
    # Using continuous-space loc (not grid index) so points sit correctly on km axes
    if !isnothing(sensor_locs)
        n_sens = maximum(length(mset.measurements) for mset in sensor_locs)
        sensor_xyz_km = fill(Float32(NaN), n_iter, n_sens, 3)
        for i in 1:n_iter
            for (j, m) in enumerate(sensor_locs[i].measurements)
                sensor_xyz_km[i, j, 1] = m.loc[1] / 1000f0   # x km
                sensor_xyz_km[i, j, 2] = m.loc[2] / 1000f0   # y km
                sensor_xyz_km[i, j, 3] = m.loc[3] / 1000f0   # z km
            end
        end
    else
        sensor_xyz_km = zeros(Float32, 0, 0, 3)
    end

    jldsave(outpath;
            recon_xz=recon_xz, truth_xz=truth_xz,
            recon_xy=recon_xy, truth_xy=truth_xy,
            recon_yz=recon_yz, truth_yz=truth_yz,
            rmse_gmf=rmse_gmf_vec, rmse_kf=rmse_kf_vec,
            rmse_ls=Float32.(rmse_ls),
            total_var=totvar_vec,
            x_km=x_km, y_km=y_km, z_km=z_km,
            z_slice_km=z_slice_km, x_slice_km=x_slice_km,
            sensor_xyz_km=sensor_xyz_km,
            n_iter=Int32(n_iter))
    @info "Saved filter history → $outpath  ($n_iter iterations)"
end


function display_comparisons(filter_history::Vector{GMFilter}, β_ls::Vector{Vector{Float32}}, β_kf::Vector{Vector{Float32}})
    


end