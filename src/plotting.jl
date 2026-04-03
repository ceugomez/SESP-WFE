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
               
  
function plot_gaussian_mixture(g::GMFilter, coeff_idx::Int=1)
    @info("GMFilter: K=$(g.K) components, L=$(g.L) modes")
    @info("  weights: $(g.weights)")
    @info("  means[:, $(coeff_idx)]:  $(g.means[:, coeff_idx])")
    @info("  vars[:, $(coeff_idx)]:   $([g.vars[k, coeff_idx] for k in 1:g.K])")
    components = [Normal(g.means[k, coeff_idx], sqrt(g.vars[k, coeff_idx])) for k in 1:g.K]
    mixture = MixtureModel(components, g.weights)
    plot(mixture,
         components=false,
         label="Total Mixture Density",
         linewidth=4,
         color=:black,
         title="GMM: Sum vs. Components (Coeff $(coeff_idx))",
         xlabel="Coefficient value",
         ylabel="Density",
         legend=:topright)
    plot!(mixture,
          components=true,
          label="", # Prevents 10 identical entries in the legend
          linestyle=:dash,
          alpha=0.5,
          linewidth=1.5)
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