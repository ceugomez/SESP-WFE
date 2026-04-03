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
    @info("  vars[:, $(coeff_idx)]:   $([g.covs[k, coeff_idx, coeff_idx] for k in 1:g.K])")
    components = [Normal(g.means[k, coeff_idx], sqrt(g.covs[k, coeff_idx, coeff_idx])) for k in 1:g.K]
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
        components = [Normal(g.means[k, coeff_idx], sqrt(g.covs[k, coeff_idx, coeff_idx])) for k in 1:g.K]
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

function plot_reconstruction_field_error_over_time(gvec::Vector{GMFilter}, truth_field::NCfield)
end