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
    @info("  vars[:, $(coeff_idx)]:   $(g.vars[:, coeff_idx])")
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