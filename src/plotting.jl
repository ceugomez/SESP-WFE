# cgf cego6160@colorado.edu 
# 4.1.26
# Plotting utilities for use on POD modal decomposition and GM 
using Distributions, Plots, StatsPlots


# ─────────────────────────────────────────────────────────────────────────────
# RMSE convergence plot — one line per estimator across filter iterations.
# pod_floor: optional irreducible error from basis truncation (horizontal line).
# ─────────────────────────────────────────────────────────────────────────────
function plot_rmse_convergence(rmse_gmf::Vector{Float64}, rmse_kf::Vector{Float64},
                               rmse_ls::Vector{Float64};
                               pod_floor::Union{Float64,Nothing}=nothing,
                               title::String="RMSE convergence",
                               outdir::String="/home/cego6160/workspace/prediction/output")
    iters = 1:length(rmse_gmf)

    # y-axis from GMF/KF range, ignoring LS blowup
    ref_vals = vcat(rmse_gmf, rmse_kf, isnothing(pod_floor) ? Float64[] : [pod_floor])
    pad = (maximum(ref_vals) - minimum(ref_vals)) * 0.1
    ylo = max(0.0, minimum(ref_vals) - pad)
    yhi = maximum(ref_vals) + pad

    p = plot(iters, rmse_gmf, label="GMF (MAP)", color=:steelblue,   linewidth=2)
    plot!(p, iters, rmse_kf,  label="KF",        color=:darkorange,  linewidth=2)
    plot!(p, iters, rmse_ls,  label="LS (sequential)", color=:forestgreen, linewidth=1.5, linestyle=:dash)
    if !isnothing(pod_floor)
        hline!(p, [pod_floor], label="POD floor", color=:black, linewidth=1, linestyle=:dot)
    end
    plot!(p, xlabel="Filter iteration", ylabel="Field RMSE  (m s⁻¹)",
          title=title, legend=:topright, grid=true, dpi=150, size=(800, 500),
          ylims=(ylo, yhi))
    fpath = joinpath(outdir, "rmse_convergence.png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-coefficient tracking: GMF MAP, GMF mean ±1σ, KF ±1σ, LS, and truth
# for the leading n_modes POD coefficients across filter iterations.
# Layout is a grid of subplots (up to 5 columns).
# ─────────────────────────────────────────────────────────────────────────────
# truth_coeffs may be a Vector{Float32} (static: same truth at every step)
# or a Matrix{Float32} of shape (n_iter, L) (dynamic: one row per assimilation step).
function plot_coeff_tracking(filter_history::Vector{GMFilter},
                             kf_history::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                             basis::PODBasis,
                             Y::Vector{measSet},
                             truth_coeffs::Union{Vector{Float32}, Matrix{Float32}};
                             n_modes::Int=4,
                             outdir::String="/home/cego6160/workspace/prediction/output")
    n_iter = length(filter_history) - 1
    iters  = 1:n_iter
    L      = min(n_modes, basis.L)

    gmf_map  = Matrix{Float64}(undef, L, n_iter)
    gmf_mean = Matrix{Float64}(undef, L, n_iter)
    gmf_sig  = Matrix{Float64}(undef, L, n_iter)
    kf_mean  = Matrix{Float64}(undef, L, n_iter)
    kf_sig   = Matrix{Float64}(undef, L, n_iter)
    ls_val   = Matrix{Float64}(undef, L, n_iter)

    for i in 1:n_iter
        gm  = filter_history[i+1]
        μ   = vec(Float64.(gm.weights' * gm.means))

        gmf_map[:, i]  = Float64.(gm.means[argmax(gm.weights), 1:L])
        gmf_mean[:, i] = μ[1:L]

        within  = vec(Float64.(gm.weights' * gm.vars))
        between = vec(sum(gm.weights[k] .* (Float64.(gm.means[k, :]) .- μ).^2 for k in 1:gm.K))
        gmf_sig[:, i]  = sqrt.(within[1:L] .+ between[1:L])

        kf_mean[:, i]  = kf_history[i+1][1][1:L]
        kf_sig[:, i]   = sqrt.(max.(kf_history[i+1][2][1:L], 0.0))

        ls_val[:, i]   = leastsquares(basis, Y[1:i])[1:L]
    end

    ncols = 2
    nrows = ceil(Int, L / ncols)
    panels = []

    # Normalise truth_coeffs to (n_iter, L) regardless of static/dynamic input
    truth_mat = truth_coeffs isa Vector ?
        repeat(Float64.(truth_coeffs[1:L])', n_iter, 1) :
        Float64.(truth_coeffs[:, 1:L])

    for l in 1:L
        truth_l = truth_mat[:, l]   # Vector{Float64} length n_iter

        # y-axis bounds from GMF/KF range + truth, ignoring LS blowup
        all_vals = vcat(gmf_map[l, :], gmf_mean[l, :], kf_mean[l, :], truth_l)
        pad = max(maximum(gmf_sig[l, :]), maximum(kf_sig[l, :]), maximum(abs.(truth_l)) * 0.1)
        ylo = minimum(all_vals) - pad
        yhi = maximum(all_vals) + pad

        p = plot(iters, gmf_map[l, :],
                 label="GMF MAP", color=:steelblue, linewidth=2)
        plot!(p, iters, gmf_mean[l, :],
              ribbon=gmf_sig[l, :], fillalpha=0.15,
              label="GMF mean ±1σ", color=:steelblue,
              linewidth=1.2, linestyle=:dash, fillcolor=:steelblue)
        plot!(p, iters, kf_mean[l, :],
              ribbon=kf_sig[l, :], fillalpha=0.15,
              label="KF ±1σ", color=:darkorange,
              linewidth=1.5, fillcolor=:darkorange)
        plot!(p, iters, ls_val[l, :],
              label="LS", color=:forestgreen,
              linewidth=1.2, linestyle=:dash)
        plot!(p, iters, truth_l,
              label="Truth", color=:red,
              linewidth=1.5, linestyle=:dot)
        plot!(p, title="Mode $l", xlabel="Assimilation step", ylabel="coeff",
              ylims=(ylo, yhi),
              legend=false,
              grid=true, titlefontsize=10, guidefontsize=8, tickfontsize=8)
        push!(panels, p)
    end

    # Legend-only phantom panel: empty series so only the key entries render
    legend_panel = plot(Float64[], Float64[], label="GMF MAP", color=:steelblue, linewidth=2)
    plot!(legend_panel, Float64[], Float64[], label="GMF mean ±1σ", color=:steelblue,
          linewidth=1.2, linestyle=:dash)
    plot!(legend_panel, Float64[], Float64[], label="KF ±1σ", color=:darkorange, linewidth=1.5)
    plot!(legend_panel, Float64[], Float64[], label="LS", color=:forestgreen,
          linewidth=1.2, linestyle=:dash)
    plot!(legend_panel, Float64[], Float64[], label="Truth", color=:red,
          linewidth=1.5, linestyle=:dot)
    plot!(legend_panel, axis=false, grid=false, ticks=false, border=:none,
          legend=:inside, legendfontsize=9, background_color=:white)
    push!(panels, legend_panel)

    layout = @layout [grid(nrows, ncols) a{0.18w}]

    p = plot(panels..., layout=layout,
             size=(ncols * 380 + 220, nrows * 300), dpi=150)
    fpath = joinpath(outdir, "coeff_tracking.png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Mixture weight evolution.
# Works with both fixed-K (runtime_loop) and variable-K (runtime_loop_resample).
# Plots: max component weight, effective number of components (1/Σwk²),
# and active component count K per iteration.
# ─────────────────────────────────────────────────────────────────────────────
function plot_weight_evolution(filter_history::Vector{GMFilter};
                               outdir::String="/home/cego6160/workspace/prediction/output")
    n_iter   = length(filter_history) - 1
    iters    = 1:n_iter
    max_w    = Vector{Float64}(undef, n_iter)
    eff_K    = Vector{Float64}(undef, n_iter)
    active_K = Vector{Int}(undef, n_iter)

    for i in 1:n_iter
        w = filter_history[i+1].weights
        max_w[i]    = maximum(w)
        eff_K[i]    = 1.0 / sum(w .^ 2)
        active_K[i] = filter_history[i+1].K
    end

    p1 = plot(iters, max_w, color=:steelblue, linewidth=2, label="",
              ylabel="Max component weight", xlabel="", grid=true,
              title="Mixture weight evolution", titlefontsize=10)
    p2 = plot(iters, eff_K,    color=:darkorange, linewidth=2, label="Effective K",
              ylabel="Effective K  (1/Σwk²)", xlabel="", grid=true)
    plot!(p2, iters, Float64.(active_K), color=:gray, linewidth=1.2,
          linestyle=:dash, label="Active K")
    plot!(p2, legend=:topright)

    p = plot(p1, p2, layout=(2, 1), size=(800, 550), dpi=150)
    fpath = joinpath(outdir, "weight_evolution.png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Prior visualization: for each of the leading n_modes coefficients, plot the
# mixture PDF (solid) with individual component Gaussians (dashed), and a
# vertical line at the truth value. Useful for comparing init_prior vs
# init_prior_uniform before any measurements are assimilated.
# ─────────────────────────────────────────────────────────────────────────────
function plot_prior(prior::GMFilter, truth_coeffs::Vector{Float32};
                    n_modes::Int=4,
                    outdir::String="/home/cego6160/workspace/prediction/output",
                    label::String="prior")
    L      = min(n_modes, prior.L)
    ncols  = min(4, L)
    nrows  = ceil(Int, L / ncols)
    panels = []

    for l in 1:L
        truth_l = Float64(truth_coeffs[l])
        components = [Normal(Float64(prior.means[k, l]), sqrt(Float64(prior.vars[k, l])))
                      for k in 1:prior.K]
        mixture = MixtureModel(components, prior.weights)

        # x range: cover ±3σ of mixture marginal, always include truth
        μ_mix = sum(prior.weights[k] * Float64(prior.means[k, l]) for k in 1:prior.K)
        σ_mix = sqrt(sum(prior.weights[k] * Float64(prior.vars[k, l]) for k in 1:prior.K) +
                     sum(prior.weights[k] * (Float64(prior.means[k, l]) - μ_mix)^2 for k in 1:prior.K))
        xlo = min(μ_mix - 3σ_mix, truth_l - σ_mix)
        xhi = max(μ_mix + 3σ_mix, truth_l + σ_mix)
        xs  = range(xlo, xhi, length=400)

        col = mod1(l, ncols)
        p = plot(xs, x -> pdf(mixture, x), color=:steelblue, linewidth=2,
                 label=(l == 1 ? "Mixture" : ""), title="Mode $l",
                 ylabel=(col == 1 ? "density" : ""), grid=true,
                 titlefontsize=9, guidefontsize=7, tickfontsize=7,
                 xrotation=45, bottom_margin=8Plots.mm,
                 left_margin=(col == 1 ? 6Plots.mm : 2Plots.mm))
        for k in 1:prior.K
            plot!(p, xs, x -> prior.weights[k] * pdf(components[k], x),
                  color=:steelblue, linewidth=0.5, alpha=0.3, linestyle=:dash, label="")
        end
        vline!(p, [truth_l], color=:red, linewidth=1.5, linestyle=:dot,
               label=(l == 1 ? "Truth" : ""))
        plot!(p, legend=(l == 1 ? :topright : false))
        push!(panels, p)
    end

    p = plot(panels..., layout=(nrows, ncols),
             size=(ncols * 300, nrows * 260), dpi=550)
    fpath = joinpath(outdir, "prior_$(label).png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
end

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
  
# ─────────────────────────────────────────────────────────────────────────────
# Final posterior uncertainty field + flight track overlay.
# Shows σ_u in the xz plane (mid-y) computed from the full GMF posterior
# covariance (within + between component), with all sensor positions from Y
# scattered and colored by iteration number to show the trajectory.
# ─────────────────────────────────────────────────────────────────────────────
function plot_uncertainty_field(gm_final::GMFilter, basis::PODBasis, Y::Vector{measSet};
                                y_idx::Int=-1,
                                outdir::String="/home/cego6160/workspace/prediction/output")
    grid = basis.grid
    iy   = y_idx > 0 ? y_idx : grid.Ny ÷ 2
    n    = grid.n_grid

    # posterior variance per coefficient: within + between component
    μ        = vec(Float64.(gm_final.weights' * gm_final.means))
    within   = vec(Float64.(gm_final.weights' * gm_final.vars))
    between  = vec(sum(gm_final.weights[k] .* (Float64.(gm_final.means[k,:]) .- μ).^2
                       for k in 1:gm_final.K))
    σ²_coeff = within .+ between                                      # (L,)

    # propagate to physical space: σ²_j = Σ_l Φ[j,l]² * σ²_coeff[l]
    # modes is (state_dim, L); squaring column-wise then weighting by σ²_coeff
    σ²_state = (basis.modes .^ 2) * σ²_coeff                         # (state_dim,)
    σ_u = reshape(sqrt.(σ²_state[1:n]), grid.Nx, grid.Ny, grid.Nz)   # u-component σ

    x_km = grid.xPos[:, iy, 1]          ./ 1000
    z_km = grid.zPos[grid.Nx÷2, iy, :] ./ 1000

    p = heatmap(x_km, z_km, σ_u[:, iy, :]',
                color=:plasma, colorbar_title="σ_u  (m s⁻¹)",
                xlabel="x  (km)", ylabel="z  (km)",
                title="Posterior u-uncertainty + flight track",
                dpi=150, size=(900, 500))

    # overlay sensor positions colored by iteration
    n_iter = length(Y)
    cmap   = cgrad(:viridis, n_iter, categorical=true)
    for (i, mset) in enumerate(Y)
        for m in mset.measurements
            # use sensor's z-index evaluated at the heatmap's reference x (Nx÷2)
            # so the dot sits on the correct row of the sigma-coordinate z-axis
            z_plot = grid.zPos[grid.Nx÷2, iy, m.locidx[3]] / 1000
            scatter!(p, [m.loc[1]/1000], [z_plot],
                     color=cmap[i], markersize=4, markerstrokewidth=0,
                     marker=:circle, label="", colorbar=false)
        end
    end

    # colorbar-like annotation: first and last iteration markers for reference
    scatter!(p, [NaN], [NaN], color=cmap[1],       markersize=5, label="iter 1")
    scatter!(p, [NaN], [NaN], color=cmap[n_iter],  markersize=5, label="iter $n_iter")
    plot!(p, legend=:topright)

    fpath = joinpath(outdir, "uncertainty_field.png")
    savefig(p, fpath)
    @info "Saved $fpath"
    return p
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
# 2×2 grid of xy mode shapes for the first 4 POD modes.
# Each panel shows the u-component in the xy plane at near-surface z.
# ─────────────────────────────────────────────────────────────────────────────
function plot_mode_shapes_xy(basis::PODBasis; z_idx::Int=-1,
                              outdir::String="/home/cego6160/workspace/prediction/output")
    grid = basis.grid
    iz   = z_idx > 0 ? z_idx : max(1, grid.Nz ÷ 8)
    n    = grid.n_grid

    x_km = grid.xPos[:, 1, 1]   ./ 1000
    y_km = grid.yPos[1, :, iz]  ./ 1000

    panels = []
    for mode_idx in 1:4
        ψ    = basis.modes[:, mode_idx]
        u_xy = reshape(ψ[1:n], grid.Nx, grid.Ny, grid.Nz)[:, :, iz]
        clim_val = max(maximum(abs.(u_xy)), eps(Float32))
        energy_frac = round(100 * basis.eigenvalues[mode_idx] / sum(basis.eigenvalues), digits=1)
        p = heatmap(x_km, y_km, u_xy',
                    color=:RdBu, clims=(-clim_val, clim_val),
                    title="Mode $mode_idx  ($(energy_frac)%)",
                    xlabel="x (km)", ylabel="y (km)",
                    titlefontsize=10, guidefontsize=8, tickfontsize=7)
        push!(panels, p)
    end

    z_km_val = round(grid.zPos[1, 1, iz] / 1000, digits=2)
    p = plot(panels..., layout=(2, 2), size=(1200, 900), dpi=150,
             plot_title="POD mode shapes — u component, xy plane at z≈$(z_km_val) km")
    fpath = joinpath(outdir, "mode_shapes_xy_2x2.png")
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

