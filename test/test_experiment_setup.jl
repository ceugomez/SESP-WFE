# test_experiment_setup.jl — the shared OSSE setup (src/experiment.jl, src/regions.jl) must
# reproduce, index for index, the sensor geometry that the drivers used
# when each carried its own inline copy (frozen below). Needs the ensemble output on disk.
#   julia --startup-file=no test/test_experiment_setup.jl
using Test
include(joinpath(@__DIR__, "..", "scripts", "paths.jl"))
include(joinpath(SRC_DIR, "kf_adapters.jl"))   # as the drivers load it
include(joinpath(SRC_DIR, "experiment.jl"))

config = load_ensemble_config(CONFIG_PATH)
grid   = load_grid(config.truth_member.files[1])
domain = AnalysisDomain(grid; margin_m=MARGIN_M)

# ---- frozen copies of the pre-2026-11-04 inline code ----
function legacy_rake(grid, domain)
    crest = argmax(grid.zPos[:, :, 1])
    crest_y = grid.yPos[crest[1], crest[2], 1]
    locs = Vector{Vector{Float32}}()
    for dy in Float32[-500, -250, 0, 250, 500], z in Float32[10, 50, 100]
        i, j, k = nearest_cell(grid, domain, 3000f0, crest_y + dy, z)
        push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
    end
    return locs
end
function legacy_lattice(grid, domain; sxy=50, sz=10)
    crest = argmax(grid.zPos[:, :, 1])
    crest_x = grid.xPos[crest[1], crest[2], 1]
    ix_dom_lo = findfirst(i -> any(domain.mask[i, :, :]), 1:grid.Nx)
    ix_hi     = findlast( i -> any(domain.mask[i, :, :]), 1:grid.Nx)
    iy_lo     = findfirst(j -> any(domain.mask[:, j, :]), 1:grid.Ny)
    iy_hi     = findlast( j -> any(domain.mask[:, j, :]), 1:grid.Ny)
    ix_lo     = max(ix_dom_lo, findfirst(i -> grid.xPos[i, 1, 1] >= crest_x, 1:grid.Nx))
    locs = Vector{Vector{Float32}}()
    for i in ix_lo:sxy:ix_hi, j in iy_lo:sxy:iy_hi
        terr = grid.zPos[i, j, 1]
        for k in 2:sz:grid.Nz
            (grid.zPos[i, j, k] - terr) <= 250f0 || break
            domain.mask[i, j, k] || continue
            push!(locs, Float32[grid.xPos[i, j, k], grid.yPos[i, j, k], grid.zPos[i, j, k]])
        end
    end
    return locs
end

@testset "shared OSSE setup == legacy inline code" begin
    @test region_rows(make_region("full", grid, domain), domain) == collect(1:domain.state_dim)
    @test_throws ErrorException make_region("wake", grid, domain)   # removed 2026-11-05
    @test rake_locations(grid, domain) == legacy_rake(grid, domain)
    @test length(rake_locations(grid, domain)) == 15
    lat = dense_lattice_locations(grid, domain)
    @test lat == legacy_lattice(grid, domain)
    @test length(lat) == 175
    @test dense_lattice_locations(grid, domain; stride_xy=40) == legacy_lattice(grid, domain; sxy=40)
    @test resolve_member(config, "truth") === config.truth_member
    @test resolve_member(config, "03").id == "03"
    @test_throws ErrorException resolve_member(config, "nope")
end
