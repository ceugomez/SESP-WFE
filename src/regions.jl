# regions.jl
# cgf cego6160@colorado.edu 11.4.26
#
# THE scoring regions for SESP-WFE. Every script that reports an error over a spatial region
# gets it from here — never by redefining a box inline. A Region selects a subset of the
# analysis-domain cells (the masked state layout of fasteddy_io.jl), so in the stacked
# [u; v; w] state a region's rows are `region_rows(region, domain)`.
#
# Regions:
#   "full" — the whole analysis domain (sponge margin excluded).
# The wake scoring box was removed 2026-11-05: it is being redesigned for the new ensemble
# setup. Add the revised region here (a constructor + a make_region branch); every scoring
# driver should then take it by name.
include("fasteddy_io.jl")   # GridInfo, AnalysisDomain

struct Region
    name        :: String
    description :: String
    cells       :: BitVector   # (domain.n_grid,) true = cell in region, masked-cell order
end

n_cells(r::Region) = count(r.cells)

"Row indices of `region` in the stacked [u; v; w] state vector (length 3·n_cells), ascending."
function region_rows(region::Region, domain::AnalysisDomain)
    length(region.cells) == domain.n_grid ||
        error("region '$(region.name)' was built for a different domain")
    p = findall(region.cells)
    n = domain.n_grid
    return vcat(p, p .+ n, p .+ 2n)
end

"(x, y) in metres of the hill crest = the highest terrain cell."
function hill_crest(grid::GridInfo)
    c = argmax(grid.zPos[:, :, 1])
    return grid.xPos[c[1], c[2], 1], grid.yPos[c[1], c[2], 1]
end

"Whole analysis domain (every masked cell)."
region_full(domain::AnalysisDomain) =
    Region("full", "entire analysis domain (sponge margin excluded)", trues(domain.n_grid))

"Look up a region by name. The single entry point drivers should use."
function make_region(name::AbstractString, grid::GridInfo, domain::AnalysisDomain)
    name == "full" && return region_full(domain)
    error("unknown region '$name' (defined: full)")
end
