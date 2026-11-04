# regions.jl
# cgf cego6160@colorado.edu 11.4.26
#
# THE scoring regions for SESP-WFE. Every script that reports an error over a spatial region
# gets it from here — never by redefining a box inline. A Region selects a subset of the
# analysis-domain cells (the masked state layout of fasteddy_io.jl), so in the stacked
# [u; v; w] state a region's rows are `region_rows(region, domain)`.
#
# Current regions (2026-11-04): only "full" = the whole analysis domain (sponge margin
# excluded). The hill+wake subregion is still to be decided; add it here when it is.
include("fasteddy_io.jl")   # GridInfo, AnalysisDomain

struct Region
    name        :: String
    description :: String
    cells       :: BitVector   # (domain.n_grid,) true = cell in region, masked-cell order
end

n_cells(r::Region) = count(r.cells)

"Row indices of `region` in the stacked [u; v; w] state vector (length 3·n_cells)."
function region_rows(region::Region, domain::AnalysisDomain)
    length(region.cells) == domain.n_grid ||
        error("region '$(region.name)' was built for a different domain")
    p = findall(region.cells)
    n = domain.n_grid
    return vcat(p, p .+ n, p .+ 2n)
end

"Whole analysis domain (every masked cell)."
region_full(domain::AnalysisDomain) =
    Region("full", "entire analysis domain (sponge margin excluded)", trues(domain.n_grid))

"Look up a region by name. The single entry point drivers should use."
function make_region(name::AbstractString, grid::GridInfo, domain::AnalysisDomain)
    name == "full" && return region_full(domain)
    error("unknown region '$name' (defined: full)")
end
