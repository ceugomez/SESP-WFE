# make_heldout_region_residual.jl
# cgf 2026-06-26
#
# Leave-one-member-out residual energy vs. truncation rank, split by SPATIAL REGION
# (full / wake / ambient) and by COMPONENT within the wake (all / horizontal uv / w-only).
# This is the projection-floor / family-closure figure: at rank r the rank-r subspace
# (basis built WITHOUT the held-out member) cannot represent that member's realization;
# the unrepresented fraction is the ceiling on any filter using this basis.
#
# Runs on the RAW (uncentered) state, matching the manuscript's representativeness-error
# definition (Eq. 12-14: e^(i) = u^(i) - <u^(i),Pr>, no mean removal) and the basis itself,
# which is deliberately built WITHOUT removing the mean flow (Sec 2a: "we choose not to
# remove the mean flow state ... such that the forcing distribution is intrinsic to the
# basis"). Centering the evaluation field here while the basis spans the full (mean+
# fluctuation) family would test something the basis was never built to represent.
#
# Exact region-restricted residual energy WITHOUT forming L rank-r reconstructions.
# For a region S (a 0/1 row-selector of the [u;v;w] state) and held-out member with
# snapshots X (state_dim×T), modes Pr (orthonormal globally), coeffs a = PrᵀX (L×T):
#
#   E_resid^S(r) = E_x^S  − 2 Σ_{i≤r} d_i  +  Σ_{i,j≤r} (Gᔆ ∘ A)_{ij}
#       E_x^S = Σ_t ⟨x_t,x_t⟩_S            (region energy)
#       d_i   = Σ_t a_{i,t} Pᔆ_{i,t},  Pᔆ = PrᵀS X   (region-restricted projection, L×T)
#       Gᔆ    = PrᵀS Pr   (region-restricted Gram, L×L; =I only for S=full,all-comp)
#       A     = a aᵀ      (temporal coeff covariance, L×L)
# Normalised by region energy E_x^S (no mean removal). y = E_resid/E_x = fraction unrepresented.
#
# Cross-check: for S=full,all-comp, Gᔆ=I, Pᔆ=a ⇒ E_resid(50)=Σ‖x‖²−Σ‖a‖² (the Parseval
# residual of heldout_residual_curve.csv) — the full panel must match that curve.
using LinearAlgebra, JLD2, Statistics, Printf

include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, HELDOUT_DIR, SRC_DIR, ...
const WORKDIR     = SRC_DIR
const OUTPUT_DIR  = BASES_DIR             # reads the LOO bases
const ANALYSIS_DIR= HELDOUT_DIR           # region-residual CSV + headline land here
const T_SPINUP    = 900.0
const MARGIN_M    = 600f0
const MEMBER_IDS  = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
const OPERATING_R = 50              # filter runs at full L; endpoint = operating-rank floor
const CHUNK       = 32              # snapshot-column chunk (bounds peak memory)

# Wake box (terrain-following): downstream of the ridge, low level. See AskUser 2026-06-26.
const WAKE_X = (1500f0, 4450f0)     # m
const WAKE_Y = (2000f0, 3500f0)     # m
const WAKE_ZAGL_MAX = 600f0         # m AGL

include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8")))
@info "BLAS threads: $(BLAS.get_num_threads())"

# Geometry → (Nx,Ny,Nz) Bool wake box (terrain-following AGL).
function wake_geom(grid::GridInfo)
    g = falses(grid.Nx, grid.Ny, grid.Nz)
    @inbounds for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        x = grid.xPos[i,j,k]; y = grid.yPos[i,j,k]
        zagl = grid.zPos[i,j,k] - grid.zPos[i,j,1]
        g[i,j,k] = (WAKE_X[1] <= x <= WAKE_X[2]) && (WAKE_Y[1] <= y <= WAKE_Y[2]) &&
                   (zagl <= WAKE_ZAGL_MAX)
    end
    return g
end

# Build a length-state_dim 0/1 selector for a region: 1 on rows whose masked cell is in
# `cellsel` AND whose component c is in `comps`; 0 elsewhere. cellsel is the length-n
# (masked-cell) BitVector of in-region cells.
function region_selector(cellsel::AbstractVector{Bool}, comps, n::Int)
    s = zeros(Float32, 3n)
    for c in comps
        @inbounds for p in 1:n
            cellsel[p] && (s[(c-1)*n + p] = 1f0)
        end
    end
    return s
end

# Region-restricted residual-energy curve E_resid^S(1:L) on the RAW state x (no mean removal,
# consistent with the basis and with the manuscript's representativeness-error definition).
# It starts at 1 (r=0) and decays as rank increases.
#   E_x^S = Σ_t⟨x_t,x_t⟩_S
#   a = Prᵀx  (coeffs), A = a aᵀ
#   d_i = Σ_t a_{i,t} Pᔆ_{i,t},  Pᔆ = PrᵀS x   (restricted),  Gᔆ = PrᵀS Pr
#   E_resid^S(r) = E_x^S − 2 Σ_{i≤r} d_i + Σ_{i,j≤r} (Gᔆ∘A)_{ij}
# X holds the raw snapshots; a the raw (uncentered) coeffs.
function region_curve(Pr, a, A, X::AbstractMatrix, s::Vector{Float32}, L::Int, T::Int)
    rowE = zeros(Float64, size(Pr, 1))                     # Σ_t x²  per row (chunked)
    d    = zeros(Float64, L)                              # Σ_t a_{i,t} Pᔆ_{i,t}
    for c0 in 1:CHUNK:T
        cc  = c0:min(c0+CHUNK-1, T)
        Xcc = @view X[:, cc]
        SXcc = s .* Xcc                                   # region-selected block
        rowE .+= vec(sum(abs2, Xcc; dims=2))             # unselected; region selector applied below
        Pcc  = Pr' * SXcc                                  # (L×|cc|) restricted projection of x
        d  .+= vec(sum(Float64.(a[:, cc]) .* Float64.(Pcc); dims=2))
    end
    E_x = sum(s .* rowE)                                  # Σ_t ⟨x_t,x_t⟩_S
    G = Pr' * (s .* Pr)                                     # restricted Gram (L×L)
    M = Float64.(G) .* A                                  # Gᔆ ∘ A
    dcum = cumsum(d)
    resid = zeros(Float64, L)
    s2 = 0.0
    @inbounds for r in 1:L
        s2 += M[r, r]
        for j in 1:r-1
            s2 += M[r, j] + M[j, r]
        end
        resid[r] = E_x - 2*dcum[r] + s2
    end
    return resid, E_x
end

config = load_ensemble_config(CONFIG_PATH)
id2m   = Dict(m.id => m for m in config.prior_members)

# region name → (component tuple, uses-wake-cells?) ; ambient handled via complement mask
const REGIONS = ["full_all", "wake_all", "wake_uv", "wake_w", "ambient_all"]
curves   = Dict(r => Dict{String,Vector{Float64}}() for r in REGIONS)   # region → memberid → resid-fraction(1:L)
fluct    = Dict(r => Dict{String,Float64}() for r in REGIONS)
present  = String[]

for mid in MEMBER_IDS
    bfile = basis_filename(OUTPUT_DIR, mid)
    isfile(bfile) || (@warn "no LOO basis for $mid, skipping"; continue)
    haskey(id2m, mid) || (@warn "member $mid not in config"; continue)
    @info "=== held-out member $mid ==="
    basis = load_pod_basis(bfile)
    # LEAKAGE GUARD: basis must have been trained WITHOUT this member.
    basis.truth_id == mid || error("basis $bfile truth_id=$(basis.truth_id) ≠ held-out $mid")
    mid in basis.member_ids && error("LEAKAGE: held-out $mid is in basis.member_ids $(basis.member_ids)")
    @info "  basis trained on $(length(basis.member_ids)) members (held-out $mid excluded) ✓"

    grid   = load_grid(id2m[mid].files[1])
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)
    n      = domain.n_grid
    L      = basis.L
    Pr      = basis.modes

    X  = load_member_snapshots(id2m[mid]; t_spinup=T_SPINUP, domain)   # (3n×T), raw (no mean removal)
    T  = size(X, 2)
    a  = project(basis, X)                                            # (L×T) = Prᵀx  (raw coeffs)
    A  = Float64.(a) * Float64.(a)'                                  # (L×L)

    # cell selectors (length-n masked-cell BitVectors)
    wk3       = wake_geom(grid) .& domain.mask
    wake_sel  = wk3[domain.mask]                                      # length-n
    amb_sel   = .!wake_sel                                            # in-domain, outside wake box

    regions_s = Dict(
        "full_all"    => region_selector(trues(n),  (1,2,3), n),
        "wake_all"    => region_selector(wake_sel,  (1,2,3), n),
        "wake_uv"     => region_selector(wake_sel,  (1,2),   n),
        "wake_w"      => region_selector(wake_sel,  (3,),    n),
        "ambient_all" => region_selector(amb_sel,   (1,2,3), n),
    )
    for rg in REGIONS
        resid, E_x = region_curve(Pr, a, A, X, regions_s[rg], L, T)
        curves[rg][mid] = resid ./ E_x
        fluct[rg][mid]  = E_x
        @info @sprintf("  %-12s  E_x=%.3e  resid_frac[r=%d]=%.4f", rg, E_x, min(OPERATING_R,L), (resid./E_x)[min(OPERATING_R,L)])
    end
    push!(present, mid)
    X = nothing; a = nothing; GC.gc()
end

isempty(present) && error("no members processed")
Lmax = length(curves["full_all"][present[1]])

# ---- write tidy CSV: region,component,rank,<mean>,<per-member cols...> ----
mkpath(ANALYSIS_DIR)
csv = joinpath(ANALYSIS_DIR, "heldout_region_residual.csv")
open(csv, "w") do io
    println(io, "region,rank,mean," * join(present, ","))
    for rg in REGIONS
        Mmat = hcat([curves[rg][m] for m in present]...)              # (L×Mpres)
        mu   = vec(mean(Mmat; dims=2))
        for r in 1:Lmax
            row = join((@sprintf("%.6e", Mmat[r,k]) for k in 1:length(present)), ",")
            println(io, "$rg,$r,$(@sprintf("%.6e", mu[r])),$row")
        end
    end
end
@info "wrote $csv"

# ---- headline scalar: wake w-only fraction unrepresented at operating rank ----
ropr = min(OPERATING_R, Lmax)
wkw  = [curves["wake_w"][m][ropr] for m in present]
μ = mean(wkw); σ = std(wkw)
hl = @sprintf("HEADLINE: at operating rank r=%d, the LOO subspace leaves %.1f%% ± %.1f%% (mean ± across-member sd, n=%d) of the held-out WAKE-region w-component energy UNREPRESENTABLE.",
              ropr, 100μ, 100σ, length(present))
@info hl
open(joinpath(ANALYSIS_DIR, "heldout_region_residual_headline.txt"), "w") do io
    println(io, hl)
    println(io, "members: $(join(present, ", "))")
    for m in present
        println(io, @sprintf("  %s: wake_w resid@r%d = %.4f", m, ropr, curves["wake_w"][m][ropr]))
    end
end
@info "=== Done ==="
