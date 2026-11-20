# run_repr_error.jl
# cgf cego6160@colorado.edu 11.4.26
#
# Time-resolved REPRESENTATION ERROR of each LOO fold: the part of the held-out truth that
# its fold basis cannot represent at rank r,
#     e_r(t) = x(t) − P_r P_rᵀ x(t),      P_r = leading r modes of the fold's basis,
# for every rank r = 1..L and every post-spinup snapshot t, per velocity component. This is
# the "subspace oracle" floor no filter on that basis can beat.
#
# EXACT, without forming L reconstructions per step. For component block c with mode rows
# P_c and truth rows x_c, and a = Pᵀx (full-state projection):
#     ‖x_c − (P_r a_{1:r})_c‖² = ‖x_c‖² − 2 Σ_{i≤r} a_i (P_cᵀx_c)_i + a_{1:r}ᵀ (P_cᵀP_c)_{1:r,1:r} a_{1:r}
# evaluated in Float64 from the Float32 data and modes. No orthonormality is assumed (the Gram
# blocks are the actual ones), so this equals the direct residual to Float64 round-off; the
# script CHECKS that against a direct reconstruction at sample steps and errors otherwise.
#
# Output: results/<tag>/repr/repr_error_<fold>.nc (REPR_DIR), one per fold, NetCDF:
#   dims  time (K), rank (L), component (3: u, v, w)
#   sse(component, rank, time)  Σ over region cells of e²     [m² s⁻²]
#   energy(component, time)     Σ over region cells of x²     [m² s⁻²]  (for fractions)
#   time(time)                  simulation time [s]
#   attrs: fold, region, n_cells, basis_members, margin_m, t_spinup, + provenance
# Component-RMS (the metric used throughout so far) = sqrt(Σ_c sse / (3·n_cells)).
#
# Usage:  FOLDS=00,01,truth julia --threads=8 scripts/offline/run_repr_error.jl
#         (default FOLDS: all 12 LOO folds + truth). REGION=full (src/regions.jl).
using LinearAlgebra, Statistics, Printf, NCDatasets
include(joinpath(@__DIR__, "..", "paths.jl"))
include(joinpath(SRC_DIR, "experiment.jl"))     # T_SPINUP, MARGIN_M, shared setup
const REGION   = get(ENV, "REGION", "full")
const FOLDS    = String.(split(get(ENV, "FOLDS",
    "00,01,02,03,04,05,06,07,08,09,10,11,truth"), ','))
const CHUNK    = 16          # snapshot columns per Float64 GEMM chunk
const CHECK_RANKS = (1, 4, 50)
include(joinpath(SRC_DIR, "fasteddy_io.jl"))
include(joinpath(SRC_DIR, "pod.jl"))
include(joinpath(SRC_DIR, "regions.jl"))
include(joinpath(SRC_DIR, "provenance.jl"))

repr_error_filename(fold) = joinpath(REPR_DIR, "repr_error_$(fold).nc")

BLAS.set_num_threads(Threads.nthreads())
config = load_ensemble_config(CONFIG_PATH)

for fold in FOLDS
    t0 = time()
    @info "=== fold $fold ==="
    basis  = load_pod_basis(basis_filename(BASES_DIR, fold))
    member = resolve_member(config, fold)
    fold in basis.member_ids && error("fold $fold is IN its own basis — not a held-out test")
    grid   = load_grid(member.files[1])
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)
    region = make_region(REGION, grid, domain)
    L, n   = basis.L, domain.n_grid
    size(basis.modes, 1) == domain.state_dim || error("basis/domain state_dim mismatch")

    times = Float64.(member.sim_times[_spinup_mask(member, T_SPINUP)])
    X     = load_member_snapshots(member; t_spinup=T_SPINUP, domain)     # (3n × K) Float32
    K     = size(X, 2)
    @info "  loaded truth: $(size(X)) in $(round(time() - t0, digits=0)) s; region=$(region.name) ($(n_cells(region)) cells)"

    # Per-component region rows and Float64 mode blocks (state is [u; v; w] blocks of n).
    cells = findall(region.cells)
    rows  = [cells .+ (c - 1) * n for c in 1:3]
    Pc    = [Float64.(basis.modes[rows[c], :]) for c in 1:3]             # (n_reg × L) each
    Gc    = [Matrix(Symmetric(Pc[c]' * Pc[c])) for c in 1:3]              # region Gram blocks
    Pfull = Float64.(basis.modes)                                         # for a = Pᵀx (full state)

    sse    = zeros(Float64, 3, L, K)
    energy = zeros(Float64, 3, K)
    A      = zeros(Float64, L, K)                                         # a = Pᵀx (oracle coeffs)
    for c0 in 1:CHUNK:K
        cc = c0:min(c0 + CHUNK - 1, K)
        Xc = Float64.(X[:, cc])
        A[:, cc] = Pfull' * Xc
        for c in 1:3
            Xr = Xc[rows[c], :]
            B  = Pc[c]' * Xr                                              # (L × |cc|)
            for (j, k) in enumerate(cc)
                a  = @view A[:, k]
                ex = sum(abs2, @view Xr[:, j])
                energy[c, k] = ex
                # cumulative over r: cross term and quadratic form, built incrementally
                cross = 0.0; quad = 0.0
                Ga = zeros(Float64, L)                                    # G[1:r,1:r] a[1:r], grown
                for r in 1:L
                    cross += a[r] * B[r, j]
                    # quad(r) = quad(r-1) + 2 a_r Σ_{i<r} G_ri a_i + G_rr a_r²
                    s = 0.0
                    @inbounds for i in 1:r-1
                        s += Gc[c][r, i] * a[i]
                    end
                    quad += 2a[r] * s + Gc[c][r, r] * a[r]^2
                    sse[c, r, k] = ex - 2cross + quad
                end
            end
        end
    end
    any(<(0), sse) && @warn "  negative sse (round-off) min=$(minimum(sse)) — clamping"
    clamp!(sse, 0.0, Inf)

    # --- Verify against a DIRECT Float64 reconstruction at sample steps ---
    for k in unique([1, K ÷ 2, K]), r in CHECK_RANKS
        xk = Float64.(X[:, k])
        e  = xk .- Pfull[:, 1:r] * A[1:r, k]
        for c in 1:3
            direct = sum(abs2, @view e[rows[c]])
            rel = abs(direct - sse[c, r, k]) / direct
            rel < 1e-6 || error("fold $fold k=$k r=$r c=$c: Gram-form sse $(sse[c,r,k]) ≠ direct $direct (rel $rel)")
        end
    end
    @info "  direct-reconstruction check passed (ranks $(CHECK_RANKS), steps 1/mid/end)"

    rmse(r) = sqrt.(vec(sum(sse[:, r, :]; dims=1)) ./ (3 * n_cells(region)))
    for r in (4, L)
        v = rmse(r)
        @info @sprintf("  r=%-2d component-RMSE over time: mean %.4f  std %.4f  [min %.4f max %.4f] m/s",
                       r, mean(v), std(v), minimum(v), maximum(v))
    end

    out = repr_error_filename(fold)
    isfile(out) && rm(out)
    NCDataset(out, "c") do ds
        defDim(ds, "component", 3); defDim(ds, "rank", L); defDim(ds, "time", K)
        v = defVar(ds, "sse", Float64, ("component", "rank", "time"),
                   attrib = Dict("units" => "m2 s-2",
                                 "long_name" => "sum over region cells of squared representation error"))
        v[:, :, :] = sse
        v = defVar(ds, "energy", Float64, ("component", "time"),
                   attrib = Dict("units" => "m2 s-2", "long_name" => "sum over region cells of squared truth"))
        v[:, :] = energy
        v = defVar(ds, "time", Float64, ("time",), attrib = Dict("units" => "s", "long_name" => "simulation time"))
        v[:] = times
        v = defVar(ds, "rank", Int32, ("rank",)); v[:] = Int32.(1:L)
        v = defVar(ds, "component", String, ("component",)); v[:] = ["u", "v", "w"]
        attrs = merge(provenance_attrs(@__FILE__), Dict{String,Any}(
            "fold" => String(fold), "region" => region.name, "region_description" => region.description,
            "n_cells" => n_cells(region), "basis_members" => join(basis.member_ids, ","),
            "basis_file" => basename(basis_filename(BASES_DIR, fold)),
            "margin_m" => Float64(MARGIN_M), "t_spinup" => T_SPINUP,
            "metric_note" => "component RMSE = sqrt(sum_c sse / (3 n_cells))"))
        for (k, val) in attrs
            ds.attrib[k] = val
        end
    end
    @info "  wrote $out  ($(round(time() - t0, digits=0)) s total)"
    X = nothing; Pfull = nothing; Pc = nothing; GC.gc()
end
@info "=== done: $(join(FOLDS, ", ")) ==="
