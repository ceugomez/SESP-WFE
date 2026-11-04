# make_kf_field_snapshot.jl
# cgf — reproject saved filter coefficients back into the full field and write a single-time
# xy slice at a target AGL for: truth, filter estimate Prα, oracle Prα* (truth-projected),
# and the ensemble-mean prior Prα0. Answers "what does the filter's wind field estimate
# actually look like" — does it carry the wake deficit, or just the bulk wind?
# Output: output/kf_field_snapshot_truth_<ID>.nc  (small, one z-slice).
using LinearAlgebra, JLD2, NCDatasets, Statistics
include(joinpath(@__DIR__, "paths.jl"))  # CONFIG_PATH, BASES_DIR, KF_DIR, FIELDS_DIR, SRC_DIR, ...
const WORKDIR     = SRC_DIR
const T_SPINUP    = 900.0
const MARGIN_M    = 600f0
# Folds to render: env FOLDS="truth 00 01 ..." overrides; default = all on disk.
const Z_AGL       = parse(Float32, get(ENV, "Z_AGL", "50"))   # slice height, m AGL
include(joinpath(WORKDIR, "fasteddy_io.jl"))
include(joinpath(WORKDIR, "pod.jl"))
include(joinpath(WORKDIR, "measurement.jl"))
include(joinpath(WORKDIR, "statmodel.jl"))
include(joinpath(WORKDIR, "reperror.jl"))
include(joinpath(WORKDIR, "kf_adapters.jl"))

config = load_ensemble_config(CONFIG_PATH)
default_folds = vcat("truth", [m.id for m in config.prior_members])
FOLDS = get(ENV, "FOLDS", "") == "" ? default_folds : split(ENV["FOLDS"])

function render_fold(TRUTH_ID::AbstractString, config)
    basis  = load_pod_basis(basis_filename(BASES_DIR, String(TRUTH_ID)))
    truth_member = config.truth_member.id == TRUTH_ID ? config.truth_member :
        config.prior_members[findfirst(m -> m.id == TRUTH_ID, config.prior_members)]
    grid   = load_grid(truth_member.files[1])
    domain = AnalysisDomain(grid; margin_m=MARGIN_M)

    res = load(kf_result_filename(KF_DIR, String(TRUTH_ID)))
    α            = res["α"]              # (L × K) filter mean coeffs
    truth_coeffs = res["truth_coeffs"]   # (L × K) oracle (truth projected onto basis)
    sensor_locs  = get(res, "sensor_locs", nothing)
    rank_used    = get(res, "L", size(α, 1))

    # the KF run may have used a truncated rank r < basis.L (see RANK in
    # run_simple_KF_assimilation.jl); match the basis width to what α/truth_coeffs were
    # actually computed against, else basis.modes * α is a dimension mismatch.
    if rank_used < basis.L
        basis = PODBasis(basis.coeffs[1:rank_used, :], basis.modes[:, 1:rank_used],
                          basis.eigenvalues[1:rank_used], rank_used, basis.n_snapshots,
                          basis.grid, basis.member_ids, basis.truth_id)
        @info "fold=$TRUTH_ID  truncated basis to stored KF rank L=$rank_used"
    end

    truth_U = load_member_snapshots(truth_member; t_spinup=T_SPINUP, domain)  # (state_dim × K)
    K = size(α, 2)
    k = get(ENV, "KIDX", "") == "" ? (K ÷ 2) : parse(Int, ENV["KIDX"])   # mid-record snapshot
    @info "fold=$TRUTH_ID  K=$K  using snapshot k=$k  z_agl=$Z_AGL m"

    α0 = vec(mean(basis.coeffs, dims=2))        # ensemble-mean prior coeffs (Eq. 10)

    # reconstruct physical states (x = Prα; modes reconstruct physical units)
    recon = Dict("truth" => truth_U[:, k], "est" => basis.modes * α[:, k],
                 "oracle" => basis.modes * truth_coeffs[:, k], "prior" => basis.modes * α0)

    # unpack flat masked [u;v;w] → 3D (Nx,Ny,Nz), zeros outside domain
    fields = Dict{String,Any}()
    for (name, x) in recon
        u3, v3, w3 = unpack_snapshot(Float32.(x), grid, domain)
        fields[name] = (u3, v3, w3)
    end

    # per-(i,j) k-level nearest to Z_AGL above local topo; build 2-D slices
    topo = grid.zPos[:, :, 1]    # surface height proxy (first level)
    klev = [argmin(abs.(grid.zPos[i, j, :] .- (topo[i, j] + Z_AGL))) for i in 1:grid.Nx, j in 1:grid.Ny]
    slice2d(F) = [F[i, j, klev[i, j]] for i in 1:grid.Nx, j in 1:grid.Ny]

    outpath = joinpath(FIELDS_DIR, "kf_field_snapshot_truth_$(String(TRUTH_ID)).nc")
    isfile(outpath) && rm(outpath)
    NCDataset(outpath, "c") do ds
        defDim(ds, "xIndex", grid.Nx); defDim(ds, "yIndex", grid.Ny)
        d2 = ("xIndex", "yIndex")
        defVar(ds, "xPos", Float32, d2)[:, :] = grid.xPos[:, :, 1]
        defVar(ds, "yPos", Float32, d2)[:, :] = grid.yPos[:, :, 1]
        defVar(ds, "topo", Float32, d2)[:, :] = topo
        defVar(ds, "mask", Int8,    d2)[:, :] = Int8.(domain.mask[:, :, 1])
        for (name, (u3, v3, w3)) in fields
            spd = sqrt.(u3 .^ 2 .+ v3 .^ 2)
            defVar(ds, "$(name)_u",     Float32, d2)[:, :] = slice2d(u3)
            defVar(ds, "$(name)_v",     Float32, d2)[:, :] = slice2d(v3)
            defVar(ds, "$(name)_w",     Float32, d2)[:, :] = slice2d(w3)
            defVar(ds, "$(name)_speed", Float32, d2)[:, :] = slice2d(spd)
        end
        ds.attrib["snapshot_k"] = k
        ds.attrib["z_agl_m"]    = Z_AGL
        ds.attrib["truth_id"]   = String(TRUTH_ID)
        if sensor_locs !== nothing
            sl = sensor_locs[k]   # (3 × nm) [x,y,z]
            defDim(ds, "sensor", size(sl, 2)); defDim(ds, "xyz3", 3)
            defVar(ds, "sensor_xyz", Float32, ("sensor", "xyz3"))[:, :] = permutedims(sl)
        end
    end
    @info "wrote $outpath"
end  # render_fold

for fold in FOLDS
    try
        render_fold(fold, config)
    catch e
        @warn "fold $fold failed" exception=(e, catch_backtrace())
    end
    GC.gc()   # free the 7.5 GB basis before the next fold
end
@info "all folds done"
