# test_kalman.jl — unit tests for src/kalman.jl (pure linear algebra, no data needed).
#   julia --startup-file=no test/test_kalman.jl
#
# Checks that (1) the gain and information/Woodbury update forms agree to round-off,
# (2) the filter is consistent (mean NIS ≈ 1) on data simulated from its own model,
# (3) the doubling Riccati solver matches the filter's converged posterior covariance, and
# (4) rank truncation with tail clutter yields a valid StaticObs.
using Test, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "src", "kalman.jl"))

@testset "kalman.jl" begin
    Random.seed!(1)
    r, m, K = 6, 40, 400
    A = randn(r, r); Σ0 = A * A' + I
    B = randn(r, r); Q  = 1e-2 * (B * B') + 1e-4I
    H = randn(m, r); C = randn(m, m); R = 0.25I + 0.05 * (C * C') / m
    model = KFModel(zeros(r), Σ0, Q)
    obs   = StaticObs(H, R)

    # simulate the random-walk state-space model the filter assumes
    α = cholesky(Σ0).L * randn(r)
    Y = zeros(m, K)
    LQ, LR = cholesky(Symmetric(Q)).L, cholesky(Symmetric(R)).L
    for k in 1:K
        α = α + LQ * randn(r)
        Y[:, k] = H * α + LR * randn(m)
    end

    hg = run_filter(model, Y, obs; form=:gain)
    hi = run_filter(model, Y, obs; form=:information)
    @test hi.form === :information
    @test run_filter(model, Y, obs).form === :information          # :auto with m > r
    @test maximum(abs, hg.α_post - hi.α_post) < 1e-10
    @test maximum(abs, hg.P_post - hi.P_post) < 1e-10
    @test maximum(abs, hg.chi2 - hi.chi2) < 1e-8
    @test 0.9 < mean_nis(hg) < 1.1

    P∞, its = riccati_steady_state(model, obs)
    @test its < 100
    @test maximum(abs, P∞ - hg.P_post[:, :, end]) < 1e-10
    ρ, τ = closed_loop_decay(model, obs, P∞)
    @test 0 < ρ < 1 && isfinite(τ)

    # rank truncation: leading block + PSD tail clutter folded into R
    tm = truncate_model(zeros(r), Σ0, Q, 3; qscale=2.0)
    @test state_rank(tm) == 3 && tm.Q ≈ 2 .* Q[1:3, 1:3]
    to = truncated_obs(H, R, Σ0, 3)
    @test size(to.H) == (m, 3)
    @test to.R ≈ R + H[:, 4:r] * Σ0[4:r, 4:r] * H[:, 4:r]'
    @test tail_clutter(H, Σ0, r) == zeros(m, m)

    # unobservable directions are detected, and the steady-state solver refuses them
    thin = StaticObs(H[1:2, :], R[1:2, 1:2])
    @test n_unobservable(thin) == r - 2
    @test_throws ErrorException riccati_steady_state(model, thin)
end
