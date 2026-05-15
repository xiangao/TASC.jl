using Test
using TASC
using LinearAlgebra
using Random
using Statistics

@testset "TASC fit and prediction" begin
    Y, _, signal = simulate_tasc(N=12, T=50, d=3, seed=11)
    T0 = 30
    model = fit_tasc(Y; d=3, T0=T0, n_em=20, tol=1e-5)
    pred = predict_counterfactual(model, Y)

    @test length(pred.target) == size(Y, 2)
    @test size(pred.donors) == (size(Y, 1) - 1, size(Y, 2))
    @test all(isfinite, pred.target)
    @test all(>=(0), pred.variance)

    naive = fill(mean(Y[1, 1:T0]), size(Y, 2) - T0)
    @test sqrt(mean((pred.target[(T0 + 1):end] - signal[1, (T0 + 1):end]) .^ 2)) <
          5 * sqrt(mean((naive - signal[1, (T0 + 1):end]) .^ 2))
end

@testset "Ported utilities" begin
    Y, _, _ = simulate_tasc(N=8, T=35, d=2, seed=3)
    model = fit_tasc(Y; d=2, T0=20, n_em=3, n_post=1, treated_rows=[1, 2])
    pred = predict_counterfactual(model, Y)
    plt = tasc_plot(model, Y)

    @test size(pred.target) == (2, 35)
    @test size(pred.variance) == (2, 35)
    @test plt isa TASCPlot
    @test length(predict_post_intervention(model, Y)) == 30

    sc = fit_synthetic_control(permutedims(Y[2:end, 1:20]), Y[1, 1:20]; method=:ridge, lambda=0.1)
    @test length(predict(sc, permutedims(Y[2:end, :]))) == 35
    @test isfinite(score(sc, permutedims(Y[2:end, 1:20]), Y[1, 1:20]))

    M = panel_matrix(permutedims(Y), 20; target=1)
    transform(M)
    denoise(M; num_sv=2)
    inverse_transform(M)
    @test size(hsvt(Y; rank=2)) == size(Y)

    theta = gen_dirichlet_params(d=2, N=5, seed=1)
    @test size(generate_model_data(theta; T=10, seed=1)) == (5, 10)
    @test size(generate_rank_k_matrix(4, 10, 2, 0.1)) == (10, 4)
    @test size(generate_sine_dataset_A(5, 20, 0.1, 2)) == (20, 5)
end

@testset "Bug fixes" begin
    # Simplex with fit_intercept: weights should sum to 1 independently of intercept
    rng = MersenneTwister(99)
    X = randn(rng, 30, 4)
    y = X[:, 1] .* 0.5 .+ X[:, 2] .* 0.5 .+ 2.0 .+ 0.01 .* randn(rng, 30)
    sc_simplex = fit_synthetic_control(X[1:20, :], y[1:20]; method=:simplex, fit_intercept=true)
    @test isapprox(sum(sc_simplex.coef), 1.0; atol=1e-4)  # donor weights sum to 1
    @test all(sc_simplex.coef .>= -1e-6)                   # weights non-negative
    @test isapprox(sc_simplex.intercept, 2.0; atol=0.2)    # intercept is free, near true 2.0

    # Lasso correctness: sparse signal recovery
    rng2 = MersenneTwister(7)
    X2 = randn(rng2, 50, 10)
    true_coef = zeros(10); true_coef[1] = 1.5; true_coef[3] = -0.8
    y2 = X2 * true_coef .+ 0.05 .* randn(rng2, 50)
    sc_lasso = fit_synthetic_control(X2[1:40, :], y2[1:40]; method=:lasso, lambda=0.05, fit_intercept=false)
    @test isapprox(sc_lasso.coef[1], true_coef[1]; atol=0.2)
    @test isapprox(sc_lasso.coef[3], true_coef[3]; atol=0.2)

    # generate_multiple_layers: layers should have distinct parameters
    sigs, obs = generate_multiple_layers(d_true=2, N=5, T=20, k=3, seed=1)
    @test size(obs) == (3, 20, 5)
    # Each layer's observations should differ (distinct seeds → distinct params)
    @test !isapprox(obs[1, :, :], obs[2, :, :]; atol=1e-10)
    @test !isapprox(obs[2, :, :], obs[3, :, :]; atol=1e-10)

    # PCA initializer reproducibility: same seed → same result
    Y3, _, _ = simulate_tasc(N=6, T=20, d=2, seed=5)
    m1 = fit_tasc(Y3; d=3, T0=15, n_em=2, seed=42)
    m2 = fit_tasc(Y3; d=3, T0=15, n_em=2, seed=42)
    @test m1.params.H == m2.params.H
end
