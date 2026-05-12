using Test
using TASC
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
