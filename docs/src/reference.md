# Reference

```@meta
CurrentModule = TASC
```

## TASC Estimation

```@docs
StateSpaceParams
TASCResult
fit_tasc
predict_counterfactual
predict_post_intervention
TASCPlot
```

## Preprocessing

Core helpers:

- `panel_matrix(data, T0; target = 1, donors = nothing)`
- `transform(M; method = :standard)`
- `inverse_transform(M)`
- `hsvt(X; rank = 2, p = 1.0)`
- `denoise(M; num_sv, p = 1.0, filter_method = :HSVT, do_transform = false)`
- `get_energy(s)`
- `get_approx_rank(s; threshold = 0.95)`

## Synthetic Control Baselines

Core helpers:

- `fit_synthetic_control(pre_donor, pre_target; method = :ols, lambda = nothing, fit_intercept = true)`
- `predict(model, donor)`
- `score(model, donor, target)`
- `predict_and_mse(model, donor, target_true)`

## Simulation

```@docs
simulate_tasc
```

Additional generators:

- `gen_A`, `gen_H`, `gen_cov`, and `gen_dirichlet_params`
- `generate_model_data` and `generate_multiple_layers`
- `generate_rank_1_matrix` and `generate_rank_k_matrix`
- `generate_sine_wave`, `generate_linear_dataset`, `generate_new_sine_dataset`, `generate_sine_dataset_A`, and `generate_sine_dataset_B`
- `make_approx_low_rank`
