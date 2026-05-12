# TASC.jl

`TASC.jl` implements Time-Aware Synthetic Control for panel data with temporal dependence. It fits a low-rank linear Gaussian state-space model on the pre-treatment panel, then treats post-treatment treated outcomes as missing and uses donor outcomes to infer the untreated counterfactual path.

The package includes:

- `fit_tasc` and `predict_counterfactual` for TASC estimation.
- Model-based pointwise uncertainty from the smoothed latent-state covariance.
- Multiple treated unit support through `treated_rows`.
- A lightweight `RecipesBase` plotting recipe through `tasc_plot`.
- Classical synthetic-control baselines and matrix preprocessing utilities ported from the Python package.
- Synthetic data generators for simulation studies.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/xiangao/TASC.jl")
```

For local development:

```julia
using Pkg
Pkg.develop(path = "/home/xao/projects/software/TASC.jl")
```

## Quick Start

```@example quickstart
using TASC
using Statistics
using Random

Random.seed!(123)

Y, params, signal = simulate_tasc(N = 16, T = 60, d = 3, seed = 123)
T0 = 35

model = fit_tasc(Y; d = 3, T0 = T0, n_em = 20, tol = 1e-4)
pred = predict_counterfactual(model, Y)

att = mean(pred.effect[(T0 + 1):end])
ci_lower = pred.target .- 1.96 .* sqrt.(max.(pred.variance, 0.0))
ci_upper = pred.target .+ 1.96 .* sqrt.(max.(pred.variance, 0.0))

round(att, digits = 3)
```

See the vignettes for a complete workflow.
