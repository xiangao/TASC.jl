# TASC.jl

`TASC.jl` implements Time-Aware Synthetic Control for panel data. It fits a low-rank linear Gaussian state-space model on pre-treatment outcomes, masks treated outcomes after treatment, and uses Kalman filtering plus Rauch-Tung-Striebel smoothing to estimate untreated counterfactual paths.

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

```julia
using TASC
using Statistics

Y, params, signal = simulate_tasc(N = 20, T = 80, d = 3, seed = 11)
T0 = 50

model = fit_tasc(Y; d = 3, T0 = T0, n_em = 30, tol = 1e-4)
pred = predict_counterfactual(model, Y)

att = mean(pred.effect[(T0 + 1):end])
```

`pred.target` is the TASC counterfactual path for the treated unit. `pred.effect` is the observed treated outcome minus that counterfactual. `pred.variance` is the model-based pointwise variance from the smoothed latent-state covariance.

## Documentation

Documentation and vignettes are published at:

https://xiangao.github.io/TASC.jl/

To build them locally:

```bash
julia --project=docs docs/make.jl
```

The generated docs are written to `docs/build`.
