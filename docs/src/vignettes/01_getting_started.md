# Getting Started

```@meta
CurrentModule = TASC
```

This vignette fits Time-Aware Synthetic Control on simulated panel data. Rows are units and columns are time periods. By default, row 1 is the treated unit and rows `2:N` are donors.

```@example getting_started
using TASC
using Statistics
using Random

Random.seed!(11)

Y, true_params, signal = simulate_tasc(N = 20, T = 80, d = 3, seed = 11)
T0 = 50

model = fit_tasc(
    Y;
    d = 3,
    T0 = T0,
    n_em = 30,
    tol = 1e-4,
)

pred = predict_counterfactual(model, Y)

post = (T0 + 1):size(Y, 2)
att = mean(pred.effect[post])
rmse = sqrt(mean((pred.target[post] .- signal[1, post]) .^ 2))

(att = round(att, digits = 3), rmse = round(rmse, digits = 3))
```

The returned prediction stores the counterfactual path, treatment-effect path, donor fitted values, and the smoothed state distribution:

```@example getting_started
keys(pred)
```

The model-based pointwise interval used in the TASC paper comes from the smoothed latent covariance:

```@example getting_started
se = sqrt.(max.(pred.variance, 0.0))
lower = pred.target .- 1.96 .* se
upper = pred.target .+ 1.96 .* se

round.((lower[end], pred.target[end], upper[end]), digits = 3)
```

To use the plotting recipe, load `Plots` and pass the wrapper returned by `tasc_plot`:

```@example getting_started
using Plots

plt = plot(
    tasc_plot(model, Y);
    ci = true,
    title = "TASC counterfactual",
    xlabel = "Time",
    ylabel = "Outcome",
)

savefig(plt, "getting-started-tasc.svg") # hide
nothing # hide
```

![](getting-started-tasc.svg)
