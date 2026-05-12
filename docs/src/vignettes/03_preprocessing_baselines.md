# Preprocessing and Baselines

The package also includes preprocessing helpers and classical synthetic-control baselines ported from the Python package.

## Matrix Helpers

`PanelMatrix` stores data in the Python helper orientation, with rows as time periods and columns as units.

```@example preprocessing
using TASC
using Statistics

Y, params, signal = simulate_tasc(N = 10, T = 50, d = 2, seed = 3)
T0 = 30

M = panel_matrix(permutedims(Y), T0; target = 1)
transform(M; method = :standard)
denoise(M; num_sv = 2)
inverse_transform(M)

size(M.data), M.denoised
```

Hard singular value thresholding is also available directly:

```@example preprocessing
Y_denoised = hsvt(Y; rank = 2)
size(Y_denoised)
```

## Classical Synthetic Control

The synthetic-control baseline expects donor matrices with rows as time periods and columns as donor units.

```@example preprocessing
pre_donor = permutedims(Y[2:end, 1:T0])
pre_target = Y[1, 1:T0]

sc = fit_synthetic_control(pre_donor, pre_target; method = :simplex)

all_donor = permutedims(Y[2:end, :])
path = predict(sc, all_donor)
pre_r2 = score(sc, pre_donor, pre_target)

(length(path), round(pre_r2, digits = 3))
```

Available methods are `:ols`, `:ridge`, `:lasso`, and `:simplex`.
