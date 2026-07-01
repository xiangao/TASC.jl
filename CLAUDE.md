# TASC.jl — project notes for Claude

Julia implementation of **Time-Aware Synthetic Control (TASC)**, ported from
the author's earlier Python codebase (see `genData/SSM.py`, `genData/Sine.py`,
and the Python `Matrix` helper referenced in comments — that Python code is
not in this repo). Single-file package: all ~1240 lines live in
`src/TASC.jl`.

## What it does

TASC fits a low-rank linear Gaussian state-space model (Kalman filter + RTS
smoother, with parameters learned by EM) on a panel `Y` (`N` units x `T`
periods). Row 1 is the treated unit by default (`treated_rows` for multiple
treated units); the rest are donors. Post-treatment treated outcomes are
masked as missing, and the RTS smoother borrows cross-sectional information
from donors and temporal structure from the fitted transition `A` to infer
the treated unit's untreated counterfactual path. This is a
state-space/matrix-completion take on synthetic control, distinct from
the local `SynthDiD.jl` package (listed as a dependency, used for docs
cross-linking, not actually called from `src/TASC.jl`).

The state-space model (see `StateSpaceParams` docstring):
```
x_t = A x_{t-1} + q_t,   q_t ~ N(0, Q)
y_t = H x_t + r_t,       r_t ~ N(0, R)
```
`d` (latent dimension) should be much smaller than `min(N, T0)` — `fit_tasc`
warns if not.

## Main API

- `fit_tasc(Y; d, T0, n_em=100, n_post=0, tol=1e-4, treated_rows=[1], ...)` —
  fits on `Y[:, 1:T0]` via EM (init options `:pca` (default), `:naive`,
  `:dirichlet`). Optional `n_post` phase re-runs EM with post-period treated
  rows imputed from the fit, for models that keep adapting after treatment.
- `predict_counterfactual(model, Y)` — returns a named tuple
  `(target, donors, variance, effect, state_mean, state_covariance)`.
  `variance` is model-based pointwise variance from the smoothed latent-state
  covariance (not a bootstrap/placebo variance).
- `predict_post_intervention(model, Y)` — pure forward-propagation forecast
  using `A`, ignoring post-period donor data entirely (contrast with
  `predict_counterfactual`, which does use post-period donors via the
  Kalman filter/smoother).
- `tasc_plot(model, Y)` + `@recipe` — `RecipesBase` plotting, mirrors the
  `SynthDiD.jl` recipe style. Needs `Plots` (or another RecipesBase backend)
  loaded by the caller; `CairoMakie` is a package dependency but is not
  wired into the recipe.
- Classical synthetic control baselines: `fit_synthetic_control` (`:ols`,
  `:ridge`, `:lasso` via coordinate descent, `:simplex` via projected
  gradient) + `predict`/`score`/`predict_and_mse`.
- Matrix utilities ported from the Python `Matrix` helper: `panel_matrix`,
  `transform`/`inverse_transform`, `denoise`/`hsvt` (hard singular value
  thresholding), `get_energy`, `get_approx_rank`.
- Synthetic data generators (ported from `genData/SSM.py`, `genData/Sine.py`):
  `simulate_tasc` (the one used in tests/vignettes), plus
  `generate_model_data`, `generate_multiple_layers`, `generate_rank_k_matrix`,
  `generate_sine_dataset_A/B`, etc. — mostly for simulation studies, not
  needed for a basic TASC fit.

## EM non-convergence warnings — expected during tests

`fit_tasc` warns via `@warn "TASC EM did not converge after N iterations..."`
(`src/TASC.jl:450`) whenever `_params_delta` between EM iterations never
drops below `tol` within `n_em` (+ `n_post`) iterations. `test/runtests.jl`
deliberately caps `n_em` very low (4, 2, 20) purely to keep the suite fast —
it is not asserting convergence, just checking output shapes/sanity and
recovery of known parameters under generous tolerances. **These warnings are
benign and expected on every test run** — they are a side effect of the
tests' short EM budgets, not a sign of an algorithm bug. If you see a
non-convergence warning from real applied usage (not tests), increase `n_em`
or loosen `tol` per the warning's own suggestion; the EM loop itself
(alternating `_kalman_filter` → `_rts_smoother` → `_m_step`) is a standard
Gaussian-LDS EM and looks numerically sound (Joseph-form covariance updates,
Cholesky with jitter fallback via `_chol`, floors on `Q`/`R` diagonals).

## Tests

```bash
cd ~/projects/software/TASC.jl
julia --project=. test/runtests.jl
```

24/24 pass as of 2026-07-01 (3 testsets: "TASC fit and prediction" 5,
"Ported utilities" 10, "Bug fixes" 9), ~25-30s, with the 4 EM warnings above.
`test/runtests.jl` covers: full fit/predict round trip against the known
simulated signal, multi-treated-unit fitting + plotting, all
preprocessing/generator utilities, and a "Bug fixes" testset that pins down
previously-found bugs (simplex weights summing to 1 independent of
intercept, lasso sparse recovery, distinct params per layer in
`generate_multiple_layers`, PCA-init seed reproducibility).

## Docs

Documenter.jl site at https://xiangao.github.io/TASC.jl/, deployed by
`.github/workflows/docs.yml` on push to `master`/`main`. Source in
`docs/src/`: `index.md` (quick start, uses a live `@example` block) +
3 vignettes (`01_getting_started`, `02_multiple_treated_units`,
`03_preprocessing_baselines`) + `reference.md`. Build locally with
`julia --project=docs docs/make.jl` (output to `docs/build`, gitignored
except what's checked in already). `docs/Project.toml` also needs
`SynthDiD.jl` (unregistered — `Pkg.develop`'d from
`github.com/xiangao/SynthDiD.jl` in the workflow) even though `src/TASC.jl`
doesn't call it directly.

## CI

`.github/workflows/CI.yml` exists and runs `julia-actions/julia-runtest@v1`
on Julia 1.10 and 1 (current stable) on every push/PR — added
2026-05-23 (commits `4861ec9`, `e00ac34`, `f33a969`). It also develops the
unregistered `SynthDiD.jl` dependency before testing. So, contrary to what
you might expect for a small single-file package, test CI is already wired
up here — don't assume it's missing without checking `.github/workflows/`.
