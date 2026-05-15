module TASC

using LinearAlgebra
using Random
using RecipesBase
using Statistics

export StateSpaceParams,
    TASCResult,
    TASCPlot,
    SyntheticControlModel,
    PanelMatrix,
    fit_tasc,
    predict_counterfactual,
    predict_post_intervention,
    tasc_plot,
    fit_synthetic_control,
    predict,
    score,
    predict_and_mse,
    panel_matrix,
    transform,
    inverse_transform,
    denoise,
    hsvt,
    get_energy,
    get_approx_rank,
    unif_sphere,
    laplace_sample,
    mse,
    gen_cov,
    gen_A,
    gen_H,
    gen_dirichlet_params,
    generate_model_data,
    generate_multiple_layers,
    data_flatten,
    generate_rank_1_matrix,
    generate_rank_k_matrix,
    generate_sine_wave,
    zero_out_fraction,
    generate_linear_dataset,
    generate_new_sine_dataset,
    generate_sine_dataset_A,
    generate_sine_dataset_B,
    make_approx_low_rank,
    simulate_tasc

"""
    StateSpaceParams(A, H, Q, R, m0, P0)

Parameters for the linear Gaussian state-space model

    x_t = A * x_{t-1} + q_t,  q_t ~ N(0, Q)
    y_t = H * x_t + r_t,      r_t ~ N(0, R)

where rows of `Y` are units and columns are time periods.
"""
struct StateSpaceParams
    A::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    m0::Vector{Float64}
    P0::Matrix{Float64}
end

"""
    TASCResult

Fitted Time-Aware Synthetic Control model. The treated unit is assumed to be
the first row of the input matrix.
"""
struct TASCResult
    params::StateSpaceParams
    T0::Int
    d::Int
    treated_rows::Vector{Int}
    iterations::Int
    converged::Bool
    loglikelihood::Float64
end

struct FilterResult
    m_filt::Matrix{Float64}
    P_filt::Array{Float64,3}
    m_pred::Matrix{Float64}
    P_pred::Array{Float64,3}
    loglikelihood::Float64
end

struct SmoothResult
    m_smooth::Matrix{Float64}
    P_smooth::Array{Float64,3}
    G::Array{Float64,3}
end

_sym(M) = Matrix(Symmetric((M + M') / 2))

function _chol(M::AbstractMatrix{<:Real}; jitter::Float64=1e-8)
    S = _sym(Matrix{Float64}(M))
    I0 = Matrix{Float64}(I, size(S, 1), size(S, 2))
    for scale in (0.0, jitter, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
        try
            return cholesky(Symmetric(S + scale * I0))
        catch err
            err isa PosDefException || rethrow()
        end
    end
    vals, vecs = eigen(Symmetric(S))
    vals = max.(vals, jitter)
    return cholesky(Symmetric(vecs * Diagonal(vals) * vecs'))
end

function _cov_from_samples(X::AbstractMatrix{<:Real}; floor::Float64=1e-6)
    n = size(X, 2)
    if n <= 1
        return Matrix{Float64}(I, size(X, 1), size(X, 1)) .* floor
    end
    Xc = X .- mean(X; dims=2)
    return _sym((Xc * Xc') / (n - 1) + floor * I)
end

function _pca_initialize(Y::AbstractMatrix{<:Real}, d::Int, rng::AbstractRNG)
    N, T = size(Y)
    d <= 0 && throw(ArgumentError("d must be positive"))

    F = svd(Matrix{Float64}(Y); full=false)
    rank = length(F.S)
    k = min(d, rank)

    H = zeros(Float64, N, d)
    X = zeros(Float64, d, T)
    if k > 0
        H[:, 1:k] = F.U[:, 1:k] * Diagonal(F.S[1:k])
        X[1:k, :] = F.Vt[1:k, :]
    end
    if d > k
        H[:, (k + 1):d] .= 0.01 .* randn(rng, N, d - k)
        X[(k + 1):d, :] .= 0.01 .* randn(rng, d - k, T)
    end

    ridge = 1e-6
    if T > 1
        X0 = X[:, 1:(T - 1)]
        X1 = X[:, 2:T]
        A = X1 * X0' / _sym(X0 * X0' + ridge * I)
        transition_resid = X1 - A * X0
        Q = _cov_from_samples(transition_resid; floor=1e-4)
    else
        A = Matrix{Float64}(I, d, d)
        Q = Matrix{Float64}(I, d, d) .* 0.1
    end

    residual = Matrix{Float64}(Y) - H * X
    rdiag = vec(mean(residual .^ 2; dims=2))
    scale = mean(abs2, Y)
    rfloor = max(1e-4, 1e-4 * scale)
    R = Diagonal(max.(rdiag, rfloor)) |> Matrix

    m0 = X[:, 1]
    P0 = Matrix{Float64}(I, d, d)
    return StateSpaceParams(Matrix{Float64}(A), H, _sym(Q), _sym(R), m0, _sym(P0))
end

function _initialize_params(
    Y::AbstractMatrix{<:Real},
    d::Int;
    init::Symbol=:pca,
    q_diag::Bool=true,
    r_diag::Bool=true,
    q_zero::Bool=false,
    a_method::Symbol=:qr,
    h_method::Symbol=:dirichlet,
    seed::Union{Nothing,Int}=nothing,
)
    N = size(Y, 1)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    if init == :pca
        return _pca_initialize(Y, d, rng)
    elseif init == :naive
        A = Matrix{Float64}(I, d, d)
        H = randn(rng, N, d)
        Q = Matrix{Float64}(I, d, d) .* 0.1
        R = Matrix{Float64}(I, N, N) .* 0.1
        m0 = zeros(Float64, d)
        P0 = Matrix{Float64}(I, d, d)
        return StateSpaceParams(A, H, Q, R, m0, P0)
    elseif init == :dirichlet
        return gen_dirichlet_params(
            d=d,
            N=N,
            Q_diag=q_diag,
            R_diag=r_diag,
            Q_zero=q_zero,
            A_method=a_method,
            H_method=h_method,
            seed=seed,
        )
    else
        throw(ArgumentError("init must be :pca, :naive, or :dirichlet"))
    end
end

function _post_observed_rows(N::Int, treated_rows::AbstractVector{<:Integer})
    treated = sort(unique(Int.(treated_rows)))
    all(1 .<= treated .<= N) || throw(ArgumentError("treated_rows must be valid row indices"))
    observed = setdiff(collect(1:N), treated)
    isempty(observed) && throw(ArgumentError("at least one donor row is required after T0"))
    return observed
end

function _kalman_filter(
    params::StateSpaceParams,
    Y::AbstractMatrix{<:Real};
    T_use::Int=size(Y, 2),
    T0::Int=T_use,
    mask_post_target::Bool=false,
    treated_rows::AbstractVector{<:Integer}=[1],
)
    A, H, Q, R, m0, P0 = params.A, params.H, params.Q, params.R, params.m0, params.P0
    N, T = size(Y)
    d = length(m0)
    T_use <= T || throw(ArgumentError("T_use cannot exceed number of columns in Y"))

    m_filt = zeros(Float64, d, T_use + 1)
    P_filt = zeros(Float64, d, d, T_use + 1)
    m_pred = zeros(Float64, d, T_use)
    P_pred = zeros(Float64, d, d, T_use)
    m_filt[:, 1] = m0
    P_filt[:, :, 1] = P0

    ll = 0.0
    log2pi = log(2pi)
    obs_all  = collect(1:N)
    obs_post = mask_post_target ? _post_observed_rows(N, treated_rows) : obs_all

    for t in 1:T_use
        mp = A * m_filt[:, t]
        Pp = _sym(A * P_filt[:, :, t] * A' + Q)
        m_pred[:, t] = mp
        P_pred[:, :, t] = Pp

        obs = (mask_post_target && t > T0) ? obs_post : obs_all
        yt = Vector{Float64}(Y[obs, t])
        Ht = H[obs, :]
        Rt = R[obs, obs]

        innovation = yt - Ht * mp
        S = _sym(Ht * Pp * Ht' + Rt)
        cholS = _chol(S)
        K = (cholS \ (Ht * Pp))'
        m_filt[:, t + 1] = mp + K * innovation
        P_filt[:, :, t + 1] = _sym(Pp - K * S * K')

        solved = cholS \ innovation
        ll -= 0.5 * (length(obs) * log2pi + logdet(cholS) + dot(innovation, solved))
    end

    return FilterResult(m_filt, P_filt, m_pred, P_pred, ll)
end

function _rts_smoother(params::StateSpaceParams, filter::FilterResult)
    A, Q = params.A, params.Q
    d, T1 = size(filter.m_filt)
    T = T1 - 1

    m_smooth = copy(filter.m_filt)
    P_smooth = copy(filter.P_filt)
    G = zeros(Float64, d, d, T)

    for s in T:-1:1
        Pp = _sym(A * filter.P_filt[:, :, s] * A' + Q)
        cholPp = _chol(Pp)
        Gs = (cholPp \ (A * filter.P_filt[:, :, s]))'
        G[:, :, s] = Gs
        m_smooth[:, s] = filter.m_filt[:, s] + Gs * (m_smooth[:, s + 1] - A * filter.m_filt[:, s])
        P_smooth[:, :, s] = _sym(filter.P_filt[:, :, s] + Gs * (P_smooth[:, :, s + 1] - Pp) * Gs')
    end

    return SmoothResult(m_smooth, P_smooth, G)
end

function _m_step(
    Y::AbstractMatrix{<:Real},
    smooth::SmoothResult;
    q_diag::Bool=true,
    r_diag::Bool=true,
    floor::Float64=1e-6,
)
    N, T = size(Y)
    d = size(smooth.m_smooth, 1)

    Sigma = zeros(Float64, d, d)
    Phi = zeros(Float64, d, d)
    B = zeros(Float64, N, d)
    C = zeros(Float64, d, d)
    D = zeros(Float64, N, N)

    for t in 1:T
        mt = smooth.m_smooth[:, t + 1]
        Pt = smooth.P_smooth[:, :, t + 1]
        mprev = smooth.m_smooth[:, t]
        Pprev = smooth.P_smooth[:, :, t]
        yt = Vector{Float64}(Y[:, t])

        Sigma += Pt + mt * mt'
        Phi += Pprev + mprev * mprev'
        B += yt * mt'
        C += Pt * smooth.G[:, :, t]' + mt * mprev'
        D += yt * yt'
    end

    Sigma = _sym(Sigma / T + floor * I)
    Phi = _sym(Phi / T + floor * I)
    B ./= T
    C ./= T
    D = _sym(D / T + floor * I)

    A = C / Phi
    H = B / Sigma
    Q = _sym(Sigma - C * A' - A * C' + A * Phi * A')
    R = _sym(D - B * H' - H * B' + H * Sigma * H')

    if q_diag
        Q = Matrix(Diagonal(max.(diag(Q), floor)))
    else
        Q += floor * I
    end
    if r_diag
        R = Matrix(Diagonal(max.(diag(R), floor)))
    else
        R += floor * I
    end

    m0 = copy(smooth.m_smooth[:, 1])
    P0 = _sym(smooth.P_smooth[:, :, 1] + floor * I)
    return StateSpaceParams(Matrix{Float64}(A), H, _sym(Q), _sym(R), m0, P0)
end

function _params_delta(a::StateSpaceParams, b::StateSpaceParams)
    return maximum((
        norm(a.A - b.A),
        norm(a.H - b.H),
        norm(a.Q - b.Q),
        norm(a.R - b.R),
        norm(a.m0 - b.m0),
        norm(a.P0 - b.P0),
    ))
end

"""
    fit_tasc(Y; d, T0, n_em=100, n_post=0, tol=1e-4, q_diag=true, r_diag=true)

Fit Time-Aware Synthetic Control on the pre-intervention columns `1:T0`.
`Y` must be an `N x T` matrix. By default, row 1 is treated and rows `2:N`
are donors. Post-intervention treated entries are ignored during counterfactual
prediction. Pass `treated_rows` to mask multiple treated units after `T0`.
"""
function fit_tasc(
    Y::AbstractMatrix{<:Real};
    d::Int,
    T0::Int,
    n_em::Int=100,
    n_post::Int=0,
    tol::Float64=1e-4,
    q_diag::Bool=true,
    r_diag::Bool=true,
    learn_q::Bool=true,
    learn_r::Bool=true,
    require_thresh::Bool=false,
    thresh::Float64=1e-4,
    init::Symbol=:pca,
    q_zero::Bool=false,
    a_method::Symbol=:qr,
    h_method::Symbol=:dirichlet,
    seed::Union{Nothing,Int}=nothing,
    treated_rows::AbstractVector{<:Integer}=[1],
)
    N, T = size(Y)
    1 <= T0 <= T || throw(ArgumentError("T0 must be between 1 and size(Y, 2)"))
    d < min(N, T0) || @warn "TASC is intended for d much smaller than min(N, T0)"
    _post_observed_rows(N, treated_rows)

    Ypre = Matrix{Float64}(Y[:, 1:T0])
    params = _initialize_params(
        Ypre,
        d;
        init=init,
        q_diag=q_diag,
        r_diag=r_diag,
        q_zero=q_zero,
        a_method=a_method,
        h_method=h_method,
        seed=seed,
    )
    iterations = 0
    converged = false
    ll = -Inf

    for iter in 1:n_em
        filter = _kalman_filter(params, Ypre; T_use=T0, T0=T0, mask_post_target=false)
        smooth = _rts_smoother(params, filter)
        newparams = _m_step(Ypre, smooth; q_diag=q_diag, r_diag=r_diag, floor=require_thresh ? thresh : 1e-6)
        if !learn_q
            newparams = StateSpaceParams(newparams.A, newparams.H, params.Q, newparams.R, newparams.m0, newparams.P0)
        end
        if !learn_r
            newparams = StateSpaceParams(newparams.A, newparams.H, newparams.Q, params.R, newparams.m0, newparams.P0)
        end
        delta = _params_delta(newparams, params)
        params = newparams
        iterations = iter
        ll = filter.loglikelihood
        if delta <= tol
            converged = true
            break
        end
    end

    Yfull = Matrix{Float64}(Y)
    if n_post > 0
        # Reset convergence: the post-intervention loop is a separate phase;
        # pre-EM convergence does not imply post-EM convergence.
        converged = false
        for iter in 1:n_post
            filter = _kalman_filter(params, Yfull; T_use=T, T0=T0, mask_post_target=true, treated_rows=treated_rows)
            smooth = _rts_smoother(params, filter)
            fitted = params.H * smooth.m_smooth[:, 2:(T + 1)]
            Yaug = copy(Yfull)
            Yaug[Int.(treated_rows), (T0 + 1):T] .= fitted[Int.(treated_rows), (T0 + 1):T]
            newparams = _m_step(Yaug, smooth; q_diag=q_diag, r_diag=r_diag, floor=require_thresh ? thresh : 1e-6)
            if !learn_q
                newparams = StateSpaceParams(newparams.A, newparams.H, params.Q, newparams.R, newparams.m0, newparams.P0)
            end
            if !learn_r
                newparams = StateSpaceParams(newparams.A, newparams.H, newparams.Q, params.R, newparams.m0, newparams.P0)
            end
            delta = _params_delta(newparams, params)
            params = newparams
            iterations += 1
            ll = filter.loglikelihood
            if delta <= tol
                converged = true
                break
            end
        end
    end

    converged || @warn "TASC EM did not converge after $iterations iterations; try increasing n_em$(n_post > 0 ? "/n_post" : "") or loosening tol."

    final_filter = _kalman_filter(params, Ypre; T_use=T0, T0=T0, mask_post_target=false)
    return TASCResult(params, T0, d, sort(unique(Int.(treated_rows))), iterations, converged, final_filter.loglikelihood)
end

"""
    predict_counterfactual(model, Y)

Use a fitted `TASCResult` to estimate the treated unit's untreated
counterfactual path. The filter uses all units before `T0`; after `T0`, it
uses only donor rows, then an RTS smoother borrows information across time.

Returns a named tuple with `target`, `donors`, `variance`, `effect`,
`state_mean`, and `state_covariance`.
"""
function predict_counterfactual(model::TASCResult, Y::AbstractMatrix{<:Real})
    params = model.params
    N, T = size(Y)
    size(params.H, 1) == N || throw(ArgumentError("Y has a different number of rows than the fitted model"))

    filter = _kalman_filter(params, Y; T_use=T, T0=model.T0, mask_post_target=true, treated_rows=model.treated_rows)
    smooth = _rts_smoother(params, filter)
    fitted = params.H * smooth.m_smooth[:, 2:(T + 1)]

    variances = zeros(Float64, length(model.treated_rows), T)
    for (j, row) in pairs(model.treated_rows)
        h = vec(params.H[row, :])
        variances[j, :] = [dot(h, smooth.P_smooth[:, :, t + 1] * h) for t in 1:T]
    end
    target = fitted[model.treated_rows, :]
    effect = Matrix{Float64}(Y[model.treated_rows, :]) - target

    return (
        target=length(model.treated_rows) == 1 ? vec(target) : target,
        donors=fitted[_post_observed_rows(N, model.treated_rows), :],
        variance=length(model.treated_rows) == 1 ? vec(variances) : variances,
        effect=effect,
        state_mean=smooth.m_smooth,
        state_covariance=smooth.P_smooth,
    )
end

"""
    predict_post_intervention(model, Y)

Forecast the treated path after `T0` by propagating the smoothed state at `T0`
forward with `A`, without using post-intervention donor observations.
"""
function predict_post_intervention(model::TASCResult, Y::AbstractMatrix{<:Real})
    params = model.params
    prefilter = _kalman_filter(params, Y[:, 1:model.T0]; T_use=model.T0, T0=model.T0)
    presmooth = _rts_smoother(params, prefilter)
    T = size(Y, 2)
    x = params.A * presmooth.m_smooth[:, model.T0 + 1]
    out = zeros(Float64, length(model.treated_rows), T - model.T0)
    for j in 1:(T - model.T0)
        j > 1 && (x = params.A * x)
        out[:, j] = params.H[model.treated_rows, :] * x
    end
    return length(model.treated_rows) == 1 ? vec(out) : out
end

"""
    TASCPlot

Lightweight plotting wrapper for `RecipesBase`/`Plots.jl`. Construct with
`tasc_plot(model, Y)` and render with `plot(...)` after loading `Plots`.
"""
struct TASCPlot
    model::TASCResult
    Y::Matrix{Float64}
    prediction::NamedTuple
end

function tasc_plot(model::TASCResult, Y::AbstractMatrix{<:Real})
    return TASCPlot(model, Matrix{Float64}(Y), predict_counterfactual(model, Y))
end

function _as_treated_matrix(x)
    x isa AbstractVector && return reshape(collect(Float64, x), 1, :)
    return Matrix{Float64}(x)
end

"""
    plot(tasc_plot(model, Y); show_effect=false, ci=true, ci_level=1.96)

Plot observed treated outcomes against the TASC counterfactual trajectory.
The recipe follows the lightweight `RecipesBase` style used in the local
`SynthDiD.jl` package.
"""
@recipe function f(tp::TASCPlot; show_effect=false, ci=true, ci_level=1.96)
    model = tp.model
    Y = tp.Y
    pred = tp.prediction
    T = size(Y, 2)
    times = collect(1:T)
    target_rows = model.treated_rows

    observed = vec(mean(Y[target_rows, :]; dims=1))
    counterfactual = vec(mean(_as_treated_matrix(pred.target); dims=1))
    effect = observed - counterfactual

    layout --> (show_effect ? (2, 1) : (1, 1))
    size --> (760, show_effect ? 560 : 380)
    link --> :x

    @series begin
        subplot := 1
        title --> "TASC Counterfactual"
        xlabel --> (show_effect ? "" : "Period")
        ylabel --> "Outcome"
        label := "Observed treated"
        color := :firebrick
        linewidth := 2
        times, observed
    end

    @series begin
        subplot := 1
        label := "Counterfactual"
        color := :steelblue
        linewidth := 2
        times, counterfactual
    end

    if ci && length(target_rows) == 1
        se = sqrt.(max.(vec(pred.variance), 0.0))
        @series begin
            subplot := 1
            seriestype := :path
            fillrange := counterfactual .- ci_level .* se
            fillalpha := 0.18
            linewidth := 0
            label := "Interval"
            color := :steelblue
            times, counterfactual .+ ci_level .* se
        end
    end

    @series begin
        subplot := 1
        seriestype := :vline
        label := ""
        color := :gray40
        linestyle := :dash
        linewidth := 1
        [model.T0 + 0.5]
    end

    if show_effect
        @series begin
            subplot := 2
            xlabel --> "Period"
            ylabel --> "Effect"
            label := "Observed - counterfactual"
            color := :black
            linewidth := 2
            times, effect
        end

        @series begin
            subplot := 2
            seriestype := :hline
            label := ""
            color := :gray40
            linestyle := :dash
            linewidth := 1
            [0.0]
        end

        @series begin
            subplot := 2
            seriestype := :vline
            label := ""
            color := :gray40
            linestyle := :dash
            linewidth := 1
            [model.T0 + 0.5]
        end
    end
end

# ---------------------------------------------------------------------------
# Matrix preprocessing and robust synthetic-control utilities

mutable struct PanelMatrix
    data::Matrix{Float64}          # T x N, matching the Python Matrix helper
    T0::Int
    target::Int
    donors::Vector{Int}
    transformed::Bool
    denoised::Bool
    transform_method::Symbol
    shift::Vector{Float64}
    scale::Vector{Float64}
end

function panel_matrix(data::AbstractMatrix{<:Real}, T0::Int; target::Int=1, donors=nothing)
    T, N = size(data)
    1 <= T0 <= T || throw(ArgumentError("T0 must be between 1 and number of rows"))
    1 <= target <= N || throw(ArgumentError("target must be a valid column index"))
    donor_cols = donors === nothing ? setdiff(collect(1:N), [target]) : Int.(donors)
    return PanelMatrix(Matrix{Float64}(data), T0, target, donor_cols, false, false, :none, zeros(T), ones(T))
end

pre_target(M::PanelMatrix) = M.data[1:M.T0, M.target:M.target]
post_target(M::PanelMatrix) = M.data[(M.T0 + 1):end, M.target:M.target]
pre_donor(M::PanelMatrix) = M.data[1:M.T0, M.donors]
post_donor(M::PanelMatrix) = M.data[(M.T0 + 1):end, M.donors]
target(M::PanelMatrix) = M.data[:, M.target:M.target]
donor(M::PanelMatrix) = M.data[:, M.donors]

function transform(M::PanelMatrix; method::Symbol=:standard)
    M.transformed && throw(ArgumentError("data are already transformed"))
    donors = M.data[:, M.donors]
    if method == :standard
        shift = vec(mean(donors; dims=2))
        scale = vec(std(donors; dims=2, corrected=true))
        scale[scale .<= eps(Float64)] .= 1.0
    elseif method == :minmax
        mins = vec(minimum(donors; dims=2))
        maxs = vec(maximum(donors; dims=2))
        shift = mins
        scale = maxs - mins
        scale[scale .<= eps(Float64)] .= 1.0
    else
        throw(ArgumentError("method must be :standard or :minmax"))
    end
    M.data .= (M.data .- shift) ./ scale
    M.transformed = true
    M.transform_method = method
    M.shift = shift
    M.scale = scale
    return M
end

function inverse_transform(M::PanelMatrix)
    M.transformed || throw(ArgumentError("data are not transformed"))
    M.data .= M.data .* M.scale .+ M.shift
    M.transformed = false
    M.transform_method = :none
    return M
end

function hsvt(X::AbstractMatrix{<:Real}; rank::Int=2, p::Float64=1.0)
    p > 0 || throw(ArgumentError("p must be greater than 0"))
    F = svd(Matrix{Float64}(X); full=false)
    s = copy(F.S)
    rank < length(s) && (s[(rank + 1):end] .= 0.0)
    vals = F.U * Diagonal(s) * F.Vt
    p < 1 && (vals ./= p)
    return vals
end

function denoise(M::PanelMatrix; num_sv::Int, p::Float64=1.0, filter_method::Symbol=:HSVT, do_transform::Bool=false)
    do_transform && transform(M)
    filter_method == :HSVT || throw(ArgumentError("only filter_method=:HSVT is implemented"))
    M.data[:, M.donors] .= hsvt(M.data[:, M.donors]; rank=num_sv, p=p)
    M.denoised = true
    return M
end

function get_energy(s::AbstractVector{<:Real})
    s2 = Float64.(s) .^ 2
    return cumsum(s2) ./ sum(s2)
end

function get_approx_rank(s::AbstractVector{<:Real}; threshold::Float64=0.95)
    cumulative = cumsum(Float64.(s)) ./ sum(s)
    idx = findfirst(>(threshold), cumulative)
    return idx === nothing ? length(s) : idx
end

function unif_sphere(d::Int, n::Int; seed::Union{Nothing,Int}=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    X = randn(rng, n, d)
    norms = sqrt.(sum(abs2, X; dims=2))
    return X ./ norms
end

function laplace_sample(d::Int, n::Int, b::Real; seed::Union{Nothing,Int}=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    radius = [rand(rng, Bool) ? randexp(rng) * b : -randexp(rng) * b for _ in 1:n]
    return unif_sphere(d, n; seed=seed) .* reshape(radius, n, 1)
end

mse(truth::Real, preds::AbstractVector{<:Real}) = mean((truth .- preds) .^ 2)

# ---------------------------------------------------------------------------
# Classical synthetic-control baselines

struct SyntheticControlModel
    method::Symbol
    coef::Vector{Float64}
    intercept::Float64
    fit_intercept::Bool
    feature_names::Vector{Int}
end

function _add_intercept(X::Matrix{Float64})
    return hcat(ones(size(X, 1)), X)
end

function _soft_threshold(x::Float64, λ::Float64)
    x > λ && return x - λ
    x < -λ && return x + λ
    return 0.0
end

function _lasso_cd(X::Matrix{Float64}, y::Vector{Float64}, λ::Float64; maxiter::Int=10_000, tol::Float64=1e-8)
    p = size(X, 2)
    β = zeros(Float64, p)
    xnorm = vec(sum(abs2, X; dims=1))
    r = copy(y)  # running residual: y - X*β; starts as y since β = 0
    for _ in 1:maxiter
        max_change = 0.0
        for j in 1:p
            r .+= X[:, j] .* β[j]        # restore partial residual (exclude col j)
            ρ = dot(X[:, j], r)
            β_new = xnorm[j] <= eps(Float64) ? 0.0 : _soft_threshold(ρ, λ) / xnorm[j]
            max_change = max(max_change, abs(β_new - β[j]))
            r .-= X[:, j] .* β_new        # update residual with new β[j]
            β[j] = β_new
        end
        max_change <= tol * (1 + norm(β)) && break
    end
    return β
end

function _project_simplex(v::AbstractVector{<:Real})
    u = sort(Float64.(v), rev=true)
    cssv = cumsum(u) .- 1
    ρ = findlast(i -> u[i] - cssv[i] / i > 0, eachindex(u))
    θ = cssv[ρ] / ρ
    return max.(Float64.(v) .- θ, 0.0)
end

function _simplex_regression(X::Matrix{Float64}, y::Vector{Float64}; fit_intercept::Bool=true, maxiter::Int=20_000, tol::Float64=1e-9)
    Xfit = fit_intercept ? _add_intercept(X) : X
    n_coef = size(X, 2)
    p = size(Xfit, 2)
    # Initialise: zero intercept, uniform weights on simplex
    β = fit_intercept ? vcat(0.0, fill(1.0 / n_coef, n_coef)) : fill(1.0 / p, p)
    L = opnorm(Xfit)^2
    η = 1 / max(L, eps(Float64))
    for _ in 1:maxiter
        old = copy(β)
        grad = Xfit' * (Xfit * β - y)
        if fit_intercept
            # Intercept is unconstrained; only donor weights are simplex-projected.
            β[1] -= η * grad[1]
            β[2:end] = _project_simplex(β[2:end] - η * grad[2:end])
        else
            β = _project_simplex(β - η * grad)
        end
        norm(β - old) <= tol * (1 + norm(old)) && break
    end
    if fit_intercept
        return β[2:end], β[1]
    else
        return β, 0.0
    end
end

function fit_synthetic_control(
    pre_donor::AbstractMatrix{<:Real},
    pre_target::AbstractVecOrMat{<:Real};
    method::Symbol=:ols,
    lambda::Union{Nothing,Real}=nothing,
    fit_intercept::Bool=true,
)
    X = Matrix{Float64}(pre_donor)
    y = vec(Float64.(pre_target))
    if method == :ols
        Xfit = fit_intercept ? _add_intercept(X) : X
        β = Xfit \ y
        intercept = fit_intercept ? β[1] : 0.0
        coef = fit_intercept ? β[2:end] : β
    elseif method == :ridge
        λ = Float64(lambda === nothing ? 1.0 : lambda)
        Xfit = fit_intercept ? _add_intercept(X) : X
        P = Matrix{Float64}(I, size(Xfit, 2), size(Xfit, 2))
        fit_intercept && (P[1, 1] = 0.0)
        β = (Xfit' * Xfit + λ * P) \ (Xfit' * y)
        intercept = fit_intercept ? β[1] : 0.0
        coef = fit_intercept ? β[2:end] : β
    elseif method == :lasso
        λ = Float64(lambda === nothing ? 1.0 : lambda)
        if fit_intercept
            μx = vec(mean(X; dims=1))
            μy = mean(y)
            Xc = X .- μx'
            yc = y .- μy
            coef = _lasso_cd(Xc, yc, λ)
            intercept = μy - dot(μx, coef)
        else
            coef = _lasso_cd(X, y, λ)
            intercept = 0.0
        end
    elseif method == :simplex
        coef, intercept = _simplex_regression(X, y; fit_intercept=fit_intercept)
    else
        throw(ArgumentError("method must be :ols, :ridge, :lasso, or :simplex"))
    end
    return SyntheticControlModel(method, coef, intercept, fit_intercept, collect(1:size(X, 2)))
end

function predict(model::SyntheticControlModel, donor::AbstractMatrix{<:Real})
    return Matrix{Float64}(donor) * model.coef .+ model.intercept
end

function score(model::SyntheticControlModel, donor::AbstractMatrix{<:Real}, target::AbstractVecOrMat{<:Real})
    y = vec(Float64.(target))
    yhat = predict(model, donor)
    ss_res = sum((y - yhat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    return ss_tot <= eps(Float64) ? NaN : 1 - ss_res / ss_tot
end

function predict_and_mse(model::SyntheticControlModel, donor::AbstractMatrix{<:Real}, target_true::AbstractVecOrMat{<:Real})
    y = vec(Float64.(target_true))
    yhat = predict(model, donor)
    return mean((yhat - y) .^ 2)
end

# ---------------------------------------------------------------------------
# Synthetic data generators from genData/SSM.py and genData/Sine.py

function _rng(seed::Union{Nothing,Int})
    return seed === nothing ? Random.default_rng() : MersenneTwister(seed)
end

function _dirichlet_sample(rng::AbstractRNG, α::AbstractVector{<:Real})
    x = [rand(rng, GammaLike(max(Float64(a), eps(Float64)))) for a in α]
    return x ./ sum(x)
end

struct GammaLike
    α::Float64
end

function Random.rand(rng::AbstractRNG, g::GammaLike)
    # Marsaglia-Tsang for Gamma(shape, scale=1), with the standard boost for α < 1.
    α = g.α
    if α < 1
        return rand(rng, GammaLike(α + 1)) * rand(rng)^(1 / α)
    end
    d = α - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        v <= 0 && continue
        u = rand(rng)
        if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
            return d * v
        end
    end
end

function gen_cov(
    d::Int;
    noise_min::Float64=0.05,
    noise_max::Float64=0.2,
    seed::Union{Nothing,Int}=nothing,
    diag::Bool=true,
)
    rng = _rng(seed)
    if diag
        vals = abs.(randn(rng, d))
        return Matrix(Diagonal(vals))
    end
    Q = rand(rng, d, d) .* (noise_max - noise_min) .+ noise_min
    Q ./= sqrt(d)
    Q = Q * Q'
    Q += 1e-6 * I
    signs = ones(Float64, d, d)
    for i in 1:d, j in (i + 1):d
        signs[i, j] = rand(rng, Bool) ? 1.0 : -1.0
        signs[j, i] = signs[i, j]
    end
    Q .*= signs
    vals, vecs = eigen(Symmetric(_sym(Q)))
    vals = max.(vals, 1e-6)
    return _sym(vecs * Diagonal(vals) * vecs')
end

function gen_A(; d::Int=15, method::Symbol=:qr, seed::Union{Nothing,Int}=nothing)
    rng = _rng(seed)
    if method == :qr || method == :QR
        F = qr(randn(rng, d, d))
        return Matrix(F.Q)
    elseif method == :dirichlet
        return reduce(vcat, [_dirichlet_sample(rng, rand(rng, d))' for _ in 1:d])
    elseif method == :noisy_dirichlet
        return reduce(vcat, [_dirichlet_sample(rng, rand(rng, d) + abs.(randn(rng, d)) .* 0.1)' for _ in 1:d])
    else
        throw(ArgumentError("unknown A generation method"))
    end
end

function gen_H(; d::Int=15, N::Int=15, method::Symbol=:dirichlet, seed::Union{Nothing,Int}=nothing)
    rng = _rng(seed)
    if method == :dirichlet
        return reduce(vcat, [_dirichlet_sample(rng, rand(rng, d))' for _ in 1:N])
    elseif method == :noisy_dirichlet
        return reduce(vcat, [_dirichlet_sample(rng, rand(rng, d) + abs.(randn(rng, d)) .* 0.1)' for _ in 1:N])
    else
        throw(ArgumentError("unknown H generation method"))
    end
end

function gen_dirichlet_params(;
    d::Int=5,
    N::Int=15,
    noise_min_q::Float64=0.01,
    noise_max_q::Float64=0.1,
    noise_min_r::Float64=0.01,
    noise_max_r::Float64=0.1,
    noise_min_p0::Float64=0.01,
    noise_max_p0::Float64=0.1,
    seed::Union{Nothing,Int}=nothing,
    Q_diag::Bool=true,
    R_diag::Bool=true,
    Q_zero::Bool=false,
    A_method::Symbol=:qr,
    H_method::Symbol=:dirichlet,
)
    rng = _rng(seed)
    A = gen_A(d=d, method=A_method, seed=seed)
    H = gen_H(d=d, N=N, method=H_method, seed=seed)
    Q = Q_zero ? zeros(Float64, d, d) : gen_cov(d; noise_min=noise_min_q, noise_max=noise_max_q, seed=seed, diag=Q_diag)
    R = gen_cov(N; noise_min=noise_min_r, noise_max=noise_max_r, seed=seed, diag=R_diag)
    m0 = rand(rng, d)
    P0 = gen_cov(d; noise_min=noise_min_p0, noise_max=noise_max_p0, seed=seed, diag=false)
    return StateSpaceParams(A, H, Q, R, m0, P0)
end

gen_dirchelet_params(; kwargs...) = gen_dirichlet_params(; kwargs...)

function generate_model_data(
    theta::StateSpaceParams;
    T::Int,
    seed::Union{Nothing,Int}=nothing,
    burn_time::Int=3,
    return_signal::Bool=false,
)
    rng = _rng(seed)
    A, H, Q, R, m0, P0 = theta.A, theta.H, theta.Q, theta.R, theta.m0, theta.P0
    d = size(A, 1)
    N = size(H, 1)
    x = m0 + _chol(P0).L * randn(rng, d) + _chol(Q + 1e-12I).L * randn(rng, d)
    Y = zeros(Float64, N, T + burn_time)
    signal = zeros(Float64, N, T + burn_time)
    for t in 1:(T + burn_time)
        signal[:, t] = H * x
        Y[:, t] = signal[:, t] + _chol(R).L * randn(rng, N)
        x = A * x + _chol(Q + 1e-12I).L * randn(rng, d)
    end
    Y = Y[:, (burn_time + 1):end]
    signal = signal[:, (burn_time + 1):end]
    return return_signal ? (signal, Y) : Y
end

function generate_multiple_layers(;
    d_true::Int,
    N::Int,
    T::Int,
    k::Int=1,
    noise_min::Float64=0.0,
    noise_max::Float64=1.0,
    seed::Union{Nothing,Int}=nothing,
    burn_time::Int=3,
)
    rng = _rng(seed)
    # Each layer gets a distinct derived seed so parameters differ across layers.
    theta_list = [
        gen_dirichlet_params(d=d_true, N=N, noise_min_q=noise_min, noise_max_q=noise_max,
            noise_min_r=noise_min, noise_max_r=noise_max,
            seed=seed === nothing ? nothing : seed + layer - 1,
            Q_diag=false, R_diag=false)
        for layer in 1:k
    ]
    mb = rand(rng, d_true)
    Pb = gen_cov(d_true; noise_min=noise_min, noise_max=noise_max, seed=seed, diag=false)
    b_list = [mb + _chol(Pb).L * randn(rng, d_true) for _ in 1:N]
    mc = rand(rng, d_true)
    Pc = gen_cov(d_true; noise_min=noise_min, noise_max=noise_max, seed=seed, diag=false)
    c_list = [mc + _chol(Pc).L * randn(rng, d_true) for _ in 1:k]

    sigs = zeros(Float64, k, T, N)
    obs = zeros(Float64, k, T, N)
    for layer in 1:k
        θ = theta_list[layer]
        Hnew = reduce(vcat, [(b_list[i] .* c_list[layer])' for i in 1:N])
        θnew = StateSpaceParams(θ.A, Hnew, θ.Q, θ.R, θ.m0, θ.P0)
        signal, Y = generate_model_data(θnew; T=T, seed=seed, burn_time=burn_time, return_signal=true)
        sigs[layer, :, :] = signal'
        obs[layer, :, :] = Y'
    end
    return sigs, obs
end

function data_flatten(Y::AbstractArray{<:Real,3}; method::Symbol=:time)
    k, T, N = size(Y)
    if method == :time
        return reshape(permutedims(Y, (2, 1, 3)), T, k * N)
    elseif method == :unit
        return reshape(permutedims(Y, (3, 1, 2)), N, k * T)
    else
        throw(ArgumentError("method must be :time or :unit"))
    end
end

function _moving_average(v::Vector{Float64}, window_size::Int)
    return [mean(@view v[i:(i + window_size - 1)]) for i in 1:(length(v) - window_size + 1)]
end

function generate_rank_1_matrix(n::Int, m::Int, noise_level::Real; smooth::Bool=true, seed::Union{Nothing,Int}=nothing)
    rng = _rng(seed)
    u = rand(rng, n)
    v = smooth ? _moving_average(rand(rng, BetaLike(2, 2), m + 4), 5) : rand(rng, m)
    return (u * v' + randn(rng, n, m) .* noise_level)'
end

function generate_rank_k_matrix(n::Int, m::Int, k::Int, noise_level::Real; smooth::Bool=true, seed::Union{Nothing,Int}=nothing)
    rng = _rng(seed)
    u = rand(rng, n, k)
    V = zeros(Float64, k, m)
    for j in 1:k
        V[j, :] = smooth ? _moving_average(rand(rng, BetaLike(2, 2), m + 4), 5) : rand(rng, m)
    end
    return (u * V + randn(rng, n, m) .* noise_level)'
end

struct BetaLike
    α::Float64
    β::Float64
end

function Random.rand(rng::AbstractRNG, b::BetaLike)
    x = rand(rng, GammaLike(b.α))
    y = rand(rng, GammaLike(b.β))
    return x / (x + y)
end

function Random.rand(rng::AbstractRNG, b::BetaLike, dims::Integer...)
    return [rand(rng, b) for _ in 1:prod(dims)] |> x -> reshape(x, dims...)
end

function generate_sine_wave(alpha::Real, omega::Real, phi::Real, noise_level::Real, num_time::Int; seed::Union{Nothing,Int}=nothing)
    rng = _rng(seed)
    time = collect(0:(num_time - 1)) .* 10pi
    signal = alpha .* sin.(2pi .* omega .* time ./ 360 .+ phi)
    noise_level > 0 && (signal .+= randn(rng, num_time) .* noise_level)
    return signal
end

function zero_out_fraction(df::AbstractMatrix{<:Real}; p::Float64=0.9, seed::Union{Nothing,Int}=nothing)
    0 <= p <= 1 || throw(ArgumentError("p must be between 0 and 1"))
    rng = _rng(seed)
    out = Matrix{Float64}(df)
    total = length(out)
    nzero = floor(Int, (1 - p) * total)
    idx = randperm(rng, total)[1:nzero]
    out[idx] .= 0.0
    return out, idx
end

function generate_linear_dataset(
    num_samples::Int,
    num_time::Int,
    noise_level::Real;
    mean_slope::Real=3,
    std_slope::Real=1,
    noise_type::Symbol=:normal,
    seed::Union{Nothing,Int}=nothing,
)
    rng = _rng(seed)
    times = collect(0:(num_time - 1)) ./ num_time
    slope = randn(rng, num_samples) .* std_slope .+ mean_slope
    intercept = rand(rng, num_samples) .* 2 .- 1
    data = slope * times' .+ intercept
    if noise_type == :normal
        data .+= randn(rng, num_samples, num_time) .* noise_level
    elseif noise_type == :cauchy
        data .+= tan.(pi .* (rand(rng, num_samples, num_time) .- 0.5)) .* noise_level
    else
        throw(ArgumentError("noise_type must be :normal or :cauchy"))
    end
    return data'
end

function generate_new_sine_dataset(
    num_samples::Int,
    num_time::Int,
    noise_level::Real,
    num_signals::Int;
    low::Real=1,
    high::Real=10,
    alpha=nothing,
    omega=nothing,
    phi=nothing,
    seed::Union{Nothing,Int}=nothing,
)
    rng = _rng(seed)
    basis = zeros(Float64, num_signals, num_time)
    for i in 1:num_signals
        a = alpha === nothing ? rand(rng, BetaLike(2, 2)) : alpha[i]
        o = omega === nothing ? rand(rng) * (high - low) + low : omega[i]
        p = phi === nothing ? randn(rng) : phi[i]
        wave_seed = seed === nothing ? nothing : seed + i
        y = generate_sine_wave(a, o, p, 0, floor(Int, num_time * 1.2); seed=wave_seed)
        basis[i, :] = y[(floor(Int, 0.2 * num_time) + 1):end]
    end
    weights = rand(rng, num_samples, num_signals)
    return (weights * basis + randn(rng, num_samples, num_time) .* noise_level)'
end

function generate_sine_dataset_A(num_samples::Int, num_time::Int, noise_level::Real, num_signals::Int; noise_type::Symbol=:normal, seed::Union{Nothing,Int}=nothing)
    return _generate_sine_dataset(num_samples, num_time, noise_level, num_signals; low=1, high=3, beta=(2, 2), noise_type=noise_type, seed=seed)
end

function generate_sine_dataset_B(num_samples::Int, num_time::Int, noise_level::Real, num_signals::Int; noise_type::Symbol=:normal, seed::Union{Nothing,Int}=nothing)
    return _generate_sine_dataset(num_samples, num_time, noise_level, num_signals; low=3, high=6, beta=(2, 5), noise_type=noise_type, seed=seed)
end

function _generate_sine_dataset(num_samples, num_time, noise_level, num_signals; low, high, beta, noise_type, seed)
    rng = _rng(seed)
    basis = zeros(Float64, num_signals, num_time)
    for i in 1:num_signals
        a = rand(rng, BetaLike(beta[1], beta[2]))
        o = rand(rng) * (high - low) + low
        p = randn(rng)
        # Derive a distinct seed per basis function so noise sequences differ.
        wave_seed = seed === nothing ? nothing : seed + i
        y = generate_sine_wave(a, o, p, 0, floor(Int, num_time * 1.2); seed=wave_seed)
        basis[i, :] = y[(floor(Int, 0.2 * num_time) + 1):end]
    end
    final = rand(rng, num_samples, num_signals) * basis
    if noise_type == :normal
        final .+= randn(rng, num_samples, num_time) .* noise_level
    elseif noise_type == :cauchy
        final .+= tan.(pi .* (rand(rng, num_samples, num_time) .- 0.5)) .* noise_level
    else
        throw(ArgumentError("noise_type must be :normal or :cauchy"))
    end
    return final'
end

function make_approx_low_rank(dataset::AbstractMatrix{<:Real}; k::Int=5)
    F = svd(Matrix{Float64}(dataset); full=false)
    s = copy(F.S)
    k < length(s) && (s[(k + 1):end] .*= 0.1)
    return F.U * Diagonal(s) * F.Vt
end

"""
    simulate_tasc(; N=20, T=80, d=3, seed=1)

Generate synthetic panel data from the TASC state-space model. Returns
`(Y, params, signal)`, where `Y` and `signal` are `N x T`.
"""
function simulate_tasc(; N::Int=20, T::Int=80, d::Int=3, seed::Int=1)
    rng = MersenneTwister(seed)
    Araw = randn(rng, d, d)
    A = Araw / max(1.2 * opnorm(Araw), 1.0)
    H = randn(rng, N, d)
    Q = Matrix(Diagonal(fill(0.03, d)))
    R = Matrix(Diagonal(fill(0.10, N)))
    m0 = randn(rng, d)
    P0 = Matrix(Diagonal(fill(0.20, d)))
    params = StateSpaceParams(A, H, Q, R, m0, P0)

    x = m0 + _chol(P0).L * randn(rng, d)
    Y = zeros(Float64, N, T)
    signal = zeros(Float64, N, T)
    for t in 1:T
        x = A * x + _chol(Q).L * randn(rng, d)
        signal[:, t] = H * x
        Y[:, t] = signal[:, t] + _chol(R).L * randn(rng, N)
    end

    return Y, params, signal
end

end
