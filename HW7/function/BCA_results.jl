# =============================================================================
# CKM / BCA figures and small plotting helpers (depends on Plots.jl).
# =============================================================================

using Plots
using LaTeXStrings

"""
    bca_qlabel_to_year(s::AbstractString) -> Float64

Convert a BEA-style quarter label to a fractional calendar year:
`"Q1-1948"` → `1948.0`, `"Q2-1948"` → `1948.25`, ….
"""
function bca_qlabel_to_year(s::AbstractString)
    parts = split(s, "-")
    q = parse(Int, replace(first(parts), "Q" => ""))
    parse(Int, last(parts)) + (q - 1) / 4
end

# Notebook alias (not `const`, so re-`include` in a session does not error).
_qlabel_to_year(s::AbstractString) = bca_qlabel_to_year(s)

"""
    plot_ckm_figure1_output_measured_wedges(wedges_hat, ZVAR_n; kwargs...) -> Plot

CKM-style **Figure 1**: detrended per-capita output and measured efficiency / labor /
investment wedges, each indexed to 100 at `base_index` (default `1` = first row,
typically 1948Q1 in the HW7 sample).

# Arguments
- `wedges_hat`: table with columns `time_label`, `log_ẑ`, `τ̂_ℓ`, `τ̂_x` (e.g. a `DataFrame`).
- `ZVAR_n`: `T×4` detrended observables; **column 1** is output `y`.

# Keyword arguments
- `base_index::Int = 1` — row used as index 100 for every series.
- `title`, `ylabel` — plot labels.
- `plot_size::Tuple{Int,Int} = (900, 450)` — `Plots.jl` canvas size.
"""
function plot_ckm_figure1_output_measured_wedges(
    wedges_hat,
    ZVAR_n::AbstractMatrix;
    base_index::Int = 1,
    title::AbstractString = "Figure 1.  U.S. Output and Measured Wedges",
    ylabel::AbstractString = "Index (1948Q1 = 100)",
    plot_size::Tuple{Int,Int} = (900, 450),
)
    T = length(wedges_hat.time_label)
    Base.size(ZVAR_n, 1) ≥ T ||
        error("ZVAR_n must have at least length(wedges_hat.time_label) rows")

    years_w = bca_qlabel_to_year.(wedges_hat.time_label)
    eff_w = exp.(wedges_hat.log_ẑ)
    lab_w = 1.0 .- wedges_hat.τ̂_ℓ
    inv_w = 1.0 ./ (1.0 .+ wedges_hat.τ̂_x)
    out_w = ZVAR_n[1:T, 1]

    _idx100(x) = 100.0 .* x ./ x[base_index]

    plt = plot(years_w, _idx100(out_w);
               label = "output", lw = 2, color = :steelblue,
               title = title, xlabel = "Year", ylabel = ylabel,
               legend = :topleft, size = plot_size)
    plot!(plt, years_w, _idx100(eff_w); label = "Efficiency Wedge", lw = 2, color = :orange)
    plot!(plt, years_w, _idx100(lab_w); label = "Labor Wedge", lw = 2, color = :seagreen)
    plot!(plt, years_w, _idx100(inv_w); label = "Investment Wedge", lw = 2, color = :firebrick)
    return plt
end

"""
    hp_filter(y::AbstractVector{<:Real}; λ::Real = 1600) -> (trend, cycle)

Hodrick–Prescott filter.  Solves
`min_τ Σ (y_t − τ_t)^2 + λ Σ ((τ_{t+1}−τ_t) − (τ_t−τ_{t-1}))^2`
and returns the trend and `cycle = y − trend`.  Default `λ = 1600` is
the standard quarterly choice.
"""
function hp_filter(y::AbstractVector{<:Real}; λ::Real = 1600)
    T = length(y)
    T ≥ 4 || error("hp_filter needs T ≥ 4 observations, got T = $T")
    # Second-difference matrix D: (T-2) × T
    D = zeros(T - 2, T)
    @inbounds for t in 1:(T - 2)
        D[t, t]     = 1.0
        D[t, t + 1] = -2.0
        D[t, t + 2] = 1.0
    end
    trend = (I(T) + λ .* (D' * D)) \ collect(Float64, y)
    cycle = y .- trend
    return trend, cycle
end

"""
    plot_ckm_figure1_hp_cycle(wedges_hat, ZVAR_n;
                              λ = 1600, kwargs...) -> Plot

Same four series as `plot_ckm_figure1_output_measured_wedges` but the y-axis
is the **HP-filtered cyclical component** of each series (percent deviation
from its HP trend).  For log variables (output and the efficiency-wedge
`log ẑ`) this is literally the log-deviation × 100; for the labor wedge
`1 − τ̂_ℓ` and investment wedge `1/(1+τ̂_x)` we filter `log(series)` so the
cyclical component is still interpretable as a percent deviation.

# Keyword arguments
- `λ::Real = 1600` — HP smoothing parameter (quarterly default).
- `title`, `ylabel` — plot labels.
- `plot_size::Tuple{Int,Int} = (900, 450)`.
"""
function plot_ckm_figure1_hp_cycle(
    wedges_hat,
    ZVAR_n::AbstractMatrix;
    λ::Real = 1600,
    title::AbstractString = "Figure 1′.  HP-filtered cyclical components (λ = $λ)",
    ylabel::AbstractString = "Percent deviation from HP trend",
    plot_size::Tuple{Int,Int} = (900, 450),
)
    T = length(wedges_hat.time_label)
    Base.size(ZVAR_n, 1) ≥ T ||
        error("ZVAR_n must have at least length(wedges_hat.time_label) rows")

    years_w = bca_qlabel_to_year.(wedges_hat.time_label)

    # Take logs so the HP cycle is a % deviation; labor wedge 1−τ_ℓ and
    # investment wedge 1/(1+τ_x) are assumed strictly positive on the sample.
    log_y   = log.(ZVAR_n[1:T, 1])
    log_eff = wedges_hat.log_ẑ
    log_lab = log.(1.0 .- wedges_hat.τ̂_ℓ)
    log_inv = log.(1.0 ./ (1.0 .+ wedges_hat.τ̂_x))

    _cycle_pct(x) = 100.0 .* last(hp_filter(x; λ))   # second element = cycle

    cy_y   = _cycle_pct(log_y)
    cy_eff = _cycle_pct(log_eff)
    cy_lab = _cycle_pct(log_lab)
    cy_inv = _cycle_pct(log_inv)

    plt = plot(years_w, cy_y;
               label = "output", lw = 2, color = :steelblue,
               title = title, xlabel = "Year", ylabel = ylabel,
               legend = :topleft, size = plot_size)
    hline!(plt, [0.0]; lw = 1, color = :gray, ls = :dash, label = "")
    plot!(plt, years_w, cy_eff; label = "Efficiency Wedge", lw = 2, color = :orange)
    plot!(plt, years_w, cy_lab; label = "Labor Wedge",      lw = 2, color = :seagreen)
    plot!(plt, years_w, cy_inv; label = "Investment Wedge", lw = 2, color = :firebrick)
    return plt
end

"""
    plot_ckm_figure2_model_vs_data(wedges_hat, ZVAR_n, run_, C_, time_labels; kwargs...) -> Plot

CKM-style **Figure 2**: 1×3 panel of output `yₜ`, labor `ℓₜ`, investment `xₜ`
— the *feed-wedges-back-into-the-model* identity from CKM (2007):

```
Y_t  =  C · X_t                    (model observation equation)
```

which, under our Kalman set-up with `R = 0`, is reproduced by
`C · X̂_{t|t}` to machine precision.  **Note that this uses `C`, not
`C̄ = C·A`, and that the comparison is at the same time index `t`.**

Curves (indexed to 100 at `base_index` within the plotted range `t = 2..T`):

1. **Data** — `log Y_t`.
2. **``C\\,\\hat X_{t|t}``** — model with the filtered state fed back.

The shifted observation form `Ȳ_k = Y_{k+1}` means the filter never sees
`Y_1`, so `X̂_{1|1}` does not reproduce `Y_1` and the first period is
dropped to avoid contaminating the normalisation.  For `t = 2..T` the two
curves overlap to machine precision when `R = 0`.

# Arguments
- `wedges_hat`: table with column `time_label` (length `T`).
- `ZVAR_n`: `T×4` detrended observables (first `T` rows are read).
- `run_`: the return value of `kalman_run`; must contain `Xtt` (`T×6`).
- `C_`: the **4×6 observation matrix `C`** (not `C̄ = C·A`).
- `time_labels`: vector of quarter labels of length `≥ T` (e.g.
  `dt.mled_detrended.time_label`); rows `1:T` set the x-axis.

# Keyword arguments
- `base_index::Int = 1` — row used as index 100 for every series.
- `plot_size::Tuple{Int,Int} = (1600, 430)` — canvas size.
- `check_identity::Bool = true` — print `max|log Y_t − C·X̂_{t|t}|` sanity
  check (should be ~1e-14 for `t ≥ 2` when `R = 0`).
"""
function plot_ckm_figure2_model_vs_data(
    wedges_hat,
    ZVAR_n::AbstractMatrix,
    run_,
    C_::AbstractMatrix,
    time_labels::AbstractVector{<:AbstractString};
    base_index::Int = 1,
    plot_size::Tuple{Int,Int} = (1600, 430),
    check_identity::Bool = true,
)
    T = length(wedges_hat.time_label)
    Base.size(ZVAR_n, 1) ≥ T ||
        error("ZVAR_n must have at least T = $(T) rows")
    length(time_labels) ≥ T ||
        error("time_labels must have at least T = $(T) entries")

    # CKM (2007) identity:  Y_t = C · X_t.  Feeding the filtered state
    # (= filtered wedges + filtered capital) back through C reproduces
    # the data to machine precision when R = 0.
    #
    # IMPORTANT: the shifted observation form Ȳ_k = Y_{k+1} means the
    # filter never sees Y_1, so X̂_{1|1} is pinned only by the prior X_0
    # and Y_2 — it does *not* reproduce Y_1.  Normalising both curves by
    # their t = 1 value would therefore bake a constant level offset
    # (= e^{resid_1}) into the whole series.  To avoid this, we drop the
    # t = 1 transient and plot only t = 2..T.
    YM_filt_full = run_.Xtt * C_'                       # C · X̂_{t|t},  t = 1..T

    t_rng = 2:T
    YM_filt  = YM_filt_full[t_rng, :]
    Z_obs    = ZVAR_n[t_rng, :]                         # data: Y_t  (same t as Xtt)
    years_b  = bca_qlabel_to_year.(time_labels[t_rng])
    base_lab = time_labels[t_rng.start]

    _idx100(x) = 100.0 .* x ./ x[base_index]
    ylabel_y   = "Index (" * base_lab * " = 100)"

    function _panel(title_str, col; with_ylabel = false)
        plt = plot(years_b, _idx100(Z_obs[:, col]);
                   label = "data",
                   lw = 2, color = :steelblue,
                   title = title_str, xlabel = "Year",
                   ylabel = with_ylabel ? ylabel_y : "",
                   legend = :topleft)
        plot!(plt, years_b, _idx100(exp.(YM_filt[:, col]));
              label = L"C\,\hat X_{t|t}",
              lw = 2, ls = :dash, color = :firebrick)
        return plt
    end

    plt_y = _panel("Output  yₜ",     1; with_ylabel = true)
    plt_l = _panel("Labor  ℓₜ",      3)
    plt_x = _panel("Investment  xₜ", 2)

    plt_fig2 = plot(plt_y, plt_l, plt_x; layout = (1, 3), size = plot_size)

    if check_identity
        resid      = log.(Z_obs[:, [1, 3, 2]]) .- YM_filt[:, [1, 3, 2]]
        resid_t1   = log.(ZVAR_n[1, [1, 3, 2]])  .- YM_filt_full[1, [1, 3, 2]]
        println("CKM identity  max|log Y_t − C·X̂_{t|t}|  (≈1e-14 when R = 0):")
        println("  t ≥ 2 (plotted):  y = ", round(maximum(abs.(resid[:, 1])); sigdigits = 3),
                "    ℓ = ", round(maximum(abs.(resid[:, 2])); sigdigits = 3),
                "    x = ", round(maximum(abs.(resid[:, 3])); sigdigits = 3))
        println("  t = 1 (dropped):  y = ", round(abs(resid_t1[1]); sigdigits = 3),
                "    ℓ = ", round(abs(resid_t1[2]); sigdigits = 3),
                "    x = ", round(abs(resid_t1[3]); sigdigits = 3),
                "   (transient — filter has not seen Y_1)")
    end

    return plt_fig2
end

# =============================================================================
#  Wedge decomposition — `pwbca.m`-style
# =============================================================================

# Short human-readable names for the four wedges.
const _WEDGE_NAMES  = ("Efficiency", "Labor", "Investment", "Government")
const _WEDGE_SYMBOL = ("z", "τℓ", "τx", "g")

# Colors / linestyles used for the four wedge-component curves.
const _WEDGE_COLORS = (:darkorange, :seagreen, :mediumorchid, :goldenrod)
const _WEDGE_LSTYLE = (:solid, :dash, :dashdot, :dot)

"""
    compute_wedge_components(param, θ̂, wedges_hat, ZVAR_n; Y0_idx = 1)

Build the `pwbca.m`-style wedge-component observables at base quarter
`Y0_idx`.  Returns a NamedTuple with:

- `YM_only[i]`  for i∈1..4 — "only wedge i" component (matches CKM Fig. B–E).
- `YM_without[i]` for i∈1..4 — "all but wedge i" leave-one-out component.
- `YM_no` — no-wedge baseline observables (all wedges frozen at `s0`).
- `YM_full` — prototype full model `Xt0 · C_full'`.
- `Ydata_log` — `T×4` log of observed data (rows of `ZVAR_n`).
- `years` — vector of fractional years from `wedges_hat.time_label`.
- `base_label` — quarter label at `Y0_idx` (e.g. `"Q1-1948"`).
- `Y0_idx`, `T_all`.

All matrices are `T_all × 4` with column order `[y, x, ℓ, g]`
(same as `ZVAR_n`).
"""
function compute_wedge_components(
    param::AbstractVector, θ̂::AbstractVector,
    wedges_hat, ZVAR_n::AbstractMatrix;
    Y0_idx::Int = 1,
)
    T_all = length(wedges_hat.time_label)
    Base.size(ZVAR_n, 1) ≥ T_all ||
        error("ZVAR_n must have at least T_all = $T_all rows")

    # Filtered state vector X_t = [log k̂, log ẑ, τ̂_ℓ, τ̂_x, log ĝ, 1]  (T × 6).
    Xt0 = hcat(wedges_hat.log_k̂, wedges_hat.log_ẑ, wedges_hat.τ̂_ℓ,
               wedges_hat.τ̂_x, wedges_hat.log_ĝ, ones(T_all))

    # Prototype full model.
    ss_full = build_bca_state_space(param, θ̂)
    C_full  = ss_full.C
    YM_full = Xt0 * C_full'

    # Base-quarter reference wedges (= `s0` in pwbca.m).
    s0_base = [wedges_hat.log_ẑ[Y0_idx], wedges_hat.τ̂_ℓ[Y0_idx],
               wedges_hat.τ̂_x[Y0_idx],  wedges_hat.log_ĝ[Y0_idx]]

    # Build observation matrices for each As.
    C0       = fixexp_state_space(param, θ̂, [0, 0, 0, 0]; s0 = s0_base).C
    C_only   = [fixexp_state_space(param, θ̂, _ei(i);   s0 = s0_base).C for i in 1:4]
    C_except = [fixexp_state_space(param, θ̂, _1m_ei(i); s0 = s0_base).C for i in 1:4]

    dX     = Xt0 .- Xt0[Y0_idx:Y0_idx, :]
    anchor = YM_full[Y0_idx:Y0_idx, :]

    YM_no       = dX * (C0 - C0)' .+ anchor                      # ≡ anchor (kept for clarity)
    YM_only     = [dX * (C_only[i]   - C0)' .+ anchor for i in 1:4]
    YM_without  = [dX * (C_except[i] - C0)' .+ anchor for i in 1:4]

    Ydata_log = log.(ZVAR_n[1:T_all, :])
    years     = bca_qlabel_to_year.(wedges_hat.time_label)
    base_lab  = wedges_hat.time_label[Y0_idx]

    return (; YM_full, YM_no, YM_only, YM_without,
            Ydata_log, years,
            base_label = base_lab,
            Y0_idx, T_all)
end

_ei(i::Int)    = (v = [0,0,0,0]; v[i] = 1; v)
_1m_ei(i::Int) = (v = [1,1,1,1]; v[i] = 0; v)

# Column in YM_* / Ydata_log for each observable name.
_col_of(var::Symbol) = var === :y ? 1 :
                       var === :x ? 2 :
                       var === :ℓ || var === :l || var === :hours ? 3 :
                       var === :g ? 4 :
                       error("unknown observable $var; use :y, :x, :ℓ, or :g")

_idx100_series(mat, col, Y0) = 100 .* exp.(mat[:, col] .- mat[Y0, col])

# q/q log-growth × 100 (%): len T-1, aligned with years[2:end].
_log_growth_series(mat, col) = 100 .* diff(@view mat[:, col])

"""
    plot_wedge_decomposition(components; mode = :only, kwargs...) -> Plot

3-panel figure (Output / Hours / Investment) with 5 curves each:

- **Data** (black), plus either
- 4 × **"only wedge i"** components (`mode = :only`), or
- 4 × **"all but wedge i"** leave-one-out components (`mode = :without`).

`components` is the NamedTuple returned by `compute_wedge_components`.

# Keyword arguments
- `mode::Symbol = :only` — `:only` or `:without`.
- `transform::Symbol = :index100` —
    - `:index100`   : level, indexed to 100 at `Y0_idx` (default).
    - `:log_growth` : q/q log-growth in % (`100·Δlog`); no HP filter.
- `panels::Tuple = (:y, :ℓ, :x)` — which observables, in what order.
- `plot_size::Tuple{Int,Int} = (1000, 1050)`.
- `title_suffix::AbstractString = ""` — appended to panel titles.
- `ylabel::AbstractString` — y-axis label (default built from transform).
"""
function plot_wedge_decomposition(
    components::NamedTuple;
    mode::Symbol = :only,
    transform::Symbol = :index100,
    panels       = (:y, :ℓ, :x),
    plot_size::Tuple{Int,Int} = (1000, 1050),
    title_suffix::AbstractString = "",
    ylabel::Union{Nothing, AbstractString} = nothing,
    year_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
)
    mode ∈ (:only, :without) ||
        error("mode must be :only or :without, got $mode")
    transform ∈ (:index100, :log_growth) ||
        error("transform must be :index100 or :log_growth, got $transform")

    YM_list = mode === :only ? components.YM_only : components.YM_without
    years   = components.years
    Y0      = components.Y0_idx
    Ydata   = components.Ydata_log

    # Choose per-panel series builder and matching y-axis (anonymous functions
    # to avoid multi-method definition when using `function` inside branches).
    _series, xs, ylab, zero_line = if transform === :index100
        ((mat, col) -> _idx100_series(mat, col, Y0),
         years,
         ylabel === nothing ? "index, $(components.base_label) = 100" : ylabel,
         false)
    else
        ((mat, col) -> _log_growth_series(mat, col),
         years[2:end],
         ylabel === nothing ? "q/q log-growth (%)" : ylabel,
         true)
    end

    # Optional year-range mask — applied to the *plot-ready* axis `xs`, so y-axis
    # autoscaling only sees the retained sample (otherwise a COVID-2020 spike
    # would compress the rest of the series visually).
    keep = year_range === nothing ? trues(length(xs)) :
           (year_range[1] .≤ xs .≤ year_range[2])
    any(keep) || error("year_range $year_range excludes the entire sample")
    xs_plot = xs[keep]

    _panel_title(var) = begin
        base = var === :y ? "Output" :
               var === :x ? "Investment" :
               var === :ℓ || var === :l || var === :hours ? "Hours" :
               var === :g ? "Government" : string(var)
        isempty(title_suffix) ? base : "$base  ($title_suffix)"
    end
    _legend_label(i) = mode === :only ?
                       "Only $(lowercase(_WEDGE_NAMES[i])) wedge" :
                       "Without $(lowercase(_WEDGE_NAMES[i])) wedge"

    function _mkpanel(var::Symbol)
        col = _col_of(var)
        plt = plot(xs_plot, _series(Ydata, col)[keep];
                   label = "Data", lw = 2, color = :black,
                   title = _panel_title(var), xlabel = "year",
                   ylabel = ylab, legend = :topleft)
        zero_line && hline!(plt, [0.0]; lw = 1, color = :gray, ls = :dash, label = "")
        for i in 1:4
            plot!(plt, xs_plot, _series(YM_list[i], col)[keep];
                  label = _legend_label(i),
                  lw    = 1.5,
                  ls    = _WEDGE_LSTYLE[i],
                  color = _WEDGE_COLORS[i])
        end
        return plt
    end

    plts = [_mkpanel(v) for v in panels]
    return plot(plts...; layout = (length(plts), 1),
                size = plot_size, link = :x)
end
