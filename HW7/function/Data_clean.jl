# US quarterly data cleaning for HW7 / BCA wedges.
#
# Layout:
#   Part 1 — BEA (NIPA) + BLS hours + OECD 15-64 population   ← HW7 main pipeline
#     Order of functions:
#       _bea_* helpers
#       read_us_total_economy_hours_worked,  merge_bea_panel_hours_worked
#       read_us_population_15_64,            merge_bea_panel_population
#       read_bea_panel                       (wide quarterly panel)
#       _fill_missing_tauc
#       build_us_quarterly_panel             (final HW7 panel, 1948Q1-2024Q4)
#       percapita_bea_ckm_from_hw7_panel     (y,h,x,g,KCD_real in dollars per 15-64 person)
#
#   Part 2 — OECD `USdata.xlsx` (cross-check only; kept lean)
#       bcag_read_usdata, bcag_clean_usdata, percapita_oecd_ckm_from_usdata
#
#   Comparison:
#       compare_percapita_bea_vs_oecd(data_pc_bea, data_pc_oecd)

using XLSX
using DataFrames
using Dates
using DelimitedFiles
using Statistics
using Interpolations

# ---------- shared micro-helpers --------------------------------------------

function _to_float(x)::Float64
    x isa Missing && return NaN
    x === nothing && return NaN
    if x isa AbstractString
        v = tryparse(Float64, strip(x)); return v === nothing ? NaN : Float64(v)
    end
    x isa Bool && return Float64(x)
    x isa Real && return Float64(x)
    return NaN
end

function _to_year(x)::Int
    x isa Integer && return Int(x)
    x isa AbstractFloat && return round(Int, x)
    if x isa AbstractString
        s = strip(x); isempty(s) && return 0
        return parse(Int, s)
    end
    v = _to_float(x); isnan(v) ? 0 : round(Int, v)
end

"""Mimic `numpy.arange(start, stop, step)` for positive `step` (stop exclusive)."""
function _numpy_arange(start::Float64, stop::Float64, step::Float64)
    step > 0 || error("step must be positive")
    out = Float64[]; x = start
    while x < stop - 1e-12; push!(out, x); x += step; end
    out
end

function _interp_linear(xs::StepRangeLen{Float64}, ys::Vector{Float64}, xq::AbstractVector{Float64})
    ext = extrapolate(scale(interpolate(ys, BSpline(Linear())), xs), Line())
    Float64[ext(x) for x in xq]
end

function _interp_cubic(xs::StepRangeLen{Float64}, ys::Vector{Float64}, xq::AbstractVector{Float64})
    ext = extrapolate(scale(interpolate(ys, BSpline(Cubic(Line(OnGrid())))), xs), Line())
    Float64[ext(x) for x in xq]
end

"""Parse year from a `Q3-1985`-style time label (nothing if malformed)."""
function _year_from_time_label(s::AbstractString)::Union{Nothing,Int}
    m = match(r"^Q([1-4])-(\d{4})\s*$", strip(s))
    m === nothing ? nothing : parse(Int, m.captures[2])
end

_safe_mean(v) = (w = Float64[x for x in v if isfinite(x)]; isempty(w) ? NaN : mean(w))
_pct_diff(a::Real, b::Real; atol::Float64 = 1e-12) =
    (isfinite(a) && isfinite(b) && abs(b) ≥ atol) ? (a - b) / b * 100.0 : NaN

# ============================================================================
# Part 1 — BEA (NIPA) + BLS hours + OECD population 15-64
# ============================================================================

# --- BEA NIPA sheet helpers --------------------------------------------------

function _bea_sheet_last_col_letters(sh)
    d = string(XLSX.get_dimension(sh)); parts = split(d, ':')
    length(parts) ≥ 2 || error("unexpected sheet dimension: $d")
    m = match(r"^([A-Z]+)", parts[end])
    m === nothing && error("cannot parse column from sheet dimension: $d")
    return m.captures[1]
end

function _bea_year_cell(x)::Union{Nothing,Int}
    (x === missing || x === nothing) && return nothing
    if x isa AbstractString
        s = strip(x); isempty(s) && return nothing
        return parse(Int, s)
    end
    v = _to_float(x); isnan(v) ? nothing : round(Int, v)
end

_bea_strip_label(x)::String = x isa AbstractString ? String(strip(x)) : ""

function _bea_find_row(D, label::AbstractString)::Int
    for i in 1:size(D, 1)
        _bea_strip_label(D[i, 2]) == String(label) && return i
    end
    error("BEA: row not found for label \"$label\"")
end

"""One quarterly series from BEA layout (header rows 6-7, data starting row 8)."""
function _bea_long_series_from_row(H, D, r::Int)
    cur_year = nothing
    time_label = String[]; vals = Float64[]
    for c in 3:size(H, 2)
        yy = _bea_year_cell(H[1, c]); yy !== nothing && (cur_year = yy)
        qcell = H[2, c]; qcell === missing && continue
        qstr = qcell isa AbstractString ? strip(string(qcell)) : ""
        (isempty(qstr) || !startswith(qstr, "Q") || cur_year === nothing) && continue
        push!(time_label, "$(qstr)-$(cur_year)")
        push!(vals, _to_float(D[r, c]))
    end
    DataFrame(; time_label, value = vals)
end

# --- BLS `total-economy-hours-employment.xlsx` ------------------------------

_excel_eq(cell, target) = cell isa AbstractString &&
    lowercase(strip(string(cell))) == lowercase(strip(String(target)))

function _mr_qtr(x)::Union{Nothing,Int}
    (x === missing || x === nothing) && return nothing
    if x isa Integer
        q = Int(x); return 1 ≤ q ≤ 4 ? q : nothing
    end
    if x isa AbstractFloat
        q = round(Int, x); return 1 ≤ q ≤ 4 ? q : nothing
    end
    if x isa AbstractString
        m = match(r"^Q?([1-4])$", uppercase(strip(string(x))))
        return m === nothing ? nothing : parse(Int, m.captures[1])
    end
    v = _to_float(x); isnan(v) && return nothing
    q = round(Int, v); 1 ≤ q ≤ 4 ? q : nothing
end

"""
    read_us_total_economy_hours_worked(path; min_year=1948, kwargs...)

BLS `total-economy-hours-employment.xlsx`, sheet **`MachineReadable`**: keep rows with
`Sector = Total economy`, `Basis = All workers`, `Component = Total U.S. economy`,
`Measure = Hours worked`. Returns `DataFrame(time_label, hours_worked_billions, year, quarter)`
in **billions of SAAR hours**.
"""
function read_us_total_economy_hours_worked(
    path::AbstractString;
    sheet::AbstractString = "MachineReadable",
    min_year::Int = 1948,
    sector::AbstractString = "Total economy",
    basis::AbstractString = "All workers",
    component::AbstractString = "Total U.S. economy",
    measure::AbstractString = "Hours worked",
)
    M = XLSX.getdata(XLSX.readxlsx(path)[sheet])
    size(M, 2) ≥ 8 || error("MachineReadable needs at least 8 columns (A:H)")
    time_label = String[]; hours_worked_billions = Float64[]
    year = Int[]; quarter = Int[]
    for i in 2:size(M, 1)
        _excel_eq(M[i, 1], sector)    || continue
        _excel_eq(M[i, 2], basis)     || continue
        _excel_eq(M[i, 3], component) || continue
        _excel_eq(M[i, 4], measure)   || continue
        yr = _to_year(M[i, 6]); yr < min_year && continue
        qtr = _mr_qtr(M[i, 7]); qtr === nothing && continue
        h = _to_float(M[i, 8]); isfinite(h) || continue
        push!(time_label, "Q$(qtr)-$(yr)"); push!(hours_worked_billions, h)
        push!(year, yr); push!(quarter, qtr)
    end
    df = DataFrame(; time_label, hours_worked_billions, year, quarter)
    isempty(df) && error("read_us_total_economy_hours_worked: no rows after filter")
    sort!(df, [:year, :quarter])
    length(unique(df.time_label)) == nrow(df) || error("duplicate time_label after filter")
    df
end

"""Left- (or inner-)join hours onto `bea` on `time_label`."""
function merge_bea_panel_hours_worked(
    bea::AbstractDataFrame, path::AbstractString; join::Symbol = :left, kw...,
)
    h = read_us_total_economy_hours_worked(path; kw...)
    join === :left  && return leftjoin(bea, h; on = :time_label)
    join === :inner && return innerjoin(bea, h; on = :time_label)
    error("merge_bea_panel_hours_worked: join must be :left or :inner")
end

# --- OECD historical population 15-64 ---------------------------------------

"""
    read_us_population_15_64(path; year_first_data=1950, year_last_data=2023,
                                   Y0=1947, Y_end=2025, kwargs...)

Read the `15 to 64` row from `population.xlsx` sheet `OECD.Stat export`, back/forward-fill
tails by the mean YoY growth rate, then cubic-spline interpolate annual → quarterly.
Returns `(quarterly, annual_extended, mean_yoy_population_growth)`.
"""
function read_us_population_15_64(
    path::AbstractString;
    sheet::AbstractString = "OECD.Stat export",
    age_row_label::AbstractString = "15 to 64",
    year_row::Int = 5, data_col_start::Int = 3,
    year_first_data::Int = 1950, year_last_data::Int = 2023,
    Y0::Int = 1947, Y_end::Int = 2025,
)
    M = XLSX.getdata(XLSX.readxlsx(path)[sheet])
    size(M, 1) ≥ year_row + 2 || error("population sheet too small")

    r15 = nothing
    for i in (year_row + 2):size(M, 1)
        lab = M[i, 1]; lab isa AbstractString || continue
        strip(lab) == age_row_label || continue
        r15 = i; break
    end
    r15 === nothing && error("read_us_population_15_64: row not found for \"$age_row_label\"")

    years_obs = Int[]; P_obs = Float64[]
    for j in data_col_start:size(M, 2)
        yy = _bea_year_cell(M[year_row, j]); yy === nothing && continue
        (year_first_data ≤ yy ≤ year_last_data) || continue
        push!(years_obs, yy); push!(P_obs, _to_float(M[r15, j]))
    end
    isempty(years_obs) && error("no year columns for $year_first_data-$year_last_data")
    perm = sortperm(years_obs); years_obs, P_obs = years_obs[perm], P_obs[perm]
    all(isfinite, P_obs) || error("non-finite population values")
    all(>(0), P_obs)     || error("population must be positive")

    g = mean(P_obs[2:end] ./ P_obs[1:end-1] .- 1.0)
    P_y = Dict{Int,Float64}(y => p for (y, p) in zip(years_obs, P_obs))
    for y in (year_first_data - 1):-1:Y0;   P_y[y] = P_y[y + 1] / (1 + g); end
    for y in (year_last_data + 1):Y_end;    P_y[y] = P_y[y - 1] * (1 + g); end

    years_ext = collect(Y0:Y_end)
    P_annual = Float64[P_y[y] for y in years_ext]
    n_aug = length(P_annual)
    xs_annual = range(1.0, step = 1.0, length = n_aug)
    iP_q = _interp_cubic(xs_annual, P_annual, _numpy_arange(1.25, Float64(n_aug) + 1.01, 0.25))
    nq = length(iP_q)
    nq == 4 * n_aug || error("expected $(4 * n_aug) quarters, got $nq")

    time_label = String[
        string("Q", mod(t - 1, 4) + 1, "-", Y0 + div(t - 1, 4)) for t in 1:nq
    ]
    (;
        quarterly = DataFrame(; time_label, iP = iP_q),
        annual_extended = DataFrame(; year = years_ext, pop_15_64_persons = P_annual),
        mean_yoy_population_growth = g,
    )
end

"""Left- (or inner-)join population onto `bea` on `time_label`."""
function merge_bea_panel_population(
    bea::AbstractDataFrame, path::AbstractString; join::Symbol = :left, kw...,
)
    q = read_us_population_15_64(path; kw...).quarterly
    join === :left  && return leftjoin(bea, q; on = :time_label)
    join === :inner && return innerjoin(bea, q; on = :time_label)
    error("merge_bea_panel_population: join must be :left or :inner")
end

# --- Integrated BEA panel ---------------------------------------------------

"""
    read_bea_panel(; path_115, path_116, path_33, path_32, path_3105,
                     path_hours_worked=nothing, path_population=nothing,
                     deflator_base_year=2009, ...)

Wide quarterly panel joined on `time_label`.

| Source            | Columns |
|-------------------|---------|
| BEA 1.1.5 nominal | `gdp`, `pce`, `private_investment`, `net_exports`, `govt_consumption_and_investment`, `durable_goods`, `nondurable_goods` |
| BEA 1.1.6 real    | `gdp_real` |
| Derived           | `gdp_deflator` (rebased so mean of `deflator_base_year` quarters = 1) |
| BEA 3.3           | `sales_taxes`, `excise_taxes_sl` |
| BEA 3.2           | `excise_taxes_federal` |
| BEA 3.10.5        | `government_consumption` |
| Derived           | `government_investment`, `social_investment` |
| BLS (optional)    | `hours_worked_billions`, `year`, `quarter` |
| OECD (optional)   | `iP` — population 15-64 (persons) |

All nominal columns are **millions USD, SAAR**.
"""
function read_bea_panel(;
    path_115::AbstractString, path_116::AbstractString, path_33::AbstractString,
    path_32::AbstractString, path_3105::AbstractString,
    path_hours_worked::Union{Nothing,AbstractString} = nothing,
    path_population::Union{Nothing,AbstractString} = nothing,
    sheet_name::AbstractString = "Table",
    last_row_115::Int = 35, last_row_116::Int = 36, last_row_33::Int = 54,
    last_row_32::Int = 54, last_row_3105::Int = 73,
    deflator_base_year::Int = 2009,
)
    function _load(path, last_row)
        sh = XLSX.readxlsx(path)[sheet_name]
        colL = _bea_sheet_last_col_letters(sh)
        XLSX.getdata(sh, "A6:$(colL)7"), XLSX.getdata(sh, "A8:$(colL)$(last_row)")
    end
    function _col(H, D, label::AbstractString, name::Symbol)
        df = _bea_long_series_from_row(H, D, _bea_find_row(D, label))
        rename!(df, :value => name); df
    end

    H1, D1 = _load(path_115, last_row_115)
    panel = nothing
    for (name, label) in [
        :gdp => "Gross domestic product",
        :pce => "Personal consumption expenditures",
        :private_investment => "Gross private domestic investment",
        :net_exports => "Net exports of goods and services",
        :govt_consumption_and_investment => "Government consumption expenditures and gross investment",
        :durable_goods => "Durable goods",
        :nondurable_goods => "Nondurable goods",
    ]
        df = _col(H1, D1, label, name)
        panel = panel === nothing ? df : innerjoin(panel, df; on = :time_label)
    end

    H6, D6 = _load(path_116, last_row_116)
    panel = innerjoin(panel, _col(H6, D6, "Gross domestic product", :gdp_real); on = :time_label)
    raw_defl = panel.gdp ./ panel.gdp_real
    base_mask = endswith.(panel.time_label, "-$(deflator_base_year)")
    any(base_mask) || error("no quarters for base year $(deflator_base_year)")
    base_vals = filter(isfinite, raw_defl[base_mask])
    isempty(base_vals) && error("no finite deflator values in base year $(deflator_base_year)")
    panel.gdp_deflator = raw_defl ./ mean(base_vals)

    H3, D3 = _load(path_33, last_row_33)
    panel = innerjoin(panel, _col(H3, D3, "Sales taxes",  :sales_taxes);     on = :time_label)
    panel = innerjoin(panel, _col(H3, D3, "Excise taxes", :excise_taxes_sl); on = :time_label)

    H32, D32 = _load(path_32, last_row_32)
    panel = innerjoin(panel, _col(H32, D32, "Excise taxes", :excise_taxes_federal); on = :time_label)

    Hg, Dg = _load(path_3105, last_row_3105)
    panel = innerjoin(panel,
        _col(Hg, Dg, "Government consumption expenditures1", :government_consumption); on = :time_label)

    panel.government_investment = panel.govt_consumption_and_investment .- panel.government_consumption
    panel.social_investment     = panel.private_investment .+ panel.government_investment

    path_hours_worked !== nothing &&
        (panel = merge_bea_panel_hours_worked(panel, path_hours_worked; join = :left))
    path_population !== nothing &&
        (panel = merge_bea_panel_population(panel, path_population; join = :left))
    panel
end

# --- Final HW7 panel --------------------------------------------------------

"""Fill `NaN` entries in `tauc` by `:none`, `:mean`, or `:quarterly_mean` (same-quarter avg)."""
function _fill_missing_tauc(tauc::AbstractVector{<:Real}, labels::AbstractVector, mode::Symbol)
    mode === :none && return collect(tauc)
    v = Float64[isfinite(x) ? Float64(x) : NaN for x in tauc]
    fin = findall(isfinite, v); isempty(fin) && return v
    if mode === :mean
        μ = mean(v[fin])
        for i in eachindex(v); isfinite(v[i]) || (v[i] = μ); end
        return v
    elseif mode === :quarterly_mean
        quarters = Vector{Int}(undef, length(labels))
        for (k, lab) in pairs(labels)
            m = match(r"^Q([1-4])-", String(lab))
            quarters[k] = m === nothing ? 0 : parse(Int, m.captures[1])
        end
        μ_q = Dict{Int,Float64}()
        for q in 1:4
            idx = [i for i in fin if quarters[i] == q]
            isempty(idx) || (μ_q[q] = mean(v[idx]))
        end
        μ_all = mean(v[fin])
        for i in eachindex(v)
            isfinite(v[i]) && continue
            v[i] = get(μ_q, quarters[i], μ_all)
        end
        return v
    end
    error("_fill_missing_tauc: unknown mode $(repr(mode))")
end

"""
    build_us_quarterly_panel(; paths..., year_start=1948, year_end=2024,
                               deflator_base_year=2009,
                               fill_tauc_missing=:quarterly_mean, kwargs...)

HW7 final quarterly panel (default 1948Q1-2024Q4).  Columns:

| Column                   | Definition |
|--------------------------|-----------|
| `time_label`             | `Q1-1948`, …, `Q4-2024` |
| `GDP`, `durable_goods`, `gross_investment`, `government_consumption`, `net_exports` | millions USD, SAAR |
| `PGDP`                   | deflator, `deflator_base_year = 1` (default 2009) |
| `iP`                     | population 15-64 (persons) |
| `total_worked_hours`     | BLS billions of hours, SAAR |
| `tauc`                   | `(3.3 Sales + 3.3 Excise + 3.2 Excise) / GDP` (pre-1958 `NaN` filled per `fill_tauc_missing`) |
"""
function build_us_quarterly_panel(;
    path_115::AbstractString, path_116::AbstractString, path_33::AbstractString,
    path_32::AbstractString, path_3105::AbstractString,
    path_hours_worked::AbstractString, path_population::AbstractString,
    year_start::Int = 1948, year_end::Int = 2024,
    deflator_base_year::Int = 2009,
    fill_tauc_missing::Symbol = :quarterly_mean,
    kwargs...,
)
    wide = read_bea_panel(;
        path_115, path_116, path_33, path_32, path_3105,
        path_hours_worked, path_population,
        deflator_base_year, kwargs...,
    )
    years = Int[something(_year_from_time_label(tl), 0) for tl in wide.time_label]
    keep = (years .≥ year_start) .& (years .≤ year_end)
    sum(keep) == 4 * (year_end - year_start + 1) ||
        @warn "build_us_quarterly_panel: got $(sum(keep)) quarters in [$year_start, $year_end]"
    w = wide[keep, :]
    tax_sum = w.sales_taxes .+ w.excise_taxes_sl .+ w.excise_taxes_federal
    tauc = _fill_missing_tauc(Float64.(tax_sum ./ w.gdp), w.time_label, fill_tauc_missing)

    DataFrame(
        time_label = w.time_label,
        GDP = w.gdp,
        PGDP = w.gdp_deflator,
        gross_investment = w.social_investment,
        government_consumption = w.government_consumption,
        net_exports = w.net_exports,
        iP = w.iP,
        total_worked_hours = w.hours_worked_billions,
        durable_goods = w.durable_goods,
        tauc = tauc,
    )
end

# --- BEA per-capita (CKM/BCA) ----------------------------------------------

"""
    percapita_bea_ckm_from_hw7_panel(panel; r_d_annual=0.04, delta_d_annual=0.25,
                                             kcd_init_factor=16.0,
                                             nominal_scale_to_dollars=1e6)

CKM/BCA per-capita series from [`build_us_quarterly_panel`](@ref).  BEA NIPA flows are stored
in **millions USD**; the final outputs are rescaled by `nominal_scale_to_dollars`
(default **`1e6`**) so `y, x, g, KCD_real` come out in **real USD per 15-64 person, SAAR**,
and `h` in **hours per person, SAAR**.

Returns `DataFrame(time_label, y, h, x, g, iP, tauc, KCD_real)` where
- `y  = (real GDP − τ_c·real GDP + (r_d^q + δ_d^q)·K^{CD}) / iP`
- `h  = total_worked_hours · 10^9 / iP`
- `x  = (real gross_investment + (1 − τ_c)·real durable_goods) / iP`
- `g  = (real government_consumption + real net_exports) / iP`
- `KCD_real[t] = (1 − δ_d^q)·KCD_real[t-1] + real durable_goods[t-1]`, seed `kcd_init_factor × real durable_goods[1]`.
"""
function percapita_bea_ckm_from_hw7_panel(
    panel::AbstractDataFrame;
    r_d_annual::Float64 = 0.04,
    delta_d_annual::Float64 = 0.25,
    kcd_init_factor::Float64 = 16.0,
    nominal_scale_to_dollars::Float64 = 1.0e6,
)
    r_d_q = (1 + r_d_annual)^(1 / 4) - 1
    delta_d_q = 1 - (1 - delta_d_annual)^(1 / 4)

    PGDP = Float64.(panel.PGDP); iP = Float64.(panel.iP); tauc = Float64.(panel.tauc)
    real_gdp = Float64.(panel.GDP)                    ./ PGDP
    real_dur = Float64.(panel.durable_goods)          ./ PGDP
    real_gi  = Float64.(panel.gross_investment)       ./ PGDP
    real_gc  = Float64.(panel.government_consumption) ./ PGDP
    real_nx  = Float64.(panel.net_exports)            ./ PGDP

    n = nrow(panel)
    KCD = Vector{Float64}(undef, n)
    KCD[1] = kcd_init_factor * real_dur[1]
    for i in 2:n
        KCD[i] = (1 - delta_d_q) * KCD[i - 1] + real_dur[i - 1]
    end

    s = nominal_scale_to_dollars
    y = (real_gdp .- tauc .* real_gdp .+ (r_d_q + delta_d_q) .* KCD) ./ iP .* s
    x = (real_gi .+ (1 .- tauc) .* real_dur) ./ iP .* s
    g = (real_gc .+ real_nx) ./ iP .* s
    h = Float64.(panel.total_worked_hours) .* 1e9 ./ iP

    DataFrame(;
        time_label = String[string(panel.time_label[i]) for i in 1:n],
        y, h, x, g, iP, tauc, KCD_real = KCD .* s,
    )
end

# ============================================================================
# Part 1b — Linear-trend detrending for wedge accounting / MLE
#
#   References:
#     sr328/mleqtrly/usdata.m   (mleq.m companion: fixed gz = 1.6% annual)
#     Project.ipynb §2.3        (calgz / maketrend: gz estimated by mean-zero
#                                log-cyclical output over the MLE sample)
#
#   Model:  y_t = y_base · (1+gz)^{t-base} · exp(ε_t),  with ε_t mean-zero
#           over the MLE sample. Then:
#             ỹ_t = y_t / [y_base · (1+gz)^{t-base}]        (fully detrended)
#             ZVAR_t = y_t / y_base · (1+gz)^{base-1}        (mleq-compatible:
#                                                             retains growth
#                                                             trend so that
#                                                             `bca_neg_loglik`
#                                                             subtracts it
#                                                             internally)
# ============================================================================

"""
    find_gz_mean_zero(y, base_idx, mle_range) → Float64

Per-capita growth rate `gz` such that the detrended log series

    log y_t − log y_{base_idx} − (t − base_idx) · log(1 + gz)

has sample mean zero over `t ∈ mle_range`. Closed-form

```
gz = exp((mean(log y[mle_range]) − log y[base_idx]) /
         mean(mle_range .− base_idx)) − 1
```

(the 1-indexed Julia analogue of `calgz`/`fsolve` in Project.ipynb §2.3).
Errors if `base_idx` coincides with the mean of `mle_range` (denominator 0).
"""
function find_gz_mean_zero(
    y::AbstractVector, base_idx::Integer,
    mle_range::AbstractUnitRange{<:Integer},
)
    last(mle_range)  ≤ length(y) || error("mle_range exceeds length(y)")
    first(mle_range) ≥ 1         || error("mle_range must start at index ≥ 1")
    yy = Float64.(y[mle_range])
    all(yy .> 0) || error("`y[mle_range]` must be strictly positive for log detrending")

    num = mean(log.(yy)) - log(Float64(y[base_idx]))
    den = mean(collect(mle_range) .- base_idx)
    abs(den) < 1e-12 &&
        error("mean(t − base_idx) ≈ 0 for the chosen mle_range; pick a base year off-centre")
    return expm1(num / den)
end

"""Locate `label` inside `labels`, with a clear error message on mismatch."""
function _locate_label(labels::AbstractVector{<:AbstractString}, label::AbstractString)
    idx = findfirst(==(label), labels)
    idx === nothing &&
        error("Label `$label` not found in data (first/last labels: " *
              "`$(first(labels))` / `$(last(labels))`)")
    return idx
end

"""
    bca_linear_detrend(data_pc; base_label,
                       mle_range_labels = nothing,
                       gz               = nothing,
                       output_col       = :y,
                       flow_cols        = [:y, :x, :g],
                       hours_col        = :h,
                       hours_scale      = 5000.0,
                       add_consumption  = true)
      → (; gz, base_idx, mle_range, y_base, hours_scale,
           mled, mled_detrended, Y_detrended, cols)

CKM / BCA-style linear-trend detrending of per-capita series (Project.ipynb
§2.3), with **all flow variables normalized by base-year output** so that the
resource constraint `c/y_base + x/y_base + g/y_base = y/y_base` holds on the
cyclical ratios.

Steps:

1. If `gz` is `nothing`, estimate it by [`find_gz_mean_zero`](@ref) on
   `output_col` over `mle_range`. Otherwise use the user-supplied value
   (e.g. the calibrated `(1.016)^(1/4) − 1`).
2. For each flow column `f ∈ flow_cols`, normalize by `y_base =
   data_pc[base_idx, output_col]`:
   - `mled[f, t]           = data_pc[f, t] / y_base · (1+gz)^{base_idx−1}`
     (retains growth trend; ready for `bca_neg_loglik` which subtracts
     `(t−1)·log(1+gz)` internally).
   - `mled_detrended[f, t] = data_pc[f, t] / y_base / (1+gz)^{t−base_idx}`
     (stationary cyclical ratio; at `t = base_idx`, flow = flow-share of GDP).
3. If `add_consumption = true`, an implied consumption column `:c =
   y − x − g` is computed on the raw series and added to the two DataFrames
   (same normalization as other flows).
4. The hours column is divided by the **annual hours cap** `hours_scale`
   (Project.ipynb: `50 weeks × 100 hours = 5000`) so that `mled[h]` is the
   fraction of the time endowment spent working (≈ 0.25–0.30 for the US).
   The same scaling is used in `mled_detrended`, and no growth trend is
   removed (hours are stationary).
5. `Y_detrended = log(mled_detrended[:, cols])` where
   `cols = [flow_cols; (:c if add_consumption); hours_col]`.
"""
function bca_linear_detrend(
    data_pc::AbstractDataFrame;
    base_label::AbstractString,
    mle_range_labels::Union{Nothing, Tuple{AbstractString,AbstractString}} = nothing,
    gz::Union{Nothing, Real} = nothing,
    output_col::Symbol       = :y,
    flow_cols::Vector{Symbol} = [:y, :x, :g],
    hours_col::Symbol        = :h,
    hours_scale::Real        = 5000.0,       # 50 wk × 100 hrs
    add_consumption::Bool    = true,
)
    hasproperty(data_pc, :time_label) ||
        error("`data_pc` must have a `time_label` column (got columns: $(names(data_pc)))")
    labels   = String.(data_pc.time_label)
    base_idx = _locate_label(labels, base_label)

    mle_range = if mle_range_labels === nothing
        1:length(labels)
    else
        s_idx = _locate_label(labels, mle_range_labels[1])
        e_idx = _locate_label(labels, mle_range_labels[2])
        s_idx ≤ e_idx || error("mle_range_labels must be in chronological order")
        s_idx:e_idx
    end

    output_col ∈ flow_cols ||
        error("`output_col` (:$output_col) must appear in `flow_cols` ($flow_cols)")
    for c in vcat(flow_cols, [hours_col])
        hasproperty(data_pc, c) ||
            error("Column `:$c` not present in `data_pc` (columns = $(names(data_pc)))")
    end
    hours_scale > 0 || error("`hours_scale` must be positive (got $hours_scale)")

    gz_val::Float64 = gz === nothing ?
        find_gz_mean_zero(data_pc[!, output_col], base_idx, mle_range) :
        Float64(gz)

    T        = nrow(data_pc)
    t_rel    = (1:T) .- base_idx                    # (t − base) in 1-indexed time
    trend_ts = (1 + gz_val) .^ t_rel                # (1+gz)^{t−base_idx}
    mleq_fac = (1 + gz_val)^(base_idx - 1)          # scalar multiplier for `mled`

    y_base = Float64(data_pc[base_idx, output_col])
    y_base > 0 || error("`data_pc[$base_idx, $output_col]` must be positive, got $y_base")

    mled           = DataFrame(time_label = labels)
    mled_detrended = DataFrame(time_label = labels)
    cols_out       = Symbol[]

    # Flow variables — all normalized by base-year OUTPUT so that
    # x/y_base + g/y_base + c/y_base = y/y_base ≡ 1 at t = base_idx.
    for c in flow_cols
        y_raw = Float64.(data_pc[!, c])
        mled[!, c]           = (y_raw ./ y_base) .* mleq_fac
        mled_detrended[!, c] = (y_raw ./ y_base) ./ trend_ts
        push!(cols_out, c)
    end

    if add_consumption
        needed = (:y, :x, :g)
        all(s -> s ∈ flow_cols, needed) ||
            error("`add_consumption = true` requires :y, :x, :g in `flow_cols`")
        :c ∉ flow_cols ||
            error("`add_consumption = true` cannot coexist with `:c` already in `flow_cols`")
        c_raw = Float64.(data_pc.y) .- Float64.(data_pc.x) .- Float64.(data_pc.g)
        mled[!, :c]           = (c_raw ./ y_base) .* mleq_fac
        mled_detrended[!, :c] = (c_raw ./ y_base) ./ trend_ts
        push!(cols_out, :c)
    end

    # Hours: fraction of time endowment (Project.ipynb: /(50 wk × 100 hrs) = /5000).
    # Stationary — no growth-trend adjustment.
    h_raw = Float64.(data_pc[!, hours_col])
    mled[!, hours_col]           = h_raw ./ hours_scale
    mled_detrended[!, hours_col] = h_raw ./ hours_scale
    push!(cols_out, hours_col)

    Y_detrended = Matrix{Float64}(undef, T, length(cols_out))
    @inbounds for (j, c) in enumerate(cols_out)
        Y_detrended[:, j] = log.(mled_detrended[!, c])
    end

    return (; gz = gz_val, base_idx, mle_range, y_base, hours_scale,
              mled, mled_detrended, Y_detrended, cols = cols_out)
end

# ============================================================================
# Part 2 — OECD `USdata.xlsx` (cross-check only)
# ============================================================================

# Excel column indices (1-based) for `Economic Outlook` sheet (pandas `skiprows=4`).
const _COL_CG    = 3
const _COL_CP    = 4
const _COL_GDP   = 5
const _COL_ITISK = 7
const _COL_MGS   = 8
const _COL_XGS   = 9
const _COL_PGDP  = 15
const _COL_ET    = 18
const _COL_HRS   = 19
const _COL_NXCD  = 21

"""Raw read of OECD `USdata.xlsx` (sheets `Economic Outlook`, `sale tax`, `population`)."""
function bcag_read_usdata(
    path::AbstractString;
    economic_range::AbstractString = "A87:W226",
    tax_rows::AbstractString = "F2:G36",
    pop_submatrix::AbstractString = "A38:D72",
)
    xf = XLSX.readxlsx(path)
    M = XLSX.getdata(xf["Economic Outlook"], economic_range)
    n = size(M, 1)
    col(j) = Float64[_to_float(M[i, j]) for i in 1:n]
    econ = DataFrame(
        time_label = Any[M[i, 1] for i in 1:n],
        GDP = col(_COL_GDP), PGDP = col(_COL_PGDP), ITISK = col(_COL_ITISK),
        CG = col(_COL_CG), XGS = col(_COL_XGS), MGS = col(_COL_MGS),
        HRS = col(_COL_HRS), ET = col(_COL_ET), CP = col(_COL_CP),
        nXCD_billions_current = col(_COL_NXCD),
    )
    taxM = XLSX.getdata(xf["sale tax"], tax_rows)
    tax = DataFrame(
        year = [_to_year(taxM[i, 1]) for i in 1:size(taxM, 1)],
        tauc_pct = Float64[_to_float(taxM[i, 2]) for i in 1:size(taxM, 1)],
    )
    popM = XLSX.getdata(xf["population"], pop_submatrix)
    population = DataFrame(
        year = [_to_year(popM[i, 1]) for i in 1:size(popM, 1)],
        pop_15_64_thousands = Float64[_to_float(popM[i, 4]) for i in 1:size(popM, 1)],
    )
    (; econ, tax, population)
end

"""OECD CKM cleaning.  `annualize_nxcd=true` multiplies NXCD by 4 so `XCD/KCD` align with
the (SAAR) flows — needed for apples-to-apples comparison with BEA SAAR panels."""
function bcag_clean_usdata(
    path::AbstractString;
    economic_range::AbstractString = "A87:W226",
    durable_scale::Float64 = 1e9,
    annualize_nxcd::Bool = false,
    deltad::Float64 = 1 - (1 - 0.25)^(1 / 4),
    kcd_init_factor::Float64 = 16.0,
    tax_rows::AbstractString = "F2:G36",
    pop_submatrix::AbstractString = "A38:D72",
)
    xf = XLSX.readxlsx(path)
    M = XLSX.getdata(xf["Economic Outlook"], economic_range)
    n = size(M, 1)
    col(j) = Float64[_to_float(M[i, j]) for i in 1:n]

    GDP = col(_COL_GDP); PGDP = col(_COL_PGDP); CG = col(_COL_CG)
    ITISK = col(_COL_ITISK); MGS = col(_COL_MGS); XGS = col(_COL_XGS)
    HRS = col(_COL_HRS); ET = col(_COL_ET)
    nXCD = col(_COL_NXCD) .* durable_scale
    annualize_nxcd && (nXCD = nXCD .* 4.0)

    taxM = XLSX.getdata(xf["sale tax"], tax_rows)
    tauc_pct = Float64[_to_float(taxM[i, 2]) for i in 1:size(taxM, 1)]
    n_annual = length(tauc_pct); n_annual ≥ 2 || error("need ≥ 2 annual tax obs")

    xs_annual = range(1.0, step = 1.0, length = n_annual)
    tauc_q = _interp_linear(xs_annual, tauc_pct,
                            _numpy_arange(1.0, Float64(n_annual) + 0.76, 0.25)) ./ 100

    popM = XLSX.getdata(xf["population"], pop_submatrix)
    P_annual = Float64[_to_float(popM[i, 4]) for i in 1:size(popM, 1)] .* 1e3
    length(P_annual) == n_annual || error("pop length ≠ tax length")
    iP_q = _interp_cubic(xs_annual, P_annual,
                         _numpy_arange(1.25, Float64(n_annual) + 1.01, 0.25))
    (length(tauc_q) == n && length(iP_q) == n) || error("interp length ≠ econ panel rows")

    XCD = nXCD ./ PGDP
    KCD = Vector{Float64}(undef, n)
    KCD[1] = kcd_init_factor * XCD[1]
    for i in 2:n
        KCD[i] = (1 - deltad) * KCD[i - 1] + XCD[i - 1]
    end
    (; GDP, PGDP, ITISK, CG, XGS, MGS, HRS, ET, tauc_q, iP_q, KCD, XCD)
end

"""
    percapita_oecd_ckm_from_usdata(usdata_path; annualize_nxcd=true, kwargs...)

CKM per-capita series from `USdata.xlsx`, aligned with `percapita_bea_ckm_from_hw7_panel`.
Outputs already in **dollars per 15-64 person, SAAR** (OECD `USdata` flows are in dollars,
so no `×1e6` is needed).  `annualize_nxcd=true` puts `XCD/KCD` on the same SAAR basis as the
BEA panel.
"""
function percapita_oecd_ckm_from_usdata(
    usdata_path::AbstractString;
    r_d_annual::Float64 = 0.04,
    delta_d_annual::Float64 = 0.25,
    kcd_init_factor::Float64 = 16.0,
    annualize_nxcd::Bool = true,
    clean_kw...,
)
    r_d_q = (1 + r_d_annual)^(1 / 4) - 1
    delta_d_q = 1 - (1 - delta_d_annual)^(1 / 4)
    c = bcag_clean_usdata(usdata_path; kcd_init_factor, annualize_nxcd, clean_kw...)
    econ = bcag_read_usdata(usdata_path).econ

    Y = c.GDP ./ c.PGDP
    y = (Y .- c.tauc_q .* Y .+ (r_d_q + delta_d_q) .* c.KCD) ./ c.iP_q
    x = (c.ITISK ./ c.PGDP .+ (1 .- c.tauc_q) .* c.XCD) ./ c.iP_q
    g = (c.CG ./ c.PGDP .+ (c.XGS .- c.MGS) ./ c.PGDP) ./ c.iP_q
    h = (c.HRS .* c.ET) ./ c.iP_q

    DataFrame(;
        time_label = String[string(econ.time_label[i]) for i in 1:nrow(econ)],
        y, h, x, g, iP = c.iP_q, tauc = c.tauc_q, KCD_real = c.KCD,
    )
end

# ============================================================================
# Comparison — BEA vs OECD per-capita tables (both already in same units)
# ============================================================================

"""
    compare_percapita_bea_vs_oecd(data_pc_bea, data_pc_oecd) -> (; quarterly, annual, summary)

Inner-join the two per-capita `DataFrame`s on `time_label` and compute
`pct_z = (BEA − OECD)/OECD × 100` for `z ∈ {y, h, x, g}`, plus annual means (simple average
of overlapping quarters within each calendar year).  Both inputs must be in compatible units
(dollars per person for `y/x/g`; hours per person for `h`).
"""
function compare_percapita_bea_vs_oecd(
    data_pc_bea::AbstractDataFrame,
    data_pc_oecd::AbstractDataFrame,
)
    b = select(data_pc_bea,  :time_label,
               :y => :y_bea, :h => :h_bea, :x => :x_bea, :g => :g_bea)
    o = select(data_pc_oecd, :time_label,
               :y => :y_oecd, :h => :h_oecd, :x => :x_oecd, :g => :g_oecd)
    j = innerjoin(b, o; on = :time_label)
    isempty(j) && error("no overlapping quarters between BEA and OECD per-capita DataFrames")

    j.pct_y = [_pct_diff(β, ω) for (β, ω) in zip(j.y_bea, j.y_oecd)]
    j.pct_h = [_pct_diff(β, ω) for (β, ω) in zip(j.h_bea, j.h_oecd)]
    j.pct_x = [_pct_diff(β, ω) for (β, ω) in zip(j.x_bea, j.x_oecd)]
    j.pct_g = [_pct_diff(β, ω) for (β, ω) in zip(j.g_bea, j.g_oecd)]

    j.year = Int[something(_year_from_time_label(tl), 0) for tl in j.time_label]
    uy = sort!(unique(filter(>(0), j.year)))
    rows = NamedTuple[]
    for yr in uy
        s = j[j.year .== yr, :]
        yb, yo = _safe_mean(s.y_bea), _safe_mean(s.y_oecd)
        hb, ho = _safe_mean(s.h_bea), _safe_mean(s.h_oecd)
        xb, xo = _safe_mean(s.x_bea), _safe_mean(s.x_oecd)
        gb, go = _safe_mean(s.g_bea), _safe_mean(s.g_oecd)
        push!(rows, (;
            year = yr,
            y_bea = yb, y_oecd = yo, pct_y = _pct_diff(yb, yo),
            h_bea = hb, h_oecd = ho, pct_h = _pct_diff(hb, ho),
            x_bea = xb, x_oecd = xo, pct_x = _pct_diff(xb, xo),
            g_bea = gb, g_oecd = go, pct_g = _pct_diff(gb, go),
        ))
    end
    annual = DataFrame(rows)

    _mae(colname) = (v = filter(isfinite, collect(annual[!, colname])); isempty(v) ? NaN : mean(abs.(v)))
    summary = (;
        n_quarters = nrow(j), n_years = nrow(annual),
        mean_abs_pct_y = _mae(:pct_y), mean_abs_pct_h = _mae(:pct_h),
        mean_abs_pct_x = _mae(:pct_x), mean_abs_pct_g = _mae(:pct_g),
    )
    (; quarterly = j, annual, summary)
end

# ============================================================================
# Part 3 — CKM / Chari–Kehoe–McGrattan (2007) MLE dataset loader
#
#   Replicates the input layout of `sr328/mleqtrly/usdata.m` (which writes
#   `uszvarq.dat`) and `sr328/mleqtrly/runmle.m` / `mleq.m`:
#
#     usdata.m (MATLAB)
#       gz   = (1.016^(1/4))^81              % cumulative trend at 1979:1
#       mled = [t,  ypc/ypc(81)*gz,          % columns 2..5 written to disk
#                   xpc/ypc(81)*gz,
#                   hpc/1300,
#                   gpc/ypc(81)*gz]
#
#     mleq.m (MATLAB) then strips  log((1+gz)^{t-1})  from columns y, x, g
#     (hours has no trend) BEFORE evaluating the Kalman filter.
#
#   Our Julia `bca_mle` expects the **already-detrended** ZVAR.  So this
#   loader returns BOTH forms:
#     - `mled`           : raw file contents (matches MATLAB `ZVAR`)
#     - `mled_detrended` : log-cyclical, ready for `bca_mle`
# ============================================================================

"""
    load_ckm_uszvarq(path; gz = (1.016)^(1/4) - 1) -> NamedTuple

Read the CKM (2007) `uszvarq.dat` file (1959:1 – 2004:3 US quarterly MLE
dataset, written by `sr328/mleqtrly/usdata.m`).  Columns of the text file
are `[time, y, x, h, g]`.

# Arguments
- `path` : absolute / relative path to `uszvarq.dat`.
- `gz`   : quarterly per-capita productivity-growth rate used to de-trend
           `y, x, g` (hours has no trend).  Defaults to the paper's
           calibration `(1.016)^(1/4) − 1 ≈ 0.003971 / quarter ≈ 1.6 % / yr`.

# Returns a NamedTuple with
- `time_label::Vector{String}`    – `"Q1-1959"`, …, `"Q3-2004"`.
- `T::Int`                        – number of quarters (= `length(time_label)`).
- `mled::Matrix{Float64}`         – `T×4`, columns `[y, x, h, g]`, trend
                                     retained (== MATLAB `ZVAR`).
- `mled_detrended::Matrix{Float64}` – `T×4`, trend `(1+gz)^{t−1}` removed
                                     from `y, x, g`; hours column unchanged.
                                     **Pass this to `bca_mle`.**
- `gz::Float64`                   – the `gz` used for de-trending.
- `base_label::String`            – base-year label at the paper's base
                                     quarter (81 → `"Q1-1979"`).
- `base_idx::Int`                 – 81 (index in `time_label`).
- `raw::DataFrame`                – all 5 columns as a `DataFrame`.

# Example
```julia
ckm  = load_ckm_uszvarq("sr328/mleqtrly/uszvarq.dat")
ZVAR = ckm.mled_detrended          # 183 × 4, ready for bca_mle
```
"""
function load_ckm_uszvarq(
    path::AbstractString;
    gz::Real = (1.016)^(1/4) - 1,
)
    isfile(path) || error("load_ckm_uszvarq: file not found at $(path)")
    raw = readdlm(path)
    size(raw, 2) == 5 ||
        error("load_ckm_uszvarq: expected 5 columns (time, y, x, h, g), got $(size(raw, 2))")

    T = size(raw, 1)
    time_num = Float64.(raw[:, 1])

    # 1959.125 -> Q1-1959,  1959.375 -> Q2-1959,  1959.625 -> Q3,  1959.875 -> Q4
    year   = Int.(floor.(time_num))
    frac   = time_num .- year
    qtr    = Int.(floor.(frac .* 4)) .+ 1
    all((1 .≤ qtr) .& (qtr .≤ 4)) ||
        error("load_ckm_uszvarq: failed to decode quarter from time column (got $qtr)")
    time_label = ["Q$(q)-$(yy)" for (q, yy) in zip(qtr, year)]

    # Base quarter — paper uses 1979:1 = row 81.
    base_idx  = something(findfirst(==("Q1-1979"), time_label), 81)
    base_lab  = time_label[base_idx]

    y_col = Float64.(raw[:, 2])
    x_col = Float64.(raw[:, 3])
    h_col = Float64.(raw[:, 4])
    g_col = Float64.(raw[:, 5])

    mled = hcat(y_col, x_col, h_col, g_col)

    # mleq.m strips log((1+gz)^{t-1}) from y, x, g before the Kalman filter.
    gz_val = Float64(gz)
    trend  = (1.0 + gz_val) .^ collect(0:T-1)                 # (1+gz)^{t-1} for t=1..T
    mled_detrended = hcat(y_col ./ trend,
                          x_col ./ trend,
                          h_col,                              # hours has no trend
                          g_col ./ trend)

    raw_df = DataFrame(
        time_label = time_label,
        time_num   = time_num,
        y = y_col, x = x_col, h = h_col, g = g_col,
    )

    return (; time_label, T,
              mled, mled_detrended,
              gz = gz_val,
              base_label = base_lab, base_idx,
              raw = raw_df)
end

"""
    ckm_runmle_theta0() -> Vector{Float64}

Paper's starting-point `x0` from `sr328/mleqtrly/runmle.m` (labeled
`% result from initpw with adja=0`), re-expressed in our Julia 30-vector
`unpack_theta_wedges` layout.  Use as an excellent warm-start for
`bca_mle` on the CKM `uszvarq.dat` dataset; the optimum lands within a few
BFGS iterations and should reproduce the paper's Table B.
"""
function ckm_runmle_theta0()
    return [
        # Sbar = [log z_s, τ_ℓs, τ_xs, log g_s]
         0.56129229542991, -0.22315683391720,  0.35940299493795, -2.31053250090644,
        # P column-major:  P[:,1]
         0.97495145690175, -0.00748563008432, -0.00597701490688,  0.0,
        # P[:,2]
         0.01728027189850,  1.01435681124639, -0.00935002415423,  0.0,
        # P[:,3]
         0.01392074289790,  0.04681660428893,  0.95238089627209,  0.0,
        # P[:,4]
         0.0,               0.0,               0.0,                0.96606868092918,
        # Q lower-triangular, column-major:  Q[1,1], Q[2,1], Q[3,1], Q[4,1]
         0.02379215892897, -0.01064290730504,  0.01295512011754,  0.0,
        # Q[2,2], Q[3,2], Q[4,2]
         0.02736479689259, -0.01874060582004,  0.0,
        # Q[3,3], Q[4,3]
         0.02732929989630,  0.0,
        # Q[4,4]
         0.11138886148534,
    ]
end
