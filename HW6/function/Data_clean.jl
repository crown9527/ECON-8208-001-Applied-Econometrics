using XLSX
using DataFrames
using Statistics
using LinearAlgebra


# ═══════════════════════════════════════════════════════════
#  Part 1 — Utilities
# ═══════════════════════════════════════════════════════════

# -------------------------------------------------------
# HP filter
# -------------------------------------------------------
function hp_filter(y, lambda)
    T = length(y)
    D = zeros(T - 2, T)
    for i in 1:(T - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    trend = (I + lambda * (D' * D)) \ y
    cycle = y - trend
    return trend, cycle
end


# -------------------------------------------------------
# Read a single data line from a BEA NIPA table Excel file.
#
# BEA table layout:
#   Row 6  : year headers  (columns 3 onward)
#   Row 8+ : data rows — column 1 = Line number, column 2 = label
#
# Returns (years::Vector{Int}, values::Vector{Float64}).
# -------------------------------------------------------
function read_bea_line(path, line_number; sheet_name="Table")
    xf   = XLSX.readxlsx(path)
    ws   = xf[sheet_name]
    raw  = ws[:, :]
    nrow_raw, ncol_raw = size(raw)

    target = string(line_number)
    data_row = nothing
    for r in 8:nrow_raw
        ln = raw[r, 1]
        if !isnothing(ln) && !ismissing(ln) && strip(string(ln)) == target
            data_row = r; break
        end
    end
    isnothing(data_row) && error("Line $line_number not found in $path")

    years_raw = [raw[6, c]        for c in 3:ncol_raw]
    vals_raw  = [raw[data_row, c] for c in 3:ncol_raw]

    function is_numeric(v)
        isnothing(v) && return false
        ismissing(v) && return false
        v isa Number && return true
        s = strip(string(v))
        return !isempty(s) && s != "---" && tryparse(Float64, s) !== nothing
    end

    valid = [is_numeric(years_raw[i]) && is_numeric(vals_raw[i])
             for i in eachindex(years_raw)]

    to_f64(v) = v isa Number ? Float64(v) : parse(Float64, strip(string(v)))
    return Int.(to_f64.(years_raw[valid])), to_f64.(vals_raw[valid])
end


# ═══════════════════════════════════════════════════════════
#  Part 2 — Data Loading & Basic Moments
# ═══════════════════════════════════════════════════════════

# -------------------------------------------------------
# Load raw data from Excel  (→ DataFrame)
# -------------------------------------------------------
function load_raw_data(path; sheet_name="Sheet1", start_year=1960)
    xf = XLSX.readxlsx(path)
    ws = xf[sheet_name]
    raw = ws[:, :]

    source_row = vec(raw[1, :])
    table_row  = vec(raw[2, :])
    var_row    = vec(raw[3, :])

    function find_col(; source_contains="", table_contains="", var_contains="")
        idx = findall(1:length(var_row)) do j
            s = isnothing(source_row[j]) || ismissing(source_row[j]) ? "" : string(source_row[j])
            t = isnothing(table_row[j])  || ismissing(table_row[j])  ? "" : string(table_row[j])
            v = isnothing(var_row[j])    || ismissing(var_row[j])    ? "" : string(var_row[j])
            occursin(source_contains, s) &&
            occursin(table_contains, t)  &&
            occursin(var_contains, v)
        end
        length(idx) != 1 && error(
            "Expected 1 column for source='$source_contains', " *
            "table='$table_contains', var='$var_contains', found $(length(idx)).")
        return idx[1]
    end

    col_year             = 1
    col_gdp_real         = find_col(source_contains="BEA", table_contains="1.1.6", var_contains="GDP 2017 prices")
    col_employment       = find_col(source_contains="BEA", table_contains="6.4A",  var_contains="Employment")
    col_gdp_nominal      = find_col(source_contains="BEA", table_contains="1.1.5", var_contains="GDP current prices")
    col_comp_employees   = find_col(source_contains="BEA", table_contains="6.2A",  var_contains="Compensation of Employees")
    col_proprietors      = find_col(source_contains="BEA", table_contains="6.12A", var_contains="Proprietors' Income")
    col_taxes            = find_col(source_contains="BEA", table_contains="3.5",   var_contains="Taxes on Production and Imports")
    col_subsidies        = find_col(source_contains="BEA", table_contains="3.13",  var_contains="Subsidies")
    col_gross_investment  = find_col(source_contains="BEA", table_contains="5.1",  var_contains="Gross Domestic Investment")
    col_depreciation     = find_col(source_contains="BEA", table_contains="5.1",   var_contains="Consumption of Fixed Capital")
    col_private_assets   = find_col(source_contains="BEA", table_contains="6.1",   var_contains="Private fixed assets")
    col_population       = find_col(source_contains="World Bank", var_contains="total")
    col_hours_employees  = find_col(source_contains="BEA", table_contains="6.9B",  var_contains="Hours Worked by Employees")
    col_self_employed    = find_col(source_contains="BEA", table_contains="6.7B",  var_contains="Self-Employed Workers")
    col_real_capital_stock = find_col(source_contains="compute", table_contains="0", var_contains="Real Capital Stock")

    function to_float_col(col_idx)
        x = raw[5:end, col_idx]
        out = Vector{Union{Missing, Float64}}(undef, length(x))
        for i in eachindex(x)
            if ismissing(x[i]) || x[i] === nothing || string(x[i]) == ""
                out[i] = missing
            else
                out[i] = Float64(x[i])
            end
        end
        return out
    end

    years = Int.(raw[5:end, col_year])

    df = DataFrame(
        year                 = years,
        gdp_real             = to_float_col(col_gdp_real),
        employment           = to_float_col(col_employment),
        gdp_nominal          = to_float_col(col_gdp_nominal),
        comp_employees       = to_float_col(col_comp_employees),
        proprietors_income   = to_float_col(col_proprietors),
        taxes_prod_imports   = to_float_col(col_taxes),
        subsidies            = to_float_col(col_subsidies),
        gross_investment     = to_float_col(col_gross_investment),
        depreciation         = to_float_col(col_depreciation),
        private_fixed_assets = to_float_col(col_private_assets),
        real_capital_stock   = to_float_col(col_real_capital_stock),
        population           = to_float_col(col_population),
        hours_employees      = to_float_col(col_hours_employees),
        self_employed_workers = to_float_col(col_self_employed),
        col_capital_stock    = to_float_col(col_real_capital_stock),
    )

    df = dropmissing(df, [
        :year, :gdp_real, :gdp_nominal, :employment, :comp_employees,
        :proprietors_income, :taxes_prod_imports, :subsidies,
        :gross_investment, :depreciation, :private_fixed_assets,
        :population, :real_capital_stock])

    df = filter(row -> row.year >= start_year, df)
    return df
end


# -------------------------------------------------------
# Construct data moments for calibration
#   → (df2, hours_df, cycle_y_pc, moments::NamedTuple)
#
# Optional: pass `fiscal` DataFrame (from compute_fiscal_states)
#   to include gov_output_ratio.
# -------------------------------------------------------
function compute_data_moments(df; hp_lambda=6.25, fiscal=nothing)
    df2 = copy(df)

    df2.y_per_capita_real = df2.gdp_real ./ df2.population

    population_growth_series       = diff(log.(df2.population))
    output_per_worker_growth_series = diff(log.(df2.y_per_capita_real))

    factor_income = df2.gdp_nominal .-
                    df2.proprietors_income .-
                    df2.taxes_prod_imports .+
                    df2.subsidies

    df2.labor_share   = df2.comp_employees ./ factor_income
    df2.capital_share = 1 .- df2.labor_share

    df2.investment_capital_ratio  = df2.gross_investment ./ df2.private_fixed_assets
    df2.investment_output_ratio   = df2.gross_investment ./ df2.gdp_nominal
    df2.consumption_output_ratio  = (df2.gdp_nominal .- df2.gross_investment) ./ df2.gdp_nominal
    df2.capital_output_ratio      = df2.private_fixed_assets ./ df2.gdp_nominal
    df2.depreciation_capital_ratio = df2.depreciation ./ df2.private_fixed_assets

    hours_mask = .!ismissing.(df2.hours_employees) .& .!ismissing.(df2.self_employed_workers)
    hours_df   = df2[hours_mask, :]

    hours_df.self_employed_hours =
        (hours_df.hours_employees ./ hours_df.employment) .* hours_df.self_employed_workers
    hours_df.total_hours   = hours_df.hours_employees .+ hours_df.self_employed_hours
    hours_df.total_workers = hours_df.employment .+ hours_df.self_employed_workers
    hours_df.potential_hours = hours_df.total_workers .* 52.0 .* 100.0 ./ 1000.0
    hours_df.h = hours_df.total_hours ./ hours_df.potential_hours
    hours_df.l = 1 .- hours_df.h

    log_y_pc = log.(df2.y_per_capita_real)
    _, cycle_y_pc = hp_filter(log_y_pc, hp_lambda)

    gov_output = NaN
    if !isnothing(fiscal)
        merged = innerjoin(df2[:, [:year, :gdp_real, :population]],
                           fiscal[:, [:year, :gov_real_pc]], on = :year)
        gov_output = mean(merged.gov_real_pc ./ (merged.gdp_real ./ merged.population))
    end

    moments = (
        population_growth        = mean(population_growth_series),
        output_per_worker_growth = mean(output_per_worker_growth_series),
        labor_share              = mean(df2.labor_share),
        capital_share            = mean(df2.capital_share),
        investment_capital_ratio  = mean(df2.investment_capital_ratio),
        investment_output_ratio  = mean(df2.investment_output_ratio),
        consumption_output_ratio = mean(df2.consumption_output_ratio),
        depreciation_capital_ratio = mean(df2.depreciation_capital_ratio),
        capital_output_ratio     = mean(df2.capital_output_ratio),
        average_hours            = mean(hours_df.h),
        average_leisure          = mean(hours_df.l),
        output_cycle_autocorr    = cor(cycle_y_pc[2:end], cycle_y_pc[1:end-1]),
        output_cycle_std         = std(cycle_y_pc),
        gov_output_ratio         = gov_output,
    )

    return df2, hours_df, cycle_y_pc, moments
end


# -------------------------------------------------------
# Estimate Solow residual and AR(1) technology parameters
#   → NamedTuple (gamma_z, rho, sigma_epsilon, log_z, years, …)
# -------------------------------------------------------
function estimate_solow_residual(hours_df, theta)
    T = size(hours_df, 1)

    Y   = Float64.(hours_df.gdp_real)
    K   = Float64.(hours_df.real_capital_stock)
    H   = Float64.(hours_df.total_hours)
    pop = Float64.(hours_df.population)

    y_pc    = Y ./ pop
    gamma_z = exp(sum(diff(log.(y_pc))) / (T - 1)) - 1
    g       = log(1 + gamma_z)

    log_tfp = log.(Y) .- theta .* log.(K) .- (1 - theta) .* log.(H)
    s       = log_tfp ./ (1 - theta)
    log_z   = s .- g .* Float64.(0:(T - 1))

    y_ar = log_z[2:end]
    x_ar = log_z[1:end-1]
    rho  = sum(x_ar .* y_ar) / sum(x_ar .^ 2)

    eps          = y_ar .- rho .* x_ar
    sigma_eps_sq = sum(eps .^ 2) / (length(eps) - 1)

    return (gamma_z          = gamma_z,
            rho              = rho,
            sigma_epsilon_sq = sigma_eps_sq,
            sigma_epsilon    = sqrt(sigma_eps_sq),
            log_z            = log_z,
            log_tfp          = log_tfp,
            years            = hours_df.year)
end


# ═══════════════════════════════════════════════════════════
#  Part 3 — Fiscal State Variables
# ═══════════════════════════════════════════════════════════

# -------------------------------------------------------
# Compute fiscal state-variable time series
#   S_t = [log z_t, τ_c, τ_h, τ_d, τ_p, log ĝ_t]
#
# Data sources:
#   τ_c : (Fed excise + S&L excise + Sales taxes) / (PCE − Revenue)
#         Table 3.5  Lines 4, 20, 23;  Table 1.1.5 Line 2
#   τ_h : (Personal current taxes + Employer social ins.) / Compensation
#         Table 2.1  Lines 26, 8, 2
#   τ_p : Taxes on corporate income / Corporate profits
#         Table 1.10 Lines 16, 15
#   τ_d : Constant calibration (default 0.25, McGrattan 2012)
#   ĝ   : Real per-capita gov spending, detrended by γ_z
#         Table 1.1.5 Line 22
#
#   → DataFrame with :year, :tau_c, :tau_h, :tau_d, :tau_p, :log_g_hat, …
# -------------------------------------------------------
function compute_fiscal_states(data_dir, df, sr; tau_d_const=0.25)
    g = log(1 + sr.gamma_z)

    tbl115 = joinpath(data_dir, "Table 1.1.5. Gross Domestic Product.xlsx")
    tbl110 = joinpath(data_dir, "Table 1.10. Gross Domestic Income by Type of Income.xlsx")
    tbl21  = joinpath(data_dir, "Table 2.1. Personal Income and Its Disposition.xlsx")
    tbl35  = joinpath(data_dir, "Table 3.5. Taxes on Production and Imports.xlsx")

    series = [
        (tbl115, 22, :gov_nominal),      # Government spending
        (tbl115,  2, :pce_nominal),      # Personal consumption expenditures
        (tbl110, 16, :corp_tax),         # Taxes on corporate income
        (tbl110, 15, :corp_prof),        # Corporate profits (before tax)
        (tbl21,   8, :employer_si),      # Employer contributions for gov social insurance
        (tbl21,  26, :personal_tax),     # Personal current taxes
        (tbl21,   2, :comp_emp),         # Compensation of employees
        (tbl35,   4, :fed_excise),       # Federal excise taxes
        (tbl35,  20, :sales_tax),        # Sales taxes (state & local)
        (tbl35,  23, :sl_excise),        # State & local excise taxes
    ]

    base = df[:, [:year, :gdp_nominal, :gdp_real, :population]]
    for (path, line, colname) in series
        yrs, vals = read_bea_line(path, line)
        tmp = DataFrame(:year => yrs, colname => vals)
        base = innerjoin(base, tmp, on = :year)
    end
    sort!(base, :year)

    # τ_c
    cons_tax_rev = base.fed_excise .+ base.sl_excise .+ base.sales_tax
    tau_c = cons_tax_rev ./ (base.pce_nominal .- cons_tax_rev)

    # τ_h
    labor_tax_rev = base.personal_tax .+ base.employer_si
    tau_h = labor_tax_rev ./ base.comp_emp

    # τ_p
    tau_p = base.corp_tax ./ base.corp_prof

    # τ_d (constant)
    T = nrow(base)
    tau_d = fill(tau_d_const, T)

    # log ĝ
    deflator    = base.gdp_nominal ./ base.gdp_real
    gov_real    = base.gov_nominal ./ deflator
    gov_real_pc = gov_real ./ base.population
    log_g_hat   = log.(gov_real_pc) .- g .* Float64.(0:T-1)

    fiscal = DataFrame(
        year          = base.year,
        tau_c         = tau_c,
        tau_h         = tau_h,
        tau_d         = tau_d,
        tau_p         = tau_p,
        log_g_hat     = log_g_hat,
        gov_real_pc   = gov_real_pc,
        cons_tax_rev  = cons_tax_rev,
        labor_tax_rev = labor_tax_rev,
        corp_tax      = base.corp_tax,
        corp_prof     = base.corp_prof,
    )

    return fiscal
end
