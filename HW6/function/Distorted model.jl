"""
    calibrate_hw6(moments, fiscal_means, sr; σ=2.0)

Calibrate (β, ψ) and steady-state fiscal parameters from data moments.

Inputs:
  moments      – NamedTuple from compute_data_moments (needs capital_share,
                 population_growth, depreciation_capital_ratio, capital_output_ratio,
                 average_hours, gov_output_ratio)
  fiscal_means – NamedTuple or anything with fields tau_c, tau_h, tau_d, tau_p
  sr           – NamedTuple from estimate_solow_residual (needs gamma_z, rho)
  σ            – CRRA parameter (default 2)

Returns NamedTuple with all calibrated parameters and the steady state.
"""
function calibrate_hw6(moments, fiscal_means, sr; σ=2.0)
    θ   = moments.capital_share
    γ_n = moments.population_growth
    γ_z = sr.gamma_z
    δ   = moments.depreciation_capital_ratio

    τ_c = fiscal_means.tau_c
    τ_h = fiscal_means.tau_h
    τ_d = fiscal_means.tau_d
    τ_p = fiscal_means.tau_p
    g_share = moments.gov_output_ratio

    growth    = (1 + γ_z) * (1 + γ_n)
    YK_ratio  = 1.0 / moments.capital_output_ratio
    β_tilde   = growth / ((1 - τ_p) * θ * YK_ratio + τ_p * δ + 1 - δ)
    β         = β_tilde / ((1 + γ_n) * (1 + γ_z)^(1 - σ))

    h_data    = moments.average_hours
    euler_rhs = growth / β_tilde - τ_p * δ - 1 + δ
    kappa     = ((1 - τ_p) * θ / euler_rhs)^(1 / (1 - θ))

    ĝ       = g_share * kappa^θ * h_data
    c_kappa  = kappa^θ + (1 - δ - growth) * kappa
    c_ss_cal = h_data * c_kappa - ĝ

    ψ = (1 - τ_h) * (1 - θ) * kappa^θ * (1 - h_data) / ((1 + τ_c) * c_ss_cal)

    ss = steady_state_hw6(θ=θ, β=β, δ=δ, σ=σ, ψ=ψ, γ_z=γ_z, γ_n=γ_n,
                          τ_c=τ_c, τ_h=τ_h, τ_d=τ_d, τ_p=τ_p, ĝ=ĝ)

    return (θ=θ, β=β, δ=δ, σ=σ, ψ=ψ, γ_z=γ_z, γ_n=γ_n, ρ_z=sr.rho,
            τ_c=τ_c, τ_h=τ_h, τ_d=τ_d, τ_p=τ_p, g_share=g_share, ĝ=ĝ,
            ss=ss)
end


"""
    steady_state_hw6(; θ, β, δ, σ, ψ, γ_z, γ_n, τ_c, τ_h, τ_d, τ_p, ĝ)

Deterministic steady state of the detrended growth model with fiscal shocks.

Euler equation with τ_p:
  growth/β̃ = (1−τ_p)r + τ_p δ + 1−δ   →  κ = ((1−τ_p)θ / (growth/β̃ − τ_p δ − 1+δ))^{1/(1−θ)}

Intratemporal FOC with τ_c, τ_h:
  ψ c/(1−h) = (1−τ_h) w̃ / (1+τ_c)

  h_ss = (w_eff + ψ ĝ) / (ψ c_κ + w_eff),  w_eff = (1−τ_h)(1−θ)κ^θ/(1+τ_c)

Returns NamedTuple with k_ss, h_ss, c_ss, r_ss, w_ss, x_inv_ss, κ_ss, β_tilde, growth, ĝ_ss
"""
function steady_state_hw6(; θ, β, δ, σ, ψ, γ_z, γ_n,
                            τ_c=0.0, τ_h=0.0, τ_d=0.0, τ_p=0.0, ĝ=0.0)
    growth  = (1 + γ_z) * (1 + γ_n)
    β_tilde = β * (1 + γ_n) * (1 + γ_z)^(1 - σ)

    euler_rhs = growth / β_tilde - τ_p * δ - 1 + δ
    kappa     = ((1 - τ_p) * θ / euler_rhs)^(1 / (1 - θ))

    c_kappa = kappa^θ + (1 - δ - growth) * kappa
    w_eff   = (1 - τ_h) * (1 - θ) * kappa^θ / (1 + τ_c)
    h_ss    = (w_eff + ψ * ĝ) / (ψ * c_kappa + w_eff)
    k_ss    = kappa * h_ss
    c_ss    = k_ss^θ * h_ss^(1 - θ) + (1 - δ) * k_ss - growth * k_ss - ĝ

    r_ss     = θ * k_ss^(θ - 1) * h_ss^(1 - θ)
    w_ss     = (1 - θ) * k_ss^θ * h_ss^(-θ)
    x_inv_ss = growth * k_ss - (1 - δ) * k_ss

    tax_rev = τ_c * c_ss + τ_h * w_ss * h_ss +
              τ_p * (r_ss * k_ss - δ * k_ss) +
              τ_d * (r_ss * k_ss - x_inv_ss - τ_p * (r_ss * k_ss - δ * k_ss))
    κ_ss = tax_rev - ĝ
    abs(κ_ss) < 1e-6 && (κ_ss = 0.0)

    return (k_ss=k_ss, h_ss=h_ss, c_ss=c_ss, r_ss=r_ss, w_ss=w_ss,
            x_inv_ss=x_inv_ss, κ_ss=κ_ss, β_tilde=β_tilde, growth=growth,
            ĝ_ss=ĝ)
end


"""
    make_return_fn(; θ, δ, σ, ψ, growth)


State  x = [log k̂, log z, τ_c, τ_h, τ_d, τ_p, log ĝ, log K̂, log H, log K̂']  (n = 10)
Control u = [log k̂', log h]                                                      (m = 2)

ĉ is determined from the household budget constraint; aggregates K̂, H, K̂'
enter through factor prices r, w̃ and government transfers κ̂.
"""
function make_return_fn(; θ, δ, σ, ψ, growth)
    function return_fn(x, u)
        k = exp(x[1]);  z = exp(x[2])
        τc = x[3]; τh = x[4]; τd = x[5]; τp = x[6]
        g  = exp(x[7])
        K  = exp(x[8]); H = exp(x[9]); Kp = exp(x[10])
        kp = exp(u[1]); h = exp(u[2])

        r_p = θ * K^(θ-1) * (z * H)^(1-θ)
        w_p = (1-θ) * K^θ * z^(1-θ) * H^(-θ)

        X_agg = growth * Kp - (1-δ) * K
        C_agg = K^θ * (z*H)^(1-θ) + (1-δ)*K - growth*Kp - g

        tax_rev = τc*C_agg + τh*w_p*H + τp*(r_p*K - δ*K) +
                  τd*(r_p*K - X_agg - τp*(r_p*K - δ*K))
        κ = tax_rev - g

        x_inv  = growth * kp - (1-δ) * k
        income = (1-τd)*((1-τp)*r_p + τp*δ)*k + (1-τh)*w_p*h + κ
        c = (income - (1-τd)*x_inv) / (1 + τc)

        l = 1 - h
        (c <= 0 || l <= 0) && return -1e12
        return (c * l^ψ)^(1-σ) / (1-σ)
    end
    return return_fn
end


"""
    estimate_fiscal_var1(fiscal, sr)

Estimate VAR(1) for the exogenous state vector
  S_t = [log z, τ_c, τ_h, τ_d, τ_p, log ĝ]  (6×1)

from fiscal data and Solow residual estimates.

τ_d is constant (not estimated) — its row in P is zero and P₀[4] = τ_d_ss.

Returns NamedTuple
  (P0, P, Σ, Q, S_mat, years, S_ss)
where
  P0   : 6×1 intercept
  P    : 6×6 coefficient matrix
  Σ    : 6×6 innovation covariance
  Q    : 6×6 Cholesky factor  (Σ = Q Q')
  S_mat: T×6 state matrix (aligned years)
  years: year vector
  S_ss : 6×1 unconditional mean  = (I - P)⁻¹ P₀
"""
function estimate_fiscal_var1(fiscal, sr)
    sr_df = DataFrame(year = Int.(sr.years), log_z = sr.log_z)
    merged = innerjoin(sr_df, fiscal[:, [:year, :tau_c, :tau_h, :tau_d, :tau_p, :log_g_hat]],
                       on = :year)
    sort!(merged, :year)

    T = nrow(merged)
    n_s = 6   # dimension of S_t

    S = hcat(merged.log_z, merged.tau_c, merged.tau_h,
             merged.tau_d, merged.tau_p, merged.log_g_hat)   # T × 6

    # τ_d (column 4) is constant → exclude from OLS, handle separately
    vary_idx = [1, 2, 3, 5, 6]      # indices of time-varying components
    fix_idx  = 4                     # τ_d
    n_v = length(vary_idx)           # 5

    Y = S[2:T, vary_idx]                              # (T-1) × 5
    X = hcat(ones(T-1), S[1:T-1, vary_idx])           # (T-1) × 6  [const, S_{t-1}^vary]

    B = (X' * X) \ (X' * Y)         # 6 × 5   (row 1 = intercept, rows 2-6 = coefficients)
    p0_v = B[1, :]                   # 5×1 intercept
    P_v  = B[2:end, :]'              # 5×5 coefficient (each row = equation)

    E = Y - X * B                    # (T-1) × 5 residuals
    Σ_v = (E' * E) / (T - 1 - n_v - 1)

    # Assemble full 6×6 matrices
    P0 = zeros(n_s)
    P  = zeros(n_s, n_s)
    Σ  = zeros(n_s, n_s)

    P0[vary_idx] = p0_v
    P0[fix_idx]  = S[1, fix_idx]     # τ_d constant → P₀ = τ_d_ss

    P[vary_idx, vary_idx] = P_v
    # P[fix_idx, :] = 0, P[:, fix_idx] = 0  (already zero)

    Σ[vary_idx, vary_idx] = Σ_v
    # Σ[fix_idx, :] = Σ[:, fix_idx] = 0

    Σ_sym = Symmetric(Σ)
    Q = zeros(n_s, n_s)
    Q[vary_idx, vary_idx] = cholesky(Symmetric(Σ_v)).L

    S_ss = (I(n_s) - P) \ P0

    return (P0 = P0, P = P, Σ = Matrix(Σ_sym), Q = Q,
            S_mat = S, years = merged.year, S_ss = S_ss)
end


"""
    estimate_fiscal_ar1(fiscal, sr)

Estimate **independent** AR(1) processes for each component of
  S_t = [log z, τ_c, τ_h, τ_d, τ_p, log ĝ]

Each variable i (except τ_d which is constant):
  S_{i,t} = p0_i + ρ_i S_{i,t-1} + σ_i ε_{i,t},   ε ~ N(0,1)

Returns the same NamedTuple format as estimate_fiscal_var1
  (P0, P, Σ, Q, S_mat, years, S_ss)
with P, Σ, Q all **diagonal**.
"""
function estimate_fiscal_ar1(fiscal, sr)
    sr_df = DataFrame(year = Int.(sr.years), log_z = sr.log_z)
    merged = innerjoin(sr_df, fiscal[:, [:year, :tau_c, :tau_h, :tau_d, :tau_p, :log_g_hat]],
                       on = :year)
    sort!(merged, :year)

    T  = nrow(merged)
    n_s = 6

    S = hcat(merged.log_z, merged.tau_c, merged.tau_h,
             merged.tau_d, merged.tau_p, merged.log_g_hat)

    vary_idx = [1, 2, 3, 5, 6]
    fix_idx  = 4

    P0   = zeros(n_s)
    rho  = zeros(n_s)
    sig2 = zeros(n_s)

    for i in vary_idx
        y_i = S[2:T, i]
        x_i = S[1:T-1, i]
        n   = length(y_i)
        x_c = hcat(ones(n), x_i)              # [1, S_{i,t-1}]
        b   = (x_c' * x_c) \ (x_c' * y_i)    # [intercept, ρ_i]
        P0[i]   = b[1]
        rho[i]  = b[2]
        e       = y_i - x_c * b
        sig2[i] = sum(e .^ 2) / (n - 2)
    end

    P0[fix_idx] = S[1, fix_idx]

    P = diagm(rho)
    Σ = diagm(sig2)
    Q = diagm(sqrt.(sig2))

    S_ss = zeros(n_s)
    for i in vary_idx
        S_ss[i] = P0[i] / (1 - rho[i])
    end
    S_ss[fix_idx] = S[1, fix_idx]

    return (P0 = P0, P = P, Σ = Σ, Q = Q,
            S_mat = S, years = merged.year, S_ss = S_ss)
end


"""
    make_transition_fn(P0_ar, P_ar)

Return a closure `transition_fn(y, u)` for the HW6 model.

y = [log k̂, log z, τ_c, τ_h, τ_d, τ_p, log ĝ]  (ny = 7)
y_{t+1} = [log k̂',  P₀ + P · X₂_t]
"""
function make_transition_fn(P0_ar, P_ar)
    function transition_fn(y, u)
        X2_next = P0_ar + P_ar * y[2:end]
        return [u[1]; X2_next]
    end
    return transition_fn
end


using Random

"""
    simulate_model_hw6(F_eq, ss, var1; θ, δ, γ_z, γ_n, T, seed)

Simulate the HW6 growth model with fiscal shocks.

Policy:  u_t = u_ss − F_eq (y_t − y_ss),   y = [log k̂, S_t]
Shocks:  S_{t+1} = P₀ + P S_t + Q ε_t,    ε ~ N(0, I₆)

Returns NamedTuple of per-capita level series (scaled by (1+γ_z)^t):
  y, c, x, g  (length T);   h, l  (length T);
  k (length T+1);   n (length T+1);   S (T+1 × 6)
"""
function simulate_model_hw6(F_eq, ss, var1;
                            θ, δ, γ_z, γ_n, T=200, seed=111)
    k_ss   = ss.k_ss;  h_ss = ss.h_ss
    growth = ss.growth
    ĝ_model = ss.ĝ_ss
    Gz = 1 + γ_z;  Gn = 1 + γ_n
    S_ss = var1.S_ss
    P0 = var1.P0;  P = var1.P;  Qm = var1.Q
    n_s = length(S_ss)

    rng = MersenneTwister(seed)

    k_hat = zeros(T+1)
    Sv    = zeros(T+1, n_s)
    y_out = zeros(T);  c_out = zeros(T);  x_out = zeros(T);  g_out = zeros(T)
    h_sim = zeros(T);  l_sim = zeros(T)
    n_sim = zeros(T+1)

    k_hat[1] = k_ss
    Sv[1, :] = S_ss
    n_sim[1] = 1.0

    y_ss_vec = [log(k_ss); S_ss]
    u_ss_vec = [log(k_ss), log(h_ss)]

    for t in 1:T
        eps = randn(rng, n_s)

        dy   = [log(k_hat[t]); Sv[t, :]] .- y_ss_vec
        u_t  = u_ss_vec .- F_eq * dy
        kp   = exp(u_t[1])
        ht   = clamp(exp(u_t[2]), 0.01, 0.99)

        zt = exp(Sv[t, 1] - S_ss[1])
        gt = ĝ_model * exp(Sv[t, 6] - S_ss[6])

        y_hat = k_hat[t]^θ * (zt * ht)^(1-θ)
        x_hat = growth * kp - (1-δ) * k_hat[t]
        c_hat = y_hat + (1-δ) * k_hat[t] - growth * kp - gt

        sc = Gz^(t-1)
        y_out[t] = y_hat * sc
        c_out[t] = max(c_hat * sc, 1e-12)
        x_out[t] = x_hat * sc
        g_out[t] = gt * sc
        h_sim[t] = ht;  l_sim[t] = 1 - ht

        k_hat[t+1]  = kp
        Sv[t+1, :]  = P0 + P * Sv[t, :] + Qm * eps
        n_sim[t+1]  = Gn * n_sim[t]
    end

    k_level = [k_hat[t] * Gz^(t-1) for t in 1:(T+1)]

    return (y=y_out, c=c_out, x=x_out, g=g_out,
            h=h_sim, l=l_sim, n=n_sim, k=k_level,
            S=Sv, k_hat=k_hat)
end


"""
    compute_model_moments_hw6(sim, θ; burn_in, hp_lambda)

Compute model moments from simulate_model_hw6 output.
Requires `hp_filter` (from Data_clean.jl).
"""
function compute_model_moments_hw6(sim, θ; burn_in=100, hp_lambda=6.25)
    idx = (burn_in+1):length(sim.c)
    c = sim.c[idx];  x = sim.x[idx];  g = sim.g[idx]
    h = sim.h[idx];  l = sim.l[idx]
    n = sim.n[idx];  k = sim.k[idx]
    y = sim.y[idx]

    _mean(v) = sum(v) / length(v)
    _std(v)  = sqrt(sum((v .- _mean(v)).^2) / (length(v) - 1))
    _cor(a,b) = begin
        ma, mb = _mean(a), _mean(b)
        sum((a .- ma) .* (b .- mb)) /
            sqrt(sum((a .- ma).^2) * sum((b .- mb).^2))
    end

    _, cycle_y = hp_filter(log.(y), hp_lambda)

    return (
        output_per_worker_growth = _mean(diff(log.(y))),
        capital_share            = θ,
        investment_output_ratio  = _mean(x ./ y),
        consumption_output_ratio = _mean(c ./ y),
        gov_output_ratio         = _mean(g ./ y),
        capital_output_ratio     = _mean(k[1:length(y)] ./ y),
        average_hours            = _mean(h),
        average_leisure          = _mean(l),
        output_cycle_autocorr    = _cor(cycle_y[2:end], cycle_y[1:end-1]),
        output_cycle_std         = _std(cycle_y),
    )
end
