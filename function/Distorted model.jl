"""
    steady_state_hw6(; θ, β, δ, σ, ψ, γ_z, γ_n, τ_c, τ_h, τ_d, τ_p, ĝ)

Deterministic steady state of the detrended growth model with fiscal shocks.

Returns NamedTuple
  `(k_ss, h_ss, c_ss, r_ss, w_ss, x_inv_ss, κ_ss, β_tilde, growth)`
"""
function steady_state_hw6(; θ, β, δ, σ, ψ, γ_z, γ_n,
                            τ_c=0.0, τ_h=0.0, τ_d=0.0, τ_p=0.0, ĝ=0.0)
    growth  = (1 + γ_z) * (1 + γ_n)
    β_tilde = β * (1 + γ_n) * (1 + γ_z)^(1 - σ)

    kappa   = (θ / (growth / β_tilde - (1 - δ)))^(1 / (1 - θ))
    c_kappa = kappa^θ + (1 - δ - growth) * kappa
    ratio   = (1 - θ) * kappa^θ / (ψ * c_kappa)
    h_ss    = ratio / (1 + ratio)
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
            x_inv_ss=x_inv_ss, κ_ss=κ_ss, β_tilde=β_tilde, growth=growth)
end


"""
    make_return_fn(; θ, δ, σ, ψ, growth)

Return a closure `return_fn(x, u)` for the HW6 distorted growth model.

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
