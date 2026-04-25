# BCA_steady_state.jl
#
# Detrended steady state for the CKM (2007) benchmark model with wedges (appendix A),
# matching the algebra in:
#   - sr328/mleannual/mle1.m   (lines 137–152, explicit ξ₃)
#   - sr328/mleqtrly/pwbca.m   (lines 58–71; uses k = (ξ₂+g)/(ξ₁+ξ₂/kl), same as ξ₃ = ξ₂/kl)
#
# Notation: ẑ, τ_l, τ_x, ĝ are *levels* as in MATLAB (z = exp(Sbar[1]), g = exp(Sbar[4])).
# β̂ = β (1+g_z)^{-σ} is the growth-adjusted discount factor.

"""
    bca_steady_state(; z, tau_l, tau_x, g, gn, gz, beta, delta, psi, sigma, theta)

Compute detrended steady-state `(k̂, ĉ, l, ŷ, x̂)` given constant wedge levels and parameters.

Returns a `NamedTuple` with fields:
  `k, c, l, y, x`, intermediate `kl, yk, xi1, xi2, xi3`, and `beta_hat`.

Reference: Chari, Kehoe & McGrattan, *Business Cycle Accounting*, appendix (steady-state block).
"""
function bca_steady_state(; z::Real, tau_l::Real, tau_x::Real, g::Real,
                          gn::Real, gz::Real, beta::Real, delta::Real,
                          psi::Real, sigma::Real, theta::Real)
    beta_hat = beta * (1 + gz)^(-sigma)
    # Capital–labor ratio k̂/l (appendix)
    kl = ((1 + tau_x) * (1 - beta_hat * (1 - delta)) / (beta_hat * theta))^(1 / (theta - 1)) * z
    yk = (kl / z)^(theta - 1)   # ŷ/k̂ in MATLAB notation (used to build resource constraint)
    xi1 = yk - (1 + gz) * (1 + gn) + 1 - delta
    xi2 = (1 - tau_l) * (1 - theta) * kl^theta * z^(1 - theta) / psi
    xi3 = xi2 / kl
    k = (xi2 + g) / (xi1 + xi3)
    c = xi1 * k - g
    l = k / kl
    y = yk * k
    x = y - c - g
    return (; k, c, l, y, x, kl, yk, xi1, xi2, xi3, beta_hat)
end


"""
    bca_steady_state_from_Sbar(Sbar, param)

Convenience: `Sbar = [log(z), τ_l, τ_x, log(g)]` (4-vector) and
`param = [gn, gz, beta, delta, psi, sigma, theta]` as in `mleq.m` / `pwbca.m`.
"""
function bca_steady_state_from_Sbar(Sbar::AbstractVector, param::AbstractVector)
    length(Sbar) == 4 || error("Sbar must have length 4: [log z, tau_l, tau_x, log g]")
    length(param) >= 7 || error("param must have at least 7 elements: gn,gz,beta,delta,psi,sigma,theta")
    gn, gz, beta, delta, psi, sigma, theta = param[1], param[2], param[3], param[4], param[5], param[6], param[7]
    z = exp(Sbar[1])
    tau_l = Sbar[2]
    tau_x = Sbar[3]
    g = exp(Sbar[4])
    return bca_steady_state(; z, tau_l=tau_l, tau_x=tau_x, g=g, gn, gz, beta, delta, psi, sigma, theta)
end
