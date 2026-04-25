# BCA_mle.jl
#
# Full state-space + Kalman-filter + MLE pipeline for the CKM / BCA wedge model
# (no capital-adjustment costs). Mirrors `sr328/mleqtrly/mleq.m`, `runmle.m`,
# `uncmin.m` / `mleseq.m`, but built as plain Julia routines.
#
# Section layout
# --------------
#   1. Steady-state bundle + helpers
#   2. Observation matrix C (4×6) — two constructions
#        (a) observation_C_from_policy   : from Vaughan (A,B,C₂,D₂) + production
#                                          + capital-accumulation identity
#        (b) observation_C_mleq_style    : direct analytical elasticities (mleq.m §5d)
#        plus compare_observation_C
#   3. Parameter (un)packing and full state-space (A,B,C,D,R) assembly
#        theta (length 30) ↔ (Sbar, P, Q)  (P lower-tri by columns; Q lower-tri)
#        build_bca_state_space           : returns (; A, B, C, D, R, X0, Y0, …)
#   4. Stationary Kalman filter for the `mleq.m` convention
#        kalman_steady_state(A, H, Qw, Rv, S)  — solves discrete-time DARE
#        kalman_run(A, Cbar, Σ, Ω, K, Ybar, X0)  — innovations recursion
#                                                   returns (Xt = X̂_{t|t-1},
#                                                            Xtt = X̂_{t|t})
#   5. Log-likelihood
#        bca_neg_loglik          : scalar `L` (same formula as mleq.m line 246)
#        bca_neg_loglik_perperiod: scalar L + vector Lt  (same as mleseq.m line 173)
#   6. MLE driver (hand-coded dense BFGS + Armijo; no Optim dependency)
#        bca_mle(theta0, param, ZVAR; …)
#        save_theta_literal(path, theta_hat)  — dump warm-start at full precision
#   7. Standard errors (OPG estimator, runmle.m lines 76-88)
#        bca_mle_standard_errors

using LinearAlgebra

# Depend on `policy_mleq_style` from BCA_policy_linear.jl. Guard so we don't
# redefine already-loaded symbols in a running session.
if !isdefined(Main, :policy_mleq_style)
    include(joinpath(@__DIR__, "BCA_policy_linear.jl"))
end

using Printf                           # only std-lib; for optional trace output

# No external optimisation package: the MLE driver uses a hand-coded dense
# BFGS with Armijo backtracking (see `_bfgs_solve` below).

# =============================================================================
# 1. Steady-state helpers (shared)
# =============================================================================

"""Wedge-model detrended steady state from `(param, P, P0)`.

Returns `(; ks, zs, tauls, tauxs, gs, ls, ys, xs, cs, tem)` with
`tem = (I-P)\\ P0` (unconditional mean of `S_t = [log z, τ_h, τ_x, log g]'`)."""
function _bca_steady_state(param::AbstractVector, P::AbstractMatrix, P0::AbstractVector)
    size(P) == (4, 4)     || error("P must be 4×4")
    length(P0) == 4       || error("P0 must have length 4")
    gn, gz, beta, delta, psi, sigma, theta = param[1:7]

    tem                 = (I(4) - P) \ P0
    zs, tauls, tauxs, gs = exp(tem[1]), tem[2], tem[3], exp(tem[4])
    beth = beta * (1 + gz)^(-sigma)
    kls  = ((1 + tauxs) * (1 - beth * (1 - delta)) / (beth * theta))^(1 / (theta - 1)) * zs
    Ares = (zs / kls)^(1 - theta) - (1 + gz) * (1 + gn) + 1 - delta
    Bres = (1 - tauls) * (1 - theta) * kls^theta * zs^(1 - theta) / psi
    ks   = (Bres + gs) / (Ares + Bres / kls)
    cs   = Ares * ks - gs
    ls   = ks / kls
    ys   = ks^theta * (zs * ls)^(1 - theta)
    xs   = ys - cs - gs
    return (; ks, zs, tauls, tauxs, gs, ls, ys, xs, cs, tem)
end

"""Given the 4×5 block `C_base` on `[log k̂, log z, τ_h, τ_x, log ĝ]` at steady state,
return `(C, phi0, X0_top, Y0)` with `phi0 = Y₀ − C_base·X₀(1:5)` so `C·X_ss = Y_ss`."""
function _attach_intercept(C_base::AbstractMatrix, ss)
    size(C_base) == (4, 5) || error("C_base must be 4×5, got $(size(C_base))")
    X0_top = [log(ss.ks), log(ss.zs), ss.tauls, ss.tauxs, log(ss.gs)]
    Y0     = [log(ss.ys), log(ss.xs), log(ss.ls), log(ss.gs)]
    phi0   = Y0 - C_base * X0_top
    return hcat(C_base, phi0), phi0, X0_top, Y0
end

# =============================================================================
# 2. Observation matrix C — two constructions
# =============================================================================

"""
    observation_C_from_policy(param, Sbar, P, P0, pol) -> NamedTuple

**Method 1.** Build the 4×6 observation matrix `C` from the log-linear policy
`pol = (A, B, C₂, D₂, …)` produced by `policy_vaughan_linear_policy`, combining:

- `ℓ̃_t = C₂ k̃_t + D₂ S_t`  (policy, given)
- `ỹ_t = θ k̃_t + (1-θ)(z̃_t + ℓ̃_t)`  (production, log-linearized)
- `N_{t+1} k_{t+1} = [(1-δ)k_t + x_t]N_t` ⇒
  `x̃_t = φ_{xk'} k̃_{t+1} + φ_{xk} k̃_t`  with
  `φ_{xk'}=(1+g_n)(1+g_z)k_ss/x_ss`, `φ_{xk}=-(1-δ)k_ss/x_ss`; then
  substitute `k̃_{t+1} = A k̃_t + B S_t`.
- Row for `log ĝ`: `[0, 0, 0, 0, 1]`.

The last column is `phi0 = Y₀ − C_{:,1:5} X₀(1:5)` so `C·X_ss = Y_ss`.
"""
function observation_C_from_policy(
    param::AbstractVector, Sbar::AbstractVector,
    P::AbstractMatrix,      P0::AbstractVector,
    pol,
)
    gn, gz, _, delta, _, _, theta = param[1:7]
    ss = _bca_steady_state(param, P, P0)

    A  = Float64(real(pol.A))
    C2 = Float64(real(pol.C2))
    B  = Float64.(real.(vec(pol.B)))
    D2 = Float64.(real.(vec(pol.D2)))
    length(B)  == 4 || error("pol.B must have 4 elements (one per wedge)")
    length(D2) == 4 || error("pol.D2 must have 4 elements (one per wedge)")

    phixk  = -ss.ks / ss.xs * (1 - delta)
    phixkp =  ss.ks / ss.xs * (1 + gz) * (1 + gn)

    row_y = [theta + (1 - theta) * C2,
             (1 - theta) * (1 + D2[1]),
             (1 - theta) *  D2[2],
             (1 - theta) *  D2[3],
             (1 - theta) *  D2[4]]
    row_x = [phixk + phixkp * A,
             phixkp * B[1], phixkp * B[2], phixkp * B[3], phixkp * B[4]]
    row_l = [C2, D2[1], D2[2], D2[3], D2[4]]
    row_g = [0.0, 0.0, 0.0, 0.0, 1.0]

    C_base = vcat(transpose(row_y), transpose(row_x), transpose(row_l), transpose(row_g))
    C, phi0, X0_top, Y0 = _attach_intercept(C_base, ss)

    return (; C, C_base, phi0, X0_top, Y0, ss,
            A, B, C2, D2, phixk, phixkp,
            method = :policy_A_B_C2_D2)
end

"""
    observation_C_mleq_style(param, Sbar, P, P0, Gamma5) -> NamedTuple

** Takes the policy row
`Gamma5 = Γ(1:5) = [γ_k, γ_z, γ_ℓ, γ_x, γ_g]` and builds each row of the base
4×5 block as `[φ_{·k}, φ_{·z}, φ_{·ℓ}, 0, φ_{·g}] + φ_{·k'}·Γ5`, with labor
elasticities from the log-linearized intratemporal FOC and `(φ_{xk}, φ_{xk'})`
from the capital-accumulation identity.
"""
function observation_C_mleq_style(
    param::AbstractVector, Sbar::AbstractVector,
    P::AbstractMatrix,      P0::AbstractVector,
    Gamma5::AbstractVector,
)
    length(Gamma5) == 5 ||
        error("Gamma5 must have 5 elements: [γ_k, γ_z, γ_ℓ, γ_x, γ_g]")
    gn, gz, _, delta, psi, _, theta = param[1:7]
    ss = _bca_steady_state(param, P, P0)
    ys, ks, ls, gs, tauls = ss.ys, ss.ks, ss.ls, ss.gs, ss.tauls

    philh = -(psi * ys * (1 - theta) +
              (1 - theta) * (1 - tauls) * ys * (1 - ls) / ls * theta +
              (1 - theta) * (1 - tauls) * ys)
    abs(philh) < 1e-18 && error("philh ≈ 0; cannot form labor comparative statics")

    philk  = (psi * ys * theta + psi * (1 - delta) * ks -
              (1 - theta) * (1 - tauls) * ys * (1 - ls) / ls * theta) / philh
    philz  = (psi * ys * (1 - theta) - (1 - theta)^2 * (1 - tauls) * ys * (1 - ls) / ls) / philh
    phill  = ((1 - theta) * (1 - tauls) * ys * (1 - ls) / ls * (1 / (1 - tauls))) / philh
    philg  = (-psi * gs) / philh
    philkp = (-psi * (1 + gz) * (1 + gn) * ks) / philh

    phiyk  = theta + (1 - theta) * philk
    phiyz  = (1 - theta) * (1 + philz)
    phiyl  = (1 - theta) * phill
    phiyg  = (1 - theta) * philg
    phiykp = (1 - theta) * philkp

    phixk  = -ks / ss.xs * (1 - delta)
    phixkp =  ks / ss.xs * (1 + gz) * (1 + gn)

    Γ5 = Vector{Float64}(real.(Gamma5))
    row_y = [phiyk, phiyz, phiyl, 0.0, phiyg] .+ phiykp .* Γ5
    row_x = [phixk, 0.0,   0.0,   0.0, 0.0]   .+ phixkp .* Γ5
    row_l = [philk, philz, phill, 0.0, philg] .+ philkp .* Γ5
    row_g = [0.0,   0.0,   0.0,   0.0, 1.0]

    C_base = vcat(transpose(row_y), transpose(row_x), transpose(row_l), transpose(row_g))
    C, phi0, X0_top, Y0 = _attach_intercept(C_base, ss)

    return (; C, C_base, phi0, X0_top, Y0, ss,
            philh, philk, philz, phill, philg, philkp,
            phiyk, phiyz, phiyl, phiyg, phiykp,
            phixk, phixkp, Gamma5 = Γ5,
            method = :mleq_style_elasticities)
end

"""
    compare_observation_C(param, Sbar, P, P0, pol, Gamma5) -> NamedTuple

Run both constructions of `C` and report `max|C₁ − C₂|`. At a correctly-solved
policy function, the two matrices agree up to numerical noise.
"""
function compare_observation_C(
    param::AbstractVector, Sbar::AbstractVector,
    P::AbstractMatrix,      P0::AbstractVector,
    pol,                    Gamma5::AbstractVector,
)
    m1 = observation_C_from_policy(param, Sbar, P, P0, pol)
    m2 = observation_C_mleq_style(param, Sbar, P, P0, Gamma5)
    return (; method1 = m1, method2 = m2,
            max_abs_diff      = maximum(abs.(m1.C .- m2.C)),
            max_abs_diff_base = maximum(abs.(m1.C_base .- m2.C_base)),
            max_abs_diff_phi0 = maximum(abs.(m1.phi0 .- m2.phi0)))
end

# =============================================================================
# 3. Parameter (un)packing + full state-space (A, B, C, D, R)
# =============================================================================

"""
    unpack_theta_wedges(theta) -> (; Sbar, P, Q)

Unpack a 30-vector in the same index convention as `mleq.m` /  `mleseq.m`
(lines 60-89 of mleq.m):

- `theta[1:4]`   = `Sbar` (log z, τ_h, τ_x, log g)
- `theta[5:20]`  = `P` in column order, `P(i,j) = theta[5 + (j-1)*4 + (i-1)]`
- `theta[21:30]` = `Q` lower-triangular, column-major
  (`Q(1,1)=θ21; Q(2,1)=θ22; Q(3,1)=θ23; Q(4,1)=θ24;
    Q(2,2)=θ25; Q(3,2)=θ26; Q(4,2)=θ27;
    Q(3,3)=θ28; Q(4,3)=θ29;
    Q(4,4)=θ30`).
"""
function unpack_theta_wedges(theta::AbstractVector)
    length(theta) == 30 ||
        error("theta must have length 30 (4 Sbar + 16 P + 10 Q)")
    Sbar = collect(Float64, theta[1:4])
    P = zeros(4, 4)
    @inbounds for j in 1:4, i in 1:4
        P[i, j] = theta[4 + (j - 1) * 4 + i]
    end
    Q = zeros(4, 4)
    Q[1, 1] = theta[21]; Q[2, 1] = theta[22]; Q[3, 1] = theta[23]; Q[4, 1] = theta[24]
    Q[2, 2] = theta[25]; Q[3, 2] = theta[26]; Q[4, 2] = theta[27]
    Q[3, 3] = theta[28]; Q[4, 3] = theta[29]
    Q[4, 4] = theta[30]
    return (; Sbar, P, Q)
end

"""Inverse of [`unpack_theta_wedges`](@ref)."""
function pack_theta_wedges(Sbar::AbstractVector, P::AbstractMatrix, Q::AbstractMatrix)
    size(P) == (4, 4) && size(Q) == (4, 4) || error("P, Q must be 4×4")
    length(Sbar) == 4 || error("Sbar must have length 4")
    theta = zeros(30)
    theta[1:4] .= Sbar
    @inbounds for j in 1:4, i in 1:4
        theta[4 + (j - 1) * 4 + i] = P[i, j]
    end
    theta[21] = Q[1, 1]; theta[22] = Q[2, 1]; theta[23] = Q[3, 1]; theta[24] = Q[4, 1]
    theta[25] = Q[2, 2]; theta[26] = Q[3, 2]; theta[27] = Q[4, 2]
    theta[28] = Q[3, 3]; theta[29] = Q[4, 3]
    theta[30] = Q[4, 4]
    return theta
end

"""
    build_bca_state_space(param, theta; D = zeros(4,4), R = zeros(4,4)) -> NamedTuple

Build the 6×6 `A`, 6×4 `B`, 4×6 `C` of the CKM / BCA system exactly as in
`mleq.m §5a-5d`:

```
X_{t+1} = A X_t + B ε_{t+1},   X_t = [log k̂_t, log z_t, τ_ℓt, τ_xt, log ĝ_t, 1]'
Y_t     = C X_t + ω_t,         Y_t = [log ŷ_t, log x̂_t, log ℓ_t, log ĝ_t]'
ω_t     = D ω_{t-1} + η_t,     Cov(η) = R
```

with `A = [γ_k  γ'  γ₀;  0  P  P0;  0  0  1]`, `B = [0; Q; 0]`, and `C` the
4×6 observation matrix (rows y, x, ℓ, g; cols = states + intercept).
"""
function build_bca_state_space(
    param::AbstractVector, theta::AbstractVector;
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
)
    u = unpack_theta_wedges(theta)
    Sbar, P, Q = u.Sbar, u.P, u.Q
    P0 = (I(4) - P) * Sbar

    m  = policy_mleq_style(Sbar, P, P0, param)
    γk  = Float64(real(m.gammak))
    γ   = Vector{Float64}(real.(m.gamma))
    γ0  = Float64(real(m.gamma0))

    A = zeros(6, 6)
    A[1, 1]   = γk
    A[1, 2:5] .= γ
    A[1, 6]   = γ0
    A[2:5, 2:5] .= P
    A[2:5, 6]   .= P0
    A[6, 6]   = 1.0

    B = zeros(6, 4)
    B[2:5, :] .= Q

    C  = Matrix{Float64}(m.C)
    X0 = [log(m.ks), Sbar[1], Sbar[2], Sbar[3], Sbar[4], 1.0]
    Y0 = [log(m.ys), log(m.xs), log(m.ls), Sbar[4]]

    return (; A, B, C, D = Matrix(D), R = Matrix(R),
            Sbar, P, P0, Q, X0, Y0,
            ss = (; ks = m.ks, ys = m.ys, xs = m.xs, ls = m.ls,
                    zs = m.tem[1] |> exp, gs = m.tem[4] |> exp),
            gamma_k = γk, gamma = γ, gamma0 = γ0, policy = m)
end

"""
    res_wedge_masked(Z, param, s0, As) -> Real

Euler residual with wedges selectively *masked*, mirroring
`sr328/mleqtrly/res_wedge2.m` (and the Python port in the paper repo):

- `As[i] = 1` → wedge `i` follows the input `Z` (as in the prototype).
- `As[i] = 0` → wedge `i` locked at the reference value `s0[i]`, so its
  derivative in the finite-difference Jacobian is **exactly zero** and the
  wedge drops out of the linearised policy.

`Z` is the 11-vector laid out by `steady_Z_from_Sbar`:

    Z = [log k₂, log k₁, log k,
         log z′, log z,       ← wedge 1 (z)       at indices 4, 5
         τ_ℓ′,   τ_ℓ,          ← wedge 2 (τ_ℓ)     at indices 6, 7
         τ_x′,   τ_x,          ← wedge 3 (τ_x)     at indices 8, 9
         log g′, log g]        ← wedge 4 (g)       at indices 10, 11
"""
function res_wedge_masked(Z::AbstractVector, param::AbstractVector,
                          s0::AbstractVector, As::AbstractVector)
    length(Z) ≥ 11 || error("Z must have length ≥ 11")
    length(s0) == 4 && length(As) == 4 || error("s0, As must have length 4")
    Zm = collect(Float64, Z)
    pairs = ((4, 5), (6, 7), (8, 9), (10, 11))
    @inbounds for k in 1:4
        if iszero(As[k])
            Zm[pairs[k][1]] = s0[k]
            Zm[pairs[k][2]] = s0[k]
        end
    end
    return res_wedge(Zm, param)
end

"""
    fixexp_state_space(param, theta, As; s0 = nothing, D, R) -> NamedTuple

CKM "single-wedge-economy" state-space, faithfully ported from
`sr328/mleqtrly/fixexp.m` (and the Python reference in
`Mypaper/BCA rep/MLE.ipynb`).  Unlike the prototype, the **capital policy
`(γ_k, γ, γ_0)` is re-derived for each `As`** by linearising a *masked*
Euler residual `res_wedge_masked(Z, param, s0, As)`: wedges flagged
`As[i] = 0` are frozen at the reference value `s0[i]` inside the residual,
so the Jacobian entries `∂R/∂(wedge i)` vanish and the corresponding
`γ_i` change.  This is what makes
`C_full − C_{As=[0,0,0,0]}` project the *direct* plus *policy-channel*
contribution of each wedge, so e.g. `(C3 − C0)` is **non-zero** even though
`τ_x` has no direct row in `C` (its effect enters through `γ_x` inside
`phi·k'·Γ`).

`As` is a 4-vector of 0/1 flags indexing the 4 wedges `[z, τ_ℓ, τ_x, g]`.

`s0` is the reference wedge vector used whenever `As[i] = 0`.  `pwbca.m`
calls this with the base-year wedges `[log ẑ(Y₀), τ̂_ℓ(Y₀), τ̂_x(Y₀), log ĝ(Y₀)]`.
If omitted, it defaults to the unconditional mean `Sbar`.

Returns a NamedTuple with `(A, B, C, C_base, phi0, Sbar, P, P0, Q, X0, Y0,
As, s0, gamma_k, gamma, gamma0, ss)`.
"""
function fixexp_state_space(
    param::AbstractVector, theta::AbstractVector, As::AbstractVector;
    s0::Union{AbstractVector, Nothing} = nothing,
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
)
    length(As) == 4 || error("As must be a 4-vector (flags for z, τ_ℓ, τ_x, g)")
    u = unpack_theta_wedges(theta)
    Sbar, P, Q = u.Sbar, u.P, u.Q
    P0 = (I(4) - P) * Sbar

    s0_ = s0 === nothing ? collect(Float64, Sbar) : collect(Float64, s0)
    length(s0_) == 4 || error("s0 must have length 4 ([log z, τ_ℓ, τ_x, log g])")

    # Steady state at the prototype linearisation point (same as mleq.m §5d)
    gn, gz, beta, delta, psi, sigma, theta_ = param[1:7]
    beth  = beta * (1 + gz)^(-sigma)
    zs    = exp(Sbar[1]); tauls = Sbar[2]; tauxs = Sbar[3]; gs = exp(Sbar[4])
    kls   = ((1 + tauxs) * (1 - beth * (1 - delta)) / (beth * theta_))^(1 / (theta_ - 1)) * zs
    Ares  = (zs / kls)^(1 - theta_) - (1 + gz) * (1 + gn) + 1 - delta
    Bres  = (1 - tauls) * (1 - theta_) * kls^theta_ * zs^(1 - theta_) / psi
    ks    = (Bres + gs) / (Ares + Bres / kls)
    cs    = Ares * ks - gs
    ls    = ks / kls
    ys    = ks^theta_ * (zs * ls)^(1 - theta_)
    xs    = ys - cs - gs

    # ---- Masked Jacobian of the Euler residual ------------------------------
    #      dR[i] = ∂R(Z_masked) / ∂Z[i]   with wedges governed by `As`.
    Z  = steady_Z_from_Sbar(Sbar, ks)
    f  = z -> res_wedge_masked(z, param, s0_, As)
    dR = central_gradient(f, Z)

    a0, a1, a2 = dR[1], dR[2], dR[3]
    b0_row     = transpose(dR[[4, 6,  8, 10]])
    b1_row     = transpose(dR[[5, 7,  9, 11]])

    # ---- Quadratic root for γ_k, 4×4 solve for γ (same algebra as mleq.m) ---
    roots_ = quadratic_roots(a0, a1, a2)
    stable = filter(λ -> abs(λ) < 1, roots_)
    isempty(stable) && error("fixexp_state_space: no stable root for As = $As")
    gammak = stable[argmin(abs.(stable))]
    rhs    = (b0_row * P + b1_row)'
    gamma  = -((a0 * gammak + a1) * I(4) + a0 * P') \ rhs
    gamma0 = (1 - gammak) * log(ks) - dot(gamma, [log(zs); tauls; tauxs; log(gs)])

    γk  = Float64(real(gammak))
    γ   = Vector{Float64}(real.(gamma))
    γ0  = Float64(real(gamma0))
    Γ5  = vcat(γk, γ)                                       # [γ_k, γ_z, γ_ℓ, γ_x, γ_g]

    # ---- Observation-equation elasticities (identical to mleq.m §5d) -------
    philh  = -(psi * ys * (1 - theta_) +
               (1 - theta_) * (1 - tauls) * ys * (1 - ls) / ls * theta_ +
               (1 - theta_) * (1 - tauls) * ys)
    philk  = (psi * ys * theta_ + psi * (1 - delta) * ks -
              (1 - theta_) * (1 - tauls) * ys * (1 - ls) / ls * theta_) / philh
    philz  = (psi * ys * (1 - theta_) -
              (1 - theta_)^2 * (1 - tauls) * ys * (1 - ls) / ls) / philh
    phill  = ((1 - theta_) * (1 - tauls) * ys * (1 - ls) / ls * (1 / (1 - tauls))) / philh
    philg  = (-psi * gs) / philh
    philkp = (-psi * (1 + gz) * (1 + gn) * ks) / philh
    phiyk  = theta_ + (1 - theta_) * philk
    phiyz  = (1 - theta_) * (1 + philz)
    phiyl  = (1 - theta_) * phill
    phiyg  = (1 - theta_) * philg
    phiykp = (1 - theta_) * philkp
    phixk  = -ks / xs * (1 - delta)
    phixkp = ks / xs * (1 + gz) * (1 + gn)

    # ---- State-space assembly ----------------------------------------------
    A = zeros(6, 6)
    A[1, 1]     = γk
    A[1, 2:5]  .= γ
    A[1, 6]     = γ0
    A[2:5, 2:5] .= P
    A[2:5, 6]   .= P0
    A[6, 6]     = 1.0

    B = zeros(6, 4)
    B[2:5, :] .= Q

    a1_, a2_, a3_, a4_ = Float64(As[1]), Float64(As[2]), Float64(As[3]), Float64(As[4])
    # Direct loadings on [log k, log z, τ_ℓ, τ_x, log g]; τ_x has no direct row
    # (follows `fixexp.m` line 105-108).  Policy-channel contribution `phi·kp·Γ5`
    # differs across `As` now — that is the whole point.
    r_y = [phiyk,  phiyz * a1_, phiyl * a2_, 0.0, phiyg * a4_] .+ phiykp .* Γ5
    r_x = [phixk,  0.0,         0.0,         0.0, 0.0]         .+ phixkp .* Γ5
    r_l = [philk,  philz * a1_, phill * a2_, 0.0, philg * a4_] .+ philkp .* Γ5
    r_g = [0.0,    0.0,         0.0,         0.0, a4_]
    C_base = vcat(transpose(r_y), transpose(r_x), transpose(r_l), transpose(r_g))

    X0_top = [log(ks), Sbar[1], Sbar[2], Sbar[3], Sbar[4]]
    Y0     = [log(ys), log(xs), log(ls), Sbar[4]]
    phi0   = Y0 - C_base * X0_top
    C      = hcat(C_base, phi0)
    X0     = vcat(X0_top, 1.0)

    return (; A, B, C, C_base, phi0, D = Matrix(D), R = Matrix(R),
            Sbar, P, P0, Q, X0, Y0,
            As = collect(Float64, As), s0 = s0_,
            ss = (; ks, ys, xs, ls, zs, gs),
            gamma_k = γk, gamma = γ, gamma0 = γ0, dR, a0, a1, a2)
end

# =============================================================================
# 4. Stationary Kalman filter — Appendix A.3.2 notation
#
#    Filtered-observation state-space form:
#        X_{t+1} = A X_t + B ε_{t+1}
#        Ȳ_t    = C̄ X_t + C B ε_{t+1} + η_{t+1},    Cov(η) = R
#
#    with C̄ = C A − D C coming from Y_{t+1} − D Y_t. The stationary filter
#    solves (Appendix A.3.2, last three displayed equations):
#
#        Ω   = C̄ Σ C̄' + R + C B B' C'
#        K   = (B B' C' + A Σ C̄') Ω⁻¹
#        Σ₊₁ = A Σ A' + B B' − (B B' C' + A Σ C̄') Ω⁻¹ (C̄ Σ A' + C B B')
# =============================================================================

"""
    kalman_steady_state(A, B, C, C̄, R; tol, max_iter)
      → (; Σ, K, Ω, iter, converged)

Solve the discrete-time Algebraic Riccati Equation for the BCA state-space
system in Appendix A.3.2 notation

```
X_{t+1} = A X_t + B ε_{t+1}
Ȳ_t     = C̄ X_t + C B ε_{t+1} + η_{t+1},    Cov(η) = R
```

with `C̄ = C A − D C`. Returns the stationary predicted-state covariance `Σ`,
the Kalman gain `K`, and the innovation covariance `Ω`, following

```
Ω = C̄ Σ C̄' + R + C B B' C'
K = (B B' C' + A Σ C̄') Ω⁻¹
Σ = A Σ A' + B B' − (B B' C' + A Σ C̄') Ω⁻¹ (C̄ Σ A' + C B B')
```

Plain fixed-point iteration — no external DARE solver / AD package required.
"""
function kalman_steady_state(
    A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
    C̄::AbstractMatrix, R::AbstractMatrix;
    tol::Real = 1e-10, max_iter::Int = 2000,
    Σ0::Union{Nothing,AbstractMatrix} = nothing,
)
    n  = size(A, 1)
    ny = size(C, 1)
    size(A, 2) == n             || error("A must be square")
    size(B, 1) == n             || error("B must have $(n) rows (got $(size(B)))")
    size(C, 2) == n             || error("C must have $(n) columns (got $(size(C)))")
    size(C̄)   == (ny, n)        || error("C̄ must be $(ny)×$(n) (got $(size(C̄)))")
    size(R)   == (ny, ny)       || error("R must be $(ny)×$(ny) (got $(size(R)))")

    BBᵀ    = B * B'             # 6×6
    BBᵀCᵀ  = BBᵀ * C'           # 6×4  (appears in K and Σ update)
    CBBᵀ   = C * BBᵀ            # 4×6  (transpose of BBᵀCᵀ, kept separately for clarity)
    CBBᵀCᵀ = C * BBᵀCᵀ          # 4×4  (state-noise contribution to Ω)

    # Warm-start with previous Σ if supplied (a neighbouring θ usually shares
    # a Σ within ~5 fixed-point iterations, instead of ~300 from zero).
    Σ         = Σ0 === nothing ? zeros(n, n) : Matrix{Float64}(Σ0)
    iter      = 0
    converged = false
    for k in 1:max_iter
        Ω  = C̄ * Σ * C̄' + R + CBBᵀCᵀ
        Ω  = (Ω + Ω') / 2                                # enforce symmetry
        Kt = (BBᵀCᵀ + A * Σ * C̄') / Ω
        Σₙ = A * Σ * A' + BBᵀ - Kt * (C̄ * Σ * A' + CBBᵀ)
        Σₙ = (Σₙ + Σₙ') / 2
        diff = maximum(abs.(Σₙ .- Σ))
        Σ    = Σₙ
        iter = k
        if diff < tol
            converged = true
            break
        end
    end
    Ω = C̄ * Σ * C̄' + R + CBBᵀCᵀ
    Ω = (Ω + Ω') / 2
    K = (BBᵀCᵀ + A * Σ * C̄') / Ω
    return (; Σ, K, Ω, iter, converged)
end

"""
    kalman_run(A, C̄, Σ, Ω, K, Ȳ, X̂₀)
      → (; innov, Xt, Xtt, Lt, logdetΩ, Ωi, Ku)

Innovations recursion for the filtered state-space system (Appendix A.3.2):

```
X̂_{1|0} = X̂₀                                   (prior)
u_t       = Ȳ_t − C̄ X̂_{t|t-1}                 (innovation)
X̂_{t|t}  = X̂_{t|t-1} + K_u u_t                (observation update,  K_u = Σ C̄' Ω⁻¹)
X̂_{t+1|t} = A X̂_{t|t-1} + K u_t               (one-step-ahead prediction)
```

`Xt[t, :]  = X̂_{t|t-1}`   (predicted, used for innovations/likelihood)
`Xtt[t, :] = X̂_{t|t}`     (filtered, use this for wedge accounting)

Per-period Gaussian log-likelihood contribution
`Lt[t] = ½ (log|Ω| + u_t' Ω⁻¹ u_t)` (mleq.m §6).
"""
function kalman_run(
    A::AbstractMatrix, C̄::AbstractMatrix,
    Σ::AbstractMatrix,
    Ω::AbstractMatrix, K::AbstractMatrix,
    Ȳ::AbstractMatrix, X̂₀::AbstractVector,
)
    T, ny = size(Ȳ)
    nx = length(X̂₀)
    size(A)  == (nx, nx) || error("A must be $(nx)×$(nx)")
    size(C̄)  == (ny, nx) || error("C̄ must be $(ny)×$(nx)")
    size(Σ)  == (nx, nx) || error("Σ must be $(nx)×$(nx)")
    size(Ω)  == (ny, ny) || error("Ω must be $(ny)×$(ny)")
    size(K)  == (nx, ny) || error("K must be $(nx)×$(ny)")

    # Cholesky-based replacement of `inv(Ω) + log(det(Ω))` — for a 4×4 PSD Ω
    # the factorisation is 3–5× cheaper than `inv`, and logdet is free.
    cΩ       = cholesky(Symmetric(Ω))
    logdetΩ  = 2.0 * sum(log, diag(cΩ.U))
    Ωi       = cΩ \ Matrix{Float64}(I, ny, ny)
    Ku       = Σ * C̄' * Ωi                         # updating gain for X̂_{t|t}
    Xt       = zeros(T, nx)
    Xtt      = zeros(T, nx)
    innov    = zeros(T, ny)
    Lt       = zeros(T)
    Xt[1, :] .= X̂₀
    innov[1, :] .= Ȳ[1, :] .- C̄ * X̂₀
    Xtt[1, :]   .= Xt[1, :] .+ Ku * innov[1, :]
    Lt[1] = 0.5 * (logdetΩ + dot(innov[1, :], Ωi * innov[1, :]))
    @inbounds for i in 2:T
        Xt[i, :]    .= A * Xt[i - 1, :] .+ K * innov[i - 1, :]
        innov[i, :] .= Ȳ[i, :] .- C̄ * Xt[i, :]
        Xtt[i, :]   .= Xt[i, :] .+ Ku * innov[i, :]
        Lt[i] = 0.5 * (logdetΩ + dot(innov[i, :], Ωi * innov[i, :]))
    end
    return (; innov, Xt, Xtt, Lt, logdetΩ, Ωi, Ku)
end

# =============================================================================
# 5. Log-likelihood for the BCA state-space system (mleq.m §5-6 port)
#
# `ZVAR` is a `T × 4` matrix of *already-detrended* observations
#   `[y_cyc, x_cyc, h_cyc, g_cyc]` (log-cyclical ratios produced by
#   `bca_linear_detrend(...).mled_detrended`).
# The filter forms `Ȳ_t = Y_{t+1} − D·Y_t` (so when `D = 0` it simply drops
# the first observation) and runs the stationary Kalman filter.
# =============================================================================

"""Penalty used in `mleq.m` to keep `max|eig(P)| ≤ cap`:

```
penalty = weight · max(λmax − cap, 0)²
```

Matches the original MATLAB implementation exactly (non-smooth kink at
`λmax = cap`). This gives a hard-ish "wall" around the stationarity
boundary, which is what the appendix uses."""
function _P_eigen_penalty(
    P::AbstractMatrix;
    cap::Real = 0.995, weight::Real = 500_000.0,
)
    λmax = maximum(abs.(eigvals(P)))
    return weight * max(λmax - cap, 0.0)^2     # one-sided: penalty = 0 when λmax ≤ cap
end

# Large penalty returned when the model cannot be solved (negative steady-state
# capital, non-finite likelihood, singular innovation covariance, …). Same
# defensive trick as MATLAB's `mleq.m` silently does via NaN → large L fallback.
const _BCA_MLE_INFEASIBLE_PENALTY = 1.0e10

"""
    bca_neg_loglik(theta, param, ZVAR; D, R) -> Float64

Penalized negative log-likelihood of the BCA state-space model on a `T × 4`
matrix `ZVAR` of detrended per-capita ratios `[y_cyc, x_cyc, h_cyc, g_cyc]`.
Matches `mleq.m` line 246:

```
L = ½ (T (log|Ω| + tr(Ω⁻¹ · innov'innov / T)) + penalty)
```

with `penalty = 500 000 · max(|eig(P)|_max − 0.995, 0)²` (`mleq.m` original
non-smooth penalty, see [`_P_eigen_penalty`](@ref)). Smaller is better
(pass directly to any minimizer).
"""
function bca_neg_loglik(
    theta::AbstractVector, param::AbstractVector, ZVAR::AbstractMatrix;
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
)
    local L
    try
        ss      = build_bca_state_space(param, theta; D, R)
        A, B, C = ss.A, ss.B, ss.C
        penalty = _P_eigen_penalty(ss.P)

        Y    = log.(ZVAR)                         # T × 4 log-cyclical obs.
        Ybar = @view(Y[2:end, :]) .- @view(Y[1:end-1, :]) * D'
        Tm1  = size(Ybar, 1)

        C̄    = C * A - D * C                      # filtered-obs. loading
        kf   = kalman_steady_state(A, B, C, C̄, R)
        run  = kalman_run(A, C̄, kf.Σ, kf.Ω, kf.K, Ybar, ss.X0)

        sum1 = run.innov' * run.innov / Tm1
        L    = 0.5 * (Tm1 * (run.logdetΩ + tr(run.Ωi * sum1)) + penalty)
    catch err
        err isa InterruptException && rethrow(err)
        return _BCA_MLE_INFEASIBLE_PENALTY
    end
    return isfinite(L) ? L : _BCA_MLE_INFEASIBLE_PENALTY
end

"""
    bca_neg_loglik_perperiod(theta, param, ZVAR; D, R) -> NamedTuple

Same as [`bca_neg_loglik`](@ref) but returns the per-period contributions
`Lt[t] = ½ (log|Ω| + innovₜ' Ω⁻¹ innovₜ) + ½·penalty/T` (mleseq.m 168-181)
for OPG standard errors, plus the filter output (`innov`, `Xt`, `kf`, `ss`).
"""
function bca_neg_loglik_perperiod(
    theta::AbstractVector, param::AbstractVector, ZVAR::AbstractMatrix;
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
)
    ss      = build_bca_state_space(param, theta; D, R)
    A, B, C = ss.A, ss.B, ss.C
    penalty = _P_eigen_penalty(ss.P)

    Y    = log.(ZVAR)
    Ybar = @view(Y[2:end, :]) .- @view(Y[1:end-1, :]) * D'
    Tm1  = size(Ybar, 1)

    C̄   = C * A - D * C
    kf  = kalman_steady_state(A, B, C, C̄, R)
    run = kalman_run(A, C̄, kf.Σ, kf.Ω, kf.K, Ybar, ss.X0)

    sum1 = run.innov' * run.innov / Tm1
    L    = 0.5 * (Tm1 * (run.logdetΩ + tr(run.Ωi * sum1)) + penalty)
    Lt   = run.Lt .+ 0.5 * penalty / Tm1
    return (; L, Lt, state_space = ss, kf, innov = run.innov,
            Xt = run.Xt, Xtt = run.Xtt)
end

"""Safe wrapper around [`bca_neg_loglik_perperiod`](@ref) that returns the
infeasible-region penalty (instead of crashing) so OPG finite-differencing
never blows up on a bad step."""
function _bca_neg_loglik_perperiod_safe(
    theta::AbstractVector, param::AbstractVector, ZVAR::AbstractMatrix;
    D::AbstractMatrix, R::AbstractMatrix,
)
    try
        return bca_neg_loglik_perperiod(theta, param, ZVAR; D, R)
    catch err
        err isa InterruptException && rethrow(err)
        Tm1 = size(ZVAR, 1) - 1
        return (; L = _BCA_MLE_INFEASIBLE_PENALTY,
                  Lt = fill(_BCA_MLE_INFEASIBLE_PENALTY / Tm1, Tm1))
    end
end

# =============================================================================
# 6. MLE driver — hand-coded dense BFGS (no external dependency)
#
# Replaces the earlier `Optim.LBFGS` driver.  All speed tricks below are
# built-in and require no extra packages:
#
#   (a) The MLE loop builds a **caching** objective closure.  The stationary
#       Kalman DARE (`kalman_steady_state`) is warm-started from the previous
#       call's `Σ`.  For nearby θ perturbations this cuts the DARE iteration
#       count from ~few-hundred to ~5.
#   (b) Gradient is computed with **one-sided (forward) differences**:
#       `∂f/∂θᵢ ≈ (f(θ+Δᵢ) − f(θ))/Δᵢ`.  That is half the evaluations of the
#       previous central scheme, and accurate enough for BFGS line search.
#   (c) `kalman_run`'s log-det / Ω⁻¹ are computed via Cholesky (see §4).
# =============================================================================

"""Forward-difference gradient with `mleq.m`-style per-coordinate step
`Δ_i = max(|x_i| · step_rel, step_min)`.  `f0 = f(x)` is supplied so the
reference evaluation is shared across all coordinates (one call, not n)."""
function _forward_grad!(g::AbstractVector, f, x::AbstractVector, f0::Real;
                        step_rel::Real = 1e-4, step_min::Real = 1e-8)
    n = length(x)
    xp = copy(x)
    @inbounds for i in 1:n
        Δ     = max(abs(x[i]) * step_rel, step_min)
        xp[i] = x[i] + Δ
        g[i]  = (f(xp) - f0) / Δ
        xp[i] = x[i]
    end
    return g
end

"""Central-difference gradient (kept for OPG / diagnostics)."""
function _central_grad!(g::AbstractVector, f, x::AbstractVector;
                        step_rel::Real = 1e-4, step_min::Real = 1e-8)
    n = length(x)
    xp = copy(x); xm = copy(x)
    @inbounds for i in 1:n
        Δ = max(abs(x[i]) * step_rel, step_min)
        xp[i] = x[i] + Δ
        xm[i] = x[i] - Δ
        g[i]  = (f(xp) - f(xm)) / (2 * Δ)
        xp[i] = x[i]
        xm[i] = x[i]
    end
    return g
end

"""Dense BFGS with Armijo backtracking line search (hand-coded; no Optim).

Designed for our 30-parameter **penalised** MLE; the penalty surface has a
one-sided kink at `λmax(P) = cap`, which means a full Newton step `p = -H·g`
is occasionally far too large and the line search then wastes many
evaluations backtracking from `α=1`.  The safeguards below address this:

* `H` is the inverse-Hessian approximation, initialised to `I`.
* Search direction `p = -H g`.  If `gᵀp ≥ 0` (descent lost — typically at a
  penalty kink), `H` is reset to `I` and `p = -g`.
* `step_bound` caps `‖p‖_∞`: if the Newton step overshoots we shrink `p`
  before line-searching.  This is the key fix for warm-starts that land on
  the penalty boundary with `|g|_∞ ∼ 10⁴-10⁵`.
* **Line search**: first try to *extrapolate* `α` upward (doubling) while
  Armijo is satisfied and the step stays feasible, then fall back to
  backtracking (halving).  This gives full Newton steps in the interior but
  tiny steps near a kink — both with the same code.
* BFGS update is skipped when the curvature condition `yᵀs > 0` fails.
* The Hessian is periodically reset (`hess_reset` iterations) so stale
  curvature information from early "large gradient" iterations does not
  dominate the final polishing stage.

Returns a NamedTuple mirroring the fields the notebook uses:
`(; minimizer, minimum, iterations, converged, g_norm, f_calls, g_calls)`.
"""
function _bfgs_solve(f, g!, x0::AbstractVector;
                     g_tol::Real       = 1e-5,
                     max_iter::Int     = 2000,
                     α_min::Real       = 1e-16,
                     step_bound::Real  = 0.2,
                     hess_reset::Int   = 0,         # 0 = never reset
                     show_trace::Bool  = false,
                     print_every::Int  = 25)
    n  = length(x0)
    x  = collect(Float64, x0)
    g  = zeros(n); g!(g, x)
    fx = f(x)
    H  = Matrix{Float64}(I, n, n)
    f_calls = 1; g_calls = 1
    g_norm  = maximum(abs, g)
    converged = false
    iter = 0
    last_reset = 0
    show_trace && @printf("iter %4d  f = %-14.6f  |g|∞ = %-10.3e\n", 0, fx, g_norm)
    for k in 1:max_iter
        iter = k
        if g_norm < g_tol
            converged = true
            break
        end
        if hess_reset > 0 && k - last_reset >= hess_reset
            H .= 0.0; @inbounds for i in 1:n; H[i, i] = 1.0; end
            last_reset = k
        end
        p  = -H * g
        gp = dot(g, p)
        if !isfinite(gp) || gp >= 0       # direction lost → steepest descent reset
            H .= 0.0; @inbounds for i in 1:n; H[i, i] = 1.0; end
            last_reset = k
            p  = -g
            gp = dot(g, p)
        end
        # Cap the Newton step — crucial near the P-eigenvalue penalty kink.
        pinf = maximum(abs, p)
        if pinf > step_bound
            scale = step_bound / pinf
            p    .*= scale
            gp   *= scale                 # gᵀp scales linearly with p
        end

        c₁  = 1e-4
        α   = 1.0
        f_trial = f(x .+ α .* p); f_calls += 1
        armijo(α, ft) = isfinite(ft) && ft <= fx + c₁ * α * gp

        if armijo(α, f_trial)
            # Extrapolation: try doubling α up to ~4× while Armijo holds and
            # f keeps improving.  Stops the solver from being stuck at α=1
            # after we've capped p.
            while α < 4.0
                α2    = 2.0 * α
                f2    = f(x .+ α2 .* p); f_calls += 1
                if armijo(α2, f2) && f2 < f_trial
                    α, f_trial = α2, f2
                else
                    break
                end
            end
        else
            # Backtracking.
            while !armijo(α, f_trial)
                α *= 0.5
                if α < α_min; break; end
                f_trial = f(x .+ α .* p); f_calls += 1
            end
        end
        if α < α_min; break; end
        s     = α .* p
        x_new = x .+ s
        g_new = similar(g); g!(g_new, x_new); g_calls += 1
        y     = g_new .- g
        ys    = dot(y, s)
        if isfinite(ys) && ys > 1e-14     # only update if curvature is positive
            ρ  = 1.0 / ys
            sy = s * y'
            ys_ = y * s'
            # H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
            H  = (I - ρ .* sy) * H * (I - ρ .* ys_) + ρ .* (s * s')
        end
        x, g, fx = x_new, g_new, f_trial
        g_norm   = maximum(abs, g)
        if show_trace && (k == 1 || k % print_every == 0)
            @printf("iter %4d  f = %-14.6f  |g|∞ = %-10.3e  α = %-10.3e  |Δx|∞ = %-10.3e\n",
                    k, fx, g_norm, α, maximum(abs, s))
        end
    end
    show_trace && @printf("stopped: iter = %d  f = %.6f  |g|∞ = %.3e  converged = %s\n",
                          iter, fx, g_norm, converged)
    return (; minimizer  = x, minimum    = fx,
              iterations = iter,
              converged,
              g_norm,
              f_calls, g_calls)
end

"""
    bca_mle(theta0, param, ZVAR;
            max_iter = 2000, g_tol = 1e-5, show_trace = false,
            gradient = :central, step_rel, step_min = 1e-10,
            polish = true, polish_iter = 200, polish_step_rel,
            step_bound = 0.2, hess_reset = 0,
            D = zeros(4,4), R = zeros(4,4))
      → (; theta_hat, L_hat, A_hat, B_hat, C_hat, state_space,
           iterations, converged, g_norm, f_calls, g_calls, result)

Minimise `bca_neg_loglik` starting from `theta0` using a hand-coded dense
BFGS (`_bfgs_solve`) with an optional polishing pass.  **No external
optimisation package is used.**

Two-phase design (since the `λmax(P) ≤ 0.995` penalty has a one-sided kink
that forward differences cross with large numerical artefacts):

1. **Coarse**  `_bfgs_solve` with `step_rel ≈ 1e-6` central-diff or
   `≈ 1e-7` forward-diff gradients (whichever is chosen by `gradient`),
   step length bounded by `step_bound`, tolerance `g_tol`.
2. **Polish**  A second `_bfgs_solve` from phase-1's optimum with a much
   finer gradient step (`polish_step_rel ≈ 3e-6` central / `1e-8` forward)
   and tighter `g_tol / 10`.  Default `polish = true`.

Gradient choice:
- `gradient = :central` (default)   — best accuracy, 2n function evals.
- `gradient = :forward`             — fast but less accurate near optimum.

The objective closure caches the stationary Kalman covariance `Σ` between
calls; a new call warm-starts the DARE fixed-point from the previous `Σ`.

The returned tuple duplicates the convergence info at the top level
(`iterations`, `converged`, `g_norm`, `f_calls`, `g_calls`) and also exposes
`result` = same info as a NamedTuple (no external optimiser package).
"""
function bca_mle(
    theta0::AbstractVector, param::AbstractVector, ZVAR::AbstractMatrix;
    max_iter::Int     = 2000,
    g_tol::Real       = 1e-5,
    show_trace::Bool  = false,
    print_every::Int  = 25,
    gradient::Symbol  = :central,
    step_rel::Union{Real,Nothing} = nothing,
    step_min::Real    = 1e-10,
    polish::Bool      = true,
    polish_iter::Int  = 200,
    polish_step_rel::Union{Real,Nothing} = nothing,
    step_bound::Real  = 0.2,
    hess_reset::Int   = 0,
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
    method::Symbol    = :BFGS,
)
    method === :BFGS || @warn "bca_mle: `method=$method` ignored; using hand-coded :BFGS"
    gradient ∈ (:forward, :central) || error("gradient must be :forward or :central")

    # Sensible default step sizes.  For a roughly 1e-11 function-evaluation
    # noise floor (Kalman log-likelihood), the optimal finite-diff step is
    # ~sqrt(noise) ≈ 3e-6 for forward, ~cbrt(noise) ≈ 2e-4 for central.  We
    # pick slightly smaller central step since the optimum has O(1) curvature.
    sr      = step_rel === nothing ?
                  (gradient === :central ? 1e-5 : 1e-7) : Float64(step_rel)
    polish_sr = polish_step_rel === nothing ?
                  (gradient === :central ? 3e-6 : 1e-8) : Float64(polish_step_rel)

    # Precompute filtered-observation matrix Ȳ once (same for every θ)
    Y    = log.(ZVAR)
    Ybar = Matrix(Y[2:end, :] .- Y[1:end-1, :] * D')
    Tm1  = size(Ybar, 1)

    # Σ cache: warm-starts DARE across successive objective evaluations.
    Σ_cache = Ref{Union{Nothing, Matrix{Float64}}}(nothing)

    function nll(θ::AbstractVector)
        try
            ss      = build_bca_state_space(param, θ; D, R)
            A, B, C = ss.A, ss.B, ss.C
            penalty = _P_eigen_penalty(ss.P)
            C̄       = C * A - D * C
            kf      = kalman_steady_state(A, B, C, C̄, R; Σ0 = Σ_cache[])
            Σ_cache[] = kf.Σ
            run_    = kalman_run(A, C̄, kf.Σ, kf.Ω, kf.K, Ybar, ss.X0)
            sum1    = run_.innov' * run_.innov / Tm1
            L       = 0.5 * (Tm1 * (run_.logdetΩ + tr(run_.Ωi * sum1)) + penalty)
            return isfinite(L) ? L : _BCA_MLE_INFEASIBLE_PENALTY
        catch err
            err isa InterruptException && rethrow(err)
            return _BCA_MLE_INFEASIBLE_PENALTY
        end
    end

    make_grad(sr_local) =
        if gradient === :central
            (g, θ) -> _central_grad!(g, nll, θ; step_rel = sr_local, step_min)
        else
            (g, θ) -> (f0 = nll(θ);
                       _forward_grad!(g, nll, θ, f0; step_rel = sr_local, step_min))
        end

    x0  = collect(Float64, theta0)
    show_trace && println("── Phase 1  BFGS (gradient = $gradient, step_rel = $sr) ──")
    res = _bfgs_solve(nll, make_grad(sr), x0;
                      g_tol, max_iter, step_bound, hess_reset,
                      show_trace, print_every)

    # Optional polishing pass with a finer gradient step.
    if polish && (res.iterations < max_iter || !res.converged)
        show_trace && println("── Phase 2  polish (step_rel = $polish_sr, g_tol = $(g_tol/10)) ──")
        Σ_cache[] = nothing                    # force DARE from scratch for a clean start
        res2 = _bfgs_solve(nll, make_grad(polish_sr), res.minimizer;
                           g_tol     = g_tol / 10,
                           max_iter  = polish_iter,
                           step_bound,
                           hess_reset,
                           show_trace, print_every)
        if res2.minimum < res.minimum
            res = (; minimizer  = res2.minimizer,
                     minimum    = res2.minimum,
                     iterations = res.iterations + res2.iterations,
                     converged  = res2.converged,
                     g_norm     = res2.g_norm,
                     f_calls    = res.f_calls + res2.f_calls,
                     g_calls    = res.g_calls + res2.g_calls)
        end
    end

    θ̂ = res.minimizer
    L̂ = res.minimum
    ss = build_bca_state_space(param, θ̂; D, R)
    return (; theta_hat = θ̂, L_hat = L̂,
            A_hat = ss.A, B_hat = ss.B, C_hat = ss.C,
            state_space = ss,
            iterations = res.iterations,
            converged  = res.converged,
            g_norm     = res.g_norm,
            f_calls    = res.f_calls,
            g_calls    = res.g_calls,
            result     = res)
end

# -----------------------------------------------------------------------------
# Helper: dump a θ vector to disk as an `include()`-able Julia literal at full
# Float64 precision.  Use this after an MLE run so that the next invocation can
# warm-start with the *exact* prior estimate (no rounding loss when pasting
# 4-5 decimal tables from Part 3 output).
# -----------------------------------------------------------------------------
"""
    save_theta_literal(path, theta; name = "theta0")

Write `theta` to `path` as a Julia include-file.  Reloading with
`include(path)` redefines `<name>` at full 17-digit precision.
"""
function save_theta_literal(path::AbstractString, theta::AbstractVector;
                            name::AbstractString = "theta0")
    length(theta) == 30 ||
        @warn "save_theta_literal: expected 30-vector, got length $(length(theta))"
    Sbar = theta[1:4]
    Pmat = reshape(theta[5:20], 4, 4)
    Qvec = theta[21:30]
    fmt(x) = rpad(repr(Float64(x)), 24)
    open(path, "w") do io
        println(io, "# Auto-generated by save_theta_literal — full Float64 precision.")
        println(io, "# Include this file to redefine `$name`.")
        println(io, "$name = [")
        println(io, "    # Sbar = [log zs, τ_hs, τ_xs, log gs]")
        println(io, "     ", join(fmt.(Sbar), ", "), ",")
        println(io, "    # P (column-major, 4×4)")
        for j in 1:4
            println(io, "     ", join(fmt.(Pmat[:, j]), ", "), ",")
        end
        println(io, "    # Q (lower-triangular, stacked column-by-column)")
        idx = 1
        for j in 1:4, i in j:4
            end_mark = idx == 10 ? "" : ","
            println(io, "     ", fmt(Qvec[idx]), end_mark,
                    "    # Q[$i,$j]")
            idx += 1
        end
        println(io, "]")
    end
    return path
end

# =============================================================================
# 7. Standard errors — OPG estimator on the per-period scores (runmle.m §se)
# =============================================================================

"""
    bca_mle_standard_errors(theta_hat, param, ZVAR;
                            D, R, step_rel = 1e-4, step_min = 1e-8)
      → (; se, V, score_t)

OPG (outer-product-of-gradients) standard errors, identical in spirit to
`runmle.m` lines 76-88. The per-period score is obtained by central
finite differences on `Lt` from [`bca_neg_loglik_perperiod`](@ref):

```
score_t[i, t] = (Lt_i(+Δ_i) − Lt_i(−Δ_i)) / (2 Δ_i)
V = (Σ_t score_t · score_t')⁻¹,   se = √diag(V)
```

`step = max(|θ̂|·step_rel, step_min)` elementwise.
"""
function bca_mle_standard_errors(
    theta_hat::AbstractVector, param::AbstractVector, ZVAR::AbstractMatrix;
    D::AbstractMatrix = zeros(4, 4),
    R::AbstractMatrix = zeros(4, 4),
    step_rel::Real = 1e-4,
    step_min::Real = 1e-8,
)
    θ0 = collect(Float64, theta_hat)
    n  = length(θ0)
    Δ  = max.(abs.(θ0) .* step_rel, step_min)

    Lt0 = _bca_neg_loglik_perperiod_safe(θ0, param, ZVAR; D, R).Lt
    T   = length(Lt0)
    score_t = zeros(n, T)
    @inbounds for i in 1:n
        θp = copy(θ0); θp[i] += Δ[i]
        θm = copy(θ0); θm[i] -= Δ[i]
        Lt_p = _bca_neg_loglik_perperiod_safe(θp, param, ZVAR; D, R).Lt
        Lt_m = _bca_neg_loglik_perperiod_safe(θm, param, ZVAR; D, R).Lt
        score_t[i, :] .= (Lt_p .- Lt_m) ./ (2 * Δ[i])
    end
    H = score_t * score_t'
    V = inv(H)
    se = sqrt.(max.(diag(V), 0.0))
    return (; se, V, score_t)
end
