# BCA_policy_linear.jl
#
# Log-linear capital policy for the CKM wedge model, in two numerical routes:
#
#   Route ①  Jacobian of the Euler residual → quadratic for the stable root → 4×4 linear solve
#              for γ in  log k_{t+1} = γ₀ + γ_k log k_t + γ' S_t.
#
#   Route ②  Stacked pencil (A₁, A₂) + Vaughan generalized eigenproblem for (A, C₂), then an
#              8×8 linear solve for (B, D₂) in
#                 tilde k_{t+1} = A tilde k_t + B S_t,
#                 tilde h_t     = C₂ tilde k_t + D₂ S_t,
#                 E_t S_{t+1}   = P S_t.

using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "Newton.jl"))
include(joinpath(@__DIR__, "BCA_steady_state.jl"))

# =============================================================================
# 1. Intratemporal labor block (shared building block)
# =============================================================================

"Static labor FOC residual `ψ c l / y - (1-τ_h)(1-θ)(1-l)` evaluated at an explicit `l`."
function labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l)
    y = k^theta * (z * l)^(1 - theta)
    c = y - (1 + gz) * (1 + gn) * k1 + (1 - delta) * k - g
    return psi * c * l / y - (1 - taul) * (1 - theta) * (1 - l)
end

"Solve the intratemporal labor FOC for `l` at given `(k, k', z, τ_h, g)` by Newton iteration."
function solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi;
                             max_iter::Int=40, step::Real=1e-6)
    l = 1 / (1 + 0.75 * psi / (1 - taul) / (1 - theta))
    for _ in 1:max_iter
        res  = labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l)
        resp = labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l + step)
        dres = (resp - res) / step
        l   -= res / dres
    end
    return l
end

"Labor FOC residual on the 11-vector `Z` (uses current-date entries of `Z`)."
function labor_intratemporal_residual(Z::AbstractVector, param::AbstractVector)
    gn, gz, _, delta, psi, _, theta = param[1:7]
    k, k1, z = exp(Z[3]), exp(Z[2]), exp(Z[5])
    taul, g  = Z[7], exp(Z[11])
    l = solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi)
    return labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l)
end

"Derivative of the labor FOC residual w.r.t. `ln h` at the steady-state labor `l₀`."
function labor_dlog_h_fd(Z::AbstractVector, param::AbstractVector; ϵ::Real=1e-5)
    gn, gz, _, delta, psi, _, theta = param[1:7]
    k, k1, z = exp(Z[3]), exp(Z[2]), exp(Z[5])
    taul, g  = Z[7], exp(Z[11])
    l0 = solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi)
    rp = labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l0 * exp(ϵ))
    rm = labor_foc_residual(k, k1, z, taul, g, gn, gz, delta, theta, psi, l0 * exp(-ϵ))
    return (rp - rm) / (2ϵ)
end

"Row `[∂res/∂ln k_t, ∂res/∂ln k_{t+1}, ∂res/∂ln h_t]` at the steady-state `Z`.
Partial derivatives of `labor_foc_residual` at fixed `ℓ = ℓ_ss` (not the Newton-solved
residual, whose derivative is trivially ≈ 0)."
function labor_intratemporal_jacobian(Z::AbstractVector, param::AbstractVector; ϵ::Real=1e-6)
    gn, gz, _, delta, psi, _, theta = param[1:7]
    k, k1, z = exp(Z[3]), exp(Z[2]), exp(Z[5])
    taul, g  = Z[7], exp(Z[11])
    l0 = solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi)
    rfoc = (kv, k1v, zv, tauv, gv, lv) ->
        labor_foc_residual(kv, k1v, zv, tauv, gv, gn, gz, delta, theta, psi, lv)
    dk  = (rfoc(k*exp(ϵ), k1, z, taul, g, l0) -
           rfoc(k*exp(-ϵ), k1, z, taul, g, l0)) / (2ϵ)
    dk1 = (rfoc(k, k1*exp(ϵ), z, taul, g, l0) -
           rfoc(k, k1*exp(-ϵ), z, taul, g, l0)) / (2ϵ)
    dl  = labor_dlog_h_fd(Z, param)
    return [dk, dk1, dl]
end

# =============================================================================
# 2. Euler residual `R(Z)` and its Jacobian on the 11-vector `Z`
# =============================================================================
#   Z = [ log k₂, log k₁, log k,
#         log z′, log z,
#         τ_h′,   τ_h,
#         τ_x′,   τ_x,
#         log g′, log g ]

"Scalar Euler (capital intertemporal FOC) residual at a stacked `Z`."
function res_wedge(Z::AbstractVector, param::AbstractVector)
    length(Z) ≥ 11 || error("Z must have length ≥ 11")
    gn, gz, beta, delta, psi, sigma, theta = param[1:7]
    beth = beta * (1 + gz)^(-sigma)

    k2, k1, k  = exp(Z[1]), exp(Z[2]), exp(Z[3])
    z1, z      = exp(Z[4]), exp(Z[5])
    taul1, taul = Z[6], Z[7]
    taux1, taux = Z[8], Z[9]
    g1, g       = exp(Z[10]), exp(Z[11])

    l  = solve_labor_given_k(k,  k1, z,  taul,  g,  gn, gz, delta, theta, psi)
    l1 = solve_labor_given_k(k1, k2, z1, taul1, g1, gn, gz, delta, theta, psi)

    y  = k^theta  * (z  * l )^(1 - theta)
    y1 = k1^theta * (z1 * l1)^(1 - theta)
    c  = y  - (1 + gz) * (1 + gn) * k1 + (1 - delta) * k  - g
    c1 = y1 - (1 + gz) * (1 + gn) * k2 + (1 - delta) * k1 - g1

    return (1 + taux) * c^(-sigma) * (1 - l)^(psi * (1 - sigma)) -
           beth * c1^(-sigma) * (1 - l1)^(psi * (1 - sigma)) *
                 (theta * y1 / k1 + (1 - delta) * (1 + taux1))
end

"Finite-difference Jacobian `dR[i] = ∂R/∂Z[i]` (length 11)."
jacobian_res_wedge(Z::AbstractVector, param::AbstractVector) =
    central_gradient(z -> res_wedge(z, param), Z)

"Stack a steady `Z` from the unconditional wedge mean `Sbar = [log z, τ_h, τ_x, log g]` and `ks`."
function steady_Z_from_Sbar(Sbar::AbstractVector, ks::Real)
    lk, lz, lg   = log(ks), Sbar[1], Sbar[4]
    tauls, tauxs = Sbar[2], Sbar[3]
    return [lk, lk, lk, lz, lz, tauls, tauls, tauxs, tauxs, lg, lg]
end

# =============================================================================
# 3. Route ① — quadratic root + 4×4 linear solve for `γ` on `S_t`
# =============================================================================

"Roots of `a₀ λ² + a₁ λ + a₂ = 0` (handles degenerate / negative-discriminant cases)."
function quadratic_roots(a0::Real, a1::Real, a2::Real)
    abs(a0) < 1e-14 && abs(a1) < 1e-14 && error("Degenerate polynomial [a0,a1,a2] = [$a0,$a1,$a2]")
    abs(a0) < 1e-14 && return ComplexF64[-a2 / a1]
    disc = a1^2 - 4a0 * a2
    s    = sqrt(Complex(disc))
    return ComplexF64[(-a1 + s) / (2a0), (-a1 - s) / (2a0)]
end

"""
    policy_mleq_style(Sbar, P, P0, param)

Log-linear capital policy `log k_{t+1} = γ₀ + γ_k log k_t + γ' S_t` from the Euler Jacobian:
stable root of `a₀λ² + a₁λ + a₂ = 0` for `γ_k`, then a 4×4 solve for `γ` on
`S_t = [log z, τ_h, τ_x, log g]'`.

Also builds the **4×6 observation matrix `C`** for
`Y_t = C X_t + ω_t` with `X_t = [log k̂_t, log z_t, τ_{ℓt}, τ_{xt}, log ĝ_t, 1]'` and
`Y_t = [log ŷ_t, log x̂_t, log ℓ_t, log ĝ_t]'`: five columns are derivatives w.r.t. the first
five states plus **`k_{t+1}`** effects via `Γ(1:5) = [γ_k, γ']`, and the sixth column is the
intercept `φ₀ = Y₀ − C_{1:5} X₀(1:5)` at steady state (same construction as `mleq.m` §5d).
"""
function policy_mleq_style(Sbar::AbstractVector, P::AbstractMatrix, P0::AbstractVector, param::AbstractVector)
    size(P) == (4, 4) || error("P must be 4×4")
    length(Sbar) == length(P0) == 4 || error("Sbar, P0 must have length 4")
    gn, gz, beta, delta, psi, sigma, theta = param[1:7]

    tem = (I(4) - P) \ P0
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

    Z  = steady_Z_from_Sbar(tem, ks)
    dR = jacobian_res_wedge(Z, param)

    a0, a1, a2 = dR[1], dR[2], dR[3]
    b0_row     = transpose(dR[[4, 6, 8, 10]])
    b1_row     = transpose(dR[[5, 7, 9, 11]])

    roots_ = quadratic_roots(a0, a1, a2)
    stable = filter(λ -> abs(λ) < 1, roots_)
    isempty(stable) && error("No stable root for [a0,a1,a2]; roots = $roots_")
    gammak = stable[argmin(abs.(stable))]

    rhs    = (b0_row * P + b1_row)'
    gamma  = -((a0 * gammak + a1) * I(4) + a0 * P') \ rhs
    gamma0 = (1 - gammak) * log(ks) - dot(gamma, [log(zs); tauls; tauxs; log(gs)])
    Gamma  = vcat(gammak, gamma, gamma0)

    # --- observation matrix C (4×6), same algebra as mleq.m §5d ----------------
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

    phixk  = -ks / xs * (1 - delta)
    phixkp = ks / xs * (1 + gz) * (1 + gn)

    Γ5 = vcat(Float64(real(gammak)), Float64.(real.(gamma[1:4])))
    r1 = [phiyk, phiyz, phiyl, 0.0, phiyg] .+ phiykp .* Γ5
    r2 = [phixk, 0.0, 0.0, 0.0, 0.0] .+ phixkp .* Γ5
    r3 = [philk, philz, phill, 0.0, philg] .+ philkp .* Γ5
    r4 = [0.0, 0.0, 0.0, 0.0, 1.0]
    C_base = vcat(transpose(r1), transpose(r2), transpose(r3), transpose(r4))

    X0_top = [log(ks), log(zs), tauls, tauxs, log(gs)]
    Y0     = [log(ys), log(xs), log(ls), log(gs)]
    phi0   = Y0 - C_base * X0_top
    C_meas = hcat(C_base, phi0)

    return (; gammak, gamma, gamma0, Gamma, dR, a0, a1, a2, b0_row, b1_row,
            ks, Z, tem, ys, xs, ls, cs,
            philh, philk, philz, phill, philg, philkp,
            phiyk, phiyz, phiyl, phiyg, phiykp, phixk, phixkp,
            C_base, phi0, C = C_meas)
end

# =============================================================================
# 4. Route ② — stacked pencil (A₁, A₂) + Vaughan + 8×8 solve for (B, D₂)
# =============================================================================

"""
    build_A1_A2_from_dR(Z, param, dR)

Build `A₁`, `A₂` for `A₁ X_{t+1} + A₂ X_t = 0` with `X_t = [k̃_t, k̃_{t+1}, h̃_t]`:
row 1 is the identity on `k̃_{t+1}`; row 2 is the linearized intratemporal FOC; row 3 is
the linearized Euler on capital.
"""
function build_A1_A2_from_dR(Z::AbstractVector, param::AbstractVector, dR::AbstractVector)
    a_row2 = labor_intratemporal_jacobian(Z, param)
    a2e, a1e, a0e = dR[3], dR[2], dR[1]
    A1 = [1.0 0.0 0.0;
          0.0 0.0 0.0;
          0.0 a0e 0.0]
    A2 = [0.0        -1.0        0.0;
          a_row2[1]   a_row2[2]  a_row2[3];
          a2e         a1e        0.0]
    return A1, A2
end

"""
    vaughan_stable_AC(A₁, A₂)

Solve `A₂ V = (-A₁) V Λ`, select the stable `|λ| < 1`, and return
`A = λ`, `C₂ = v₃/v₁`, `C_kh = [A, C₂]` along with the full eigendata.
"""
function vaughan_stable_AC(A1::AbstractMatrix{<:Real}, A2::AbstractMatrix{<:Real})
    size(A1) == size(A2) == (3, 3) || error("A1, A2 must be 3×3")
    F = eigen(A2, -A1)
    vals, vecs = F.values, F.vectors
    stable_idx = [i for i in eachindex(vals) if abs(vals[i]) < 1]
    isempty(stable_idx) && error("No stable generalized eigenvalue: $vals")
    j = stable_idx[argmin(abs.(vals[stable_idx]))]
    v = real.(vecs[:, j])
    abs(v[1]) < 1e-14 && error("Eigenvector first component ≈ 0")
    A  = real(vals[j])
    C2 = length(v) ≥ 3 ? v[3] / v[1] : 0.0
    return (; lambda_stable = A, eigenvalues = vals, eigenvectors = vecs,
            j_stable = j, v_normalized = v ./ v[1], A, C2, C_kh = [A, C2])
end

"Shock-row blocks on `S_t`: labor intratemp by FD at fixed `ℓ = ℓ_ss` (`G₂`);
Euler current (`G₃`) and lead (`H₃`) from `dR`."
function shock_rows_from_dR(Z::AbstractVector, param::AbstractVector, dR::AbstractVector; ϵ::Real=1e-6)
    gn, gz, _, delta, psi, _, theta = param[1:7]
    k, k1, z = exp(Z[3]), exp(Z[2]), exp(Z[5])
    taul, g  = Z[7], exp(Z[11])
    l0 = solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi)
    rfoc = (kv, k1v, zv, tauv, gv, lv) ->
        labor_foc_residual(kv, k1v, zv, tauv, gv, gn, gz, delta, theta, psi, lv)
    dz = (rfoc(k, k1, z*exp(ϵ), taul, g, l0) -
          rfoc(k, k1, z*exp(-ϵ), taul, g, l0)) / (2ϵ)
    dtauh = (rfoc(k, k1, z, taul+ϵ, g, l0) -
             rfoc(k, k1, z, taul-ϵ, g, l0)) / (2ϵ)
    dtaux = 0.0                         # labor FOC doesn't depend on τ_x
    dg = (rfoc(k, k1, z, taul, g*exp(ϵ), l0) -
          rfoc(k, k1, z, taul, g*exp(-ϵ), l0)) / (2ϵ)
    G2 = [dz, dtauh, dtaux, dg]
    G3 = Vector(dR[[5, 7, 9, 11]])
    H3 = Vector(dR[[4, 6, 8, 10]])
    return G2, G3, H3
end

"Unpack scalars `(a₁,a₂,a₃, b₁,b₂,b₃,b₄,b₅)` from the stacked pencil."
function abc_from_A1A2(A1::AbstractMatrix, A2::AbstractMatrix)
    a1, a2, a3 = A2[2, 1], A2[2, 2], A2[2, 3]
    b1, b2, b4 = A2[3, 1], A2[3, 2], A2[3, 3]
    b3, b5     = A1[3, 2], A1[3, 3]
    return (; a1, a2, a3, b1, b2, b3, b4, b5)
end

"""
    policy_vaughan_linear_policy(A₁, A₂, P, Z, param, dR)

Full log-linear policy from the stacked pencil: Vaughan gives `(A, C₂)`; with
`E_t S_{t+1} = P S_t`, the coefficients on `S_t` yield an 8×8 linear system in
`(B', D₂')`. Residuals `res_k_*` / `res_S_*` should be ≈ 0 at the solution.
"""
function policy_vaughan_linear_policy(
    A1::AbstractMatrix{<:Real}, A2::AbstractMatrix{<:Real},
    P::AbstractMatrix{<:Real},  Z::AbstractVector,
    param::AbstractVector,      dR::AbstractVector,
)
    size(P) == (4, 4) || error("P must be 4×4")
    v = vaughan_stable_AC(A1, A2)
    A, C2 = v.A, v.C2
    C = reshape([A, C2], 2, 1)

    (; a1, a2, a3, b1, b2, b3, b4, b5) = abc_from_A1A2(A1, A2)
    G2, G3, H3 = shock_rows_from_dR(Z, param, dR)
    PT = Matrix(transpose(P))
    I4 = Matrix{Float64}(I, 4, 4)

    M = zeros(8, 8)
    M[1:4, 1:4] .= a2 .* I4
    M[1:4, 5:8] .= a3 .* I4
    M[5:8, 1:4] .= (b2 + b3 * A + b5 * C2) .* I4 .+ b3 .* PT
    M[5:8, 5:8] .= b4 .* I4 .+ b5 .* PT
    rhs = vcat(.-G2, .-(G3 .+ PT * H3))
    w   = M \ rhs

    Bv, D2v = w[1:4], w[5:8]
    B  = reshape(Bv, 1, 4)
    D2 = reshape(D2v, 1, 4)

    res_k_intratemp = a1 + a2 * A + a3 * C2
    res_k_euler     = b1 + b2 * A + b3 * A^2 + b4 * C2 + b5 * C2 * A
    res_S_intra = a2 .* Bv .+ a3 .* D2v .+ G2
    res_S_euler = b2 .* Bv .+ b3 .* A .* Bv .+ b3 .* PT * Bv .+
                  b4 .* D2v .+ b5 .* C2 .* Bv .+ b5 .* PT * D2v .+ G3 .+ PT * H3

    return (; A, C, C2, B, D2, res_k_intratemp, res_k_euler,
            res_S_intra, res_S_euler, M_cond = cond(M),
            lambda_stable = v.lambda_stable, eigenvalues = v.eigenvalues,
            eigenvectors  = v.eigenvectors, j_stable = v.j_stable,
            v_normalized  = v.v_normalized, C_kh = v.C_kh)
end

# =============================================================================
# 5. Diagnostics and bundled comparison
# =============================================================================

"∂ ln h / ∂ ln k along the steady-state diagonal (`Z[2]`, `Z[3]` move together); `h = 1 − leisure`."
function labor_h_ln_k_elasticity(Z::AbstractVector, param::AbstractVector; δ::Real=1e-5)
    gn, gz, _, delta, psi, _, theta = param[1:7]
    function get_h(Zc)
        k, k1, z = exp(Zc[3]), exp(Zc[2]), exp(Zc[5])
        taul, g  = Zc[7], exp(Zc[11])
        return 1 - solve_labor_given_k(k, k1, z, taul, g, gn, gz, delta, theta, psi)
    end
    Zp = copy(Z); Zp[2] += δ; Zp[3] += δ
    Zm = copy(Z); Zm[2] -= δ; Zm[3] -= δ
    return (log(get_h(Zp)) - log(get_h(Zm))) / (2δ)
end

"""
    bca_policy_compare(Sbar, P, P0, param)

Run Route ① (`policy_mleq_style`) and Route ② (`policy_vaughan_linear_policy`) on the same
steady state and report `|γ_k − λ_stable|` and a labor elasticity for both.
"""
function bca_policy_compare(Sbar::AbstractVector, P::AbstractMatrix, P0::AbstractVector, param::AbstractVector)
    m      = policy_mleq_style(Sbar, P, P0, param)
    A1, A2 = build_A1_A2_from_dR(m.Z, param, m.dR)
    linear = policy_vaughan_linear_policy(A1, A2, P, m.Z, param, m.dR)
    gk     = Float64(real(m.gammak))
    h_elas = labor_h_ln_k_elasticity(m.Z, param)
    return (; mleq = m, linear, A1, A2,
            diff_gammak_lambda = abs(gk - linear.lambda_stable),
            C_Z_X_mleq    = [gk,                  h_elas],
            C_Z_X_vaughan = [linear.lambda_stable, h_elas])
end

"Default quarterly calibration; run both routes and print a summary."
function run_bca_policy_linear_tests(; sigma::Real = 1.001)
    P     = Matrix(0.995 * I(4))
    Sbar  = [log(1.0), 0.05, 0.0, log(0.07)]
    P0    = (I(4) - P) * Sbar
    gn    = (1.015)^(1 / 4) - 1
    gz    = (1.016)^(1 / 4) - 1
    beta  = 0.9722^(1 / 4)
    delta = 1 - (1 - 0.0464)^(1 / 4)
    param = [gn, gz, beta, delta, 2.24, sigma, 0.35]

    r = bca_policy_compare(Sbar, P, P0, param)
    println("=== BCA linear policy — two routes ===")
    println("  γ_k  (Route ①, quadratic root)          = ", Float64(real(r.mleq.gammak)))
    println("  λ    (Route ②, stable gen. eigenvalue)  = ", r.linear.lambda_stable)
    println("  |γ_k − λ|                               = ", r.diff_gammak_lambda)
    println("  C_ZX (Route ①)  [k̃_{t+1}/k̃_t, ∂ln h/∂ln k] = ", r.C_Z_X_mleq)
    println("  C_ZX (Route ②)                             = ", r.C_Z_X_vaughan)
    println("  Route ② residuals: max|res_S| = ",
            maximum(abs.(r.linear.res_S_intra)), ", ",
            maximum(abs.(r.linear.res_S_euler)),
            "  (intratemp / Euler on S_t)")
    return r
end

# =============================================================================
# 6. Closed-form benchmark (HW1-style limit: σ = δ = 1, g_n = g_z = ψ = 0)
# =============================================================================

"""
    appendix_a221_analytical(; beta, theta)

`A = θ` and `B_row = [1−θ, 0, −1+βθ, 0]` on `S_t = [log z, τ_h, τ_x, log g]'`.
"""
appendix_a221_analytical(; beta::Real, theta::Real) =
    (; A = theta, B_row = [1 - theta, 0.0, -1 + beta * theta, 0.0])

"""
    run_appendix_a221_test(; beta, theta, rho_z, psi_floor)

Compare Route ① and Route ② against `appendix_a221_analytical` using a tiny `ψ` so
the labor FOC stays non-degenerate.
"""
function run_appendix_a221_test(; beta::Real = 0.96, theta::Real = 0.35,
                                rho_z::Real = 0.9, psi_floor::Real = 1e-7)
    param = [0.0, 0.0, beta, 1.0, psi_floor, 1.0, theta]
    Sbar  = [log(1.0), 0.0, 0.0, log(1e-12)]
    P           = zeros(4, 4); P[1, 1] = rho_z
    P0          = (I(4) - P) * Sbar

    ref    = appendix_a221_analytical(; beta, theta)
    m      = policy_mleq_style(Sbar, P, P0, param)
    A1, A2 = build_A1_A2_from_dR(m.Z, param, m.dR)
    linear = policy_vaughan_linear_policy(A1, A2, P, m.Z, param, m.dR)

    gk  = Float64(real(m.gammak))
    B_m = Vector{Float64}(real.(m.gamma))
    B_v = vec(linear.B)
    err_route1_A = abs(gk       - ref.A)
    err_route2_A = abs(linear.A - ref.A)
    err_route1_B = maximum(abs.(B_m .- ref.B_row))
    err_route2_B = maximum(abs.(B_v .- ref.B_row))

    println("=== Appendix A.2.2.1 — analytical vs numerical (ψ = $(psi_floor)) ===")
    println("  Analytical A = θ = ", ref.A)
    println("  Route ① γ_k = ", gk,       "  |Δ| = ", err_route1_A)
    println("  Route ② A   = ", linear.A, "  |Δ| = ", err_route2_A)
    println("  Analytical B (z, τ_h, τ_x, log g) = ", ref.B_row)
    println("  Route ① γ   = ", B_m, "  max|Δ| = ", err_route1_B)
    println("  Route ② B   = ", B_v, "  max|Δ| = ", err_route2_B)
    return (; ref, mleq = m, linear,
            err_route1_A, err_route2_A, err_route1_B, err_route2_B)
end
