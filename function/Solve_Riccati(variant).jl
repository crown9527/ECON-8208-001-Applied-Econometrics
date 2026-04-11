using LinearAlgebra


# ══════════════════════════════════════════════════════════════════════════════
#  Part 0: Shared utilities — finite-difference Hessians and Jacobians
# ══════════════════════════════════════════════════════════════════════════════

"""    compute_QWR(r, x_bar, u_bar; δ=1e-5)

Compute Q, W, R for the LQ approximation of return function r(x, u).

Returns (Q, W, R):
- `Q` : n×n symmetric matrix  (= ½ r_xx)
- `W` : n×m matrix            (= ½ r_xu)
- `R` : m×m symmetric matrix  (= ½ r_uu)
"""
function compute_QWR(r, x_bar, u_bar; δ=1e-5)
    n = length(x_bar)
    m = length(u_bar)

    H_xx = zeros(n, n)
    for i in 1:n
        ei = zeros(n); ei[i] = δ
        for j in i:n
            ej = zeros(n); ej[j] = δ
            H_xx[i, j] = (r(x_bar + ei + ej, u_bar) - r(x_bar + ei - ej, u_bar)
                        - r(x_bar - ei + ej, u_bar) + r(x_bar - ei - ej, u_bar)) / (4δ^2)
            H_xx[j, i] = H_xx[i, j]
        end
    end

    H_uu = zeros(m, m)
    for i in 1:m
        ei = zeros(m); ei[i] = δ
        for j in i:m
            ej = zeros(m); ej[j] = δ
            H_uu[i, j] = (r(x_bar, u_bar + ei + ej) - r(x_bar, u_bar + ei - ej)
                        - r(x_bar, u_bar - ei + ej) + r(x_bar, u_bar - ei - ej)) / (4δ^2)
            H_uu[j, i] = H_uu[i, j]
        end
    end

    H_xu = zeros(n, m)
    for i in 1:n
        ei_x = zeros(n); ei_x[i] = δ
        for j in 1:m
            ej_u = zeros(m); ej_u[j] = δ
            H_xu[i, j] = (r(x_bar + ei_x, u_bar + ej_u) - r(x_bar + ei_x, u_bar - ej_u)
                        - r(x_bar - ei_x, u_bar + ej_u) + r(x_bar - ei_x, u_bar - ej_u)) / (4δ^2)
        end
    end

    return H_xx / 2, H_xu / 2, H_uu / 2
end


"""    compute_AB(g, x_bar, u_bar; δ=1e-5)

Compute A, B for the LQ approximation of transition function g(x, u).

Returns (A, B):
- `A` : n×n matrix  (= g_x at steady state)
- `B` : n×m matrix  (= g_u at steady state)
"""
function compute_AB(g, x_bar, u_bar; δ=1e-5)
    n = length(x_bar)
    m = length(u_bar)

    A = zeros(n, n)
    for j in 1:n
        ej = zeros(n); ej[j] = δ
        A[:, j] = (g(x_bar + ej, u_bar) - g(x_bar - ej, u_bar)) / (2δ)
    end

    B = zeros(n, m)
    for j in 1:m
        ej = zeros(m); ej[j] = δ
        B[:, j] = (g(x_bar, u_bar + ej) - g(x_bar, u_bar - ej)) / (2δ)
    end

    return A, B
end


"""    transform_to_standard(Q, W, R, A, B, β)

Transform a discounted LQ problem with cross-term into standard form
(no cross-term, no discounting).

Returns `(Q̃, Ã, B̃)`.
"""
function transform_to_standard(Q, W, R, A, B, β)
    Rinv_Wt = R \ W'
    Q_tilde = Q - W * Rinv_Wt
    A_tilde = sqrt(β) * (A - B * Rinv_Wt)
    B_tilde = sqrt(β) * B
    return Q_tilde, A_tilde, B_tilde
end


# ══════════════════════════════════════════════════════════════════════════════
#  Part 1: Variant LQ — Riccati iteration
# ══════════════════════════════════════════════════════════════════════════════

"""    compute_transformed_blocks(Q, W, R, Ay, By, β, ny; Az=nothing)

Compute standard-form blocks for the variant LQ (eqs. 36–38):
    Ãy = √β (Ay − By R⁻¹ Wy')
    Ãz = √β (Az − By R⁻¹ Wz')
    B̃y = √β By
    Q̃  = Q − W R⁻¹ W'  →  Q̃y, Q̃z

Returns NamedTuple `(Ay_tilde, Az_tilde, By_tilde, Qy_tilde, Qz_tilde, Q_tilde)`.
"""
function compute_transformed_blocks(Q, W, R, Ay, By, β, ny; Az=nothing)
    n  = size(Q, 1)
    nz = n - ny

    Q_tilde = Q - W * (R \ W')

    Wy = W[1:ny, :]
    Wz = W[ny+1:n, :]

    if Az === nothing
        Az = zeros(ny, nz)
    end

    Ay_tilde = sqrt(β) * (Ay - By * (R \ Wy'))
    Az_tilde = sqrt(β) * (Az - By * (R \ Wz'))
    By_tilde = sqrt(β) * By

    Qy_tilde = Q_tilde[1:ny, 1:ny]
    Qz_tilde = Q_tilde[1:ny, ny+1:n]

    return (Ay_tilde=Ay_tilde, Az_tilde=Az_tilde, By_tilde=By_tilde,
            Qy_tilde=Qy_tilde, Qz_tilde=Qz_tilde, Q_tilde=Q_tilde)
end


"""    transform_market_clearing(Theta, Psi, R, W, ny)

Map market-clearing coefficients to the transformed problem:
    T = I + Ψ R⁻¹ Wz'
    Θ̃ = T⁻¹ (Θ − Ψ R⁻¹ Wy')
    Ψ̃ = T⁻¹ Ψ

Returns `(Theta_tilde, Psi_tilde)`.
"""
function transform_market_clearing(Theta, Psi, R, W, ny)
    Wy = W[1:ny, :]
    Wz = W[ny+1:end, :]
    nz = size(Psi, 1)
    T  = I(nz) + Psi * (R \ Wz')
    Theta_tilde = T \ (Theta - Psi * (R \ Wy'))
    Psi_tilde   = T \ Psi
    return Theta_tilde, Psi_tilde
end


"""    partition_and_build(Ay_tilde, Az_tilde, By_tilde, Qy_tilde, Qz_tilde, R, Theta_tilde, Psi_tilde)

Build modified block matrices from pre-computed standard-form blocks:
    Â = Ãy + Ãz Θ̃
    Q̂ = Q̃y + Q̃z Θ̃
    B̂ = B̃y + Ãz Ψ̃
    Ā = Ãy − B̃y R⁻¹ Ψ̃' Q̃z'

Returns NamedTuple `(A_hat, Q_hat, B_hat, A_bar, ...)`.
"""
function partition_and_build(Ay_tilde, Az_tilde, By_tilde, Qy_tilde, Qz_tilde,
                             R, Theta_tilde, Psi_tilde)
    A_hat = Ay_tilde + Az_tilde * Theta_tilde
    Q_hat = Qy_tilde + Qz_tilde * Theta_tilde
    B_hat = By_tilde + Az_tilde * Psi_tilde
    A_bar = Ay_tilde - By_tilde * (R \ (Psi_tilde' * Qz_tilde'))

    return (A_hat=A_hat, Q_hat=Q_hat, B_hat=B_hat, A_bar=A_bar,
            By_tilde=By_tilde, Ay_tilde=Ay_tilde, Az_tilde=Az_tilde,
            Qy_tilde=Qy_tilde, Qz_tilde=Qz_tilde)
end


"""    solve_modified_riccati(A_hat, Q_hat, B_hat, A_bar, By_tilde, R; ...)

Iterate the modified Riccati equation to convergence.
    F̃ = (R + B̃y' P B̂)⁻¹ B̃y' P Â
    P ← Q̂ + Ā' P (Â − B̂ F̃)

Returns `(F_tilde, P)`.
"""
function solve_modified_riccati(A_hat, Q_hat, B_hat, A_bar, By_tilde, R;
                                P0=nothing, tol=1e-10, maxiter=20000, verbose=true)
    n = size(A_hat, 1)
    P = P0 === nothing ? (-Matrix(I, n, n)) : Matrix(P0)
    P = (P + P') / 2

    F_tilde = zeros(size(B_hat, 2), n)
    for iter in 1:maxiter
        F_tilde = (R + By_tilde' * P * B_hat) \ (By_tilde' * P * A_hat)
        P_new = Q_hat + A_bar' * P * (A_hat - B_hat * F_tilde)
        P_new = (P_new + P_new') / 2

        err = maximum(abs.(P_new - P))
        P = P_new
        if err < tol
            verbose && println("Modified Riccati converged in $iter iterations (err ≈ $(round(err, sigdigits=3)))")
            return F_tilde, P
        end
    end

    @warn "Modified Riccati did not converge after $maxiter iterations"
    return F_tilde, P
end


# ══════════════════════════════════════════════════════════════════════════════
#  Part 2: Variant LQ — Vaughan's method (generalized eigenvalue pencil)
# ══════════════════════════════════════════════════════════════════════════════

"""    build_modified_pencil(A_hat, Q_hat, B_hat, A_bar, By_tilde, R)

Construct the generalized eigenvalue pencil from the Hamiltonian system

Forward Hamiltonian:  H = L⁻¹N   (z_{t+1} = H z_t)

`eigen(L, N)` solves  L v = λ N v,  i.e.  N⁻¹L v = λ v  →  eigenvalues of H⁻¹.
Since H and H⁻¹ share eigenvectors (eigenvalues are reciprocals), selecting
|λ| > 1 from `eigen(L, N)` picks the stable roots of H (|λ_H| < 1).

Returns `(L, N)`.
"""
function build_modified_pencil(A_hat, Q_hat, B_hat, A_bar, By_tilde, R)
    n = size(A_hat, 1)
    M = B_hat * (R \ By_tilde')
    L = [Matrix(1.0I, n, n)  M;
         zeros(n, n)          A_bar']
    N = [A_hat          zeros(n, n);
         -Q_hat         Matrix(1.0I, n, n)]
    return L, N
end


"""    solve_modified_vaughan_P(A_hat, Q_hat, B_hat, A_bar, By_tilde, R; verbose=true)

Compute P via the generalized eigenvalue pencil  L v = λ N v.
`eigen(L, N)` returns eigenvalues of H⁻¹ = N⁻¹L (reciprocals of the forward
Hamiltonian H = L⁻¹N).  Select n eigenvectors with |λ| > 1 (= stable roots
of H), then P = V₂₁ V₁₁⁻¹.

Returns `(F_tilde, P, L, N)`.
"""
function solve_modified_vaughan_P(A_hat, Q_hat, B_hat, A_bar, By_tilde, R; verbose=true)
    n = size(A_hat, 1)
    L, N_mat = build_modified_pencil(A_hat, Q_hat, B_hat, A_bar, By_tilde, R)

    eig = eigen(L, N_mat)
    vals = eig.values
    vecs = eig.vectors

    finite_mask = isfinite.(vals)
    idx_outside = findall(i -> finite_mask[i] && abs(vals[i]) > 1, 1:length(vals))
    if length(idx_outside) == n
        idx_pick = idx_outside
    else
        ord = sortperm(abs.(vals), rev=true)
        idx_pick = ord[1:n]
    end

    V = vecs[:, idx_pick]
    V11 = V[1:n, :]
    V21 = V[n+1:2n, :]

    P = real.(V21 / V11)
    P = (P + P') / 2

    F_tilde = (R + By_tilde' * P * B_hat) \ (By_tilde' * P * A_hat)

    finite_vals = filter(isfinite, vals)
    verbose && println("Modified Vaughan: |eigs| = ",
                       round.(sort(abs.(finite_vals), rev=true), digits=4))

    return F_tilde, P, L, N_mat
end


# ══════════════════════════════════════════════════════════════════════════════
#  Part 3: solve_modified_lq — end-to-end solver (dispatches Riccati/Vaughan)
# ══════════════════════════════════════════════════════════════════════════════

"""    solve_modified_lq(Q, W, R, Ay, By, β, ny, Theta, Psi; method, Az, kwargs...)

End-to-end solver for the variant LQ problem with distortions.

Pipeline:
1) `compute_transformed_blocks`  → Ãy, Ãz, B̃y, Q̃y, Q̃z  (eqs. 36–38)
2) `transform_market_clearing`   → Θ̃, Ψ̃
3) `partition_and_build`         → Â, Q̂, B̂, Ā
4) Solve via Riccati or Vaughan
5) Recover equilibrium policy F_eq from F̃

**Un-transformation (step 5).**
The cross-term removal is  ũ = u + R⁻¹(Wy'y + Wz'X₃).
At market clearing  X₃ = Θy + Ψu,  solving for u gives:

    (I + R⁻¹Wz'Ψ) u = -(F̃ + R⁻¹Wy' + R⁻¹Wz'Θ) y

so the equilibrium policy is:

    F_eq = (I + R⁻¹Wz'Ψ)⁻¹ (F̃ + R⁻¹(Wy' + Wz'Θ))

Returns NamedTuple with fields:
  F_eq, F_tilde, P, A_hat, Q_hat, B_hat, A_bar, ...
"""
function solve_modified_lq(Q, W, R, Ay, By, β, ny, Theta, Psi;
                           method=:riccati, Az=nothing, kwargs...)
    n  = size(Q, 1)
    nz = n - ny
    Wy = W[1:ny, :]
    Wz = W[ny+1:n, :]

    tb = compute_transformed_blocks(Q, W, R, Ay, By, β, ny; Az=Az)
    Theta_tilde, Psi_tilde = transform_market_clearing(Theta, Psi, R, W, ny)
    blk = partition_and_build(tb.Ay_tilde, tb.Az_tilde, tb.By_tilde,
                              tb.Qy_tilde, tb.Qz_tilde, R, Theta_tilde, Psi_tilde)

    if method == :riccati
        F_tilde, P = solve_modified_riccati(
            blk.A_hat, blk.Q_hat, blk.B_hat, blk.A_bar, blk.By_tilde, R; kwargs...)
    elseif method == :vaughan
        F_tilde, P, _, _ = solve_modified_vaughan_P(
            blk.A_hat, blk.Q_hat, blk.B_hat, blk.A_bar, blk.By_tilde, R; kwargs...)
    else
        error("Unknown method = $method. Use :riccati or :vaughan")
    end

    m = size(R, 1)
    T_eq = I(m) + R \ (Wz' * Psi)
    F_eq = T_eq \ (F_tilde + R \ (Wy' + Wz' * Theta))

    return (F_eq=F_eq, F_tilde=F_tilde, P=P,
            A_hat=blk.A_hat, Q_hat=blk.Q_hat, B_hat=blk.B_hat, A_bar=blk.A_bar,
            By_tilde=blk.By_tilde, Ay_tilde=blk.Ay_tilde, Az_tilde=blk.Az_tilde,
            Qy_tilde=blk.Qy_tilde, Qz_tilde=blk.Qz_tilde)
end


# ══════════════════════════════════════════════════════════════════════════════
#  Part 4: Standard LQ — Riccati iteration (for verification / HW5-style)
# ══════════════════════════════════════════════════════════════════════════════

"""    solve_standard_riccati(Q̃, R, Ã, B̃, W; tol, maxiter)

Iterate the standard Riccati equation (no discounting, no cross-term in
the transformed problem).  Returns `(F, P)` in **original** coordinates.
"""
function solve_standard_riccati(Q_tilde, R, A_tilde, B_tilde, W;
                                 tol=1e-10, maxiter=10000)
    n = size(A_tilde, 1)
    P = zeros(n, n)

    for iter in 1:maxiter
        BtPB = B_tilde' * P * B_tilde
        BtPA = B_tilde' * P * A_tilde
        F_tilde = (R + BtPB) \ BtPA
        P_new = Q_tilde + A_tilde' * P * A_tilde - BtPA' * F_tilde

        err = maximum(abs.(P_new - P))
        P = P_new
        if err < tol
            println("Standard Riccati converged in $iter iterations (err ≈ $(round(err, sigdigits=3)))")
            F = F_tilde + R \ W'
            return F, P
        end
    end

    @warn "Did not converge after $maxiter iterations"
    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = F_tilde + R \ W'
    return F, P
end


# ══════════════════════════════════════════════════════════════════════════════
#  Part 5: Standard LQ — Vaughan's method (for verification / HW5-style)
# ══════════════════════════════════════════════════════════════════════════════

"""    compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

Build the 2n×2n Hamiltonian matrix H for the standard LQ Vaughan method.
"""
function compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)
    Ainv = inv(A_tilde)
    M = Ainv * B_tilde * (R \ B_tilde')
    H = [Ainv            M;
         Q_tilde*Ainv    Q_tilde*M + A_tilde']
    return H
end


"""    solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)

Solve the standard LQ via Vaughan's eigenvalue decomposition.
Returns `(F, P)` in original coordinates.
"""
function solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)
    n = size(A_tilde, 1)

    H = compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n,     :]
    V21 = V[n+1:2n,  :]

    P = real.(V21 / V11)
    P = (P + P') / 2

    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = real.(F_tilde + R \ W')

    println("Vaughan: eigenvalues of H = ", round.(sort(abs.(vals), rev=true), digits=4))
    return F, P
end
