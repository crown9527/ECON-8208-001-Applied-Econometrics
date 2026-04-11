""" Compute Q, W, R for the LQ approximation of return function r(x, u).


Arguments
- `r`     : return function r(x, u), scalar-valued.
- `x_bar` : n-vector, steady-state state.
- `u_bar` : m-vector, steady-state control.
- `δ`     : step size for finite differences (default 1e-5).

Returns (Q, W, R):
- `Q` : n×n symmetric matrix  (=1/2 r_xx)
- `W` : n×m matrix            (=1/2 r_xu)
- `R` : m×m symmetric matrix  (=1/2 r_uu)
"""

using LinearAlgebra


function compute_QWR(r, x_bar, u_bar; δ=1e-5)
    n = length(x_bar)
    m = length(u_bar)
    r0 = r(x_bar, u_bar)

    # ── r_xx: n×n Hessian of r w.r.t. x ──
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

    # ── r_uu: m×m Hessian of r w.r.t. u ──
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

    # ── r_xu: n×m cross-Hessian ──
    H_xu = zeros(n, m)
    for i in 1:n
        ei_x = zeros(n); ei_x[i] = δ
        for j in 1:m
            ej_u = zeros(m); ej_u[j] = δ
            H_xu[i, j] = (r(x_bar + ei_x, u_bar + ej_u) - r(x_bar + ei_x, u_bar - ej_u)
                        - r(x_bar - ei_x, u_bar + ej_u) + r(x_bar - ei_x, u_bar - ej_u)) / (4δ^2)
        end
    end

    Q = H_xx / 2
    W = H_xu / 2
    R = H_uu / 2

    return Q, W, R
end


""" Compute A, B for the LQ approximation of transition function g(x, u).


Arguments
- `g`     : transition function g(x, u), returns n-vector.
- `x_bar` : n-vector, steady-state state.
- `u_bar` : m-vector, steady-state control.
- `δ`     : step size for finite differences (default 1e-5).

Returns (A, B):
- `A` : n×n matrix  (= g_x evaluated at steady state)
- `B` : n×m matrix  (= g_u evaluated at steady state)
"""


function compute_AB(g, x_bar, u_bar; δ=1e-5)
    n = length(x_bar)
    m = length(u_bar)

    # ── A = g_x: n×n Jacobian of g w.r.t. x ──
    A = zeros(n, n)
    for j in 1:n
        ej = zeros(n); ej[j] = δ
        A[:, j] = (g(x_bar + ej, u_bar) - g(x_bar - ej, u_bar)) / (2δ)
    end

    # ── B = g_u: n×m Jacobian of g w.r.t. u ──
    B = zeros(n, m)
    for j in 1:m
        ej = zeros(m); ej[j] = δ
        B[:, j] = (g(x_bar, u_bar + ej) - g(x_bar, u_bar - ej)) / (2δ)
    end

    return A, B
end


"""
    compute_ABC(g, x_bar, u_bar, n_ε; δ=1e-5)

Compute A, B, C for the stochastic LQ approximation:
    X_{t+1} = g(X_t, u_t, ε_{t+1}) ≈ A X_t + B u_t + C ε_{t+1}
where the linearization is around (x̄, ū, 0).

Arguments
- `g`     : transition function g(x, u, ε), returns n-vector.
- `x_bar` : n-vector, steady-state state.
- `u_bar` : m-vector, steady-state control.
- `n_ε`   : dimension of the shock vector ε.
- `δ`     : step size for finite differences (default 1e-5).

Returns (A, B, C)
- `A` : n×n matrix  (= ∂g/∂x at steady state)
- `B` : n×m matrix  (= ∂g/∂u at steady state)
- `C` : n×n_ε matrix  (= ∂g/∂ε at steady state, ε=0)
"""

function compute_ABC(g, x_bar, u_bar, n_ε; δ=1e-5)
    n = length(x_bar)
    m = length(u_bar)
    ε_bar = zeros(n_ε)

    A = zeros(n, n)
    for j in 1:n
        ej = zeros(n); ej[j] = δ
        A[:, j] = (g(x_bar + ej, u_bar, ε_bar) - g(x_bar - ej, u_bar, ε_bar)) / (2δ)
    end

    B = zeros(n, m)
    for j in 1:m
        ej = zeros(m); ej[j] = δ
        B[:, j] = (g(x_bar, u_bar + ej, ε_bar) - g(x_bar, u_bar - ej, ε_bar)) / (2δ)
    end

    C = zeros(n, n_ε)
    for j in 1:n_ε
        ej = zeros(n_ε); ej[j] = δ
        C[:, j] = (g(x_bar, u_bar, ε_bar + ej) - g(x_bar, u_bar, ε_bar - ej)) / (2δ)
    end

    return A, B, C
end



"""Solve the LQ Riccati equation via fixed-point iteration.

# Inputs
- `Q` : n×n matrix  (½ r_xx)
- `W` : n×m matrix  (½ r_xu)
- `R` : m×m matrix  (½ r_uu)
- `A` : n×n matrix  (g_x at steady state)
- `B` : n×m matrix  (g_u at steady state)
- `β` : discount factor

# Returns
- `F` : m×n feedback matrix  (optimal policy: û = −Fx̂)
- `P` : n×n value function matrix  (V̂(x̂) = x̂′Px̂)
"""
function solve_riccati(Q, W, R, A, B, β; tol=1e-10, max_iter=10000)
    n = size(A, 1)
    m = size(B, 2)

    P = zeros(n, n)
    F = zeros(m, n)

    for iter in 1:max_iter
        H_uu = R + β * B' * P * B
        H_ux = W' + β * B' * P * A
        F = H_uu \ H_ux
        A_cl = A - B * F
        P_new = Q - F' * W' - W * F + F' * R * F + β * A_cl' * P * A_cl

        err = maximum(abs.(P_new - P))
        P = P_new

        if err < tol
            println("Converged in $iter iterations (error ≈ $(round(err, sigdigits=3)))")
            return F, P
        end
    end

    @warn "Did not converge after $max_iter iterations"
    return F, P
end



"""
    transform_to_standard(Q, W, R, A, B, β)

Transform a discounted LQ problem **with cross-term** into standard form
(no cross-term, no discounting).

# Arguments
- `Q` : n×n  (½ Hessian of return w.r.t. state)
- `W` : n×m  (½ cross-Hessian, state × control)
- `R` : m×m  (½ Hessian of return w.r.t. control)
- `A` : n×n  (Jacobian of transition w.r.t. state)
- `B` : n×m  (Jacobian of transition w.r.t. control)
- `β` : scalar discount factor (here β̃ for detrended model)

# Returns
`(Q̃, Ã, B̃)` — matrices of the standard-form LQ problem.
"""
function transform_to_standard(Q, W, R, A, B, β)
    Rinv_Wt = R \ W'                           # R⁻¹ W'  (m×n)
    Q_tilde = Q - W * Rinv_Wt                   # Q̃ = Q − W R⁻¹ W'
    A_tilde = sqrt(β) * (A - B * Rinv_Wt)       # Ã = √β (A − B R⁻¹ W')
    B_tilde = sqrt(β) * B                       # B̃ = √β B
    return Q_tilde, A_tilde, B_tilde
end


"""
    solve_standard_riccati(Q̃, R, Ã, B̃, W; tol, maxiter)

# Arguments
- `Q_tilde` : n×n  standard-form state cost
- `R`       : m×m  control cost (same in both forms)
- `A_tilde` : n×n  standard-form state transition
- `B_tilde` : n×m  standard-form control effect
- `W`       : n×m  original cross-term (needed to recover F)
- `tol`     : convergence tolerance on max|Pⁿ⁺¹ − Pⁿ|
- `maxiter` : maximum number of iterations

# Returns
- `F` : m×n feedback matrix in **original** coordinates
- `P` : n×n value-function matrix (V̂(x̂) ≈ x̂'P x̂)
"""
function solve_standard_riccati(Q_tilde, R, A_tilde, B_tilde, W;
                                 tol=1e-10, maxiter=10000)
    n = size(A_tilde, 1)
    P = zeros(n, n)                              # P⁰ = 0

    for iter in 1:maxiter
        BtPB = B_tilde' * P * B_tilde             # B̃'PB̃  (m×m)
        BtPA = B_tilde' * P * A_tilde             # B̃'PÃ  (m×n)
        F_tilde = (R + BtPB) \ BtPA               # F̃ = (R + B̃'PB̃)⁻¹ B̃'PÃ
        P_new = Q_tilde + A_tilde' * P * A_tilde - BtPA' * F_tilde   # Riccati update

        err = maximum(abs.(P_new - P))
        P = P_new
        if err < tol
            println("Standard Riccati converged in $iter iterations (err ≈ $(round(err, sigdigits=3)))")
            F = F_tilde + R \ W'                   # undo cross-term: F = F̃ + R⁻¹W'
            return F, P
        end
    end

    @warn "Did not converge after $maxiter iterations"
    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = F_tilde + R \ W'
    return F, P
end





"""
    compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

Build the 2n x 2n Hamiltonian matrix H for Vaughan's method,
from the standard-form LQ matrices (no discount, no cross-term).

The eigenvalues of H come in reciprocal pairs (mu, 1/mu).

# Arguments
- `A_tilde` : n x n  standard-form state transition
- `B_tilde` : n x m  standard-form control effect
- `Q_tilde` : n x n  standard-form state cost
- `R`       : m x m  control cost

# Returns
- `H` : 2n x 2n  Hamiltonian matrix
"""
function compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)
    Ainv = inv(A_tilde)
    M = Ainv * B_tilde * (R \ B_tilde')       # A^{-1} B R^{-1} B'
    H = [Ainv            M;
         Q_tilde*Ainv    Q_tilde*M + A_tilde']
    return H
end


"""
    solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)

Solve the LQ problem via Vaughan's (1970) eigenvalue decomposition
of the Hamiltonian matrix H.

# Algorithm
1. Build H from the standard-form matrices.
2. Eigendecompose H.  Its 2n eigenvalues come in reciprocal pairs
   (mu_i, 1/mu_i).  Select the n eigenvectors whose eigenvalues
   have |mu| > 1  (the "unstable" eigenvalues of H, which correspond
   to the *stable* eigenvalues of the forward system H^{-1}).
3. Stack those eigenvectors as columns:
       V = [V_11; V_21]   (each n x n)
   Then the value-function matrix is  P = V_21 * V_11^{-1}.
4. Recover the feedback:
       F_tilde = R^{-1} B_tilde' P
       F       = F_tilde + R^{-1} W'     (undo cross-term)

# Arguments
- `Q_tilde` : n x n
- `R`       : m x m
- `A_tilde` : n x n
- `B_tilde` : n x m
- `W`       : n x m  original cross-term (needed to recover F)

# Returns
- `F` : m x n  feedback matrix in original coordinates
- `P` : n x n  value-function matrix
"""
function solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)
    n = size(A_tilde, 1)

    H = compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    # Select eigenvectors with |eigenvalue| > 1 (unstable for H,
    # stable for forward system H^{-1})
    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n,     :]
    V21 = V[n+1:2n,  :]

    P = real.(V21 / V11)
    P = (P + P') / 2                            # enforce symmetry

    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = real.(F_tilde + R \ W')

    println("Vaughan: eigenvalues of H = ", round.(sort(abs.(vals), rev=true), digits=4))
    return F, P
end


"""
    compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

Build the 2n x 2n Hamiltonian matrix H for Vaughan's method,
from the standard-form LQ matrices (no discount, no cross-term).

The eigenvalues of H come in reciprocal pairs (mu, 1/mu).

# Arguments
- `A_tilde` : n x n  standard-form state transition
- `B_tilde` : n x m  standard-form control effect
- `Q_tilde` : n x n  standard-form state cost
- `R`       : m x m  control cost

# Returns
- `H` : 2n x 2n  Hamiltonian matrix
"""
function compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)
    Ainv = inv(A_tilde)
    M = Ainv * B_tilde * (R \ B_tilde')       # A^{-1} B R^{-1} B'
    H = [Ainv            M;
         Q_tilde*Ainv    Q_tilde*M + A_tilde']
    return H
end


"""
    solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)

Solve the LQ problem via Vaughan's (1970) eigenvalue decomposition
of the Hamiltonian matrix H.

# Algorithm
1. Build H from the standard-form matrices.
2. Eigendecompose H.  Its 2n eigenvalues come in reciprocal pairs
   (mu_i, 1/mu_i).  Select the n eigenvectors whose eigenvalues
   have |mu| > 1  (the "unstable" eigenvalues of H, which correspond
   to the *stable* eigenvalues of the forward system H^{-1}).
3. Stack those eigenvectors as columns:
       V = [V_11; V_21]   (each n x n)
   Then the value-function matrix is  P = V_21 * V_11^{-1}.
4. Recover the feedback:
       F_tilde = R^{-1} B_tilde' P
       F       = F_tilde + R^{-1} W'     (undo cross-term)

# Arguments
- `Q_tilde` : n x n
- `R`       : m x m
- `A_tilde` : n x n
- `B_tilde` : n x m
- `W`       : n x m  original cross-term (needed to recover F)

# Returns
- `F` : m x n  feedback matrix in original coordinates
- `P` : n x n  value-function matrix
"""
function solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)
    n = size(A_tilde, 1)

    H = compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    # Select eigenvectors with |eigenvalue| > 1 (unstable for H,
    # stable for forward system H^{-1})
    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n,     :]
    V21 = V[n+1:2n,  :]

    P = real.(V21 / V11)
    P = (P + P') / 2                            # enforce symmetry

    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = real.(F_tilde + R \ W')

    println("Vaughan: eigenvalues of H = ", round.(sort(abs.(vals), rev=true), digits=4))
    return F, P
end



