using LinearAlgebra

"""
    newton_scalar(f, x0; h=1e-7, tol=1e-10, maxiter=1000)

Find the fixed point of a scalar function `f` using Newton's method
with central-difference numerical derivative.

Solves g(x) = f(x) - x = 0.
"""
function newton_scalar(f, x0; h=1e-7, tol=1e-10, maxiter=1000)
    x = Float64(x0)
    for i in 1:maxiter
        gx  = f(x) - x
        dgx = ((f(x + h) - (x + h)) - (f(x - h) - (x - h))) / (2h)
        abs(dgx) < 1e-14 && error("Numerical derivative of g(x) ≈ 0, Newton step undefined.")
        x_new = x - gx / dgx
        abs(x_new - x) < tol && return x_new
        x = x_new
    end
    @warn "newton_scalar did not converge within $maxiter iterations."
    return x
end


"""
    newton_vector(f, x0; h=1e-7, tol=1e-10, maxiter=1000)

Find the fixed point of a vector-valued function `f : ℝⁿ → ℝⁿ` using
Newton's method with central-difference numerical Jacobian.

Solves g(x) = f(x) - x = 0.
"""
function newton_vector(f, x0; h=1e-7, tol=1e-10, maxiter=1000)
    n = length(x0)
    x = copy(Float64.(x0))

    for i in 1:maxiter
        gx = f(x) - x

        Jg = zeros(n, n)
        for j in 1:n
            e_j = zeros(n); e_j[j] = h
            Jg[:, j] = (f(x + e_j) - (x + e_j) - f(x - e_j) + (x - e_j)) / (2h)
        end

        dx = Jg \ gx
        x_new = x - dx
        norm(x_new - x) < tol && return x_new
        x = x_new
    end
    @warn "newton_vector did not converge within $maxiter iterations."
    return x
end


# --- Central-difference Jacobians / gradients (for BCA and other callers) ----

"""
    central_gradient(F, x; del)

`F : ℝⁿ → ℝ` scalar. Returns `g` with `g[i] ≈ ∂F/∂xᵢ` (central difference).

`del` may be an `n`-vector of step sizes; default matches CKM `mleq.m`:
`del[i] = max(|x[i]|·1e-5, 1e-8)`.
"""
function central_gradient(F, x::AbstractVector; del=nothing)
    x = float.(collect(x))
    n = length(x)
    del === nothing && (del = max.(abs.(x) * 1e-5, 1e-8))
    g = zeros(n)
    for i in 1:n
        xp = copy(x)
        xm = copy(x)
        xp[i] += del[i]
        xm[i] -= del[i]
        g[i] = (F(xp) - F(xm)) / (2 * del[i])
    end
    return g
end


"""
    central_gradient_components(F, x, inds; δ)

`F : ℝⁿ → ℝ`. Returns a vector of length `length(inds)` with
`[∂F/∂x_{inds[1]}, …]` using step `δ` on each listed coordinate only.
"""
function central_gradient_components(F, x::AbstractVector, inds; δ::Real=1e-6)
    x = float.(collect(x))
    return [begin
                xp = copy(x)
                xm = copy(x)
                xp[i] += δ
                xm[i] -= δ
                (F(xp) - F(xm)) / (2δ)
            end for i in inds]
end
