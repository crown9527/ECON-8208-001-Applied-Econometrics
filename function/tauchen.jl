using Distributions

""" tauchen(ρ, σ_ε; N=5, m=3)

Approximate an AR(1) process  log z' = ρ log z + ε,  ε ~ N(0, σ_ε²)
by a finite-state Markov chain using the Tauchen (1986) method.

# Arguments
- `ρ`   : persistence parameter (|ρ| < 1)
- `σ_ε` : std dev of innovation ε
- `N`   : number of grid points (default 5)
- `m`   : grid spans ±m unconditional std devs (default 3)

# Returns
- `z_grid` : length-N vector of grid values for log z
- `Π`      : N×N transition matrix, Π[i,j] = Pr(z'=z_j | z=z_i)
"""
function tauchen(ρ, σ_ε; N=5, m=3)
    σ_y = σ_ε / sqrt(1 - ρ^2)

    z_max = m * σ_y
    z_grid = range(-z_max, z_max, length=N) |> collect
    w = z_grid[2] - z_grid[1]

    d = Normal(0, σ_ε)
    Π = zeros(N, N)

    for i in 1:N
        μ = ρ * z_grid[i]
        for j in 1:N
            if j == 1
                Π[i, j] = cdf(d, z_grid[1] - μ + w/2)
            elseif j == N
                Π[i, j] = 1 - cdf(d, z_grid[N] - μ - w/2)
            else
                Π[i, j] = cdf(d, z_grid[j] - μ + w/2) - cdf(d, z_grid[j] - μ - w/2)
            end
        end
    end

    return z_grid, Π
end