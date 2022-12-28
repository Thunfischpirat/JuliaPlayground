using LinearAlgebra
using Plots

### a)

# Matrix B corresponding to midpoint rule.
B_(n) = 1/n * tril(ones(n,n))
# Functions from the exercise.
g_(x) = x^3 - 2x^2
f_(x) = 3x^2 - 4x

### b)

## Generate data
n = 100

g = [g_(i/n) for i = 1:n]
B = B_(n)

## Calculate F according to the formula from the lecture notes on page 70.

# Perform SVD on B.
U = svd(B).U
V = svd(B).V
S = svd(B).S

# Define filter function for Tikhonov regularization.
F(σ,γ) = σ^2/(σ^2 + γ^2)

# Calculate f(γ).
f(γ, G) = V * (F.(S,γ) .* (1 ./ S) .* (U' * G))

# Calculate f(γ) for different values of γ.
f_γ = [f(γ, g) for γ = [0.1, 0.001, 0.0001]]
# Plot f_γ and f(x) for comparison.
pt = plot(f_γ, label=["γ = 0.1" "γ = 0.001" "γ = 0.0001"],
                linestyle = [:solid :dot :dashdotdot],
                title="Tikhonov-Approximation of f(x)",
                xlabel="x",
                ylabel="f(x)")
plot!(pt, [f_(i/n) for i = 1:n], label="f(x)")

### c)

# Generate noisy data by adding uniform noise from [-0.01, 0.01] to g.
noise_level = 0.1
g_noisy = g + noise_level * (rand(n) .*2 .- 1)

# Calculate the approximation of g for different values of γ.
g_approx = [B*f(γ, g_noisy) for γ = [0.1, 0.001, 0.0001]]
# Plot g_approx and g for comparison.
pt = plot(g_approx, label=["γ = 0.1" "γ = 0.001" "γ = 0.0001"],
                linestyle = [:solid :dot :dashdotdot],
                title="Tikhonov-Approximation of g(x) plus noise (ε=$(noise_level))",
                xlabel="x",
                ylabel="g(x)")
plot!(pt, g, label="g(x)", linestyle=:dashdot)
plot!(pt, g_noisy, label="g(x) + ε", linestyle=:dashdotdot)
