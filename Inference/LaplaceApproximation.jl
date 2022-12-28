### Implement the Laplace approximation to the posterior distribution of θ of a beta-bernoulli model.
### Based on https://probml.github.io/pml-book/book2.html 7.4.3.

using Zygote
using Distributions

# Dataset D.
D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# Define the prior distribution of θ
α = 1
β = 1
prior = Beta(α, β)


# Marginal likelihood of the data.
marginal_likelihood(D) = beta(α + sum(D), β + length(D) - sum(D) / beta(α, β))


# Define the likelihood function
function likelihood(θ, y)
    return pdf.(Bernoulli(θ), y)
end

# Define the posterior distribution of θ
function posterior(θ, D)
    return prod(likelihood.(θ, D)) * pdf(prior, θ)
end


# MAP estimate of θ.
θ_map = (α + sum(D) - 1) / (α + β + length(D) - 2)

# Joint distribution of θ and D.
joint(θ) = prod(likelihood.(θ, D)) * pdf(prior, θ)

# Energy function of the joint distribution.
energy(θ) = -log(joint(θ))

# Hessian of the energy function at θ_map.
H = hessian(energy, θ_map)

# Approximation of posterior distribution of θ.
posterior(θ) = pdf(Normal(θ_map, sqrt(1/H)), θ)

# Plot the posterior distribution of θ.
θ_grid = range(0, 1, length=100)
plot(θ_grid, posterior.(θ_grid), xlabel="θ", ylabel="Density", label="Laplace approximation", line=:dash, )

# Plot the theoretical posterior distribution of θ.
theoretical_posterior = [posterior(θ, D) for θ in θ_grid] / marginal_likelihood(D)
plot!(θ_grid, theoretical_posterior, line=:dashdot, label="Theoretical posterior")