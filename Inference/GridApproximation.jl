### Approximate the posterior distribution of θ of a beta-bernoulli model using a grid approximation.
### Based on https://probml.github.io/pml-book/book2.html 7.4.2.

using Distributions
using SpecialFunctions
using Plots

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

# Define the grid approximation
function grid_approximation(D, n)
    # Define the grid
    param_grid = range(0, 1, length=n)
    # Compute the posterior distribution on the grid
    posterior_grid = [posterior(θ, D) for θ in param_grid]
    # Normalize the posterior distribution
    posterior_grid ./= (1/length(param_grid) * sum(posterior_grid))
    # Return the grid and the posterior distribution
    return param_grid, posterior_grid
end

# Define D consisting of 10 heads (0) and 1 tail (1).
D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


# Compute the posterior distribution of θ using a grid approximation with 100 points.
param_grid, posterior_grid = grid_approximation(D, 100)

# Plot the grid approximation of the posterior distribution of θ
plot(param_grid, posterior_grid, xlabel="θ", ylabel="Density", line=:dash, legend=false)
# Plot the theoretical posterior distribution of θ
theoretical_posterior = [posterior(θ, D) for θ in param_grid] / marginal_likelihood(D)
plot!(param_grid, theoretical_posterior, line=:dashdot, label="Theoretical posterior")