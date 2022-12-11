using LinearAlgebra
using Random
using Plots

Random.seed!(1234)

# 20x1 Matrix as dummy data.
X = rand(200,10) .+ 1
weight = ones(10,1)
# Add some noise to the data
y = vec(3*X*weight + 0*rand(200, 1))

# Obtain singular values of X
sigma = maximum(svd(X).S)


function Landweber(X, y, sigma, max_iter=100, tol=1e-4)
    # Initialize w
    w = zeros(size(X, 2))
    # Initialize error
    error = Inf
    # Initialize iteration counter
    iter = 0
    # Initialize step size
    omega = 1/sigma^2
    # Initialize y_hat
    y_hat = X*w
    # Initialize error vector
    error_vec = []
    while error > tol && iter < max_iter
        # Update w
        w = w - omega*(X'*(y_hat - y))
        # Update y_hat
        y_hat = X*w
        # Update error
        error = norm(y - y_hat)/2
        # Update iteration counter
        iter += 1
        # Update error vector
        push!(error_vec, error)
    end
    return w, error_vec
end

# Plot error_vec for different multiples of sigma
pt = plot(title="Landweber-Iteration", xlabel="Iteration", ylabel="Error")
for i in [0.8, 1, 2, 4, 8]
    w, error_vec = Landweber(X, y, i*sigma)
    plot!(pt, error_vec, label="sigma = $(round(i*sigma, digits=2))")
    # print weight vector
    println("w = $(w)")
end

display(pt)





