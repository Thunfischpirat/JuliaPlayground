using LinearAlgebra
using Random
using Plots

Random.seed!(1234)

# Generate 200x10 random matrix with integers between 0 and 10.
X = rand(collect(0:10), 200, 10) / 20
# Make sure that X has a non-trivial nullspace.
X[:,1] = ones(200)
X[:,2] = -ones(200)

# Add some noise to the data
error_strength = 0
y = vec(3*X*ones(10,1) + error_strength*rand(200, 1))

# Obtain singular values of X
singular_values = svd(X).S
sigma = maximum(singular_values)


function Landweber(X, y, beta, initial_w="zeros", max_iter=1000, tol=1e-4)
    # Initialize w either as the zero vector or as an arbitrary vector from R^10.
    if initial_w == "zeros"
        w = zeros(size(X, 2))
    else
       w = initial_w
    end
    # Initialize error
    error = Inf
    # Initialize iteration counter
    iter = 0
    # Initialize y_hat
    y_hat = X*w
    # Initialize error vector
    error_vec = []
    while error > tol && iter < max_iter
        # Update w
        w = w - beta*(X'*(y_hat - y))
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


## Plot error_vec for different multiples of sigma
pt = plot(title="Landweber-Iteration (σ≈$(round(sigma, digits=2)))", xlabel="Iteration", ylabel="Error")
for b in [0.25, 0.5,  1, 1.25, 2] 
    w, error_vec = Landweber(X, y, b*1/sigma^2)
    plot!(pt, error_vec, label="β = $(b)1/σ²")
end
# Show plots
display(pt)

## Verify convergence of Landweber when the initial w is the zero vector.
# Moore-Penrose Pseudoinverse of X
X_pinv = pinv(X) 

# Calculate w using Moore-Penrose Pseudoinverse
w_pinv = X_pinv*y

# Calculate w using Landweber
w_landweber, _ = Landweber(X, y, 1/sigma^2)

print("Compare w using Moore-Penrose Pseudoinverse and Landweber:\n")
print("w_theo = $(w_pinv)\n")
print("w_landweber = $(w_landweber)\n")

## Verify convergence of Landweber when the initial w is an arbitrary vector from R^10.
# Initialize w
w = rand(10)
# Project w onto the nullspace of X
X_null = nullspace(X)
w_proj = X_null*inv(X_null'*X_null)*X_null'*w

# Calculate w using Landweber
w_landweber, _ = Landweber(X, y, 1/sigma^2, w)

# Theoretical value of w
w_theo = X_pinv*y + w_proj

print("Compare w using Landweber and Moore-Penrose plus nullspace projection:\n")
print("w_theo = $(w_theo)\n")
print("w_landweber = $(w_landweber)")