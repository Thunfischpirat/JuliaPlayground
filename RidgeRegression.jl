using LinearAlgebra
using Plots

struct RidgeRegression
    w::Vector{Float64}
end

function fit(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64)
    X = [X ones(size(X, 1))]
    w = inv(X'*X + lambda*I)*X'*y
    RidgeRegression(w)
end 

function predict(model::RidgeRegression, X::Matrix{Float64})
    X = [X ones(size(X, 1))]
    X*model.w
end

# 20x1 Matrix as dummy data.
X = reshape(collect(range(0.0, stop=10.0, length=20)), (20, 1)) .+ 1
# Add some noise to the data
y = vec(X+3*rand(20, 1))

model = fit(X, y, 1.0)
y_pred = predict(model, X)

# Plot the data
plot(X, y, seriestype=:scatter, label="data", title="Ridge Regression")
# Plot regression line
plot!(X, y_pred, label="regression line")
xlabel!("x")
ylabel!("y")
