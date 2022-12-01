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

# Median filter for RGB image
function median_filter(img::Array{Float64, 3}, k::Int)
    # Get the size of the image
    m, n, c = size(img)
    # Create a new image
    img_new = zeros(m, n, c)
    # Loop over the image
    for i in 1:m
        for j in 1:n
            for l in 1:c
                # Get the window
                window = img[max(1, i-k):min(m, i+k), max(1, j-k):min(n, j+k), l]
                # Get the median
                img_new[i, j, l] = median(vec(window))
            end
        end
    end
    img_new
end

# function calculating the median of an Array
function median(arr::Array{Float64, 1})
    # Sort the array
    arr = sort(arr, Dims=1)
    # Get the length of the array
    n = length(arr)
    # Check if the length is even or odd
    if iseven(n)
        # If even, return the average of the two middle elements
        (arr[n÷2] + arr[n÷2+1])/2
    else
        # If odd, return the middle element
        arr[n÷2+1]
    end
end