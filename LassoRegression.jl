
using CSV
using DataFrames
using Parameters
using LinearAlgebra
using Statistics
using Plots
import Random

# Set seed for reproducibility.
Random.seed!(1234)

# Load data (https://hastie.su.domains/ElemStatLearn/datasets/prostate.data)
df = CSV.read("Data/ESL/prostate.txt", delim="\t", DataFrame)
# Get rows where train is true.
df_train = df[df[:, end] .== true, :]

# Get explanatory variables and response variable from df.
X = Matrix(df_train[:, 2:end-2])
y = df_train[:, end-1]

# https://stats.stackexchange.com/questions/287370/standardization-vs-normalization-for-lasso-ridge-regression
mean_X = mean(X, dims=1)
std_X = std(X, dims=1)
X = (X .- mean_X)./std_X

# Create test data.
df_test = df[df[:, end] .== false, :]
X_test = Matrix(df_test[:, 2:end-2])
X_test = (X_test .- mean_X)./std_X
y_test = df_test[:, end-1]


# https://stackoverflow.com/questions/65945827/how-can-i-set-default-parameters-for-mutable-structs-in-julia
@with_kw mutable struct LassoRegression
    # Model for lasso regression.
    #
    # Fields:
    # w: Vector of coefficients.
    ############################
    w::Union{Vector{Float64}, Nothing} = nothing
end

function proximal_mapping(y, τ)
    # Calculate the proximal mapping of the lasso penalty.
    # 
    # Input:
    # y: Response variable.
    # τ: Regularization parameter.
    # 
    # Output:
    # w: Vector of coefficients.   
    ############################
    if y > τ
        return y - τ
    elseif y < -τ
        return y + τ
    else
        return 0
    end
end

function fit(model::LassoRegression ,X, y, λ=1, η=0.01, max_iter=1000)
    # Calculate the solution to the lasso regression problem.
    # 
    # Input:
    # model: Model for lasso regression.
    # X: Matrix of explanatory variables.
    # y: Vector of response variable.
    # λ: Regularization parameter.
    # η: Learning rate.
    # max_iter: maximum number of iterations.
    # 
    # Output:
    # w: Vector of coefficients.
    ############################
    # https://datascience.stackexchange.com/questions/43506/attach-1-for-the-bias-unit-in-neural-networks-what-does-it-mean
    X = [X ones(size(X, 1))]
    w = zeros(size(X, 2))
    i = 1
    while i <= max_iter
        for j in 1:size(X, 2)
            # For derivation of a_j and c_j see https://probml.github.io/pml-book/book1.html p. 383.
            a_j = sum(X[:, j].^2)
            c_j = sum(X[:, j].*(y - X*w + w[j]*X[:, j]))
            # Update coefficient. (See https://probml.github.io/pml-book/book1.html p. 394.)
            w[j] = proximal_mapping(w[j] - η*(a_j*w[j] - c_j) , η*λ)
        end
        i += 1
    end
    model.w = w
    return model
end

function predict(model::LassoRegression, X::Matrix{Float64})
    # Predict response variable.
    #
    # Input:
    # model: Model for lasso regression.
    # X: Matrix of explanatory variables.
    #
    # Output:
    # y_pred: Vector of predicted response variable.
    ############################
    X = [X ones(size(X, 1))]
    X*model.w
end

# Compare MSE of the model on the training data and the test data for different values of λ.
train_mse = []
test_mse = []
weights = []
for λ in [0., 0.1, 0.3, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]
    # Training
    local model = fit(LassoRegression(), X, y, λ)
    local y_pred = predict(model, X)
    # MSE of the model on the training data.
    local mse_train = sum((y_pred - y).^2)/length(y)
    push!(train_mse, mse_train)
    push!(weights, model.w[1:end-1])

    # Test
    local test_preds = predict(model, X_test)
    # MSE of the model on the test data.
    local mse_test = sum((test_preds - y_test).^2)/length(y_test)
    push!(test_mse, mse_test)
end


# Plot MSE of the model on the training data and the test data for different values of λ.
p1 = scatter([train_mse, test_mse],
        label=["train_mse" "test_mse"],
        xlabel="λ",
        ylabel="MSE",
        title="Training MSE for different values of λ",
        legend=:topright)
# https://github.com/JuliaPlots/Plots.jl/issues/140
plot!([train_mse, test_mse], label="")

# Plot weights for different values of λ.
# https://discourse.julialang.org/t/how-to-convert-vector-of-vectors-to-matrix/72609/4
weights = reduce(hcat, weights)'
p2 = scatter(weights,
        label=[ "lcavol" "lweight" "age" "lbph" "svi" "lcp" "gleason" "pgg45"],
        xlabel="Coefficient",
        ylabel="Weight",
        title="Weights for different values of λ",
        legend=:topright)

# Display plots side by side.
# https://stackoverflow.com/questions/49168872/increase-space-between-subplots-in-plots-jl
l = @layout [a b] 
plot(p1, p2, layout=l, size=(1000, 500), bottom_margin=3*Plots.mm)




