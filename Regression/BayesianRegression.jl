
using CSV
using DataFrames
using Parameters
using LinearAlgebra
using Statistics
using Distributions
using Random
using LaTeXStrings
using Plots

## Prepare data.

# Set seed for reproducibility.
Random.seed!(32)

# Load data (https://hastie.su.domains/ElemStatLearn/datasets/ozone.data)
df = CSV.read("Data/ESL/ozone.txt", delim="\t", DataFrame)
# Get rows where train is true.
train_size = ceil(Int, 0.8 * size(df, 1))
train_idx = rand(1:size(df,1), train_size)
df_train = df[train_idx, :]


# Get "ozone" as explanatory variable.
X = reshape(df_train[:, 1], train_size, 1)
# Normalize explanatory variable.
mean_X = mean(X, dims=1)
std_X = std(X, dims=1)
X = (X .- mean_X) ./ std_X
# Add bias term.
X = [ones(size(X, 1)) X]
# Get "temperature" as response variable.
y = df_train[:, end-1]
mean_y = mean(y)
std_y = std(y)
y = (y .- mean_y) ./ std_y

## Fit model.

# Prior distribution of w.
prior_mean = [0, 0]
prior_cov = [1 0; 0 1]
prior = MvNormal(prior_mean, prior_cov)
# Posterior distribution of w. (See https://probml.github.io/pml-book/book1.html p. 401.)
Σ_hat = inv(inv(cov(prior)) + 1/std_y*X'*X)
w_hat = Σ_hat * (inv(cov(prior))*prior_mean + 1/std_y*X'*y)
posterior = MvNormal(w_hat, Σ_hat)

## Make contour plot of prior and posterior.

# Create grid.
x = range(-2, 2, length=100)
y = range(-2, 2, length=100)
Z_prior = [pdf(prior, [x, y]) for x in x, y in y]
Z_posterior = [pdf(posterior, [x, y]) for x in x, y in y]

# Create plots.
p1 = plot(x, y, Z_prior, st=:heatmap, color=:turbo,
    title=L"p(w)", 
    xlabel=L"w_1",
    ylabel=L"w_2")
p2 = plot(x, y, Z_posterior, st=:heatmap, color=:turbo,
    title=L"p(w|X, y)", 
    xlabel=L"w_1",
    ylabel=L"w_2")
# Display plots side by side.
l = @layout [a b] 
plot(p1, p2, layout=l, size=(1000, 500), bottom_margin=3*Plots.mm, left_margin=4*Plots.mm)

