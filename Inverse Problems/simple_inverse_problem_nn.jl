using Distributions
using Zygote
using Statistics
using LinearAlgebra
using Plots

# Set ε and σ and other hyper-parameters.
ε = 0.1
σ = 0.1
num_train_samples = 2^13
num_test_samples = 2^12
batch_size = 2^5

# Sample data from a 2d Gaussian
μ = [0.0, 0.0]
Σ = [1.0 0.; 0. 1.0]
X = rand(MvNormal(μ, Σ), num_train_samples + num_test_samples)

# Sample errors from a 2d Gaussian
μ = [0.0, 0.0]
Σ = [σ 0.; 0. σ]
η = rand(MvNormal(μ, Σ), num_train_samples + num_test_samples)

A = [1 1; 1 1+ε]

# Generate the data
Y = A*X + η

# Create training and test sets
X_train = X[:, 1:num_train_samples]
Y_train = Y[:, 1:num_train_samples]

X_test = X[:, num_train_samples+1:end]
Y_test = Y[:, num_train_samples+1:end]

# Initialize weight Matrix
W_forward = rand(2, 2)
W_backward = rand(2, 2)

# Forward loss
loss_forward(W,X,Y) = mean((Y - W*X).^2)

# Backward loss
loss_backward(W,Y,X) = mean((X - W*Y).^2)

forward_losses = []
backward_losses = []

for epoch in 1:Int(num_train_samples/batch_size)-1
    local X = X_train[:,epoch*batch_size: (epoch + 1)*batch_size]
    local Y = Y_train[:,epoch*batch_size: (epoch + 1)*batch_size]
    # Forward pass
    global W_forward = W_forward .- 0.01 .* gradient(W -> loss_forward(W,X,Y), W_forward)[1]

    # Backward pass
    global W_backward = W_backward .- 0.01 .* gradient(W -> loss_backward(W,Y,X), W_backward)[1]

    push!(forward_losses, loss_forward(W_forward, X_test, Y_test))
    push!(backward_losses, loss_backward(W_backward, Y_test, X_test))
end

# Tikhonov Solution
eye = [1 0; 0 1]
W_tikhonov = inv(A'*A + σ^2 .* eye) * A'
tiknonov_loss = loss_backward(W_tikhonov, Y_test, X_test)

# Plot the losses
plot(forward_losses, label="Forward")
plot!(backward_losses, label="Backward")
hline!([tiknonov_loss], label="Tikhonov")
plot!(title="Losses", xlabel="Epoch", ylabel="Loss")

