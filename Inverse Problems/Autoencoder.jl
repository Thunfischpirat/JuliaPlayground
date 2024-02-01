using Flux
using MLDatasets
using CUDA
using PlotlyJS
using Zygote
using Statistics
using MultivariateStats
using Random

# Set the seed for reproducibility
Random.seed!(1234)

X_train, _ = MLDatasets.MNIST(:train)[:]
# Normalize data.
mean_train = mean(X_train)
std_train = std(X_train)
X_train = (X_train .- mean_train) ./ std_train
X_train = reshape(X_train, 28, 28, 1, :)


loader = Flux.DataLoader(X_train |> gpu, batchsize=16)

encoder = Chain(
        Conv((3,3), 1=>4, relu, pad=1, stride=2),
        Conv((3,3), 4=>8, relu, pad=1, stride=2),
        Conv((3,3), 8=>16, relu, pad=1, stride=2),
    )

# Decoder
decoder = Chain(
        ConvTranspose((3,3), 16=>8, relu, pad=1, stride=2),
        ConvTranspose((3,3), 8=>4, relu, pad=1, stride=2),
        ConvTranspose((3,3), 4=>1, relu, pad=1, stride=2),
        ConvTranspose((4,4), 1=>1, sigmoid),
    )

# Autoencoder
autoencoder = Chain(
        encoder,
        decoder
    ) |> gpu

# Training
noise_factor=0.5
optim = Flux.setup(Adam(0.001), autoencoder)
for epoch in 1:100
    losses = []
    for x in loader
        loss, grads = Flux.withgradient(autoencoder) do model
            # Evaluate model and loss inside gradient context:
            noisy_x = x .+ noise_factor .* randn(eltype(x), size(x))
            Flux.mse(model(noisy_x), x)
        end
        update!(optim, autoencoder, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
    @info "Epoch: $epoch, Loss Training: $(mean(losses))"
end
