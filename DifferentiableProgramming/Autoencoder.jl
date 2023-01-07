using Flux
using MLDatasets
using MLUtils
using CUDA
using ProgressMeter
using Plots
using BSON: @save, @load
using Statistics
using Random
using Images

# Set the seed for reproducibility
Random.seed!(1234)

dataset = MNIST(:train)

features = Flux.flatten(dataset.features)
# Normalize the data to be between 0 and 1
features = normalise(features; dims=ndims(features), ϵ=1e-5)
labels = dataset.targets
loader = DataLoader(features |> gpu, batchsize=8)


load_model = false

if load_model

    @load "DifferentiableProgramming/autoencoder.bson" autoencoder

else
    # Encoder
    encoder = Chain(
        Dense(28^2, 128, relu),
        Dense(128, 24, relu),
        Dense(24, 2)
    )

    # Decoder
    decoder = Chain(
        Dense(2, 24, relu),
        Dense(24, 128, relu),
        Dense(128, 28^2, sigmoid)
    )

    # Autoencoder
    autoencoder = Chain(
        encoder,
        decoder
    ) |> gpu
    

    # Training
    epoch_losses = []
    optim = Flux.setup(Flux.Adam(0.1), autoencoder)
    @showprogress for epoch in 1:20
        losses = []
        for x in loader
            loss, grads = Flux.withgradient(autoencoder) do model
                # Evaluate model and loss inside gradient context:
                Flux.mse(model(x), x)
            end
            Flux.update!(optim, autoencoder, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
        push!(epoch_losses, mean(losses))
    end

    # Save model.
    @save "DifferentiableProgramming/autoencoder.bson" autoencoder

end

# Plot the 2d latent space of the autoencoder on the MNIST dataset
encoded_data = Flux.flatten(autoencoder[1](features |> gpu)) |> cpu

# Plot the data and color it by the digit
scatter(encoded_data[1, :], encoded_data[2, :], color=labels, markersize=0.5, legend=false)

# Plot the loss
plot(epoch_losses, xlabel="Iteration", ylabel="Loss", title="Loss over time")

