using Flux
using ManifoldLearning
using CUDA
using MLUtils
using ProgressMeter
using PlotlyJS
using Zygote
using BSON: @save, @load
using Statistics
using MultivariateStats
using Random
using LinearAlgebra


# Set the seed for reproducibility
Random.seed!(1234)

dataset = ManifoldLearning.swiss_roll(20000, 0, segments=10)

# Get training data.
shuffled_indices = shuffle(1:20000)
train_indices = shuffled_indices[1:15000]
features = dataset[1][:, train_indices]
labels = dataset[2][train_indices]

# Normalize features to be between 0 and 1.
mean_features = mean(features, dims=2)
std_features = std(features, dims=2)
features = (features .- mean_features) ./ std_features

# Validation_data
val_indices = shuffled_indices[15001:20000]
val_features = dataset[1][:, val_indices] 
val_features = (val_features .- mean_features) ./ std_features |> gpu
val_labels = dataset[2][val_indices]

# Plot the swiss roll data in 3d
plot(scatter(x=features[1,:], y=features[2,:], z=features[3,:],
            mode="markers",
            marker=attr(
                size=12,
                color=labels,                # set color to an array/list of desired values
                colorscale="Viridis",   # choose a colorscale
                opacity=0.8
            ),
            type="scatter3d"))

# Pass to dataloader for training.
loader = Flux.DataLoader(features |> gpu, batchsize=16)

train_model = false

if train_model
    # Encoder
    encoder = Chain(
        Dense(3, 12, relu),
        Dense(12, 24, relu),
        Dense(24, 12, relu),
        Dense(12, 2)
    )

    # Decoder
    decoder = Chain(
        Dense(2, 12, relu),
        Dense(12, 24, relu),
        Dense(24, 12, relu),
        Dense(12, 3)
    )

    # Autoencoder
    autoencoder = Chain(
        encoder,
        decoder
    ) |> gpu


    # Training
    best_loss = Inf
    patience = 10
    optim = Flux.setup(Flux.Adam(0.001), autoencoder)
    for epoch in 1:100
        losses = []
        for x in loader
            loss, grads = Flux.withgradient(autoencoder) do model
                # Evaluate model and loss inside gradient context:
                Flux.mse(model(x), x)
            end
            Flux.update!(optim, autoencoder, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
        val_loss = Flux.mse(autoencoder(val_features), val_features)
        # Early stopping
        if val_loss < best_loss
            global best_loss = val_loss
            local model = autoencoder |> cpu
            @save "DifferentiableProgramming/model_ae.bson" model
            @info "Model with best validation loss saved."
            global patience = 10
        else
            global patience -= 1
            if patience == 0
                break
            end
        end
        @info "Epoch: $epoch, Loss Training: $(mean(losses)), Loss Validation: $(val_loss)"
    end
end

@load "DifferentiableProgramming/model_ae.bson" model
autoencoder = model


# Plot the latent space representation of the swiss roll data.
latent_space = autoencoder[1](features)
p1 = plot(scatter(x=latent_space[1,:], y=latent_space[2,:],
            name="Autoencoder",
            mode="markers",
            marker=attr(
                size=6,
                color=labels,           # set color to an array/list of desired values
                colorscale="Viridis",   # choose a colorscale
                opacity=0.8
            )),
            Layout(title="Autoencoder"))

# PCA model for comparison.
pca_model = fit(PCA, features, maxoutdim=2)
pca_latent_space = predict(pca_model, features)
p2 = plot(scatter(x=pca_latent_space[1,:], y=pca_latent_space[2,:],	
            name="PCA",
            mode="markers",
            marker=attr(
                size=6,
                color=labels,           # set color to an array/list of desired values
                colorscale="Viridis",   # choose a colorscale
                opacity=0.8
            )),
            Layout(title="PCA"))

p = [p1; p2]
relayout!(p, height=600, width=600, title_text="Swiss Roll Latent Space Representation")
p
