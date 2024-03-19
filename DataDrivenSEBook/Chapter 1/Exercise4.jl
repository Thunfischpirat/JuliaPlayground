using LinearAlgebra, Statistics, Plots, Random, StatsPlots

rng = MersenneTwister(1234);

p = plot(title="Singular Values of Random Matrices",
         xlabel="Index",
         ylabel="Singular Value")

singular_values::Vector{Float64} = []
for n in [50, 200, 500, 1000]
    for i=1:100
        A::Matrix{Float64} = randn(rng, Float64, (n, n))
        U, E, V = svd(A)
        append!(singular_values, E)
    end
    boxplot!(singular_values, bins=50, label="n = $n", legend=:topleft)
end


