# Exercise 1.2

using Images, FileIO, LinearAlgebra, Plots

img = Gray.(load("DataDrivenSEBook/Data/dog.jpg"))

U, E, V = svd(img, full=false)
Σ = Diagonal(E)

ReconstrutedNorm = []
ReconstructedVariance = []
SingularValues = []
for r in 1:length(E)
    img_approx = U[:, 1:r] * Σ[1:r, 1:r] * V[:, 1:r]'
    err = norm(img - img_approx) / norm(img)
    m_var = err^2
    push!(ReconstrutedNorm, 1 - err)
    push!(ReconstructedVariance, 1 - m_var)
    
end

frob_99 = findfirst(x -> x >= 0.99, ReconstrutedNorm)
var_99 = findfirst(x -> x >= 0.99, ReconstructedVariance)

CumSingularValues = cumsum(E) / sum(E)
sv_99 = findfirst(x -> x >= 0.99, CumSingularValues)

println("Rank for 99% Frobenius Norm: ", frob_99)
println("Rank for 99% Missing Variance: ", var_99)
println("Rank for 99% Cumulative Singular Values: ", sv_99)

p = plot(ReconstrutedNorm,
         title="Reconstruction Quality",
         xlabel="Rank",
         ylabel="Error",
         label=" Relative Frobenius Norm",
         legend=:bottomright)

plot!(p, ReconstructedVariance, label="Missing Variance")
plot!(p, CumSingularValues, label="Cumulative Singular Values")