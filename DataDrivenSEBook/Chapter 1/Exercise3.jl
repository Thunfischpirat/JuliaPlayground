using MAT, LinearAlgebra, Images, Statistics, Plots

file = matread("DataDrivenSEBook/Data/allFaces.mat")
faces::Matrix{Float64} = file["faces"]
avg_face::Matrix{Float64} = mean(faces, dims=2)

faces = faces .- avg_face

U, E , V = svd(faces)

# Use method of snapshots
F = eigen(faces' * faces)

indices::Vector{Int64} = sortperm(F.values, rev=true)

eigenvalues::Vector{Float64} = sqrt.(F.values[indices])
eigenvectors::Matrix{Float64} = F.vectors[:, indices]

U_approx = faces * eigenvectors * inv(diagm(eigenvalues))

U ≈ U_approx

# p = plot(eigenvalues,
#         label="Snapshots",
#         xlabel="Index",
#         ylabel="Singular Value",
#         title="Singular Values of Faces",
#         yscale=:log10)

# plot!(p, E, label="SVD", yscale=:log10)

# display(p)

# p = plot(reshape(eigenvectors[:, 1], 192, 168),
#         label="Snapshot",
#         xlabel="Index",
#         ylabel="Eigenvector",
#         title="First Eigenvector of Faces")