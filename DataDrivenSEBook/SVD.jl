### Exercises from chapter 1 of the book "Data driven science and engineering" by Steven L. Brunton and J. Nathan Kutz

### Playing with the Singular Value Decomposition (SVD) of a matrix
using Images, FileIO, LinearAlgebra, Plots


# Create a random matrix
A =  rand([0, 1, 2, 3, 4, 5], 80, 50)

# Compute the SVD
U, E, V = svd(A)
Σ = Diagonal(E)

# Check that the factorization is correct
A ≈ U * Σ * V'

# Compare the original matrix with its SVD based low-rank approximations
errors = []
for i::Int in 1:length(E)
    A_approx = U[:, 1:i] * Σ[1:i, 1:i] * V[:, 1:i]'
    push!(errors, norm(A - A_approx))
end

p = plot(errors,
         title="Error of low-rank approximations",
         xlabel="Rank",
         ylabel="Error",
         legend=false)


### Exercises

# Exercise 1.1
img = load("LinearAlgebra/Data/dog.jpg")

U, E, V = svd(Gray.(img))

n, m = size(img)
r = 100
I_r = Matrix{Float64}(I, r, r)

U_r = U[:, 1:r]
(U_r' * U_r ≈ I_r) && println("U_r' * U_r ≈ I_r")
!(U * U' ≈ I) && println("U' * U ≠ I")

Errors = []
I_n = Matrix{Float64}(I, n, n)
for r in 1:m
    err = norm(U[:,1:r] * U[:,1:r]' - I_n)
    push!(Errors, err)
end

p = plot(Errors,
         title="UU* - I error",
         xlabel="Rank",
         ylabel="Error",
         legend=false)

# Exercise 1.2
img = load("LinearAlgebra/Data/dog.jpg")
