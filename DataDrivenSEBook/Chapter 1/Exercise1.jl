# Exercise 1.1
using Images, FileIO, LinearAlgebra, Plots

img = load("DataDrivenSEBook/Data/dog.jpg")

U, E, V = svd(Gray.(img), full=true)

n, m = size(img)
r = 100
I_r = Matrix{Float64}(I, r, r)

U_r = U[:, 1:r]
(U_r' * U_r ≈ I_r) && println("U_r' * U_r ≈ I_r")
!(U * U' ≈ I) && println("U' * U ≠ I")

Errors = []
I_n = Matrix{Float64}(I, n, n)
for r in 1:n
    err = norm(U[:,1:r] * U[:,1:r]' - I_n)
    push!(Errors, err)
end

p = plot(Errors,
         title="UU* - I error",
         xlabel="Rank",
         ylabel="Error",
         legend=false)