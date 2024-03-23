using MAT, Images, Plots, LinearAlgebra, Measures


file = matread("DataDrivenSEBook/Data/CYLINDER_ALL.mat")

vortall::Matrix{Float64} = file["VORTALL"]
nx::Int64 = file["nx"]
ny::Int64 = file["ny"]

# Perform SVD on the vortall matrix
U, E, V = svd(vortall)
Σ = Diagonal(E)

# a) Plot the singular value spectrum and eigenflows
p1 = plot(E,
          yscale=:log10,
          legend=false,
          xlabel="Index",
          ylabel="Singular Value",
          title="Singular Values of Vortall")

p2 = plot(layout = (3, 3), size = (900, 900))

for i in 1:9
    eigenflow = reshape(U[:, i], nx, ny)
    heatmap!(p2[i], 
             eigenflow,
             title="Eigenflow $i",
             color=:grays,
             colorbar=false,
             xticks=false,
             yticks=false)
end

# b) Plot reconstructed movies for various truncation values of r

total_energy = sum(E)
cumulative_energies = cumsum(E) / total_energy
thresholds = [0.9, 0.99, 0.999]
r_vals = []
k = 1
for i in eachindex(cumulative_energies)
    if cumulative_energies[i] > thresholds[k]
        push!(r_vals, i)
        global k += 1
        if k == 4
            break
        end
    end
end


rec_movie_1::Matrix{Float64} = U[:, 1:r_vals[1]] * Σ[1:r_vals[1], 1:r_vals[1]] * V[:, 1:r_vals[1]]'
rec_movie_2::Matrix{Float64} = U[:, 1:r_vals[2]] * Σ[1:r_vals[2], 1:r_vals[2]] * V[:, 1:r_vals[2]]'
rec_movie_3::Matrix{Float64} = U[:, 1:r_vals[3]] * Σ[1:r_vals[3], 1:r_vals[3]] * V[:, 1:r_vals[3]]'

err1 = round(norm(vortall - rec_movie_1), digits=2)
err2 = round(norm(vortall - rec_movie_2), digits=2)
err3 = round(norm(vortall - rec_movie_3), digits=2)

anim = @animate for i in eachindex(vortall[1, :])
    p1 = plot(Gray.(reshape(vortall[:, i], nx, ny)), title="Original")
    p2 = plot(Gray.(reshape(rec_movie_3[:, i], nx, ny)), title="r = $(r_vals[3]) \n error = $err3")
    p3 = plot(Gray.(reshape(rec_movie_2[:, i], nx, ny)), title="r = $(r_vals[2]) \n error = $err2")
    p4 = plot(Gray.(reshape(rec_movie_1[:, i], nx, ny)), title="r = $(r_vals[1]) \n error = $err1")
    plot(p1, p2, p3, p4,
        layout=(1, 4),
        size=(1200, 300),
        grid=false,
        axis=false,
        margin=1mm) 
end

# gif(anim, "animation.gif", fps = 10)

# c)

U_truncated::Matrix{Float64} = U[:, 1:10]
Σ_truncated::Matrix{Float64} = Σ[1:10, 1:10]
V_truncated::Matrix{Float64} = V[:, 1:10]

W::Matrix{Float64} = Σ_truncated * V_truncated'

X_rec = reshape((U_truncated * W)[:, 1], nx, ny)
X = reshape(vortall[:, 1], nx, ny)
Gray.(hcat(X, X_rec))

# d)

W_next::Matrix{Float64} = W[:,2:end]
W_prev::Matrix{Float64} = W[:,1:end-1]
U_w, E_w, V_w = svd(W_prev)

MP_inv::Matrix{Float64} = V_w * inv(Diagonal(E_w)) * U_w'
A::Matrix{Float64} = W_next * MP_inv

eigenvalues_A::Vector{ComplexF64} = eigvals(A)

plot(real(eigenvalues_A),
     imag(eigenvalues_A),
     seriestype=:scatter,
     legend=false,
     xlabel="Real",
     ylabel="Imaginary",
     title="Eigenvalues of A")

# e) Reconstruct flow field based on initial condition
w::Vector{Float64} = W[:,1] # w1
anim = @animate for i in 1:151
    X_rec = reshape((U_truncated * w), nx, ny)
    X_true = reshape(vortall[:, i], nx, ny)
    global w = A * w
    plot(Gray.(hcat(X_rec, X_true)),
         title="Reconstructed Flow Field \n iteration: $i",
         grid=false,
         axis=false)
end

gif(anim, "animation2.gif", fps = 10)
