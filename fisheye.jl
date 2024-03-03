### Transformation of the complex numbers that simulates the fisheye effect from photography
using Plots
using Plots.Measures

# Create sets of points in the complex plane
T = range(-4,4,100)
φ = range(0, 2pi, 100)

M1 = map(t -> t + im/2, T)
M2 = map(t -> -2 + t*im, T)

M3 = map(θ -> 0.5*(cos(θ) + im*sin(θ)), φ)
M4 = map(θ -> 0.5*(cos(θ) + im*sin(θ)) - 1, φ)

# Define the transformation function
f(z) = z / (abs(z) + 1)

# Plot the sets of points and their images under the transformation
p0 = plot()

plot!(p0, real.(M1), imag.(M1), label="M1", color=:blue)
plot!(p0, real.(M2), imag.(M2), label="M2", color=:red)
plot!(p0, real.(M3), imag.(M3), label="M3", color=:green)
plot!(p0, real.(M4), imag.(M4), label="M4", color=:purple)

xlabel!(p0, "Re")
ylabel!(p0, "Im")

p1 = plot()

plot!(p1, real.(f.(M1)), imag.(f.(M1)), label="f(M1)", color=:blue)
plot!(p1, real.(f.(M2)), imag.(f.(M2)), label="f(M2)", color=:red)
plot!(p1, real.(f.(M3)), imag.(f.(M3)), label="f(M3)", color=:green)
plot!(p1, real.(f.(M4)), imag.(f.(M4)), label="f(M4)", color=:purple)

xlabel!(p1, "Re")
ylabel!(p1, "Im")

p = plot(p0, p1, layout=(1,2), size=(800, 400), margin=2mm)
display(p)
