### Various experiments to illustrate and verify the benefits of the Monte Carlo method.

using Plots
using Distributions
show_plots = false

experiment = ["Quadrature", "Uniform Monte Carlo", "Halton Monte Carlo"]

function gaussian(x::Float64, y::Float64, z::Float64)::Float64
    return exp(-(x^2 + y^2 + z^2) / 2) * (2π)^(-3/2)
end

function mse(pred::Float64, target::Float64)::Float64
    return (pred - target)^2
end

if show_plots
    grid = -2:0.01:2
    plot(grid, gaussian.(grid), label="Gaussian function", xlabel="x", ylabel="exp(-x^2)", line=:dash)
end

# Estimate the integral via quadrature.
function quadrature(f::Function, a::Integer, b::Integer, n::Integer)::Float64
    x = y = z = range(a, b, length=n)
    dx = dy = dz = (b - a) / n
    integral = 0.0
    for x in x, y in y, z in z
        integral += f(x, y, z)
    end
    return integral * dx * dy * dz
end

# Estimate the integral via uniform Monte Carlo.
function monte_carlo_uniform(f::Function, a::Int, b::Int, n::Int)::Float64
    x = rand(Uniform(a, b), n)
    y = rand(Uniform(a, b), n)
    z = rand(Uniform(a, b), n)
    return mean(f.(x, y, z)) * (b - a)^3
end

# Estimate the integral via halton sequence based Monte Carlo.
function corput(n::Int, base::Int)::Float64
    # See: https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler (7.4.1)
    q = 0
    inv_base = 1 / base
    while (n > 0)
        q += (n % base) * inv_base
        n = div(n, base)
        inv_base *= inv_base
    end
    return q 
end

function monte_carlo_halton(f::Function, a::Int, b::Int, n::Int)::Float64
    nums = 1:n
    x = corput.(nums, 2) * (b - a) .+ a
    y = corput.(nums, 3) * (b - a) .+ a
    z = corput.(nums, 5) * (b - a) .+ a
    return mean(f.(x, y, z)) * (b - a)^3
end

if "Quadrature" in experiment
    errors_quadrature = []
    for n in [9, 27, 81]
        @time begin
        result::Float64 = quadrature(gaussian, -4, 4, n);
        error::Float64 = mse(result, 1.)
        push!(errors_quadrature, error)
        println("Quadrature estimate (n=$n):  $result  mse: $error")
        end
    end
    println()
end

# We want to use the same number of samples for the Monte Carlo methods
# as for the quadrature method. Thus n_i = n_i_quadrature^3 / 3
if "Uniform Monte Carlo" in experiment
    errors_uniform = []
    for n in [243, 6561, 177147]
        @time begin
            result::Float64 = monte_carlo_uniform(gaussian, -4, 4, n);
            error::Float64 = mse(result, 1.)
            push!(errors_uniform, error)
            println("Quadrature estimate (n=$n):  $result  mse: $error")
        end
    end
    println()
end

if "Halton Monte Carlo" in experiment
    errors_halton = []
    for n in [243, 6561, 177147]
        @time begin
            result::Float64 = monte_carlo_halton(gaussian, -4, 4, n);
            error::Float64 = mse(result, 1.)
            push!(errors_halton, error)
            println("Quadrature estimate (n=$n):  $result  mse: $error")
        end
    end
    println()
end

function plot_errors(errors_quadrature, errors_uniform, errors_halton)
    sample_sizes = [243, 6561, 177147]
    plot(sample_sizes, errors_quadrature, label="Quadrature", marker=:circle)
    plot!(sample_sizes, errors_uniform, label="Uniform Monte Carlo", marker=:square)
    plot!(sample_sizes, errors_halton, label="Halton Monte Carlo", marker=:diamond)
    plot!(legend=:topleft)
    xlabel!("Sample Size")
    ylabel!("MSE")
    title!("Comparison of MSE for Different Methods")
end

# Plot all error arrays
plot_errors(errors_quadrature, errors_uniform, errors_halton)


