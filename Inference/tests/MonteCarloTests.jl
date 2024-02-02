using Test
include("../MonteCarlo.jl")

@testset "Van der Corput sequence" begin
    @testset "Base 2" begin
        @test corput(1, 2) == 0.5
        @test corput(2, 2) == 0.25
        @test corput(3, 2) == 0.75
    end
    @testset "Base 10" begin
        @test corput(1, 10) == 0.1
        @test corput(2, 10) == 0.2
        @test corput(4, 10) == 0.4
    end
end