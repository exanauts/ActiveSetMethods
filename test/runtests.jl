using ActiveSetMethods
using GLPK
using Test

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end

# @testset "Unit tests" begin
#     include("unittests.jl")
# end

@testset "toy_example.jl" begin
    include("../examples/toy_example.jl")
    @test isapprox(xsol, -1.0, rtol=1e-4)
    @test isapprox(ysol, -1.0, rtol=1e-4)
end

# @testset "opf.jl" begin
#     include("../examples/opf.jl")
#     run_opf("../examples/cases/case3.m")
# end