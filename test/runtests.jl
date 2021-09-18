using ActiveSetMethods
using JuMP, MathOptInterface
using GLPK, Ipopt
using Test

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end

@testset "External Solver Attributes Implementation with Toy Example" begin
    include("ext_solver.jl")
    @test isapprox(xsol, -1.0, rtol=1e-4)
    @test isapprox(ysol, -1.0, rtol=1e-4)
    @test status == MOI.LOCALLY_SOLVED
end

@testset "opf.jl" begin
    include("opf.jl")
    result = run_slp_opf("../examples/acopf/case3.m", 100, "SLP-LS")
    @test isapprox(result["objective"], +5.90687949e+03, rtol=1e-3)
    result = run_slp_opf("../examples/acopf/case3.m", 100, "SLP-TR")
    @test isapprox(result["objective"], +5.90687949e+03, rtol=1e-2)
    result = run_sqp_opf("../examples/acopf/case3.m", 100)
    @test isapprox(result["objective"], +5.90687949e+03, rtol=1e-2)
end 
