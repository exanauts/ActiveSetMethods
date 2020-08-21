using ActiveSetMethods
using Test

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end

# @testset "Unit tests" begin
#     include("unittests.jl")
# end

# @testset "test.jl" begin
#     include("../examples/test.jl")
# end

# @testset "opf.jl" begin
#     include("../examples/opf.jl")
#     Options_["max_iter"] = 10
#     run_opf("../examples/cases/case3.m")
# end