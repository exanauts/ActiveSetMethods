using Test

@testset "Unit tests" begin
    include("unittests.jl")
end

@testset "test.jl" begin
    include("../examples/test.jl")
end