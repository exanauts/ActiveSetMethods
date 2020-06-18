module ProxSDP

    using MathOptInterface
    using TimerOutputs
    using Arpack
    using Printf
    using SparseArrays
    using LinearAlgebra

    import Random
    import LinearAlgebra: BlasInt

    include("structs.jl")
    include("util.jl")
    include("printing.jl")
    include("scaling.jl")
    include("equilibration.jl")
    include("pdhg.jl")
    include("residuals.jl")
    include("eigsolver.jl")
    include("prox_operators.jl")

    include("MOI_wrapper.jl")
    println("##########-------> After all includes and before MOIU");
    MOIU.@model _ProxSDPModelData () (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan) (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.PositiveSemidefiniteConeTriangle) () (MOI.SingleVariable,) (MOI.ScalarAffineFunction,) (MOI.VectorOfVariables,) (MOI.VectorAffineFunction,)
    println("##########-------> After MOIU and before Solver");
    Solver(;args...) = MOIU.CachingOptimizer(_ProxSDPModelData{Float64}(), ProxSDP.Optimizer(args))
    println("##########-------> After Solver");
end
