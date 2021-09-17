abstract type AbstractSubOptimizer end

struct QpData{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}}
    sense::MOI.OptimizationSense
    Q::Union{Nothing,Tm}
    c::Tv
    c0::T # objective functiton constant term
    A::Tm
    b::Tv
    c_lb::Tv
    c_ub::Tv
    v_lb::Tv
    v_ub::Tv
end

SubModel = Union{
    MOI.AbstractOptimizer,
    JuMP.AbstractModel,
}

include("subproblem_MOI.jl")
include("subproblem_JuMP.jl")
