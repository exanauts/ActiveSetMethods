module ActiveSetMethods

using LinearAlgebra, SparseArrays

import MathOptInterface

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

include("Parameters.jl")
include("struct.jl")
include("SLP_line_search.jl")
include("SLP_trust_region.jl")
include("lp_opt.jl")


export createASMProblem
export solveASMProblem
export ASMProblem


"Creates the ActiveSetMethods Problem"
createASMProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64}, 
    j_sparsity::Array{Tuple{Int64,Int64}}, h_sparsity::Array{Tuple{Int64,Int64}}, 
    eval_f::Function, eval_g::Function, eval_grad_f::Function, eval_jac_g::Function, eval_h, 
    parameters::Parameters) = ASMProblem(
        n, x_L, x_U, m, g_L, g_U, j_sparsity, h_sparsity,
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, parameters)

createASMProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64}, 
    j_sparsity::Array{Tuple{Int64,Int64}}, h_sparsity::Array{Tuple{Int64,Int64}}, 
    eval_f::Function, eval_g::Function, eval_grad_f::Function, eval_jac_g::Function,
    parameters::Parameters) = createASMProblem(
        n, x_L, x_U, m, g_L, g_U, 
        j_sparsity, h_sparsity, eval_f, eval_g, eval_grad_f, eval_jac_g, nothing, 
        parameters)

"Solves the ActiveSetMethods Problem"
function solveASMProblem(model::ASMProblem)
    if isnothing(model.parameters.external_optimizer)
    	model.status = -12;
        @error "`external_optimizer` parameter must be set for subproblem solutions."
    else
        if model.parameters.method == "SLP"
            env = SLP(model)
            if model.parameters.algorithm == "Line Search"
                line_search_method(env);
            end
            if model.parameters.algorithm == "Trust Region"
                trust_region_method(env);
            end
        else
            @error "The method is not defined"
        end
    end
    return model.status 
end

include("MOI_wrapper.jl")
# include("analysis.jl")
#include("MPB_wrapper.jl")

end
