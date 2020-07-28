module Nlopt
using LinearAlgebra, SparseArrays, GLPK, Ipopt
include("lp_opt.jl")
include("Options.jl")
include("struct.jl")
include("SLP_line_search.jl")
include("Ipopt_method.jl")


export createProblem, addOption, Options_
export openOutputFile, setProblemScaling, setIntermediateCallback
export solveProblem
export NloptProblem


function createNloptProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64}, j_sparsity::Array{Tuple{Int64,Int64}},
    h_sparsity::Array{Tuple{Int64,Int64}}, eval_f, eval_g, eval_grad_f, eval_jac_g,
    eval_merit = nothing, eval_D = nothing, eval_h = nothing)

    return NloptProblem(n, x_L, x_U, m, g_L, g_U, j_sparsity, h_sparsity,
                        eval_f, eval_g, eval_grad_f, eval_jac_g,
                        eval_merit, eval_D, eval_h);
end


function solveNloptProblem(model::NloptProblem)
    ret = 5;

    #ret = SLP_line_search(model);
    if (Options_["method"] == "Ipopt")
        ret = Ipopt_method(model);
    elseif (Options_["method"] == "SLP" && Options_["algorithm"] == "Line Search")
        ret = SLP_line_search(model);
    end

    println("ret: ", ret)
    model.status = Int(ret)
    return Int(ret)
end




include("MOI_wrapper.jl")
#include("MPB_wrapper.jl")

end
