module ActiveSetMethods

using LinearAlgebra, SparseArrays
using GLPK # TODO: This needs to be removed.

include("lp_opt.jl"); # containts the LP subproblem solution algorithm
include("Options.jl"); # contains the Options_ such method, algorithm, parameters, LP solver, etc.
include("struct.jl"); # constains the NoptProblem struct definition
include("SLP_line_search.jl"); # contains the SLP line search algorithm


export createNoptProblem, Options_
export solveNoptProblem
export NoptProblem


#TODO change the name convention of these variables to make it consistant and intuitive
export fx, normEx, Phix, mux, alphax, mu_numerator, mu_RHS, Dx, px, alphapx
export lamx, errx, normDfx, normLamx, normdCx
export plot_error_components

fx = Array{Float64,1}()
normEx = Array{Float64,1}()
Phix = Array{Float64,1}()
mux = Array{Float64,1}()
alphax = Array{Float64,1}()
mu_numerator = Array{Float64,1}()
mu_RHS = Array{Float64,1}()
Dx = Array{Float64,1}()
px = Array{Float64,1}()
alphapx = Array{Float64,1}()
lamx= Array{Float64,1}()
errx = Array{Float64,1}()
normDfx = Array{Float64,1}()
normLamx = Array{Float64,1}()
normdCx = Array{Float64,1}()


# Creates the ActiveSetMethods Problem
#TODO change the x_L and others to new symbols
function createNoptProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64}, j_sparsity::Array{Tuple{Int64,Int64}},
    h_sparsity::Array{Tuple{Int64,Int64}}, eval_f, eval_g, eval_grad_f, eval_jac_g,
    eval_norm_E = nothing, eval_merit = nothing, eval_D = nothing, eval_h = nothing)

    return NoptProblem(n, x_L, x_U, m, g_L, g_U, j_sparsity, h_sparsity,
                        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_norm_E,
                        eval_merit, eval_D, eval_h);
end

# Solves the ActiveSetMethods Problem
#TODO make sure all the model internal structure are defined such as status and etc.
function solveNoptProblem(model::NoptProblem)
    ret = 5;

    if (Options_["method"] == "SLP" && Options_["algorithm"] == "Line Search")
        ret = SLP_line_search(model);
    else
        println("ERROR: The method is not defined")
    end

    println("ret: ", ret)
    model.status = Int(ret)
    return Int(ret)
end




include("MOI_wrapper.jl"); # it contains the MOI wrapper functions
# include("analysis.jl")
#include("MPB_wrapper.jl")

end
