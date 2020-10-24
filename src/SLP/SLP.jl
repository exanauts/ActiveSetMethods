using Printf

"""
    AbstractSlpOptimizer

Abstract type of SLP solvers
"""
abstract type AbstractSlpOptimizer end

"""
    slp_optimize!
    
Empty function to run SLP algorithm
"""
function slp_optimize! end

include("line_search.jl")
include("lp_subproblem.jl")