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

"""
    norm_violations

Compute the normalized constraint violation
"""
function norm_violations(
    E::Tv, g_L::Tv, g_U::Tv, x::Tv, x_L::Tv, x_U::Tv, p = 1
) where {T, Tv <: AbstractArray{T}}

    m = length(E)
    n = length(x)
    viol = Tv(undef, m+n)
    fill!(viol, 0.0)
    for i = 1:m
        if E[i] > g_U[i]
            viol[i] = E[i] - g_U[i]
        elseif E[i] < g_L[i]
            viol[i] = g_L[i] - E[i]
        end
    end
    for j = 1:n
        if x[j] > x_U[j]
            viol[m+j] = x[j] - x_U[j]
        elseif x[j] < x_L[j]
            viol[m+j] = x_L[j] - x[j]
        end
    end
    return norm(viol, p)
end

include("line_search.jl")
include("lp_subproblem.jl")
