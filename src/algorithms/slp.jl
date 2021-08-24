"""
    AbstractSlpOptimizer

Abstract type of SLP solvers
"""
abstract type AbstractSlpOptimizer <: AbstractOptimizer end

function LpData(slp::AbstractSlpOptimizer)
	A = compute_jacobian_matrix(slp)
	return QpData(
        MOI.MIN_SENSE,
        nothing,
		slp.df,
		slp.f,
		A,
		slp.E,
		slp.problem.g_L,
		slp.problem.g_U,
		slp.problem.x_L,
		slp.problem.x_U)
end

"""
    KT_residuals

Compute Kuhn-Turck residuals
"""
KT_residuals(slp::AbstractSlpOptimizer) = KT_residuals(slp.df, slp.lambda, slp.mult_x_U, slp.mult_x_L, compute_jacobian_matrix(slp))

"""
    norm_complementarity

Compute the normalized complementeraity
"""
norm_complementarity(slp::AbstractSlpOptimizer, p = Inf) = norm_complementarity(
    slp.E, slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    slp.lambda, slp.mult_x_U, slp.mult_x_L, 
    p
)

"""
    norm_violations

Compute the normalized constraint violation
"""

norm_violations(slp::AbstractSlpOptimizer, p = 1) = norm_violations(
    slp.E, slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

norm_violations(slp::AbstractSlpOptimizer, x::Tv, p = 1) where {T, Tv<:AbstractArray{T}} = norm_violations(
    slp.problem.eval_g(x, zeros(slp.problem.m)), slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

function eval_functions!(slp::AbstractSlpOptimizer)
    slp.f = slp.problem.eval_f(slp.x)
    slp.problem.eval_grad_f(slp.x, slp.df)
    slp.problem.eval_g(slp.x, slp.E)
    slp.problem.eval_jac_g(slp.x, :eval, [], [], slp.dE)
end

compute_jacobian_matrix(slp::AbstractSlpOptimizer) = compute_jacobian_matrix(slp.problem.m, slp.problem.n, slp.problem.j_str, slp.dE)

include("slp_line_search.jl")
include("slp_trust_region.jl")
