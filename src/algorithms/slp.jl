"""
    AbstractSlpOptimizer

Abstract type of SLP solvers
"""
abstract type AbstractSlpOptimizer <: AbstractSqpOptimizer end

"""
    sub_optimize_MOI!

Solve LP subproblems by using either MOI
"""
sub_optimize_MOI!(slp::AbstractSlpOptimizer, Δ::Float64) = sub_optimize!(slp, MOI.instantiate(slp.options.external_optimizer), Δ)

"""
    sub_optimize_JuMP!

Solve LP subproblems by using either JuMP
"""
sub_optimize_JuMP!(slp::AbstractSlpOptimizer, Δ::Float64) = sub_optimize!(slp, JuMP.Model(slp.options.external_optimizer), Δ)

# FIXME: let's use JuMP by default.
sub_optimize!(slp::AbstractSlpOptimizer, Δ::Float64 = 1000.0) = sub_optimize_JuMP!(slp, Δ)

"""
MOI implementation is limited to the solvers that can incrementally add variables and constraints.
Many solvers do not seem to support that. Moreover, some solvers do not support SingleVariable constraints either.
"""
# sub_optimize!(slp::AbstractSlpOptimizer, Δ::Float64 = 1000.0) = sub_optimize_MOI!(slp, Δ)

"""
    compute_phi

Evaluate and return the merit function value for a given point x + α * p.

# Arguments
- `slp`: SlpLS structure
- `x`: the current solution point
- `α`: step size taken from `x`
- `p`: direction taken from `x`
"""
function compute_phi(slp::AbstractSlpOptimizer, x::Tv, α::T, p::Tv) where {T,Tv<:AbstractArray{T}}
    ϕ = 0.0
    xp = x + α * p
    E = ifelse(α == 0.0, slp.E, slp.problem.eval_g(xp, zeros(slp.problem.m)))
    if slp.feasibility_restoration
        p_slack = slp.p_slack
        ϕ = slp.prim_infeas
        for (i, v) in p_slack
            ϕ += α * sum(v)
        end
        for i = 1:slp.problem.m
            viol =
                maximum([0.0, slp.E[i] - slp.problem.g_U[i], slp.problem.g_L[i] - slp.E[i]])
            lhs = E[i] - viol
            if slp.problem.g_L[i] > -Inf && slp.problem.g_U[i] < Inf
                lhs += α * (p_slack[i][1] - p_slack[i][2])
            elseif slp.problem.g_L[i] > -Inf
                lhs += α * p_slack[i][1]
            elseif slp.problem.g_U[i] < Inf
                lhs -= α * p_slack[i][1]
            end
            ϕ +=
                slp.μ[i] *
                maximum([0.0, lhs - slp.problem.g_U[i], slp.problem.g_L[i] - lhs])
        end
    else
        ϕ = slp.problem.eval_f(xp)
        for i = 1:slp.problem.m
            if E[i] > slp.problem.g_U[i]
                ϕ += slp.μ[i] * (E[i] - slp.problem.g_U[i])
            elseif E[i] < slp.problem.g_L[i]
                ϕ += slp.μ[i] * (slp.problem.g_L[i] - E[i])
            end
        end
    end
    return ϕ
end

"""
    compute_derivative

Compute the directional derivative at current solution for a given direction.
"""
function compute_derivative(slp::AbstractSlpOptimizer)
    dfp = 0.0
    cons_viol = zeros(slp.problem.m)
    if slp.feasibility_restoration
        for (_, v) in slp.p_slack
            dfp += sum(v)
        end
        for i = 1:slp.problem.m
            viol = maximum([0.0, slp.E[i] - slp.problem.g_U[i], slp.problem.g_L[i] - slp.E[i]])
            lhs = slp.E[i] - viol
            cons_viol[i] += maximum([0.0, lhs - slp.problem.g_U[i], slp.problem.g_L[i] - lhs])
        end
    else
        dfp += slp.df' * slp.p
        for i = 1:slp.problem.m
            cons_viol[i] += maximum([
                0.0, 
                slp.E[i] - slp.problem.g_U[i],
                slp.problem.g_L[i] - slp.E[i]
            ])
        end
    end
    return compute_derivative(dfp, slp.μ, cons_viol)
end

include("slp_line_search.jl")
include("slp_trust_region.jl")
