"""
    AbstractSqpOptimizer

Abstract type of SQP solvers
"""
abstract type AbstractSqpOptimizer <: AbstractOptimizer end

macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

@def sqp_fields begin
    problem::Model{T,Tv,Tt} # problem data

    x::Tv # primal solution
    p::Tv # search direction
    p_slack::Dict{Int,Tv} # search direction at feasibility restoration phase
    lambda::Tv # Lagrangian dual multiplier
    mult_x_L::Tv # reduced cost for lower bound
    mult_x_U::Tv # reduced cost for upper bound

    # Evaluations at `x`
    f::T # objective function
    df::Tv # gradient
    E::Tv # constraint
    dE::Tv # Jacobian

    j_row::Vector{Int} # Jacobian matrix row index
    j_col::Vector{Int} # Jacobian matrix column index
    Jacobian::AbstractMatrix{T} # Jacobian matrix

    h_row::Vector{Int} # Hessian matrix row index
    h_col::Vector{Int} # Hessian matrix column index
    h_val::Tv # Hessian matrix values
    Hessian::Union{Nothing,AbstractMatrix{T}} # Hessian matrix

    prim_infeas::T # primal infeasibility at `x`
    dual_infeas::T # dual (approximate?) infeasibility
    compl::T # complementary slackness

    optimizer::Union{Nothing,AbstractSubOptimizer} # Subproblem optimizer

    options::Parameters

    feasibility_restoration::Bool # indicator for feasibility restoration
    iter::Int # iteration counter
    ret::Int # solution status
    start_time::Float64 # solution start time
    start_iter_time::Float64 # iteration start time
end

"""
    QpData

Create QP subproblem data
"""
function QpData(sqp::AbstractSqpOptimizer)
	return QpData(
        MOI.MIN_SENSE,
        sqp.Hessian,
		sqp.df,
		sqp.f,
		sqp.Jacobian,
		sqp.E,
		sqp.problem.g_L,
		sqp.problem.g_U,
		sqp.problem.x_L,
		sqp.problem.x_U)
end

"""
    sub_optimize!

Solve QP subproblem
"""
function sub_optimize!(sqp::AbstractSqpOptimizer, model::SubModel, Δ::Float64 = Inf)
    if isnothing(sqp.optimizer)
        sqp.optimizer = SubOptimizer(
            model,
            QpData(sqp),
        )
        create_model!(sqp.optimizer, sqp.x, Δ)
    else
        sqp.optimizer.data = QpData(sqp)
    end
    return sub_optimize!(
        sqp.optimizer,
        sqp.x,
        Δ,
        sqp.feasibility_restoration
    )
end

"""
    eval_functions!

Evalute the objective, gradient, constraints, and Jacobian.
"""
function eval_functions!(sqp::AbstractSqpOptimizer)
    sqp.f = sqp.problem.eval_f(sqp.x)
    sqp.problem.eval_grad_f(sqp.x, sqp.df)
    sqp.problem.eval_g(sqp.x, sqp.E)
    sqp.problem.eval_jac_g(sqp.x, :eval, [], [], sqp.dE)
    fill!(sqp.Jacobian.nzval, 0.0)
    for (i, v) in enumerate(sqp.dE)
        sqp.Jacobian[sqp.j_row[i],sqp.j_col[i]] += v
    end
    if !isnothing(sqp.problem.eval_h)
        sqp.problem.eval_h(sqp.x, :eval, [], [], 1.0, sqp.lambda, sqp.h_val)
        fill!(sqp.Hessian.nzval, 0.0)
        for (i, v) in enumerate(sqp.h_val)
            sqp.Hessian[sqp.h_row[i],sqp.h_col[i]] += v
        end
    end
end

"""
    norm_violations

Compute the normalized constraint violation
"""
norm_violations(sqp::AbstractSqpOptimizer, p = 1) = norm_violations(
    sqp.E, sqp.problem.g_L, sqp.problem.g_U, 
    sqp.x, sqp.problem.x_L, sqp.problem.x_U, 
    p
)

norm_violations(sqp::AbstractSqpOptimizer, x::Tv, p = 1) where {T, Tv<:AbstractArray{T}} = norm_violations(
    sqp.problem.eval_g(x, zeros(sqp.problem.m)), sqp.problem.g_L, sqp.problem.g_U, 
    sqp.x, sqp.problem.x_L, sqp.problem.x_U, 
    p
)

"""
    KT_residuals

Compute Kuhn-Turck residuals
"""
KT_residuals(sqp::AbstractSqpOptimizer) = KT_residuals(sqp.df, sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, sqp.Jacobian)

"""
    norm_complementarity

Compute the normalized complementeraity
"""
norm_complementarity(sqp::AbstractSqpOptimizer, p = Inf) = norm_complementarity(
    sqp.E, sqp.problem.g_L, sqp.problem.g_U, 
    sqp.x, sqp.problem.x_L, sqp.problem.x_U, 
    sqp.lambda, sqp.mult_x_U, sqp.mult_x_L, 
    p
)

"""
    compute_phi

Evaluate and return the merit function value for a given point x + α * p.

# Arguments
- `sqp`: SQP structure
- `x`: the current solution point
- `α`: step size taken from `x`
- `p`: direction taken from `x`
"""
function compute_phi(sqp::AbstractSqpOptimizer, x::Tv, α::T, p::Tv) where {T,Tv<:AbstractArray{T}}
    ϕ = 0.0
    xp = x + α * p
    E = ifelse(α == 0.0, sqp.E, sqp.problem.eval_g(xp, zeros(sqp.problem.m)))
    if sqp.feasibility_restoration
        p_slack = sqp.p_slack
        ϕ = sqp.prim_infeas
        for (i, v) in p_slack
            ϕ += α * sum(v)
        end
        for i = 1:sqp.problem.m
            viol =
                maximum([0.0, sqp.E[i] - sqp.problem.g_U[i], sqp.problem.g_L[i] - sqp.E[i]])
            lhs = E[i] - viol
            if sqp.problem.g_L[i] > -Inf && sqp.problem.g_U[i] < Inf
                lhs += α * (p_slack[i][1] - p_slack[i][2])
            elseif sqp.problem.g_L[i] > -Inf
                lhs += α * p_slack[i][1]
            elseif sqp.problem.g_U[i] < Inf
                lhs -= α * p_slack[i][1]
            end
            ϕ +=
                sqp.μ *
                maximum([0.0, lhs - sqp.problem.g_U[i], sqp.problem.g_L[i] - lhs])
        end
    else
        ϕ = sqp.problem.eval_f(xp)
        for i = 1:sqp.problem.m
            if E[i] > sqp.problem.g_U[i]
                ϕ += sqp.μ * (E[i] - sqp.problem.g_U[i])
            elseif E[i] < sqp.problem.g_L[i]
                ϕ += sqp.μ * (sqp.problem.g_L[i] - E[i])
            end
        end
    end
    return ϕ
end

"""
    compute_derivative

Compute the directional derivative at current solution for a given direction.
"""
function compute_derivative(sqp::AbstractSqpOptimizer)
    dfp = 0.0
    cons_viol = 0.0
    if sqp.feasibility_restoration
        for (_, v) in sqp.p_slack
            dfp += sum(v)
        end
        for i = 1:slp.problem.m
            viol = maximum([0.0, slp.E[i] - slp.problem.g_U[i], slp.problem.g_L[i] - slp.E[i]])
            lhs = slp.E[i] - viol
            cons_viol += maximum([0.0, lhs - slp.problem.g_U[i], slp.problem.g_L[i] - lhs])
        end
    else
        dfp += sqp.df' * sqp.p
        for i = 1:sqp.problem.m
            cons_viol += maximum([
                0.0, 
                sqp.E[i] - sqp.problem.g_U[i],
                sqp.problem.g_L[i] - sqp.E[i]
            ])
        end
    end
    return compute_derivative(dfp, sqp.μ, cons_viol)
end

include("sqp_line_search.jl")