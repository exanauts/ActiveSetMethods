"""
    Sequential quadratic programming with line search
"""
mutable struct SqpLS{T,Tv,Tt} <: AbstractSqpOptimizer
    @sqp_fields

    phi::T # merit function value
    μ::T # penalty parameters for the merit function
    directional_derivative::T # directional derivative
    alpha::T # stepsize

    function SqpLS(problem::Model{T,Tv,Tt}) where {T,Tv<:AbstractArray{T},Tt}
        sqp = new{T,Tv,Tt}()
        sqp.problem = problem
        sqp.x = Tv(undef, problem.n)
        sqp.p = zeros(problem.n)
        sqp.p_slack = Dict()
        sqp.lambda = zeros(problem.m)
        sqp.mult_x_L = zeros(problem.n)
        sqp.mult_x_U = zeros(problem.n)
        sqp.df = Tv(undef, problem.n)
        sqp.E = Tv(undef, problem.m)
        sqp.dE = Tv(undef, length(problem.j_str))

        sqp.j_row = Vector{Int}(undef, length(problem.j_str))
        sqp.j_col = Vector{Int}(undef, length(problem.j_str))
        for i=1:length(problem.j_str)
            sqp.j_row[i] = Int(problem.j_str[i][1])
            sqp.j_col[i] = Int(problem.j_str[i][2])
        end
        sqp.Jacobian = sparse(sqp.j_row, sqp.j_col, ones(length(sqp.j_row)), problem.m, problem.n)
        sqp.h_row = Vector{Int}(undef, length(problem.h_str))
        sqp.h_col = Vector{Int}(undef, length(problem.h_str))
        for i=1:length(problem.h_str)
            sqp.h_row[i] = Int(problem.h_str[i][1])
            sqp.h_col[i] = Int(problem.h_str[i][2])
        end
        sqp.h_val = Tv(undef, length(problem.h_str))
        sqp.Hessian = sparse(sqp.h_row, sqp.h_col, ones(length(sqp.h_row)), problem.n, problem.n)

        sqp.phi = Inf
        sqp.μ = -Inf

        sqp.alpha = 1.0

        sqp.prim_infeas = Inf
        sqp.dual_infeas = Inf
        sqp.compl = Inf

        sqp.options = problem.parameters
        sqp.optimizer = nothing

        sqp.feasibility_restoration = false
        sqp.iter = 1
        sqp.ret = -5
        sqp.start_time = 0.0
        sqp.start_iter_time = 0.0

        return sqp
    end
end

"""
    run!

Run the line-search SQP algorithm
"""
function run!(sqp::SqpLS)

    sqp.start_time = time()

    if sqp.options.OutputFlag == 1
        sparsity_val = ifelse(
            sqp.problem.m > 0,
            length(sqp.problem.j_str) / (sqp.problem.m * sqp.problem.n),
            0.0,
        )
        @printf("Constraint sparsity: %e\n", sparsity_val)
        add_statistics(sqp.problem, "sparsity", sparsity_val)
    else
        Logging.disable_logging(Logging.Info)
    end

    # Set initial point from MOI
    @assert length(sqp.x) == length(sqp.problem.x)
    sqp.x .= sqp.problem.x
    # Adjust the initial point to satisfy the column bounds
    for i = 1:sqp.problem.n
        if sqp.problem.x_L[i] > -Inf
            sqp.x[i] = max(sqp.x[i], sqp.problem.x_L[i])
        end
        if sqp.problem.x_U[i] > -Inf
            sqp.x[i] = min(sqp.x[i], sqp.problem.x_U[i])
        end
    end

    sqp.iter = 1
    is_valid_step = true
    while true

        # Iteration counter limit
        if sqp.iter > sqp.options.max_iter
            sqp.ret = -1
            if sqp.prim_infeas <= sqp.options.tol_infeas
                sqp.ret = 6
            end
            break
        end

        sqp.start_iter_time = time()

        # evaluate function, constraints, gradient, Jacobian
        eval_functions!(sqp)
        sqp.alpha = 0.0
        sqp.prim_infeas = norm_violations(sqp, Inf)
        sqp.dual_infeas = KT_residuals(sqp)
        sqp.compl = norm_complementarity(sqp)

        LP_time_start = time()
        # solve LP subproblem (to initialize dual multipliers)
        sqp.p, lambda, mult_x_U, mult_x_L, sqp.p_slack, status =
            sub_optimize!(sqp)

        # directions for dual multipliers
        p_lambda = lambda - sqp.lambda
        p_x_U = mult_x_U - sqp.mult_x_U
        p_x_L = mult_x_L - sqp.mult_x_L

        add_statistics(sqp.problem, "LP_time", time() - LP_time_start)

        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.INFEASIBLE]
            @warn("Unexpected LP subproblem solution status ($status)")
            sqp.ret == -3
            if sqp.prim_infeas <= sqp.options.tol_infeas
                sqp.ret = 6
            end
            break
        elseif status == MOI.INFEASIBLE
            if sqp.feasibility_restoration == true
                @info "Failed to find a feasible direction"
                if sqp.prim_infeas <= sqp.options.tol_infeas
                    sqp.ret = 6
                else
                    sqp.ret = 2
                end
                break
            else
                # println("Feasibility restoration ($(status), |p| = $(norm(sqp.p, Inf))) begins.")
                sqp.feasibility_restoration = true
                continue
            end
        end

        compute_mu!(sqp)
        sqp.phi = compute_phi(sqp, sqp.x, 0.0, sqp.p)
        sqp.directional_derivative = compute_derivative(sqp)

        # step size computation
        is_valid_step = compute_alpha(sqp)

        print(sqp)
        collect_statistics(sqp)

        if norm(sqp.p, Inf) <= sqp.options.tol_direction
            if sqp.feasibility_restoration
                sqp.feasibility_restoration = false
                sqp.iter += 1
                continue
            else
                sqp.ret = 0
                break
            end
        end

        if sqp.prim_infeas <= sqp.options.tol_infeas && sqp.compl <= sqp.options.tol_residual
            if sqp.feasibility_restoration
                sqp.feasibility_restoration = false
                sqp.iter += 1
                continue
            elseif sqp.dual_infeas <= sqp.options.tol_residual
                sqp.ret = 0
                break
            end
        end

        # Failed to find a step size
        if !is_valid_step
            @info "Failed to find a step size"
            if sqp.ret == -3
                if sqp.prim_infeas <= sqp.options.tol_infeas
                    sqp.ret = 6
                else
                    sqp.ret = 2
                end
                break
            else
                sqp.feasibility_restoration = true
            end

            sqp.iter += 1
            continue
        end

        # update primal points
        sqp.x += sqp.alpha .* sqp.p
        sqp.lambda += sqp.alpha .* p_lambda
        sqp.mult_x_U += sqp.alpha .* p_x_U
        sqp.mult_x_L += sqp.alpha .* p_x_L

        sqp.iter += 1
    end
    sqp.problem.obj_val = sqp.problem.eval_f(sqp.x)
    sqp.problem.status = Int(sqp.ret)
    sqp.problem.x .= sqp.x
    sqp.problem.g .= sqp.E
    sqp.problem.mult_g .= sqp.lambda
    sqp.problem.mult_x_U .= sqp.mult_x_U
    sqp.problem.mult_x_L .= sqp.mult_x_L
    add_statistic(sqp.problem, "iter", sqp.iter)
end

"""
    sub_optimize!

Solve QP subproblems by using JuMP
"""
sub_optimize!(sqp::SqpLS) = sub_optimize!(sqp, JuMP.Model(sqp.options.external_optimizer))

"""
    compute_mu!

Compute the penalty parameter for the merit fucntion
"""
function compute_mu!(sqp::AbstractSqpOptimizer)
    denom = max((1-sqp.options.rho)*norm_violations(sqp), 1.0e-8)
    Hess_part = max(0.5 * sqp.p' * sqp.Hessian * sqp.p, 0.0)
    sqp.μ = max(
        sqp.μ,
        (sqp.df' * sqp.p + Hess_part) / denom
    )
end

"""
    compute_alpha

Compute step size for line search
"""
function compute_alpha(sqp::AbstractSqpOptimizer)::Bool
    is_valid = true
    sqp.alpha = 1.0
    
    if norm(sqp.p, Inf) <= sqp.options.tol_direction
        return true
    end

    phi_x_p = compute_phi(sqp, sqp.x, sqp.alpha, sqp.p)
    eta = sqp.options.eta

    while phi_x_p > sqp.phi + eta * sqp.alpha * sqp.directional_derivative
        # The step size can become too small.
        if sqp.alpha < sqp.options.min_alpha
            if sqp.feasibility_restoration
                sqp.ret = -3
            end
            is_valid = false
            break
        end
        sqp.alpha *= sqp.options.tau
        phi_x_p = compute_phi(sqp, sqp.x, sqp.alpha, sqp.p)
    end
    # @show phi_x_p, sqp.phi, sqp.alpha, sqp.directional_derivative, is_valid
    return is_valid
end

"""
    print

Print iteration information.
"""
function print(sqp::SqpLS)
    if sqp.options.OutputFlag == 0
        return
    end
    if (sqp.iter - 1) % 25 == 0
        @printf("  %6s", "iter")
        @printf("  %15s", "f(x_k)")
        @printf("  %15s", "ϕ(x_k)")
        @printf("  %15s", "D(ϕ,p)")
        @printf("  %14s", "α")
        @printf("  %14s", "|p|")
        @printf("  %14s", "α|p|")
        @printf("  %14s", "inf_pr")
        @printf("  %14s", "inf_du")
        @printf("  %14s", "compl")
        @printf("  %10s", "time")
        @printf("\n")
    end
    st = ifelse(sqp.feasibility_restoration, "FR", "  ")
    @printf("%2s%6d", st, sqp.iter)
    @printf("  %+6.8e", sqp.f)
    @printf("  %+6.8e", sqp.phi)
    @printf("  %+.8e", sqp.directional_derivative)
    @printf("  %6.8e", sqp.alpha)
    @printf("  %6.8e", norm(sqp.p, Inf))
    @printf("  %6.8e", sqp.alpha * norm(sqp.p, Inf))
    @printf("  %6.8e", sqp.prim_infeas)
    @printf("  %.8e", sqp.dual_infeas)
    @printf("  %6.8e", sqp.compl)
    @printf("  %10.2f", time() - sqp.start_time)
    @printf("\n")
end

"""
    collect_statistics

Collect iteration information.
"""
function collect_statistics(sqp::SqpLS)
    if sqp.options.StatisticsFlag == 0
        return
    end
    add_statistics(sqp.problem, "f(x)", sqp.f)
    add_statistics(sqp.problem, "ϕ(x_k))", sqp.phi)
    add_statistics(sqp.problem, "D(ϕ,p)", sqp.directional_derivative)
    add_statistics(sqp.problem, "|p|", norm(sqp.p, Inf))
    add_statistics(sqp.problem, "|J|2", norm(sqp.dE, 2))
    add_statistics(sqp.problem, "|J|inf", norm(sqp.dE, Inf))
    add_statistics(sqp.problem, "inf_pr", sqp.prim_infeas)
    # add_statistics(sqp.problem, "inf_du", dual_infeas)
    add_statistics(sqp.problem, "compl", sqp.compl)
    add_statistics(sqp.problem, "alpha", sqp.alpha)
    add_statistics(sqp.problem, "iter_time", time() - sqp.start_iter_time)
    add_statistics(sqp.problem, "time_elapsed", time() - sqp.start_time)
end