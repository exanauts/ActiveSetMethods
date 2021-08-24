mutable struct SlpLS{T,Tv,Tt} <: AbstractSlpOptimizer
    problem::Model{T,Tv,Tt}

    x::Tv # primal solution
    p::Tv
    p_slack::Dict{Int,Tv}
    lambda::Tv
    mult_x_L::Tv
    mult_x_U::Tv
    
    f::T
    df::Tv
    E::Tv
    dE::Tv
    phi::T
    ν::Tv
    directional_derivative::T

    norm_E::T # norm of constraint violations
    alpha::T

    lp_infeas::T
    prim_infeas::T
    dual_infeas::T
    compl::T

    optimizer::MOI.AbstractOptimizer

    options::Parameters

    feasibility_restoration::Bool
    iter::Int
    ret::Int
    start_time::Float64

    function SlpLS(problem::Model{T,Tv,Tt}) where {T, Tv<:AbstractArray{T}, Tt}
        slp = new{T,Tv,Tt}()
        slp.problem = problem
        slp.x = Tv(undef, problem.n)
        slp.p = zeros(problem.n)
        slp.p_slack = Dict()
        slp.lambda = zeros(problem.m)
        slp.mult_x_L = zeros(problem.n)
        slp.mult_x_U = zeros(problem.n)
        slp.df = Tv(undef, problem.n)
        slp.E = Tv(undef, problem.m)
        slp.dE = Tv(undef, length(problem.j_str))
        slp.phi = Inf
        slp.ν = Tv(undef, problem.m)

        slp.norm_E = 0.0
        slp.alpha = 1.0

        slp.lp_infeas = Inf
        slp.prim_infeas = Inf
        slp.dual_infeas = Inf
        slp.compl = Inf

        slp.options = problem.parameters
        slp.optimizer = MOI.instantiate(slp.options.external_optimizer)

        slp.feasibility_restoration = false
        slp.iter = 1
        slp.ret = -5
        slp.start_time = 0.0

        return slp
    end
end

function run!(slp::SlpLS)

    slp.start_time = time()

    if slp.options.OutputFlag == 1
        sparsity_val = slp.problem.m > 0 ? length(slp.problem.j_str) / (slp.problem.m * slp.problem.n) : 0.0
        @printf("LP subproblem sparsity: %e\n", sparsity_val)
        add_statistics(slp.problem, "sparsity", sparsity_val)
    end

    Δ = slp.options.tr_size

    # Set initial point from MOI
    @assert length(slp.x) == length(slp.problem.x)
    slp.x .= slp.problem.x
    # Adjust the initial point to satisfy the column bounds
    for i = 1:slp.problem.n
        if slp.problem.x_L[i] > -Inf
            slp.x[i] = max(slp.x[i], slp.problem.x_L[i])
        end
        if slp.problem.x_U[i] > -Inf
            slp.x[i] = min(slp.x[i], slp.problem.x_U[i])
        end
    end

    slp.iter = 1
    is_valid_step = true
    while true
        
        # evaluate function, constraints, gradient, Jacobian
        eval_functions!(slp)
        slp.alpha = 0.0
        slp.norm_E = norm_violations(slp)
        slp.prim_infeas = norm_violations(slp, Inf)
        slp.dual_infeas = KT_residuals(slp)
        slp.compl = norm_complementarity(slp)
    
        LP_time_start = time()
        # solve LP subproblem (to initialize dual multipliers)
        slp.p, lambda, mult_x_U, mult_x_L, slp.p_slack, slp.lp_infeas, status = sub_optimize!(slp, Δ)
        # @show slp.lp_infeas

        add_statistics(slp.problem, "LP_time", time() - LP_time_start)

        if status ∉ [MOI.OPTIMAL, MOI.INFEASIBLE]
            @warn("Unexpected LP subproblem solution status ($status)")
            slp.ret == -3
            if slp.prim_infeas <= slp.options.tol_infeas
                slp.ret = 6
            end
            break
        elseif status == MOI.INFEASIBLE
            if slp.feasibility_restoration == true
                @printf("Failed to find a feasible direction\n")
                if slp.prim_infeas <= slp.options.tol_infeas
                    slp.ret = 6
                else
                    slp.ret = 2
                end
                break
            else
                # println("Feasibility restoration ($(status), |p| = $(norm(slp.p, Inf))) begins.")
                slp.feasibility_restoration = true
                continue
            end
        end

        # update multipliers
        slp.lambda .= lambda
        slp.mult_x_U .= mult_x_U
        slp.mult_x_L .= mult_x_L
    
        compute_nu!(slp)
        slp.phi = compute_phi(slp)
        slp.directional_derivative = compute_derivative(slp)
        
        if slp.lp_infeas <= slp.options.tol_infeas
            # step size computation
            is_valid_step = compute_alpha(slp)
        end

        print_header(slp)
        print(slp)
        collect_statistics(slp)

        # Failed to find a step size
        if slp.ret == -3
            @printf("Failed to find a step size\n")
            if slp.prim_infeas <= slp.options.tol_infeas
                slp.ret = 6
            else
                slp.ret = 2
            end
            break
        end

        if !is_valid_step
            slp.iter += 1
            continue
        end

        # Check the first-order optimality condition
        if slp.prim_infeas <= slp.options.tol_infeas &&
            slp.compl <= slp.options.tol_residual &&
            min(-slp.directional_derivative, slp.alpha * norm(slp.p, Inf)) <= slp.options.tol_direction
            if slp.feasibility_restoration
                slp.feasibility_restoration = false
            else
                slp.ret = 0
                break
            end
        end
        
        # Iteration counter limit
        if slp.iter >= slp.options.max_iter
            slp.ret = -1
            if slp.prim_infeas <= slp.options.tol_infeas
                slp.ret = 6
            end
            break
        end

        # update primal points
        slp.x += slp.alpha .* slp.p

        slp.iter += 1
    end
    slp.problem.obj_val = slp.problem.eval_f(slp.x)
    slp.problem.status = Int(slp.ret)
    slp.problem.x .= slp.x
    slp.problem.g .= slp.E
    slp.problem.mult_g .= slp.lambda
    slp.problem.mult_x_U .= slp.mult_x_U
    slp.problem.mult_x_L .= slp.mult_x_L
end

sub_optimize!(slp::SlpLS, Δ) = sub_optimize!(
	slp.optimizer,
	LpData(slp),
	Inf,
	slp.x,
	Δ,
    slp.feasibility_restoration
)

"""
    compute_alpha

Compute step size for line search
"""
function compute_alpha(slp::SlpLS)::Bool
    is_valid = true
    slp.alpha = 1.0
    phi_x_p = compute_phi(slp, slp.x, slp.alpha, slp.p)
    eta = slp.options.eta

    while phi_x_p > slp.phi + eta * slp.alpha * slp.directional_derivative
        # The step size can become too small.
        if slp.alpha < slp.options.min_alpha
            # @printf("Descent step cannot be computed.\n")
            # @printf("Feasibility restoration is required but not implemented yet.\n")
            if slp.feasibility_restoration
                slp.ret = -3
            else
                slp.feasibility_restoration = true
            end
            is_valid = false
            break
        end
        slp.alpha *= slp.options.tau
        phi_x_p = compute_phi(slp, slp.x, slp.alpha, slp.p)
    end
    # @show phi_x_p, slp.phi, slp.directional_derivative
    return is_valid
end

# merit function

function compute_nu!(slp::SlpLS)
    if slp.iter == 1
        norm_df = ifelse(slp.feasibility_restoration, 1.0, norm(slp.df))
        J = compute_jacobian_matrix(slp)
        for i = 1:slp.problem.m
            slp.ν[i] = max(1.0, norm_df / max(1.0, norm(J[i,:])))
        end
    else
        for i = 1:slp.problem.m
            slp.ν[i] = max(slp.ν[i], abs(slp.lambda[i]))
        end
    end
end

compute_phi(slp::SlpLS) = compute_phi(slp, slp.x, 0.0, slp.p)
function compute_phi(slp::SlpLS, x::Tv, α::T, p::Tv) where {T, Tv<:AbstractArray{T}}
    ϕ = 0.0
    xp = x + α * p
    E = ifelse(α == 0.0, slp.E, slp.problem.eval_g(xp, zeros(slp.problem.m)))
    if slp.feasibility_restoration
        p_slack = slp.p_slack
        ϕ = slp.prim_infeas
        for (i,v) in p_slack
            ϕ += α * sum(v)
        end
        for i = 1:slp.problem.m
            viol = max(0.0, max(slp.E[i] - slp.problem.g_U[i], slp.problem.g_L[i] - slp.E[i]))
            lhs = E[i] - viol
            if slp.problem.g_L[i] > -Inf && slp.problem.g_U[i] < Inf
                lhs += α * (p_slack[i][1] - p_slack[i][2])
            elseif slp.problem.g_L[i] > -Inf
                lhs += α * p_slack[i][1]
            elseif slp.problem.g_U[i] < Inf
                lhs -= α * p_slack[i][1]
            end
            ϕ += slp.ν[i]*maximum([
                0.0, 
                lhs - slp.problem.g_U[i], 
                slp.problem.g_L[i] - lhs
            ])
        end
    else
        ϕ = slp.problem.eval_f(xp)
        for i = 1:slp.problem.m
            if E[i] > slp.problem.g_U[i]
                ϕ += slp.ν[i]*(E[i] - slp.problem.g_U[i])
            elseif E[i] < slp.problem.g_L[i]
                ϕ += slp.ν[i]*(slp.problem.g_L[i] - E[i])
            end
        end
    end
    return ϕ
end

# directional derivative
function compute_derivative(slp::SlpLS)
    D = 0.0
    if slp.feasibility_restoration
        for (i,v) in slp.p_slack
            D += sum(v)
        end
        for i = 1:slp.problem.m
            viol = max(0.0, max(slp.E[i] - slp.problem.g_U[i], slp.problem.g_L[i] - slp.E[i]))
            lhs = slp.E[i] - viol
            D -= slp.ν[i]*maximum([
                0.0, 
                lhs - slp.problem.g_U[i], 
                slp.problem.g_L[i] - lhs
            ])
        end
    else
        D = slp.df' * slp.p
        for i = 1:slp.problem.m
            if slp.E[i] > slp.problem.g_U[i]
                D -= slp.ν[i]*(slp.E[i] - slp.problem.g_U[i])
            elseif slp.E[i] < slp.problem.g_L[i]
                D -= slp.ν[i]*(slp.problem.g_L[i] - slp.E[i])
            end
        end
    end
    return D
end

function print_header(slp::SlpLS)
    if slp.options.OutputFlag == 0
        return
    end
    if (slp.iter - 1) % 25 == 0
        @printf("  %6s", "iter")
        @printf("  %15s", "f(x_k)")
        # @printf("  %15s", "ϕ(x_k)")
        @printf("  %15s", "D(ϕ,p)")
        # @printf("  %15s", "∇f^Tp")
        @printf("  %14s", "α")
        @printf("  %14s", "|p|")
        @printf("  %14s", "α|p|")
        # @printf("  %14s", "|∇f|")
        @printf("  %14s", "inf_pr")
        @printf("  %14s", "inf_du")
        @printf("  %14s", "compl")
        @printf("  %10s", "time")
        @printf("\n")
    end
end

function print(slp::SlpLS)
    if slp.options.OutputFlag == 0
        return
    end
    st = ifelse(slp.feasibility_restoration, "FR", "  ")
    @printf("%2s%6d", st, slp.iter)
    @printf("  %+6.8e", slp.f)
    # @printf("  %+6.8e", slp.phi)
    @printf("  %+.8e", slp.directional_derivative)
    # @printf("  %+.8e", slp.df' * slp.p)
    @printf("  %6.8e", slp.alpha)
    @printf("  %6.8e", norm(slp.p, Inf))
    @printf("  %6.8e", slp.alpha*norm(slp.p, Inf))
    # @printf("  %.8e", norm(slp.df))
    @printf("  %6.8e", slp.prim_infeas)
    @printf("  %.8e", slp.dual_infeas)
    @printf("  %6.8e", slp.compl)
    @printf("  %10.2f", time() - slp.start_time)
    @printf("\n")
end

function collect_statistics(slp::SlpLS)
    if slp.options.StatisticsFlag == 0
        return
    end
    add_statistics(slp.problem, "f(x)", slp.f)
    add_statistics(slp.problem, "ϕ(x_k))", slp.phi)
    add_statistics(slp.problem, "D(ϕ,p)", slp.directional_derivative)
    add_statistics(slp.problem, "|p|", norm(slp.p, Inf))
    add_statistics(slp.problem, "|J|2", norm(slp.dE, 2))
    add_statistics(slp.problem, "|J|inf", norm(slp.dE, Inf))
    add_statistics(slp.problem, "inf_pr", slp.prim_infeas)
    # add_statistics(slp.problem, "inf_du", dual_infeas)
    add_statistics(slp.problem, "compl", slp.compl)
    add_statistics(slp.problem, "alpha", slp.alpha)
    add_statistics(slp.problem, "time_elapsed", time() - slp.start_time)
end