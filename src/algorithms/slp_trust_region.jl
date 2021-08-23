mutable struct SlpTR{T,Tv,Tt} <: AbstractSlpOptimizer
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
    # hLag::Tv
    phi::T

    ν::Tv # penalty factors
    Δ::T
    Δ_max::T
    alpha1::T
    alpha2::T

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

    function SlpTR(problem::Model{T,Tv,Tt}) where {T, Tv<:AbstractArray{T}, Tt}
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
        # slp.hLag = Tv(undef, length(problem.h_str))
        slp.phi = Inf

        slp.ν = Tv(undef, problem.m)
        slp.Δ = 0.4
        slp.Δ_max = 2.0
        slp.alpha1 = 0.1
        slp.alpha2 = 0.25

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

function active_set_optimize!(slp::SlpTR)

    slp.start_time = time()

    if slp.options.OutputFlag == 1
        sparsity_val = slp.problem.m > 0 ? length(slp.problem.j_str) / (slp.problem.m * slp.problem.n) : 0.0
        @printf("LP subproblem sparsity: %e\n", sparsity_val)
        add_statistics(slp.problem, "sparsity", sparsity_val)
    end

    slp.Δ = slp.options.tr_size

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
    while true

        # evaluate function, constraints, gradient, Jacobian
        eval_functions!(slp)
    
        LP_time_start = time()
        # solve LP subproblem (to initialize dual multipliers)
        slp.p, lambda, mult_x_U, mult_x_L, slp.p_slack, slp.lp_infeas, status = sub_optimize!(slp, slp.Δ)
        # @show slp.lambda

        add_statistics(slp.problem, "LP_time", time() - LP_time_start)

        if status ∉ [MOI.OPTIMAL, MOI.INFEASIBLE]
            @warn("Unexpected LP subproblem solution status")
            slp.ret == -3
            if norm_violations(slp, slp.x) <= slp.options.tol_infeas
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
                slp.feasibility_restoration = true
                continue
            end
        end

        # update multipliers
        slp.lambda .= lambda
        slp.mult_x_U .= mult_x_U
        slp.mult_x_L .= mult_x_L
    
        compute_nu!(slp)

        slp.prim_infeas = norm_violations(slp, Inf)
        slp.dual_infeas = KT_residuals(slp)
        slp.compl = norm_complementarity(slp)

        print_header(slp)
        print(slp)
        collect_statistics(slp)

        # If the LP subproblem is infeasible, increase μ_merit and resolve.
        if slp.lp_infeas > slp.options.tol_infeas
            slp.ret = 2
            break
        end

        # Check the first-order optimality condition
        if slp.prim_infeas <= slp.options.tol_infeas &&
            slp.dual_infeas <= slp.options.tol_residual &&
            slp.compl <= slp.options.tol_residual &&
            norm(slp.p, Inf) <= slp.options.tol_direction

            if slp.feasibility_restoration
                slp.feasibility_restoration = false
                slp.iter += 1
                continue
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

        # step size computation
        rho = step_quality(slp)
        if slp.ret ∈ [0,2,6]
            break
        end
        if rho >= 0
            # update primal points
            slp.x += slp.p
        end
        
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

sub_optimize!(slp::SlpTR, Δ) = sub_optimize!(
	slp.optimizer,
	LpData(slp),
	Inf,
	slp.x,
	Δ,
    slp.feasibility_restoration
)

function step_quality(slp::SlpTR)
    slp.phi = compute_phi(slp, slp.x, slp.p) - compute_phi(slp, slp.x)
    # @show compute_phi(slp, slp.x, slp.p), compute_phi(slp, slp.x)

    ϕ_pre = compute_derivative(slp)
    # if slp.feasibility_restoration
    #     for (i,v) in slp.p_slack
    #         ϕ_pre += sum(v)
    #     end
    # else
    #     ϕ_pre = slp.df' * slp.p
    # end

    ρ = 0.0
    # @show slp.phi, ϕ_pre
    if abs(ϕ_pre) > 0.0
        ρ = slp.phi / ϕ_pre
        if ρ <= 0
            slp.Δ *= slp.alpha1
        elseif ρ <= 0.25
            slp.Δ *= slp.alpha2
        elseif ρ > 0.75
            slp.Δ = min(2*slp.Δ, slp.Δ_max)
        end
    else
        ρ = -slp.phi
        if abs(slp.phi) < 1.e-8
            if slp.feasibility_restoration
                slp.feasibility_restoration = false
            else
                if slp.prim_infeas <= slp.options.tol_infeas
                    if slp.dual_infeas <= slp.options.tol_residual &&
                        slp.compl <= slp.options.tol_residual
                        slp.ret = 0
                    else
                        slp.ret = 6
                    end
                else
                    slp.ret = 2
                end
            end
        end
    end

    return ρ
end

# NOTE: taken from the line search
function compute_derivative(slp::SlpTR)
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

function compute_nu!(slp::SlpTR)
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

# merit function
function compute_phi(slp::SlpTR, x::Tv, α::T, p::Tv) where {T, Tv<:AbstractArray{T}}
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
compute_phi(slp::SlpTR, x::Tv) where {T, Tv<:AbstractArray{T}} = compute_phi(slp, x, 0.0, slp.p)
compute_phi(slp::SlpTR, x::Tv, p::Tv) where {T, Tv<:AbstractArray{T}} = compute_phi(slp, x, 1.0, p)

function print_header(slp::SlpTR)
    if slp.options.OutputFlag == 0
        return
    end
    if (slp.iter - 1) % 25 == 0
        @printf("  %6s", "iter")
        @printf("  %15s", "f(x_k)")
        @printf("  %15s", "Δ")
        @printf("  %14s", "|p|")
        @printf("  %14s", "inf_pr")
        @printf("  %14s", "inf_du")
        @printf("  %14s", "compl")
        @printf("  %10s", "time")
        @printf("\n")
    end
end

function print(slp::SlpTR)
    if slp.options.OutputFlag == 0
        return
    end
    st = ifelse(slp.feasibility_restoration, "FR", "  ")
    @printf("%2s%6d", st, slp.iter)
    @printf("  %+6.8e", slp.f)
    @printf("  %+6.8e", slp.Δ)
    @printf("  %6.8e", norm(slp.p, Inf))
    @printf("  %6.8e", slp.prim_infeas)
    @printf("  %6.8e", slp.dual_infeas)
    @printf("  %6.8e", slp.compl)
    @printf("  %10.2f", time() - slp.start_time)
    @printf("\n")
end

function collect_statistics(slp::SlpTR)
    if slp.options.StatisticsFlag == 0
        return
    end
    add_statistics(slp.problem, "f(x)", slp.f)
    add_statistics(slp.problem, "|p|", norm(slp.p, Inf))
    add_statistics(slp.problem, "|J|2", norm(slp.dE, 2))
    add_statistics(slp.problem, "|J|inf", norm(slp.dE, Inf))
    add_statistics(slp.problem, "inf_pr", slp.prim_infeas)
    # add_statistics(slp.problem, "inf_du", dual_infeas)
    add_statistics(slp.problem, "compl", slp.compl)
    add_statistics(slp.problem, "time_elapsed", time() - slp.start_time)
end
