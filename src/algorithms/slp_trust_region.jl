using Printf

mutable struct SlpTR{T,Tv,Tt} <: AbstractSlpOptimizer
    problem::Model{T,Tv,Tt}

    x::Tv # primal solution
    p::Tv
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

    optimizer::MOI.AbstractOptimizer

    options::Parameters
    statistics::Dict{String,Any}

    iter::Int
    ret::Int

    function SlpTR(problem::Model{T,Tv,Tt}) where {T, Tv<:AbstractArray{T}, Tt}
        slp = new{T,Tv,Tt}()
        slp.problem = problem
        slp.x = Tv(undef, problem.n)
        slp.p = zeros(problem.n)
        slp.lambda = zeros(problem.m)
        slp.mult_x_L = zeros(problem.n)
        slp.mult_x_U = zeros(problem.n)
        slp.df = Tv(undef, problem.n)
        slp.E = Tv(undef, problem.m)
        slp.dE = Tv(undef, length(problem.j_str))
        # slp.hLag = Tv(undef, length(problem.h_str))
        slp.phi = Inf

        slp.ν = Tv(undef, problem.m + problem.n)
        slp.Δ = 0.4
        slp.Δ_max = 2.0
        slp.alpha1 = 0.1
        slp.alpha2 = 0.25

        slp.options = problem.parameters
        slp.optimizer = MOI.instantiate(slp.options.external_optimizer)

        slp.iter = 1
        slp.ret = -5
        return slp
    end
end

function active_set_optimize!(slp::SlpTR)
    # slp.Δ = slp.options.tr_size

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
    if slp.options.StatisticsFlag != 0
    	slp.problem.statistics["f(x)"] = Array{Float64,1}()
    	slp.problem.statistics["ϕ_pre"] = Array{Float64,1}()
    	slp.problem.statistics["|p|"] = Array{Float64,1}()
    	slp.problem.statistics["|J|2"] = Array{Float64,1}() 
    	slp.problem.statistics["|J|inf"] = Array{Float64,1}() 
    	slp.problem.statistics["iter_time"] = Array{Float64,1}() 
    	slp.problem.statistics["m(p)"] = Array{Float64,1}()
    	slp.problem.statistics["inf_pr"] = Array{Float64,1}()
    	slp.problem.statistics["Sparsity"] = Array{Float64,1}()
    	slp.problem.statistics["LP_time"] = Array{Float64,1}()
    	slp.problem.statistics["iter_time"] = Array{Float64,1}()    	
    end

    while true
	    iter_time_start = time();
        if (slp.iter - 1) % 25 == 0 && slp.options.OutputFlag != 0
            @printf("%6s", "iter")
            @printf("  %15s", "f(x_k)")
            @printf("  %15s", "ϕ_pre")
            @printf("  %14s", "|p|")
            @printf("  %14s", "m(p)")
            @printf("  %14s", "inf_pr")
            @printf("  %14s", "Sparsity")
            @printf("\n")
        end

        eval_functions!(slp)
    
        LP_time_start = time()
        # solve LP subproblem (to initialize dual multipliers)
        slp.p, lambda, mult_x_U, mult_x_L, infeasibility, status = sub_optimize!(slp, slp.Δ)
        # @show slp.lambda
        if slp.options.StatisticsFlag != 0
	    	push!(slp.problem.statistics["LP_time"],time()-LP_time_start)
    	end

        # update multipliers
        slp.lambda .= lambda
        slp.mult_x_U .= mult_x_U
        slp.mult_x_L .= mult_x_L
    
        compute_nu!(slp)
        slp.phi = slp.df' * slp.p

        #prim_infeas = norm(slp.dE, Inf) > 0 ? norm_violations(slp, Inf) / norm(slp.dE, Inf) : norm_violations(slp, Inf)
        prim_infeas = norm_violations(slp, Inf)
        sparsity_val = slp.problem.m > 0 ? length(slp.problem.j_str) / (slp.problem.m * slp.problem.n) : 0.0
        
        if slp.options.OutputFlag != 0
            @printf("%6d", slp.iter)
            @printf("  %+.8e", slp.f)
            @printf("  %+.8e", slp.phi)
            @printf("  %.8e", norm(slp.p))
            @printf("  %.8e", infeasibility)
            @printf("  %.8e", prim_infeas)
            @printf("  %.8e", sparsity_val)
            @printf("\n")
        end
        
        if slp.options.StatisticsFlag != 0
	    	push!(slp.problem.statistics["f(x)"],slp.f)
	    	push!(slp.problem.statistics["ϕ_pre"],slp.phi)
	    	push!(slp.problem.statistics["|p|"],norm(slp.p, Inf))
	    	push!(slp.problem.statistics["|J|2"],norm(slp.dE, 2))
	    	push!(slp.problem.statistics["|J|inf"],norm(slp.dE, Inf))
	    	push!(slp.problem.statistics["m(p)"],infeasibility)
	    	push!(slp.problem.statistics["inf_pr"],prim_infeas)
	    	push!(slp.problem.statistics["Sparsity"],sparsity_val)
    	end

        # If the LP subproblem is infeasible, increase μ_merit and resolve.
        if infeasibility > slp.options.tol_infeas
            @printf("LP subproblem is infeasible.\n")
            @printf("Feasibility restoration is required but not implemented yet.\n")
            slp.ret = 2
            break
        end

        # Check the first-order optimality condition
        # TODO: should we check |p| < ϵ instead?
        if prim_infeas <= slp.options.tol_infeas &&
            norm(slp.p, Inf) <= slp.options.tol_residual
            slp.ret = 0
            break
        end
        
        # Iteration counter limit
        if slp.iter >= slp.options.max_iter
            slp.ret = -1
            if norm_violations(slp, slp.x) <= slp.options.tol_infeas
                slp.ret = 6
            end
            break
        end

        # step size computation
        rho = step_quality(slp)
        if rho < 0
            @show rho, slp.Δ
            continue
        end

        # update primal points
        slp.x += slp.p
        
        if slp.options.StatisticsFlag != 0
	        push!(slp.problem.statistics["iter_time"],time()-iter_time_start);
    	end
        
        slp.iter += 1
    end
    if slp.options.StatisticsFlag != 0
	    slp.problem.statistics["iter"] = slp.iter
    end
    slp.problem.obj_val = slp.problem.eval_f(slp.x)
    slp.problem.status = Int(slp.ret)
    slp.problem.x .= slp.x
    slp.problem.g .= slp.E
    slp.problem.mult_g .= slp.lambda
    slp.problem.mult_x_U .= slp.mult_x_U
    slp.problem.mult_x_L .= slp.mult_x_L
end

function LpData(slp::SlpTR)
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

sub_optimize!(slp::SlpTR, Δ) = sub_optimize!(
	slp.optimizer,
	LpData(slp),
	Inf,
	slp.x,
	Δ
)

"""
    norm_violations

Compute the normalized constraint violation
"""

norm_violations(slp::SlpTR, p = 1) = norm_violations(
    slp.E, slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

norm_violations(slp::SlpTR{T,Tv,Tt}, x::Tv, p = 1) where {T, Tv<:AbstractArray{T}, Tt} = norm_violations(
    slp.problem.eval_g(x, zeros(slp.problem.m)), slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

function eval_functions!(slp::SlpTR)
    slp.f = slp.problem.eval_f(slp.x)
    slp.problem.eval_grad_f(slp.x, slp.df)
    slp.problem.eval_g(slp.x, slp.E)
    slp.problem.eval_jac_g(slp.x, :eval, [], [], slp.dE)
    # obj_factor = 1.0
    # slp.problem.eval_h(slp.x, :eval, [], [], obj_factor, slp.lambda, slp.hLag)
    # @show slp.f, slp.df, slp.E, slp.dE
end

compute_jacobian_matrix(slp::SlpTR) = compute_jacobian_matrix(slp.problem.m, slp.problem.n, slp.problem.j_str, slp.dE)

function compute_nu!(slp::SlpTR)
    if slp.iter == 1
        norm_df = norm(slp.df)
        J = compute_jacobian_matrix(slp)
        for i = 1:slp.problem.m
            slp.ν[i] = norm_df / norm(J[i,:])
            if slp.E[i] > slp.problem.g_U[i]
            elseif slp.E[i] < slp.problem.g_L[i]
            end
        end
        for j = 1:slp.problem.n
            slp.ν[slp.problem.m+j] = norm_df
        end
    else
        for i = 1:slp.problem.m
            slp.ν[i] = max(slp.ν[i], abs(slp.lambda[i]))
        end
        for j = 1:slp.problem.n
            slp.ν[slp.problem.m+j] = max(slp.ν[slp.problem.m+j], abs(slp.mult_x_L[j]) + abs(slp.mult_x_U[j]))
        end
    end
end

function step_quality(slp::SlpTR)
    ϕ_act = compute_phi(slp, slp.x + slp.p) - compute_phi(slp, slp.x)
    ρ = ϕ_act / slp.phi
    if ρ <= 0
        slp.Δ *= slp.alpha1
    elseif ρ <= 0.25
        slp.Δ *= slp.alpha2
    elseif ρ > 0.75
        slp.Δ = min(2*slp.Δ, slp.Δ_max)
    end
    return ρ
end

# merit function
function compute_phi(slp::SlpTR, x::Tv) where {T, Tv<:AbstractArray{T}}
    E = slp.problem.eval_g(x, zeros(slp.problem.m))
    ϕ = slp.f
    for i = 1:slp.problem.m
        if E[i] > slp.problem.g_U[i]
            ϕ += slp.ν[i]*(E[i] - slp.problem.g_U[i])
        elseif E[i] < slp.problem.g_L[i]
            ϕ += slp.ν[i]*(slp.problem.g_L[i] - E[i])
        end
    end
    for j = 1:slp.problem.n
        if x[j] > slp.problem.x_U[j]
            ϕ += slp.ν[slp.problem.m+j]*(x[j] - slp.problem.x_U[j])
        elseif x[j] < slp.problem.x_L[j]
            ϕ += slp.ν[slp.problem.m+j]*(slp.problem.x_L[j] - x[j])
        end
    end
    return ϕ
end
