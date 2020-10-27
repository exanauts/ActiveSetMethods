using Printf

mutable struct SlpLS{T,Tv,Tt} <: AbstractSlpOptimizer
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
    phi::T
    directional_derivative::T

    norm_E::T # norm of constraint violations
    mu::T
    alpha::T

    options::Parameters

    iter::Int
    ret::Int

    function SlpLS(problem::Model{T,Tv,Tt}) where {T, Tv<:AbstractArray{T}, Tt}
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
        slp.phi = Inf

        slp.norm_E = 0.0
        slp.mu = problem.parameters.mu
        @assert slp.mu > 0
        slp.alpha = 1.0

        slp.options = problem.parameters

        slp.iter = 1
        slp.ret = -5
        return slp
    end
end

function slp_optimize!(slp::SlpLS)

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

    @printf("%6s  %15s  %15s  %15s  %14s  %14s  %14s  %14s  %14s\n", "iter", "f(x_k)", "ϕ(x_k)", "D(ϕ,p)", "μ", "|∇f|", "inf_pr", "KT_resid", "Sparsity")
    while true

        eval_functions!(slp)
        norm_violations!(slp)
    
        # solve LP subproblem (to initialize dual multipliers)
        slp.p, lambda, mult_x_U, mult_x_L, status = sublp_optimize!(slp, Δ)
        # @show slp.lambda
    
        compute_mu!(slp)
        compute_phi!(slp)
        compute_derivative!(slp)

        # if slp.iter == 1
        #     slp.lambda .= lambda
        #     slp.mult_x_U .= mult_x_U
        #     slp.mult_x_L .= mult_x_L
        # end

        prim_infeas = norm(slp.dE, Inf) > 0 ? norm_violations(slp, Inf) / norm(slp.dE, Inf) : norm_violations(slp, Inf)
        dual_infeas = KT_residuals(slp)
        sparsity_val = slp.problem.m > 0 ? length(slp.problem.j_str) / (slp.problem.m * slp.problem.n) : 0.0
        @printf("%6d  %+.8e  %+.8e  %+.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n", 
            slp.iter, slp.f, slp.phi, slp.directional_derivative, slp.mu, norm(slp.df), prim_infeas, dual_infeas, sparsity_val)
        if dual_infeas <= slp.options.tol_residual && prim_infeas <= slp.options.tol_infeas
            @printf("Terminated due to tolerance: primal (%e), Kuhn-Tucker residual (%e)\n", prim_infeas, dual_infeas)
            slp.ret = 0;
            break
        end

        # directional derivative of the 1-norm merit function
        if slp.directional_derivative > -1.e-8
            @printf("Terminated: directional derivative (%e)\n", slp.directional_derivative)
            slp.ret = 0
            break
        end

        # step size computation
        is_valid_step = compute_alpha(slp)
        if slp.ret == -3
            @warn "Failed to find a step size"
            break
        end
        if !is_valid_step && slp.iter < slp.options.max_iter
            slp.iter += 1
            continue
        end
        # @show slp.alpha

        # update primal points
        slp.x += slp.alpha .* slp.p

        # update multipliers
        slp.lambda += slp.alpha .* (lambda - slp.lambda)
        slp.mult_x_U += slp.alpha .* (mult_x_U - slp.mult_x_U)
        slp.mult_x_L += slp.alpha .* (mult_x_L - slp.mult_x_L)
        # @show slp.lambda
        # @show slp.mult_x_U
        # @show slp.mult_x_L

        # Iteration counter limit
        if slp.iter >= slp.options.max_iter
            slp.ret = -1
            if norm_violations(slp, slp.x) <= slp.options.tol_infeas
                slp.ret = 6
            end
            break
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

"""
    KT_residuals

Compute Kuhn-Turck residuals
"""
KT_residuals(slp::SlpLS) = KT_residuals(slp.df, slp.lambda, slp.mult_x_U, slp.mult_x_L, compute_jacobian_matrix(slp))

"""
    norm_violations

Compute the normalized constraint violation
"""

norm_violations(slp::SlpLS, p = 1) = norm_violations(
    slp.E, slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

norm_violations(slp::SlpLS{T,Tv,Tt}, x::Tv, p = 1) where {T, Tv<:AbstractArray{T}, Tt} = norm_violations(
    slp.problem.eval_g(x, zeros(slp.problem.m)), slp.problem.g_L, slp.problem.g_U, 
    slp.x, slp.problem.x_L, slp.problem.x_U, 
    p
)

function norm_violations!(slp::SlpLS)
    slp.norm_E = norm_violations(slp)
end

function eval_functions!(slp::SlpLS)
    slp.f = slp.problem.eval_f(slp.x)
    slp.problem.eval_grad_f(slp.x, slp.df)
    slp.problem.eval_g(slp.x, slp.E)
    slp.problem.eval_jac_g(slp.x, :opt, [], [], slp.dE)
    # @show slp.f, slp.df, slp.E, slp.dE
end

function compute_jacobian_matrix(slp::SlpLS)
	J = spzeros(slp.problem.m, slp.problem.n)
	for i = 1:length(slp.problem.j_str)
		J[slp.problem.j_str[i][1], slp.problem.j_str[i][2]] += slp.dE[i]
    end 
    return J
end

function compute_mu!(slp::SlpLS)
    # Update mu only for positive violation
    if slp.norm_E > 0
        denom = (1 - slp.options.rho) * slp.norm_E
        if denom > 0
            # @show denom, norm(slp.p), slp.df' * slp.p
            slp.mu = max(slp.mu, (slp.df' * slp.p) / denom)
        end
    end
    slp.mu = max(norm(slp.lambda, Inf)+1.e-4, slp.mu)
    # @show slp.mu
end

"""
    compute_alpha

Compute step size for line search
"""
function compute_alpha(slp::SlpLS)::Bool
    is_valid = true

    slp.alpha = 1.0
    phi_x_p = compute_phi(slp, slp.x .+ slp.alpha * slp.p)
    eta = slp.options.eta

    while phi_x_p > slp.phi + eta * slp.alpha * slp.directional_derivative
        # The step size can become too small.
        if slp.alpha < slp.options.min_alpha
            # This can happen when mu is not sufficiently large.
            # A descent step is valid only for a sufficiently large mu.
            slp.mu = slp.mu * 3
            is_valid = false
            #=
            if slp.mu < slp.options.max_mu
                # Increase mu, if mu is not large enough.
                slp.mu = min(slp.options.max_mu, slp.mu * 10)
                @printf("* step size too small: increase mu to %e\n", slp.mu)
                is_valid = false
            elseif eta > 1.e-6
                eta *= 0.5
                @printf("* step size too small: decrease eta to %e\n", eta)
                continue
            else
                slp.ret = -3
            end
            =#
            break
        end
        slp.alpha *= slp.options.tau
        phi_x_p = compute_phi(slp, slp.x + slp.alpha * slp.p)
        # @show phi_x_p, slp.phi, slp.alpha, slp.directional_derivative, slp.phi + slp.options.eta * slp.alpha * slp.directional_derivative
    end
    return is_valid
end

# merit function
compute_phi(f, mu, norm_E) = f + mu * norm_E
compute_phi(slp::SlpLS) = compute_phi(slp.f, slp.mu, slp.norm_E)
compute_phi(slp::SlpLS{T,Tv,Tt}, x::Tv) where {T, Tv<:AbstractArray{T}, Tt} = compute_phi(
    slp.problem.eval_f(x), slp.mu, norm_violations(slp, x))
function compute_phi!(slp::SlpLS)
    slp.phi = compute_phi(slp)
end

# directional derivative
compute_derivative(slp::SlpLS) = slp.df' * slp.p - slp.mu * slp.norm_E
function compute_derivative!(slp::SlpLS)
    slp.directional_derivative = compute_derivative(slp)
end
