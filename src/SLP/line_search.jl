mutable struct SlpLS{Tv,Tt} <: AbstractSlpOptimizer
    problem::Model{Tv,Tt}

    x::Tv # primal solution
    p::Tv
    lambda::Tv
    mult_x_L::Tv
    mult_x_U::Tv
    
    f::Float64
    df::Tv
    E::Tv
    dE::Tv
    phi::Float64
    directional_derivative::Float64

    norm_E::Float64 # norm of constraint violations
    mu::Float64
    alpha::Float64

    options::Parameters

    iter::Int
    ret::Int

    function SlpLS(problem::Model{Tv,Tt}) where {Tv<:AbstractArray{Float64},Tt}
        slp = new{Tv,Tt}()
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

function slp_optimize!(env::SlpLS)

    Δ = env.options.tr_size

    # Set initial point from MOI
    @assert length(env.x) == length(env.problem.x)
    env.x .= env.problem.x
    # Adjust the initial point to satisfy the column bounds
    for i = 1:env.problem.n
        if env.problem.x_L[i] > -Inf
            env.x[i] = max(env.x[i], env.problem.x_L[i])
        end
        if env.problem.x_U[i] > -Inf
            env.x[i] = min(env.x[i], env.problem.x_U[i])
        end
    end

    env.iter = 1

    @printf("%6s  %15s  %15s  %14s  %14s  %14s  %14s\n", "iter", "f(x_k)", "ϕ(x_k)", "|∇f|", "inf_pr", "inf_du", "Sparsity")
    while true

        eval_functions!(env)
        norm_violations!(env)
    
        # solve LP subproblem (to initialize dual multipliers)
        env.p, lambda, mult_x_U, mult_x_L, status = solve_lp(env, Δ)
        # @show env.lambda
    
        compute_mu!(env)
        compute_phi!(env)

        # if env.iter == 1
        #     env.lambda .= lambda
        #     env.mult_x_U .= mult_x_U
        #     env.mult_x_L .= mult_x_L
        # end

        prim_infeas = normalized_primal_infeasibility(env)
        dual_infeas = normalized_dual_infeasibility(env)
        Jac_matrix = compute_jacobian_matrix(env);
        sparsity_val = nnz(Jac_matrix)/length(Jac_matrix);
        @printf("%6d  %+.8e  %+.8e  %.8e  %.8e  %.8e  %.8e\n", env.iter, env.f, env.phi, norm(env.df), prim_infeas, dual_infeas, sparsity_val)
        if dual_infeas <= env.options.tol_residual && prim_infeas <= env.options.tol_infeas
            @printf("Terminated due to tolerance: primal (%e), dual (%e)\n", prim_infeas, dual_infeas)
            env.ret = 0;
            break
        end

        # directional derivative of the 1-norm merit function
        compute_derivative!(env)
        if env.directional_derivative > -1.e-8
            @printf("Terminated: directional derivative (%e)\n", env.directional_derivative)
            env.ret = 0
            break
        end

        # step size computation
        is_valid_step = compute_alpha(env)
        if env.ret == -3
            @warn "Failed to find a step size"
            break
        end
        if !is_valid_step && env.iter < env.options.max_iter
            env.iter += 1
            continue
        end
        # @show env.alpha

        # update primal points
        env.x += env.alpha .* env.p

        # update multipliers
        env.lambda += env.alpha .* (lambda - env.lambda)
        env.mult_x_U += env.alpha .* (mult_x_U - env.mult_x_U)
        env.mult_x_L += env.alpha .* (mult_x_L - env.mult_x_L)
        # @show env.lambda
        # @show env.mult_x_U
        # @show env.mult_x_L

        # Iteration counter limit
        if env.iter >= env.options.max_iter
            env.ret = -1
            if norm_violations(env, env.x) <= env.options.tol_infeas
                env.ret = 6
            end
            break
        end
        env.iter += 1
    end

    env.problem.obj_val = env.problem.eval_f(env.x)
    env.problem.status = Int(env.ret)
    env.problem.x = env.x
    env.problem.g = env.E
    env.problem.mult_g = env.lambda
    env.problem.mult_x_U = env.mult_x_U
    env.problem.mult_x_L = env.mult_x_L
end

function norm_violations(
    E::Vector{Float64}, g_L::Vector{Float64}, g_U::Vector{Float64},
    x::Vector{Float64}, x_L::Vector{Float64}, x_U::Vector{Float64},
    p = 1)

    m = length(E)
    n = length(x)
    viol = zeros(m+n)
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
norm_violations(env::SlpLS, p = 1) = norm_violations(
    env.E, 
    env.problem.g_L, 
    env.problem.g_U, 
    env.x, 
    env.problem.x_L, 
    env.problem.x_U, 
    p)
norm_violations(env::SlpLS{Tv,Tt}, x::Tv, p = 1) where {Tv<:AbstractArray{Float64}, Tt} = norm_violations(
    env.problem.eval_g(x, zeros(env.problem.m)), 
    env.problem.g_L, 
    env.problem.g_U, 
    env.x, 
    env.problem.x_L, 
    env.problem.x_U, 
    p)
function norm_violations!(env::SlpLS)
    env.norm_E = norm_violations(env)
end

function eval_functions!(env::SlpLS)
    env.f = env.problem.eval_f(env.x)
    env.problem.eval_grad_f(env.x, env.df)
    env.problem.eval_g(env.x, env.E)
    env.problem.eval_jac_g(env.x, :opt, [], [], env.dE)
    # @show env.f, env.df, env.E, env.dE
end

function compute_jacobian_matrix(env::SlpLS)
	J = spzeros(env.problem.m, env.problem.n)
	for i = 1:length(env.problem.j_str)
		J[env.problem.j_str[i][1], env.problem.j_str[i][2]] += env.dE[i]
    end 
    return J
end

"""
Normalized primal/dual infeasibilities
"""
normalized_primal_infeasibility(env::SlpLS) = norm_violations(env, Inf) / norm(env.dE, Inf)
normalized_dual_infeasibility(env::SlpLS) = compute_normalized_Kuhn_Tucker_residuals(env)

compute_normalized_Kuhn_Tucker_residuals(env::SlpLS) = compute_normalized_Kuhn_Tucker_residuals(
    env.df, env.lambda, env.mult_x_U, env.mult_x_L, compute_jacobian_matrix(env))
function compute_normalized_Kuhn_Tucker_residuals(
    df::Tv, lambda::Tv, mult_x_U::Tv, mult_x_L::Tv, J::SparseMatrixCSC{Float64,Int}
) where Tv<:AbstractArray{Float64}
    KT_res = norm(df - J' * lambda - mult_x_U - mult_x_L)
    scalar = max(1.0, norm(df))
    for i = 1:J.m
        scalar = max(scalar, abs(lambda[i]) * norm(J[i,:]))
    end
    # @show KT_res, scalar
    return KT_res / scalar
end

function compute_mu!(env::SlpLS)
    # Update mu only for positive violation
    if env.norm_E > 0
        denom = (1 - env.options.rho) * env.norm_E
        if denom > 0
            # @show denom, norm(env.p), env.df' * env.p
            env.mu = max(env.mu, (env.df' * env.p) / denom)
        end
    end
    # @show env.mu
end

function compute_alpha(env::SlpLS)::Bool
    is_valid = true

    env.alpha = 1.0
    phi_x_p = compute_phi(env, env.x .+ env.alpha * env.p)
    eta = env.options.eta

    while phi_x_p > env.phi + eta * env.alpha * env.directional_derivative
        if env.alpha < env.options.min_alpha
            if env.mu < env.options.max_mu
                env.mu = min(env.options.max_mu, env.mu * 10)
                @printf("* step size too small: increase mu to %e\n", env.mu)
                is_valid = false
            elseif eta > 1.e-6
                eta *= 0.5
                @printf("* step size too small: decrease eta to %e\n", eta)
                continue
            else
                env.ret = -3
            end
            break
        end
        env.alpha *= env.options.tau
        phi_x_p = compute_phi(env, env.x + env.alpha * env.p)
        # @show phi_x_p, env.phi, env.alpha, env.directional_derivative, env.phi + env.options.eta * env.alpha * env.directional_derivative
    end
    return is_valid
end

# merit function
compute_phi(f::Float64, mu::Float64, norm_E::Float64)::Float64 = f + mu * norm_E
compute_phi(env::SlpLS)::Float64 = compute_phi(env.f, env.mu, env.norm_E)
compute_phi(env::SlpLS{Tv,Tt}, x::Tv) where {Tv<:AbstractArray{Float64}, Tt} = compute_phi(
    env.problem.eval_f(x), env.mu, norm_violations(env, x))
function compute_phi!(env::SlpLS)
    env.phi = compute_phi(env)
end

# directional derivative
compute_derivative(env::SlpLS)::Float64 = env.df' * env.p - env.mu * env.norm_E
function compute_derivative!(env::SlpLS)
    env.directional_derivative = compute_derivative(env)
end
