using Printf

abstract type Environment end

mutable struct SLP <: Environment
    problem::NloptProblem

    x::Vector{Float64}
    p::Vector{Float64}
    lambda::Vector{Float64}
    mult_x_L::Vector{Float64}
    mult_x_U::Vector{Float64}
    
    f::Float64
    df::Vector{Float64}
    E::Vector{Float64}
    dE::Vector{Float64}
    phi::Float64
    directional_derivative::Float64

    norm_E::Float64 # norm of constraint violations
    mu::Float64
    alpha::Float64

    options::Parameters

    iter::Int
    ret::Int

    function SLP(problem::NloptProblem)
        slp = new()
        slp.problem = problem
        slp.x = Vector{Float64}(undef, problem.n)
        slp.p = zeros(problem.n)
        slp.lambda = zeros(problem.m)
        slp.mult_x_L = zeros(problem.n)
        slp.mult_x_U = zeros(problem.n)
        slp.df = Vector{Float64}(undef, problem.n)
        slp.E = Vector{Float64}(undef, problem.m)
        slp.dE = Vector{Float64}(undef, length(problem.j_str))
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

function line_search_method(env::SLP)

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

    @printf("%6s  %15s  %15s  %14s  %14s  %14s\n", "iter", "f(x_k)", "ϕ(x_k)", "|E(x_k)|", "|∇f|", "KT resid.")

    eval_functions!(env)
    norm_violations!(env)
    compute_phi!(env)

    # solve LP subproblem (to initialize dual multipliers)
    p, env.lambda, env.mult_x_U, env.mult_x_L, status = solve_lp(env, Δ)
    # @show env.lambda

    while true
        err = compute_normalized_Kuhn_Tucker_residuals(env)
        @printf("%6d  %+.8e  %+.8e  %.8e  %.8e  %.8e\n", env.iter, env.f, env.phi, env.norm_E, norm(env.df), err)
        if err <= env.options.tol_residual && env.norm_E <= env.options.tol_infeas
            @printf("Terminated: KT residuals (%e)\n", err)
            env.ret = 0;
            break
        end

        # solve LP subproblem
        env.p, lambda, mult_x_U, mult_x_L, status = solve_lp(env, Δ)
        @assert length(env.lambda) == length(lambda)

        compute_mu!(env)
        compute_phi!(env)

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

        eval_functions!(env)
        norm_violations!(env)
        compute_phi!(env)
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
    x::Vector{Float64}, x_L::Vector{Float64}, x_U::Vector{Float64})

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
    return norm(viol, 1)
end
norm_violations(env::SLP) = norm_violations(env.E, env.problem.g_L, env.problem.g_U, env.x, env.problem.x_L, env.problem.x_U)
norm_violations(env::SLP, x::Vector{Float64}) = norm_violations(env.problem.eval_g(x, zeros(env.problem.m)), env.problem.g_L, env.problem.g_U, env.x, env.problem.x_L, env.problem.x_U)
function norm_violations!(env::SLP)
    env.norm_E = norm_violations(env)
end

function eval_functions!(env::SLP)
    # TODO: creating zero vectors everytime may not be efficient.
    env.f = env.problem.eval_f(env.x)
    env.df .= env.problem.eval_grad_f(env.x, zeros(env.problem.n))
    env.E .= env.problem.eval_g(env.x, zeros(env.problem.m))
    env.dE .= env.problem.eval_jac_g(env.x, :opt, [], [], zeros(length(env.problem.j_str)))
    # @show env.f, env.df, env.E, env.dE
end

function compute_jacobian_matrix(env::SLP)
	J = spzeros(env.problem.m, env.problem.n)
	for i = 1:length(env.problem.j_str)
		J[env.problem.j_str[i][1], env.problem.j_str[i][2]] += env.dE[i]
    end 
    return J
end

compute_normalized_Kuhn_Tucker_residuals(env::SLP) = compute_normalized_Kuhn_Tucker_residuals(
    env.df, env.lambda, env.mult_x_U, env.mult_x_L, compute_jacobian_matrix(env))
function compute_normalized_Kuhn_Tucker_residuals(df::Vector{Float64}, lambda::Vector{Float64}, mult_x_U::Vector{Float64}, mult_x_L::Vector{Float64}, J::SparseMatrixCSC{Float64,Int})
    KT_res = norm(df - J' * lambda - mult_x_U - mult_x_L)
    scalar = max(1.0, norm(df))
    for i = 1:J.m
        scalar = max(scalar, abs(lambda[i]) * norm(J[i,:]))
    end
    # @show KT_res, scalar
    return KT_res / scalar
end

function compute_mu!(env::SLP)
    # Update mu only for positive violation
    if env.norm_E > 0
        denom = (1 - env.options.rho) * env.norm_E
        if denom > 0
            env.mu = max(env.mu, (env.df' * env.p) / denom)
        end
    else
        env.mu *= 0.9
        # @printf("* large penalty: reduce mu to %e\n", env.mu)
    end
    # @show env.mu
end

function compute_alpha(env::SLP)::Bool
    is_valid = true

    env.alpha = 1.0
    phi_x_p = compute_phi(env, env.x .+ env.alpha * env.p)
    eta = env.options.eta

    while phi_x_p > env.phi + eta * env.alpha * env.directional_derivative
        env.alpha *= env.options.tau
        phi_x_p = compute_phi(env, env.x + env.alpha * env.p)
        if env.alpha < env.options.min_alpha
            if env.mu < env.options.max_mu
                env.mu = min(env.options.max_mu, env.mu * 10)
                @printf("* step size too small: increase mu to %e\n", env.mu)
                is_valid = false
            else
                env.ret = -3
            end
            break
        end
        # @show phi_x_p, env.phi, env.alpha, env.directional_derivative, env.phi + env.options.eta * env.alpha * env.directional_derivative
    end
    return is_valid
end

# merit function
compute_phi(f::Float64, mu::Float64, norm_E::Float64)::Float64 = f + mu * norm_E
compute_phi(env::SLP)::Float64 = compute_phi(env.f, env.mu, env.norm_E)
compute_phi(env::SLP, x::Vector{Float64})::Float64 = compute_phi(env.problem.eval_f(x), env.mu, norm_violations(env, x))
function compute_phi!(env::SLP)
    env.phi = compute_phi(env)
end

# directional derivative
compute_derivative(env::SLP)::Float64 = env.df' * env.p - env.mu * env.norm_E
function compute_derivative!(env::SLP)
    env.directional_derivative = compute_derivative(env)
end
