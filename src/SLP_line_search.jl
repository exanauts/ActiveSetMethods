using Printf

"""
SLP_line_search(model)
This function applies SLP line search algorithm on nonlinear optimization
    problem defined in model.
if the return value is 1 it means it is converged, if 5, it means it was
    interrupted such as reached max_iter
# Examples
```julia-repl
julia> SLP_line_search(model)
1
```
"""
function SLP_line_search(model::NloptProblem)

    ret = -5 # Optimize_not_called

    num_variables = model.n;
    num_constraints = model.m;
    constraint_lb = model.g_L
    constraint_ub = model.g_U
    jacobian_sparsity = model.j_str


    c_init = spzeros(num_variables+1);
    A = spzeros(num_constraints,num_variables);

    x = zeros(num_variables)

    p = ones(num_variables)
    df = ones(num_variables)
    E = zeros(num_constraints)

    dE = zeros(length(jacobian_sparsity))
    lam = zeros(num_constraints)
    p_lam = zeros(num_constraints)
    lam_ = zeros(num_constraints)

    eta = Options_["eta"];
    tau = Options_["tau"];
    rho = Options_["rho"];
    mu = Options_["mu"];
    mu_max = Options_["mu_max"];
	Δ = Options_["TR_size"]

    # This sets x0, initial point.
    # TODO: This must be set by MOI.
    for i=1:num_variables
        if model.x_L[i] != -Inf && model.x_U[i] != Inf
            # x[i] = rand(model.x_L[i]:model.x_U[i])
            x[i] = 0.5 * (model.x_L[i] + model.x_U[i])
        end
    end

    alpha = 1;
    norm_E_a = 0.0;

    if Options_["mode"] == "Debug"
        println("model.x_L: ", model.x_L);
        println("model.x_U: ", model.x_U);
        println("constraint_lb: ", constraint_lb);
        println("constraint_ub: ", constraint_ub);
        println("####---->solveProblem(num_variables): ", num_variables);
        println("####---->solveProblem(num_constraints): ", num_constraints);
        println("####---->solveProblem(jacobian_sparsity): ", jacobian_sparsity);
    end


    @printf("%6s  %15s  %15s  %14s  %14s  %14s\n", "iter", "f(x_k)", "ϕ(x_k)", "|E(x_k)|", "|∇f|", "KT resid.")
    for i=1:Options_["max_iter"]

        f = model.eval_f(x);
        df = model.eval_grad_f(x, zeros(num_variables));
        E = model.eval_g(x, zeros(num_constraints));
        dE = model.eval_jac_g(x, :opt, [], [], zeros(length(jacobian_sparsity)));

        # Compute penalty parameter μ
        norm_E = model.eval_norm_E(x,zeros(num_constraints),constraint_lb,constraint_ub);
        mu_temp = (norm_E > 0) ? df' * p / (1 - rho) / norm_E : 1;
        #mu_temp = (norm_E > 0) ? mu / (1 - rho) / norm_E : 1;
        mu = (mu < mu_temp && norm_E > 0 && mu < mu_max) ? mu_temp : mu;


        if Options_["mode"] == "Debug"
            println("-----------------------------> itr: ", i);
            println("####---->solveProblem(x): ", x);
            println("####---->solveProblem(f): ", f);
            println("####---->solveProblem(df): ", df);
            println("####---->solveProblem(E): ", E);
            println("####---->solveProblem(dE): ", dE);
            println("####---->Before solveProblem(mu): ", mu);
        end

        c_init[1:num_variables] .= df;
        c_init[num_variables+1] = f;
        for Ai = 1:length(jacobian_sparsity)
            #TODO This might cause problem in terms of lambda
            # A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = 1.0;
            A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = dE[Ai];
        end

        (p,p_lam,p_status) = solve_lp(c_init,A,E,model.x_L,model.x_U,constraint_lb,constraint_ub,mu,Δ)
        @show length(p_lam), length(lam)
        @assert length(p_lam) == length(lam)
        #println("num_constraints: ", num_constraints);
        #println("length p_lam: ", length(p_lam));
        lam_ .= p_lam - lam;
        # println("A: ", A)
        # println("norm_E from function: ", model.eval_norm_E(x+p,E,constraint_lb,constraint_ub))
        # phi_k1 =
        # phi_k =
        alpha = 1;
        # norm_E_mu = model.eval_norm_E(x+alpha*p,zeros(num_constraints),constraint_lb,constraint_ub)

        norm_E_a = model.eval_norm_E(x+alpha*p,zeros(num_constraints),constraint_lb,constraint_ub)

        phi_x = model.eval_merit(x, norm_E, mu);
        phi_x_p = model.eval_merit(x+alpha * p, norm_E_a, mu);
        D1_x = model.eval_D(x, df, norm_E, mu, p);

        if Options_["mode"] == "Debug"
            println("--------------------------> calc_phi(x): ", phi_x)
            println("--------------------------> calc_phi(x+ap): ", phi_x_p)
            println("--------------------------> calc_D1(x): ", D1_x)
            println("--------------------------> |E(x)|: ", norm_E)
            println("--------------------------> |E(x+ap)|: ", norm_E_a)
        end


        temp_ind = 0
        # while((phi_x_p > phi_x + eta * alpha * D1_x) && (alpha > Options_["alpha_lb"]))
        while phi_x_p > phi_x + eta * alpha * D1_x
            temp_ind+=1;
            alpha = alpha * tau;
            norm_E_a = model.eval_norm_E(x+alpha*p,zeros(num_constraints),constraint_lb,constraint_ub)
            phi_x = model.eval_merit(x, norm_E, mu);
            phi_x_p = model.eval_merit(x+alpha * p, norm_E_a, mu);
            D1_x = model.eval_D(x, df, norm_E, mu, p);
            if alpha <= Options_["alpha_lb"]
                @warn "Step size too small"
                break
            end
            # if (temp_ind>Options_["max_iter_inner"])
            #     break
            # end
        end

        if alpha <= Options_["alpha_lb"]
            ret = -3
            break;
        end

        p_new = alpha .* p
        lam_new = alpha .* lam_

        #TODO Temporary
        if Options_["mode"] == "Debug"
            push!(mu_RHS, mu_temp)
            push!(mu_numerator, df' * p)
            push!(mux, mu)
            push!(fx, f)
            push!(normEx, norm_E)
            push!(Phix, phi_x)
            push!(Dx, D1_x)
            append!(px, p)
            push!(alphax, alpha)
            append!(alphapx, p_new)
            append!(lamx, lam + lam_new)

        end



        if Options_["mode"] == "Debug"
            println("-------------------------->after alpha: ", alpha)
            println("--------------------------> calc_phi(x): ", phi_x)
            println("--------------------------> calc_phi(x+ap): ", phi_x_p)
            println("--------------------------> calc_D1(x): ", D1_x)
            println("--------------------------> |E(x)|: ", norm_E)
            println("--------------------------> |E(x+ap)|: ", norm_E_a)
            println("####---->solveProblem(alpha*p): ", p_new);
        end
        # for j=1:num_constraints
        #     lam_[j] = df[1]/dE[j];
        #     plam[j] = lam_[j] - lam[j]
        # end
        #lam_[1] = df[1]/dE[1];
        #lam_[2] = df[1]/dE[2];
        #plam[1] = lam_[1] - lam[1]
        #plam[2] = lam_[2] - lam[2]
        #println("typeof(x): ", typeof(x));
        #println("length(x): ", length(x));
        #println("alpha: ", alpha);
        #println("typeof(alpha): ", typeof(alpha));
        #println("typeof(p): ", typeof(p));
        #println("length(p): ", length(p));
        # println("p: ", p);

        x .= x + p_new;
        lam .= lam + alpha .* lam_;

        E = model.eval_g(x, zeros(num_constraints));
        dE = model.eval_jac_g(x, :opt, [], [], zeros(length(jacobian_sparsity)));
        df = model.eval_grad_f(x, zeros(num_variables));

        num_temp = zeros(num_variables);
        denom_temp = zeros(num_constraints);
        dE_vec = zeros(num_constraints);

        for Ai = 1:length(jacobian_sparsity)
            # r_temp[jacobian_sparsity[Ai][2]] += p_lam[jacobian_sparsity[Ai][1]] * dE[Ai];
            num_temp[jacobian_sparsity[Ai][2]] += lam[jacobian_sparsity[Ai][1]] * dE[Ai];
            dE_vec[jacobian_sparsity[Ai][1]] += dE[Ai];
            denom_temp[jacobian_sparsity[Ai][1]] += (lam[jacobian_sparsity[Ai][1]] * dE[Ai]) ^ 2;
            # A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = 1.0;
            # A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = dE[Ai];
        end
        # err = norm(df - r_temp) / max(norm(dE), norm(lam)*norm(E))


        denom_temp =  sqrt.(denom_temp)

        (err_denom,_) = findmax(denom_temp)

        err = norm(df - num_temp) / max(norm(df), err_denom, 1.0);

        #TODO Temporary
        if Options_["mode"] == "Debug"
            push!(mu_RHS, mu_temp)
            push!(mu_numerator, df' * p)
            push!(mux, mu)
            push!(fx, f)
            push!(normDfx, norm(df))
            push!(normLamx, norm(lam))
            push!(normEx, norm_E)
            push!(Phix, phi_x)
            push!(Dx, D1_x)
            append!(px, p)
            push!(alphax, alpha)
            append!(alphapx, p_new)
            append!(lamx, lam + lam_new)
            append!(errx, err)
            append!(normdCx, norm(dE_vec))
        end


        #err = sum(df.^2)
        if Options_["mode"] == "Debug"
            println("X: ", x);
        end
        # if (sum(abs.(p)) <= Options_["epsilon"])
        #     ret = 0
        #     break;
        # end
        # if (norm_E_a + sum(abs.(df' * p_new)) <= Options_["epsilon"])
        # println("err: ", err)

        @printf("%6d  %+.8e  %+.8e  %.8e  %.8e  %.8e\n", i, f, phi_x, norm_E, norm(df), err)

        if err <= Options_["epsilon"]
            ret = 0
            break;
        end
    end


    #df = eval_g_cb(x)
    #E =
    #dE =
    #H =
    #a = eval_objective(model, [4.0])
    #gx2 = eval_g_cb([4.0], [1,2])
    #println("####---->solveProblem(gx2)x=4: ", gx2);
    #println("####---->solveProblem(model.inner.g): ", model.inner.g);

    #gx1 = eval_constraint(model.inner, [0.0,0.0], [2])

    #a = prob.eval_f_cb(4);
    #println("####---->solveProblem(gx1,2): ", gx1);
    #println("####---->solveProblem(a): ", a);
    #println("####---->solveProblem(prob): ", prob);
    model.obj_val = model.eval_f(x)
    model.status = Int(ret)
    #prob.obj_val = eval_f_cb(x);
    model.x = x;
    #println("####---->solveProblem(ret)", ret);
    return Int(ret)
end

abstract type Environment end

mutable struct SLP <: Environment
    problem::NloptProblem

    x::Vector{Float64}
    p::Vector{Float64}
    lambda::Vector{Float64}
    
    f::Float64
    df::Vector{Float64}
    E::Vector{Float64}
    dE::Vector{Float64}
    phi::Float64
    directional_derivative::Float64

    norm_E::Float64 # norm of constraint violations
    mu::Float64
    alpha::Float64

    options::Dict{String,Any}

    ret::Int

    function SLP(problem::NloptProblem, options::Dict{String,Any})
        slp = new()
        slp.problem = problem
        slp.x = Vector{Float64}(undef, problem.n)
        slp.p = Vector{Float64}(undef, problem.n)
        slp.lambda = zeros(problem.m)
        slp.df = Vector{Float64}(undef, problem.n)
        slp.E = Vector{Float64}(undef, problem.m)
        slp.dE = Vector{Float64}(undef, length(problem.j_str))

        slp.norm_E = 0.0
        slp.mu = options["mu"]
        slp.alpha = 1.0;

        slp.options = options

        slp.ret = -5
        return slp
    end
end

function line_search_method(env::SLP)

    Δ = env.options["TR_size"]

    # This sets x0, initial point.
    # TODO: This must be set by MOI.
    for i = 1:env.problem.n
        if env.problem.x_L[i] > -Inf && env.problem.x_U[i] < Inf
            env.x[i] = 0.5 * (env.problem.x_L[i] + env.problem.x_U[i])
        elseif env.problem.x_L[i] > -Inf
            env.x[i] = env.problem.x_L[i]
        elseif env.problem.x_U[i] < Inf
            env.x[i] = env.problem.x_U[i]
        else
            env.x[i] = 0.0
        end
    end

    for i = 1:env.problem.m
        if env.problem.g_U[i] < Inf && env.problem.g_L[i] > -Inf
            @error "Range constraints are not supposed."
        end
    end

    itercnt = 1

    @printf("%6s  %15s  %15s  %14s  %14s  %14s\n", "iter", "f(x_k)", "ϕ(x_k)", "|E(x_k)|", "|∇f|", "KT resid.")
    while true
        eval_functions!(env)
        norm_violations!(env)

        compute_mu!(env)
        compute_phi!(env)

        err = compute_normalized_Kuhn_Tucker_residuals(env)
        @printf("%6d  %+.8e  %+.8e  %.8e  %.8e  %.8e\n", itercnt, env.f, env.phi, env.norm_E, norm(env.df), err)

        if err <= env.options["epsilon"]
            env.ret = 0;
            break
        end

        # solve LP subproblem
        env.p, lambda, status = solve_lp(env, Δ)
        @assert length(env.lambda) == length(lambda)

        # directional derivative
        compute_derivative!(env)
        if env.directional_derivative > -1.e-6
            env.ret = 0
            break
        end

        # step size computation
        compute_alpha!(env)
        if env.ret == -3
            break
        end

        # Iteration counter limit
        if itercnt >= env.options["max_iter"]
            env.ret = -1
            break
        end
        itercnt += 1

        # update primal/dual points
        env.x += env.alpha .* env.p
        env.lambda += env.alpha .* (lambda - env.lambda)
    end

    env.problem.obj_val = env.problem.eval_f(env.x)
    env.problem.status = Int(env.ret)
    env.problem.x = env.x
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
    return norm(viol, Inf)
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
end

function compute_jacobian_matrix(env::SLP)
	J = spzeros(env.problem.m, env.problem.n)
	for i = 1:length(env.problem.j_str)
		J[env.problem.j_str[i][1], env.problem.j_str[i][2]] += env.dE[i]
    end 
    return J
end

compute_normalized_Kuhn_Tucker_residuals(env::SLP) = compute_normalized_Kuhn_Tucker_residuals(
    env.df, env.lambda, compute_jacobian_matrix(env))
function compute_normalized_Kuhn_Tucker_residuals(df::Vector{Float64}, lambda::Vector{Float64}, J::SparseMatrixCSC{Float64,Int})
    KT_res = norm(df - J' * lambda)
    scalar = max(1.0, norm(df))
    for i = 1:J.m
        scalar = max(scalar, abs(lambda[i]) * norm(J[i,:]))
    end
    return KT_res / scalar
end

function compute_mu!(env::SLP)
    # Update mu only for positive violation
    if env.norm_E > 0
        env.mu = max(
            env.mu,
            env.df' * env.p / (1 - env.options["rho"]) / env.norm_E)
    end
end

function compute_alpha!(env::SLP)
    phi_x_p = compute_phi(env, env.x .+ env.alpha * env.p)

    while phi_x_p > env.phi + env.options["eta"] * env.alpha * env.directional_derivative
        env.alpha *= env.options["tau"]
        phi_x_p = compute_phi(env, env.x + env.alpha * env.p)
        if env.alpha <= env.options["alpha_lb"]
            @warn "Step size too small"
            env.ret = -3
            break
        end
        # @show phi_x_p, env.phi, env.directional_derivative, env.phi + env.options["eta"] * env.alpha * env.directional_derivative
    end
end

# merit function
compute_phi(f::Float64, mu::Float64, norm_E::Float64) = f + mu * norm_E
compute_phi(env::SLP) = compute_phi(env.f, env.mu, env.norm_E)
compute_phi(env::SLP, x::Vector{Float64}) = compute_phi(env.problem.eval_f(x), env.mu, norm_violations(env, x))
function compute_phi!(env::SLP)
    env.phi = compute_phi(env)
end

# directional derivative
compute_derivative(env::SLP) = env.df' * env.p - env.mu * env.norm_E
function compute_derivative!(env::SLP)
    env.directional_derivative = compute_derivative(env)
end
