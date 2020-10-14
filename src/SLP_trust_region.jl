function trust_region_method(env::SLP)

    #Δ = env.options.tr_size

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
        eval_violations!(env)
    
        # solve LP subproblem (to initialize dual multipliers)
        env.p, env.lambda, env.mult_x_U, env.mult_x_L, status = solve_lp(env)
        # @show env.lambda
    
        compute_nu!(env)
        compute_phi2!(env)

        #=if env.iter == 1
            env.lambda .= lambda
            env.mult_x_U .= mult_x_U
            env.mult_x_L .= mult_x_L
        end=#
        #println("phi: ", env.phi)

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

        # step change check
        if norm(env.p, Inf) < env.options.tol_infeas && env.norm_vio
        	@error "Failed to find a step size"
        	env.ret = -3
        elseif norm(env.p, Inf) < env.options.tol_infeas
        	@printf("Terminated: The step-size has reached the user-specified accuracy (%e)\n", norm(env.p, Inf))
            	env.ret = 0
            	break
        end

        # step size computation
        is_valid_step = eval_step(env)
        #=if env.ret == -3
            @warn "Failed to find a step size"
            break
        end=#
        if is_valid_step
            env.x += env.p
        end
        # @show env.alpha

        # update primal points
        #env.x += env.alpha .* env.p

        # update multipliers
        #env.lambda += env.alpha .* (lambda - env.lambda)
        #env.mult_x_U += env.alpha .* (mult_x_U - env.mult_x_U)
        #env.mult_x_L += env.alpha .* (mult_x_L - env.mult_x_L)
        # @show env.lambda
        # @show env.mult_x_U
        # @show env.mult_x_L

        # Iteration counter limit
        if env.iter >= env.options.max_iter
            env.ret = -1
            if norm(eval_violations(env, env.x),1) <= env.options.tol_infeas
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

function compute_nu!(env::SLP)
    # Update nu if it is first iteration
    if env.iter == 1
    	for i = 1:length(env.problem.j_str)
		env.nu[env.problem.j_str[i][1]] += (env.dE[i])^2
    	end 
    	env.nu .= max.(sqrt.(env.nu),1);
    	env.nu .= max.(norm(env.df) ./ env.nu,1)
    else
    	env.nu .= max.(env.nu,[env.lambda;env.mult_x_L+env.mult_x_U]);
    end
    # @show env.nu
end

# merit function2
compute_phi2(f::Float64, nu::Vector{Float64}, vio::Vector{Float64})::Float64 = f + nu' * vio
compute_phi2(env::SLP)::Float64 = compute_phi2(env.f, env.nu, env.vio)
compute_phi2(env::SLP, x::Vector{Float64})::Float64 = compute_phi2(env.problem.eval_f(x), env.nu, eval_violations(env, x))
function compute_phi2!(env::SLP)
    env.phi = compute_phi2(env)
end


function eval_step(env::SLP)::Bool
    is_valid = true

    phi_pre = env.df' * env.p
    phi_act = compute_phi2(env, env.x .+ env.p) - compute_phi2(env, env.x)
    rho_k = phi_act / phi_pre;
    if rho_k <= 0.0
    	env.Δ = env.options.eta * env.Δ;
    	is_valid = false
    elseif rho_k <= 0.25 && rho_k > 0.0
    	env.Δ = env.options.tau * env.Δ;
    elseif rho_k <= 0.75 && rho_k > 0.25
    	env.Δ = env.Δ;
    else rho_k > 0.75
    	env.Δ = min(2 * env.Δ,env.options.tr_max);
    end
    return is_valid
end
