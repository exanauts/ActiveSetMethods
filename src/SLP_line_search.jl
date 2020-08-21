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
    ret = -5
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
        mu_nu = df' * p / (1 - rho);
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

        (p,p_lam,p_status) = solve_lp(c_init,A,E,model.x_L,model.x_U,constraint_lb,constraint_ub,mu,x)
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
