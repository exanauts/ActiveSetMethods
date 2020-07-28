function SLP_line_search(model::NloptProblem)
    println("############## ------- > n: ", model.n);
    println("############## ------- > n: ", model.m);

    final_objval = [0.0]
    ret = 0;

    num_variables = model.n;
    num_constraints = model.m;
    constraint_lb = model.g_L
    constraint_ub = model.g_U
    jacobian_sparsity = model.j_str


    c_init = spzeros(num_variables+1);
    A = spzeros(num_constraints,num_variables);
    mu = 0.01;
    x = zeros(num_variables)
    p = ones(num_variables)
    df = zeros(num_variables)
    E = zeros(num_constraints)

    dE = zeros(length(jacobian_sparsity))
    lam = zeros(num_constraints)
    plam = zeros(num_constraints)
    lam_ = zeros(num_constraints)

    eta = Options_["eta"];
    tau = Options_["tau"];
    rho = Options_["rho"];




    println("constraint_lb: ", constraint_lb);
    println("constraint_ub: ", constraint_ub);


    alpha = 1;


    println("####---->solveProblem(num_variables): ", num_variables);
    println("####---->solveProblem(num_constraints): ", num_constraints);
    println("####---->solveProblem(jacobian_sparsity): ", jacobian_sparsity);
    println("####---->solveProblem(typeof(jacobian_sparsity): ", typeof(jacobian_sparsity));
    println("####---->solveProblem(size(jacobian_sparsity): ", size(jacobian_sparsity));
    println("####---->solveProblem(length(jacobian_sparsity): ", length(jacobian_sparsity));
    println("####---->solveProblem(jacobian_sparsity[1]): ", jacobian_sparsity[1]);
    #println("####---->solveProblem(jacobian_sparsity[4][1]): ", jacobian_sparsity[4][1]);
    #println("####---->solveProblem(jacobian_sparsity[4][2]): ", jacobian_sparsity[4][2]);

    for i=1:Options_["max_iter"]
        println("-----------------------------> itr: ", i);
        f = model.eval_f(x);
        println("####---->solveProblem(f): ", f);
        df = model.eval_grad_f(x, df)
        println("####---->solveProblem(df): ", df);
        E = model.eval_g(x, E)
        println("####---->solveProblem(E): ", E);
        dE = model.eval_jac_g(x, :opt, [], [], dE)
        println("####---->solveProblem(dE): ", dE);
        mu_nu = df' * p / (1 - rho);
        println("####---->Before solveProblem(mu): ", mu);
        mu_temp = df' * p / (1 - rho) / sum(abs.(E));
        mu = (mu < mu_temp) ? mu_temp : mu;

        # calc_phi(x) = eval_f_cb(x) + mu * sum(abs.(eval_g_cb(x, E)));
        # calc_D1(x) = eval_grad_f_cb(x, df)' * p - mu * sum(abs.(eval_g_cb(x, E)));
        # calc_phi(x,mod_E) = eval_f_cb(x) + mu * mod_E;
        # calc_D1(x,mod_E) = eval_grad_f_cb(x, df)' * p - mu * mod_E;


        println("####---->After solveProblem(mu): ", mu);
        c_init[1:num_variables] .= df;
        c_init[num_variables+1] = f;
        for Ai = 1:length(jacobian_sparsity)
            A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = dE[Ai];
        end
        (p,optimality) = solve_lp(c_init,A,E,constraint_lb,constraint_ub,mu)

        # phi_k1 =
        # phi_k =
        alpha = 1;
        mod_E_x = sum(abs.(model.eval_g(x, E)))
        mod_E_x_p = sum(abs.(model.eval_g(x+alpha * p, E)))
        phi_x = model.eval_merit(x, E, mu);
        phi_x_p = model.eval_merit(x+alpha * p, E, mu);
        D1_x = model.eval_D(x, df, E, mu, p);

        mod_E_x = sum(abs.(model.eval_g(x, E)))
        mod_E_x_p = sum(abs.(model.eval_g(x+alpha * p, E)))

        println("--------------------------> calc_phi(x): ", phi_x)
        println("--------------------------> calc_phi(x+ap): ", phi_x_p)
        println("--------------------------> calc_D1(x): ", D1_x)
        println("--------------------------> |E(x)|: ", mod_E_x)
        println("--------------------------> |E(x+ap)|: ", mod_E_x_p)



        temp_ind = 0
        while((phi_x_p > phi_x + eta * alpha * D1_x) && (alpha > Options_["alpha_lb"]))
            temp_ind+=1;
            # if (phi_x_p > phi_x && mod_E_x_p > mod_E_x)
            #     println("Correction step for Maratos effect");
            #     E_x_p = eval_g_cb(x+alpha*p, E)
            #     for bi = 1:length(jacobian_sparsity)
            #         E_x_p[jacobian_sparsity[bi][1]]-=dE[bi]*p[jacobian_sparsity[bi][2]];
            #     end
            #     (pm,optimality) = solve_lp(model.lp_solver,c_init,A,E_x_p,constraint_lb,constraint_ub,model.sense,mu)
            #     println("p_old: ", p);
            #     println("p_correct: ",pm);
            #     p .= p+pm;
            #     println("p_new: ", p);
            # else
                alpha = alpha * tau;
            # end
            # mod_E_x = sum(abs.(eval_g_cb(x, E)))
            # mod_E_x_p = sum(abs.(eval_g_cb(x+alpha * p, E)))
            phi_x = model.eval_merit(x, E, mu);
            phi_x_p = model.eval_merit(x+alpha * p, E, mu);
            D1_x = model.eval_D(x, df, E, mu, p);
            #println("--------------------------> alpha: ", alpha)
            if (temp_ind>Options_["max_iter_inner"])
                break
            end
        end
        println("-------------------------->after alpha: ", alpha)
        println("####---->solveProblem(p): ", p);
        for j=1:num_constraints
            lam_[j] = df[1]/dE[j];
            plam[j] = lam_[j] - lam[j]
        end
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
        println("p: ", p);
        x .= x + alpha .* p;
        lam .= lam + alpha .* plam;
        println("X: ", x);
        if (sum(abs.(p)) <= Options_["epsilon"])
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
    model.obj_val = final_objval[1]
    model.status = Int(ret)
    #prob.obj_val = eval_f_cb(x);
    model.x = x;
    #println("####---->solveProblem(ret)", ret);

    return Int(ret)
end
