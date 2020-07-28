function Ipopt_method(model::NloptProblem)
    #@eval using Ipopt
    n = model.n;
    m = model.m;
    g_L = model.g_L
    g_U = model.g_U
    x_L = model.x_L
    x_U = model.x_U
    jacobian_sparsity = model.j_str
    eval_f = model.eval_f
    eval_g = model.eval_g
    eval_grad_f = model.eval_grad_f
    eval_jac_g = model.eval_jac_g
    eval_h = model.eval_h
    prob = createProblem(n, x_L, x_U, m, g_L, g_U, length(jacobian_sparsity), 0,
                     eval_f, eval_g, eval_grad_f, eval_jac_g)
    ret = solveProblem(prob);
    return Int(ret)
end
