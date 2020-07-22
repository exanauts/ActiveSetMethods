module Slopt
export slopt_status
using Libdl
using LinearAlgebra
include("lp_opt.jl")

if VERSION < v"1.3" || (haskey(ENV, "JULIA_IPOPT_LIBRARY_PATH") && haskey(ENV, "JULIA_IPOPT_EXECUTABLE_PATH"))
    if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
        include("../deps/deps.jl")
    else
        error("Ipopt not properly installed. Please run import Pkg; Pkg.build(\"Ipopt\")")
    end
    const libipopt_path = libipopt
    const amplexe_path = amplexe

    function amplexefun(arguments::String)
        temp_env = copy(ENV)
        for var in Ipopt.amplexe_env_var
            temp_env[var] = Ipopt.amplexe_env_val
        end
        temp_dir = abspath(dirname(Ipopt.amplexe))
        proc = run(pipeline(Cmd(`$(Ipopt.amplexe) $arguments`,env=temp_env,dir=temp_dir), stdout=stdout))
        wait(proc)
        kill(proc)
        proc.exitcode
    end
else
    import Ipopt_jll: libipopt, libipopt_path, amplexe, amplexe_path

    function amplexefun(arguments::String)
        # temp_env = copy(ENV)
        # for var in Ipopt.amplexe_env_var
        #     temp_env[var] = Ipopt.amplexe_env_val
        # end
        temp_dir = abspath(dirname(amplexe_path))
        proc = amplexe() do amplexe_path
          run(pipeline(Cmd(`$amplexe_path $arguments`,dir=temp_dir), stdout=stdout))
        end
        wait(proc)
        kill(proc)
        proc.exitcode
    end
end

export createProblem, addOption
export openOutputFile, setProblemScaling, setIntermediateCallback
export solveProblem
export SloptProblem

function __init__()
    julia_libdir = joinpath(dirname(first(filter(x -> occursin("libjulia", x), Libdl.dllist()))), "julia")
    julia_bindir = Sys.BINDIR
    ipopt_libdir = libipopt_path |> dirname
    pathsep = Sys.iswindows() ? ';' : ':'
    @static if Sys.isapple()
        global amplexe_env_var = ["DYLD_LIBRARY_PATH"]
        global amplexe_env_val = "$(julia_libdir)$(pathsep)$(get(ENV,"DYLD_LIBRARY_PATH",""))"
    elseif Sys.islinux()
        global amplexe_env_var = ["LD_LIBRARY_PATH"]
        global amplexe_env_val = "$(julia_libdir)$(pathsep)$(get(ENV,"LD_LIBRARY_PATH",""))"
    elseif Sys.iswindows()
        # for some reason windows sometimes needs Path instead of PATH
        global amplexe_env_var = ["PATH","Path","path"]
        global amplexe_env_val = "$(julia_bindir)$(pathsep)$(get(ENV,"PATH",""))"
    end

    # Still need this for AmplNLWriter to work until it uses amplexefun defined above
    # (amplexefun wraps the call to the binary and doesn't leave environment variables changed.)
    @static if Sys.isapple()
         ENV["DYLD_LIBRARY_PATH"] = string(get(ENV, "DYLD_LIBRARY_PATH", ""), ":", julia_libdir)
    elseif Sys.islinux()
         ENV["LD_LIBRARY_PATH"] = string(get(ENV, "LD_LIBRARY_PATH", ""), ":", julia_libdir, ":", ipopt_libdir)
    end
end

include("struct.jl")



###########################################################################
# Callback wrappers
###########################################################################
# Objective (eval_f)
function eval_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint,
                        obj_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    # Calculate the new objective
    new_obj = convert(Float64,
                      prob.eval_f(unsafe_wrap(Array,x_ptr, Int(n))))::Float64
    # Fill out the pointer
    unsafe_store!(obj_ptr, new_obj)
    # Done
    #println("--> eval_f_wrapper (prob): ", prob);
    #println("--> eval_f_wrapper (new_obj): ", new_obj);
    return Int32(1)
end

# Constraints (eval_g)
function eval_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint,
                        m::Cint, g_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    # Calculate the new constraint values
    new_g = unsafe_wrap(Array,g_ptr, Int(m))
    prob.eval_g(unsafe_wrap(Array,x_ptr, Int(n)), new_g)
    # Done
    #println("--> eval_g_wrapper (prob): ", prob);
    #println("--> eval_g_wrapper (new_g): ", new_g);
    return Int32(1)
end

# Objective gradient (eval_grad_f)
function eval_grad_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint,
                             grad_f_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    # Calculate the gradient
    new_grad_f = unsafe_wrap(Array,grad_f_ptr, Int(n))
    prob.eval_grad_f(unsafe_wrap(Array,x_ptr, Int(n)), new_grad_f)
    # Done
    #println("--> eval_grad_f_wrapper (prob): ", prob);
    #println("--> eval_grad_f_wrapper (new_grad_f): ", new_grad_f);
    return Int32(1)
end

# Jacobian (eval_jac_g)
function eval_jac_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, m::Cint,
                            nele_jac::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint},
                            values_ptr::Ptr{Float64}, user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    rows = unsafe_wrap(Array, iRow, Int(nele_jac))
    cols = unsafe_wrap(Array,jCol, Int(nele_jac))
    values = unsafe_wrap(Array,values_ptr, Int(nele_jac))
    prob.eval_jac_g(x, mode, rows, cols, values)
    # Done
    #println("--> eval_jac_g_wrapper (prob): ", prob);
    #println("--> eval_jac_g_wrapper (x): ", x);
    #println("--> eval_jac_g_wrapper (mode): ", mode);
    #println("--> eval_jac_g_wrapper (rows): ", rows);
    #println("--> eval_jac_g_wrapper (cols): ", cols);
    #println("--> eval_jac_g_wrapper (values): ", values);
    return Int32(1)
end

# Hessian
function eval_h_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint,
                        obj_factor::Float64, m::Cint, lambda_ptr::Ptr{Float64},
                        new_lambda::Cint, nele_hess::Cint, iRow::Ptr{Cint},
                        jCol::Ptr{Cint}, values_ptr::Ptr{Float64},
                        user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    # Did the user specify a Hessian
    if prob.eval_h === nothing
        # No Hessian provided
        return Int32(0)
    else
        # Determine mode
        mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
        x = unsafe_wrap(Array,x_ptr, Int(n))
        lambda = unsafe_wrap(Array,lambda_ptr, Int(m))
        rows = unsafe_wrap(Array,iRow, Int(nele_hess))
        cols = unsafe_wrap(Array,jCol, Int(nele_hess))
        values = unsafe_wrap(Array,values_ptr, Int(nele_hess))
        prob.eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        # Done
        return Int32(1)
    end
end

# Intermediate
function intermediate_wrapper(alg_mod::Cint, iter_count::Cint,
                              obj_value::Float64, inf_pr::Float64,
                              inf_du::Float64, mu::Float64, d_norm::Float64,
                              regularization_size::Float64, alpha_du::Float64,
                              alpha_pr::Float64, ls_trials::Cint,
                              user_data::Ptr{Cvoid})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::SloptProblem
    keepgoing = prob.intermediate(Int(alg_mod), Int(iter_count), obj_value,
                                  inf_pr, inf_du, mu, d_norm,
                                  regularization_size, alpha_du, alpha_pr,
                                  Int(ls_trials))
    # Done
    return keepgoing ? Int32(1) : Int32(0)
end

###########################################################################
# C function wrappers
###########################################################################
function createProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
    m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
    nele_jac::Int, nele_hess::Int,
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing)

    #println("--> createProblem (n): ", n);
    #println("--> createProblem (x_L): ", x_L);
    #println("--> createProblem (x_U): ", x_U);
    #println("--> createProblem (m): ", m);
    #println("--> createProblem (g_L): ", g_L);
    #println("--> createProblem (g_U): ", g_U);
    #println("--> createProblem (nele_jac): ", nele_jac);
    #println("--> createProblem (nele_hess): ", nele_hess);
    #println("--> createProblem (eval_f): ", eval_f);
    #println("--> createProblem (eval_g): ", eval_g);
    #println("--> createProblem (eval_grad_f): ", eval_grad_f);
    #println("--> createProblem (eval_jac_g): ", eval_jac_g);
    #println("--> createProblem (eval_h): ", eval_h);

    @assert n == length(x_L) == length(x_U)
    @assert m == length(g_L) == length(g_U)
    # Wrap callbacks
    eval_f_cb = @cfunction(eval_f_wrapper, Cint,
    (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}))
    eval_g_cb = @cfunction(eval_g_wrapper, Cint,
    (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cvoid}))
    eval_grad_f_cb = @cfunction(eval_grad_f_wrapper, Cint,
    (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}))
    eval_jac_g_cb = @cfunction(eval_jac_g_wrapper, Cint,
    (Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Cvoid}))
    eval_h_cb = @cfunction(eval_h_wrapper, Cint,
    (Cint, Ptr{Float64}, Cint, Float64, Cint, Ptr{Float64}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Cvoid}))

    ret = ccall((:CreateIpoptProblem, libipopt), Ptr{Cvoid},
    (Cint, Ptr{Float64}, Ptr{Float64},  # Num vars, var lower and upper bounds
    Cint, Ptr{Float64}, Ptr{Float64},  # Num constraints, con lower and upper bounds
    Cint, Cint,                        # Num nnz in constraint Jacobian and in Hessian
    Cint,                              # 0 for C, 1 for Fortran
    Ptr{Cvoid}, Ptr{Cvoid},              # Callbacks for eval_f, eval_g
    Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),  # Callbacks for eval_grad_f, eval_jac_g, eval_h
    n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, 1,
    eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)

    if ret == C_NULL
        error("SLOPT: Failed to construct problem.")
    else
        return SloptProblem(ret, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end

# TODO: Not even expose this? Seems dangerous, should just destruct
# the SloptProblem object via GC
function freeProblem(prob::SloptProblem)
    if prob.ref != C_NULL
        ccall((:FreeIpoptProblem, libipopt), Cvoid, (Ptr{Cvoid},), prob.ref)
        prob.ref = C_NULL
    end
end


function addOption(prob::SloptProblem, keyword::String, value::String)
    #/** Function for adding a string option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !(isascii(keyword) && isascii(value))
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall((:AddIpoptStrOption, libipopt),
    Cint, (Ptr{Cvoid}, Ptr{UInt8}, Ptr{UInt8}),
    prob.ref, keyword, value)
    if ret == 0
        error("SLOPT: Couldn't set option '$keyword' to value '$value'.")
    end
end


function addOption(prob::SloptProblem, keyword::String, value::Float64)
    #/** Function for adding a Number option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !isascii(keyword)
        error("SLOPT: Non ASCII parameters not supported")
    end
    ret = ccall((:AddIpoptNumOption, libipopt),
    Cint, (Ptr{Cvoid}, Ptr{UInt8}, Float64),
    prob.ref, keyword, value)
    if ret == 0
        error("SLOPT: Couldn't set option '$keyword' to value '$value'.")
    end
end


function addOption(prob::SloptProblem, keyword::String, value::Integer)
    #/** Function for adding an Int option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall((:AddIpoptIntOption, libipopt),
    Cint, (Ptr{Cvoid}, Ptr{UInt8}, Cint),
    prob.ref, keyword, value)
    if ret == 0
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
end


function openOutputFile(prob::SloptProblem, file_name::String, print_level::Int)
    #/** Function for opening an output file for a given name with given
    # *  printlevel.  Returns false, if there was a problem opening the
    # *  file. */
    if !isascii(file_name)
        error("SLOPT: Non ASCII parameters not supported")
    end
    ret = ccall((:OpenIpoptOutputFile, libipopt),
    Cint, (Ptr{Cvoid}, Ptr{UInt8}, Cint),
    prob.ref, file_name, print_level)
    if ret == 0
        error("SLOPT: Couldn't open output file.")
    end
end

# TODO: Verify this function even works! Trying it with 0.5 on HS071
# seems to change nothing.
function setProblemScaling(prob::SloptProblem, obj_scaling::Float64,
    x_scaling = nothing,
    g_scaling = nothing)
    #/** Optional function for setting scaling parameter for the NLP.
    # *  This corresponds to the get_scaling_parameters method in TNLP.
    # *  If the pointers x_scaling or g_scaling are NULL, then no scaling
    # *  for x resp. g is done. */
    x_scale_arg = (x_scaling == nothing) ? C_NULL : x_scaling
    g_scale_arg = (g_scaling == nothing) ? C_NULL : g_scaling
    ret = ccall((:SetIpoptProblemScaling, libipopt),
    Cint, (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
    prob.ref, obj_scaling, x_scale_arg, g_scale_arg)
    if ret == 0
        error("SLOPT: Error setting problem scaling.")
    end
end


function setIntermediateCallback(prob::SloptProblem, intermediate::Function)
    intermediate_cb = @cfunction(intermediate_wrapper, Cint,
    (Cint, Cint, Float64, Float64, Float64, Float64,
    Float64, Float64, Float64, Float64, Cint, Ptr{Cvoid}))
    ret = ccall((:SetIntermediateCallback, libipopt), Cint,
    (Ptr{Cvoid}, Ptr{Cvoid}), prob.ref, intermediate_cb)
    prob.intermediate = intermediate
    if ret == 0
        error("SLOPT: Something went wrong setting the intermediate callback.")
    end
end

function solveProblem1(model::SloptProblem)
    println("############## ------- > n: ", model.n);
    println("############## ------- > n: ", model.m);

    final_objval = [0.0]
    ret = 0;

    num_variables = model.n;
    num_constraints = model.m;

    c_init = spzeros(num_variables+1);
    A = spzeros(num_constraints,num_variables);
    mu = 0.01;
    x = zeros(num_variables)
    p = ones(num_variables)
    df = zeros(num_variables)
    E = zeros(num_constraints)


    # constraint_lb, constraint_ub = constraint_bounds(model)
    # println("Sense: ", model.sense);
    # println("constraint_lb: ", constraint_lb);
    # println("constraint_ub: ", constraint_ub);
    # c_init = spzeros(num_variables+1);
    # A = spzeros(num_constraints,num_variables);
    # mu = 0.01;
    # x = zeros(num_variables)
    # p = ones(num_variables)
    # df = zeros(num_variables)
    # E = zeros(num_constraints)
    # dE = zeros(length(jacobian_sparsity))
    # lam = zeros(num_constraints)
    # plam = zeros(num_constraints)
    # lam_ = zeros(num_constraints)
    # alpha = 1;
    # eta = model.options["eta"];
    # tau = model.options["tau"];
    # rho = model.options["rho"];
    #
    # println("####---->solveProblem(num_variables): ", num_variables);
    # println("####---->solveProblem(num_constraints): ", num_constraints);
    # println("####---->solveProblem(jacobian_sparsity): ", jacobian_sparsity);
    # println("####---->solveProblem(typeof(jacobian_sparsity): ", typeof(jacobian_sparsity));
    # println("####---->solveProblem(size(jacobian_sparsity): ", size(jacobian_sparsity));
    # println("####---->solveProblem(length(jacobian_sparsity): ", length(jacobian_sparsity));
    # println("####---->solveProblem(jacobian_sparsity[1]): ", jacobian_sparsity[1]);
    #
    # for i=1:model.options["max_iter"]
    #     println("-----------------------------> itr: ", i);
    #     f = eval_f_cb(x);
    #     println("####---->solveProblem(f): ", f);
    #     df = eval_grad_f_cb(x, df)
    #     println("####---->solveProblem(df): ", df);
    #     E = eval_g_cb(x, E)
    #     println("####---->solveProblem(E): ", E);
    #     dE = eval_constraint_jacobian(model, dE, x)
    #     println("####---->solveProblem(dE): ", dE);
    #     mu_nu = df' * p / (1 - model.options["rho"]);
    #     println("####---->Before solveProblem(mu): ", mu);
    #     mu_temp = df' * p / (1 - model.options["rho"]) / sum(abs.(E));
    #     mu = (mu < mu_temp) ? mu_temp : mu;
    #
    #     calc_phi(x) = eval_f_cb(x) + mu * sum(abs.(eval_g_cb(x, E)));
    #     calc_D1(x) = eval_grad_f_cb(x, df)' * p - mu * sum(abs.(eval_g_cb(x, E)));
    #     calc_phi(x,mod_E) = eval_f_cb(x) + mu * mod_E;
    #     calc_D1(x,mod_E) = eval_grad_f_cb(x, df)' * p - mu * mod_E;
    #
    #
    #     println("####---->After solveProblem(mu): ", mu);
    #     c_init[1:num_variables] .= df;
    #     c_init[num_variables+1] = f;
    #     for Ai = 1:length(jacobian_sparsity)
    #         A[jacobian_sparsity[Ai][1],jacobian_sparsity[Ai][2]] = dE[Ai];
    #     end
    #     (p,optimality) = solve_lp(model.lp_solver,c_init,A,E,constraint_lb,constraint_ub,model.sense,mu)
    #
    #     # phi_k1 =
    #     # phi_k =
    #     alpha = 1;
    #     mod_E_x = sum(abs.(eval_g_cb(x, E)))
    #     mod_E_x_p = sum(abs.(eval_g_cb(x+alpha * p, E)))
    #     phi_x = calc_phi(x,mod_E_x);
    #     phi_x_p = calc_phi(x+alpha * p, mod_E_x_p);
    #     D1_x = calc_D1(x,mod_E_x);
    #
    #     mod_E_x = sum(abs.(eval_g_cb(x, E)))
    #     mod_E_x_p = sum(abs.(eval_g_cb(x+alpha * p, E)))
    #
    #     println("--------------------------> calc_phi(x): ", phi_x)
    #     println("--------------------------> calc_phi(x+ap): ", phi_x_p)
    #     println("--------------------------> calc_D1(x): ", D1_x)
    #     println("--------------------------> |E(x)|: ", mod_E_x)
    #     println("--------------------------> |E(x+ap)|: ", mod_E_x_p)



        # temp_ind = 0
        # while((phi_x_p > phi_x + eta * alpha * D1_x) && (alpha > model.options["alpha_lb"]))
        #     temp_ind+=1;
        #     if (phi_x_p > phi_x && mod_E_x_p > mod_E_x)
        #         println("Correction step for Maratos effect");
        #         E_x_p = eval_g_cb(x+alpha*p, E)
        #         for bi = 1:length(jacobian_sparsity)
        #             E_x_p[jacobian_sparsity[bi][1]]-=dE[bi]*p[jacobian_sparsity[bi][2]];
        #         end
        #         (pm,optimality) = solve_lp(model.lp_solver,c_init,A,E_x_p,constraint_lb,constraint_ub,model.sense,mu)
        #         println("p_old: ", p);
        #         println("p_correct: ",pm);
        #         p .= p+pm;
        #         println("p_new: ", p);
        #     else
        #         alpha = alpha * tau;
        #     end
        #     mod_E_x = sum(abs.(eval_g_cb(x, E)))
        #     mod_E_x_p = sum(abs.(eval_g_cb(x+alpha * p, E)))
        #     phi_x = calc_phi(x,mod_E_x);
        #     phi_x_p = calc_phi(x+alpha * p, mod_E_x_p);
        #     D1_x = calc_D1(x,mod_E_x);
        #     #println("--------------------------> alpha: ", alpha)
        #     if (temp_ind>5)
        #         break
        #     end
        # end
        # println("-------------------------->after alpha: ", alpha)
        # println("####---->solveProblem(p): ", p);
        # for j=1:num_constraints
        #     lam_[j] = df[1]/dE[j];
        #     plam[j] = lam_[j] - lam[j]
        # end
        # println("p: ", p);
        # x .= x + alpha .* p;
        # lam .= lam + alpha .* plam;
        # println("X: ", x);
        # if (sum(abs.(p))==0)
        #     break;
        # end
end


#=function solveProblem(model::Optimizer)
    prob = model.inner
    final_objval = [0.0]
    ret = 0;
    ret = ccall((:IpoptSolve, libipopt),
    Cint, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Any),
    prob.ref, prob.x, prob.g, final_objval, prob.mult_g, prob.mult_x_L, prob.mult_x_U, prob)
    #=
    println("####---->solveProblem(prob.ref): ", prob.ref);
    println("####---->solveProblem(typeof(prob.ref)): ", typeof(prob.ref));
    println("####---->solveProblem(prob.x): ", prob.x);
    println("####---->solveProblem(typeof(prob.x)): ", typeof(prob.x));
    println("####---->solveProblem(prob.g): ", prob.g);
    println("####---->solveProblem(typeof(prob.g)): ", typeof(prob.g));
    println("####---->solveProblem(final_objval): ", final_objval);
    println("####---->solveProblem(typeof(final_objval)): ", typeof(final_objval));
    println("####---->solveProblem(prob.mult_g): ", prob.mult_g);
    println("####---->solveProblem(typeof(prob.mult_g)): ", typeof(prob.mult_g));
    println("####---->solveProblem(prob.mult_x_L): ", prob.mult_x_L);
    println("####---->solveProblem(typeof(prob.mult_x_L)): ", typeof(prob.mult_x_L));
    println("####---->solveProblem(prob.mult_x_U): ", prob.mult_x_U);
    println("####---->solveProblem(typeof(prob.mult_x_U)): ", typeof(prob.mult_x_U));=#
    a = eval_objective(model, [4])
    #a = prob.eval_f_cb(4);
    println("####---->solveProblem(a): ", a);
    println("####---->solveProblem(prob): ", prob);
    prob.obj_val = final_objval[1]
    prob.status = Int(ret)
    println("####---->solveProblem(ret)", ret);

    return Int(ret)
 end =#

include("MOI_wrapper.jl")
#include("MPB_wrapper.jl")

end
