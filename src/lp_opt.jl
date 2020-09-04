"""
solve_lp(c_init,A,b,x_L,x_U,constraint_lb,constraint_ub,mu,x_hat)
This function solved the LP subproblem of the SLP line search algorithm on
	nonlinear optimization problem defined in model.
c_init has all weights of the objective function. The last elment of it has the
	constant value of the objective function. c_init is a sparse vector
A is a constraint matrix of the LP subproblem. It is a sparse matrix
b is a vector containing the constant values of the constraint matrix A.
	b is a dense vector.
x_L is a vector containing the lower bound of the original variables of the
	nonlinear optimization problem
x_U is a vector containing the upper bound of the original variables of the
	nonlinear optimization problem
constraint_lb is a vector containing the lower bound of the original constraints
 	of the nonlinear optimization problem
constraint_ub is a vector containing the upper bound of the original constraints
 	of the nonlinear optimization problem
mu is a scalar number which defines the peanlty for constraint violations
	Δ is the size of the trust region

The function returns the slution of the LP subproblem variables, status of the
	LP subproblem solution, and duals of the constraints of the LP subproblem. 
	If the problem is dual_infeasible or has not constraints, the duals of the c
	onstraints will be an empty array.
"""

function solve_lp(
	c_init::SparseVector{Float64,Int},
	A::SparseMatrixCSC{Float64,Int},
	b::Vector{Float64},
	x_L::Vector{Float64},
	x_U::Vector{Float64},
	constraint_lb::Vector{Float64},
	constraint_ub::Vector{Float64},
	mu::Float64,
	x_k::Vector{Float64},
	Δ::Float64,
	model::MOI.AbstractOptimizer)

	# empty optimizer just in case
	MOI.empty!(model)

	n = A.n;
	m = A.m;
	@assert n > 0
	@assert m >= 0

	@assert length(c_init) == n+1
	c = c_init[1:n];
	c0 = c_init[n+1];
	
	#TODO all varialbes are defined as x ... change it to p so it is consistant with 
	# the algorithm
	# Variables are defined as a vector named x
	x = MOI.add_variables(model, n)
	
	# terms is defined for the objective function
	terms = Array{MOI.ScalarAffineTerm{Float64},1}();
	for (ind, val) in zip(c.nzind, c.nzval)
		push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
	end

	# Slacks v and u are added only for constrained problems.
	if m > 0
		u = MOI.add_variables(model, m)
		v = MOI.add_variables(model, m)
		append!(terms, MOI.ScalarAffineTerm.(mu, u));
		append!(terms, MOI.ScalarAffineTerm.(mu, v));
	end

	MOI.set(model,
		MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
		MOI.ScalarAffineFunction(terms, c0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	@assert length(x_L) == n
	@assert length(x_U) == n

	# Add a dummy trust-region to all variables
	constr_x_U = MOI.ConstraintIndex[]
	constr_x_L = MOI.ConstraintIndex[]
	for i = 1:n
		ub = min(Δ, x_U[i] - x_k[i])
		lb = max(-Δ, x_L[i] - x_k[i])
		push!(constr_x_U, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)))
		push!(constr_x_L, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)))
	end
	@assert length(constr_x_U) == n
	@assert length(constr_x_L) == n
	@assert length(constraint_lb) == m
	@assert length(constraint_ub) == m
	
	# constr is used to retrieve the dual variable of the constraints after solution
	constr = MOI.ConstraintIndex[]
	for i=1:m
		Ai = A[i,:];
		terms = Array{MOI.ScalarAffineTerm{Float64},1}();
		# u_term = Array{MOI.ScalarAffineTerm{Float64},1}();
		# v_term = Array{MOI.ScalarAffineTerm{Float64},1}();
		for (ind, val) in zip(Ai.nzind, Ai.nzval)
			push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
		end
		# push!(u_term, MOI.ScalarAffineTerm{Float64}(1.0, u[i]));
		# push!(v_term, MOI.ScalarAffineTerm{Float64}(1.0, v[i]));

		u_term = MOI.ScalarAffineTerm{Float64}(1.0, u[i]);
		v_term = MOI.ScalarAffineTerm{Float64}(1.0, v[i]);;
		push!(terms, MOI.ScalarAffineTerm{Float64}(1.0, u[i]));
		push!(terms, MOI.ScalarAffineTerm{Float64}(-1.0, v[i]));
		if constraint_lb[i] == constraint_ub[i] #This means the constraint is equality
			constr1 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.EqualTo(constraint_lb[i]));
		elseif constraint_lb[i] != -Inf && constraint_ub[i] != Inf && constraint_lb[i] < constraint_ub[i]
			constr2 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.GreaterThan(constraint_lb[i]));
			constr1 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.LessThan(constraint_ub[i]));
			push!(constr, constr2);
		elseif constraint_lb[i] != -Inf
			constr1 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.GreaterThan(constraint_lb[i]));
		elseif constraint_ub[i] != Inf
			constr1 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.LessThan(constraint_ub[i]));
		end
		push!(constr, constr1);
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction([u_term], 0.0), MOI.GreaterThan(0.0));
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction([v_term], 0.0), MOI.GreaterThan(0.0));
	end

	MOI.optimize!(model);
	status = MOI.get(model, MOI.TerminationStatus())

	Xsol = zeros(n);
	lambda = zeros(m)
	mult_x_U = zeros(n)
	mult_x_L = zeros(n)

	if status == MOI.OPTIMAL
		Xsol .= MOI.get(model, MOI.VariablePrimal(), x);
		if m > 0
			Usol = MOI.get(model, MOI.VariablePrimal(), u);
			Vsol = MOI.get(model, MOI.VariablePrimal(), v);
		end

		# extract the multipliers to constraints
		ci = 1
		for i=1:m
			lambda[i] = MOI.get(model, MOI.ConstraintDual(1), constr[ci])
			ci += 1
			if constraint_lb[i] > -Inf && constraint_ub[i] < Inf && constraint_lb[i] < constraint_ub[i]
				lambda[i] += MOI.get(model, MOI.ConstraintDual(1), constr[ci])
				ci += 1
			end
		end

		# extract the multipliers to column bounds
		mult_x_U = MOI.get(model, MOI.ConstraintDual(1), constr_x_U)
		mult_x_L = MOI.get(model, MOI.ConstraintDual(1), constr_x_L)
		# careful because of the trust region
		for j=1:n
			if Xsol[j] < x_U[j] - x_k[j]
				mult_x_U[j] = 0.0
			end
			if Xsol[j] > x_L[j] - x_k[j]
				mult_x_L[j] = 0.0
			end
		end
	elseif status == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected status: $(status)"
	end

	return Xsol, lambda, mult_x_U, mult_x_L, status
end

solve_lp(env::SLP, Δ) = solve_lp(
	sparse([env.df; env.f]),
	compute_jacobian_matrix(env),
	env.E,
	env.problem.x_L,
	env.problem.x_U,
	env.problem.g_L,
	env.problem.g_U,
	env.mu,
	env.x,
	Δ,
	env.options.external_optimizer)
