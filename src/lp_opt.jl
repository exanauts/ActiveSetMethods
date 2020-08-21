"""
solve_lp(c_init,A,b,x_L,x_U,constraint_lb,constraint_ub,mu,x_hat)
This function solved the LP subproblem of the SLP line search algorithm on
	nonlinear optimization problem defined in model.
c_init has all weights of the objective function. The last elment of it has the
	constant value of the objective function
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
x_hat is the estinmated values of the original variables of the nonlinear
	optimization problem

The function returns the slution of the LP subproblem variables, status of the
	LP subproblem solution, and duals of the constraints of the LP subproblem
"""

function solve_lp(c_init,A,b,x_L,x_U,constraint_lb,constraint_ub,mu,x_hat)

	model = Options_["LP_solver"]()
	n = A.n;
	m = A.m;
	@assert n > 0
	@assert m > 0

	c = c_init[1:n];

	c0 = c_init[n+1];

	x = MOI.add_variables(model, n)
	u = MOI.add_variables(model, m)
	v = MOI.add_variables(model, m)
	terms = Array{MOI.ScalarAffineTerm{Float64},1}();
	for (ind, val) in zip(c.nzind, c.nzval)
		push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
	end
	append!(terms, MOI.ScalarAffineTerm.(mu, u));
	append!(terms, MOI.ScalarAffineTerm.(mu, v));
	MOI.set(model,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(terms, c0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	@assert length(x_L) == n
	@assert length(x_U) == n
	for i=1:n
		term = MOI.ScalarAffineTerm{Float64}(1.0, x[i]);

		if x_L[i] != -Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction([term], x_hat[i]), MOI.GreaterThan(x_L[i]));
		end
		if x_U[i] != Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction([term], x_hat[i]), MOI.LessThan(x_U[i]));
		end
	end

	@assert length(constraint_lb) == m
	@assert length(constraint_ub) == m
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
		if constraint_lb[i] == constraint_ub[i]
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
	status = MOI.get(model, MOI.TerminationStatus());
	if Options_["mode"] == "Debug"
		print(model);
		println("Status: ", status);
	end

	# statusPrimal = MOI.get(model, MOI.PrimalStatus());
	# println("StatusPrimal: ", statusPrimal);
	#
	# statusDual = MOI.get(model, MOI.DualStatus());
	# println("StatusDual: ", statusDual);
	#
	# Pobj = MOI.get(model, MOI.ObjectiveValue());
	# println("Pobj: ", Pobj);
	#
	# Dobj = MOI.get(model, MOI.DualObjectiveValue());
	# println("Dobj: ", Dobj);

	#Dconst = MOI.get(model, MOI.ConstraintPrimal());
	#println("Dconst: ", Dconst);



	Xsol = zeros(n);
	norm_E = 0.0
	lambda = []


	if status == MOI.OPTIMAL
		Xsol = MOI.get(model, MOI.VariablePrimal(), x);
		Usol = MOI.get(model, MOI.VariablePrimal(), u);
		Vsol = MOI.get(model, MOI.VariablePrimal(), v);
		lambda = MOI.get(model, MOI.ConstraintDual(1), constr);
		if Options_["mode"] == "Debug"
			println("Xsol: ", Xsol);
			println("Usol: ", Usol);
			println("Vsol: ", Vsol);
			println("Dual Constraints constr: ", MOI.get(model, MOI.ConstraintDual(1), constr))
		end
	elseif status == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected status: $(status)"
	end

	# if MOI.get(model, MOI.DualStatus()) == FEASIBLE_POINT
	# 	println("Dual Constraints constr: ", MOI.get(model, MOI.ConstraintDual(1), constr))
	# end


	# println("Dual Status: ", MOI.get(model, MOI.DualStatus()))

	return(Xsol,lambda,status)
end
