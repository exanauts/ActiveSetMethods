function solve_lp(c_init,A,b,x_L,x_U,constraint_lb,constraint_ub,mu,x_hat)

	model = Options_["LP_solver"]()
	n = A.n;
	m = A.m;

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

	for i=1:n
		term = MOI.ScalarAffineTerm{Float64}(1.0, x[i]);

		if constraint_lb[i] != -Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction([term], x_hat[i]), MOI.GreaterThan(x_L[i]));
		end
		if constraint_ub[i] != Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction([term], x_hat[i]), MOI.LessThan(x_U[i]));
		end
	end


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
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.EqualTo(constraint_lb[i]));
		elseif constraint_lb[i] != -Inf && constraint_ub[i] != Inf && constraint_lb[i] < constraint_ub[i]
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.GreaterThan(constraint_lb[i]));
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.LessThan(constraint_ub[i]));
		elseif constraint_lb[i] != -Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.GreaterThan(constraint_lb[i]));
		elseif constraint_ub[i] != Inf
			MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.LessThan(constraint_ub[i]));
		end
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction([u_term], 0.0), MOI.GreaterThan(0.0));
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction([v_term], 0.0), MOI.GreaterThan(0.0));
	end


	MOI.optimize!(model);
	status = MOI.get(model, MOI.TerminationStatus());
	print(model);
	println("Status: ", status);

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


	if status == MOI.OPTIMAL
		Xsol = MOI.get(model, MOI.VariablePrimal(), x);
		println("Xsol: ", Xsol);
		Usol = MOI.get(model, MOI.VariablePrimal(), u);
		println("Usol: ", Usol);
		Vsol = MOI.get(model, MOI.VariablePrimal(), v);
		println("Vsol: ", Vsol);
		# # Xdual = MOI.get(model, MOI.DualObjectiveValue(), con1);
		# # println(Xdual);
		# norm_E = sum(Usol+Vsol)
	end

	return(Xsol,status)
end
