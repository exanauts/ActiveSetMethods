#=
The model solves the following minimization problem.

      min   cx
      s.b.  A1x + b1 = 0
            A2x + b2 <= 0

x  -> n x 1
c -> 1 x n
A1 -> m1 x n
b1 -> m1 x 1
A2 -> m2 x n
b1 -> m2 x 1

n=4;
m1=3;
m2=2;

c = rand(n);
A1 = rand(m1,n);
b1 = rand(m1);
A2 = rand(m2,n);
b2 = rand(m2);


c = sprand(Float64, 6, 0.75);
c0 = 2.3;
A2 = sprand(Float64, 4, 6, 0.0);
A1 =  sparse([1,2,3,4,5,6],[1,2,3,4,5,6],[1.0,1.0,1.0,1.0,1.0,1.0])
b1 = sprand(Float64, 6, 1.0);
b2 = sprand(Float64, 4, 0.0)
=#

function solve_lp(solver,c_init,A,b,constraint_lb,constraint_ub,sense)

	model = solver()
	n = A.n;
	m = A.m;

	c = c_init[1:n];

	c0 = c_init[n+1];

	x = MOI.add_variables(model, n)
	terms = Array{MOI.ScalarAffineTerm{Float64},1}();
	for (ind, val) in zip(c.nzind, c.nzval)
		push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
	end
	MOI.set(model,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(terms, c0))
	MOI.set(model, MOI.ObjectiveSense(), sense)
	con1=[]
	for i=1:m
		Ai = A[i,:];
		terms = Array{MOI.ScalarAffineTerm{Float64},1}();
		for (ind, val) in zip(Ai.nzind, Ai.nzval)
			push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
		end
		if constraint_lb[i] != -Inf
			con1= MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.GreaterThan(constraint_lb[i]));
		end
		if constraint_ub[i] != Inf
			con2 = MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction(terms, b[i]), MOI.LessThan(constraint_ub[i]));
		end
	end


	MOI.optimize!(model);
	status = MOI.get(model, MOI.TerminationStatus());
	println("Model: ", model);
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
	s = 1;


	if status == MOI.OPTIMAL
		Xsol = MOI.get(model, MOI.VariablePrimal(), x);
		println(Xsol);
		# Xdual = MOI.get(model, MOI.DualObjectiveValue(), con1);
		# println(Xdual);
	end

	return(Xsol,s)
end
