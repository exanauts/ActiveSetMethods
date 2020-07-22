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

function solve_lp(model,c,A1,b1,A2,b2,min_max=0)

	sense = MOI.MIN_SENSE

	if min_max != 0
		sense = MOI.MAX_SENSE
	end

	n = A1.n;
	m1 = A1.m;
	m2 = A2.m;

	c0 = c[n+1]


	x = MOI.add_variables(model, n)
	terms = Array{MOI.ScalarAffineTerm{Float64},1}();
	for (ind, val) in zip(c.nzind, c.nzval)
		push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
	end
	MOI.set(model,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(terms, c0))
	MOI.set(model, MOI.ObjectiveSense(), sense)

	for i=1:m1
		Ai = A1[i,:];
		terms = Array{MOI.ScalarAffineTerm{Float64},1}();
		for (ind, val) in zip(Ai.nzind, Ai.nzval)
			push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
		end
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction(terms, b1[i]), MOI.EqualTo(0.0));
	end

	for i=1:m2
		Ai = A2[i,:];
		terms = Array{MOI.ScalarAffineTerm{Float64},1}();
		for (ind, val) in zip(Ai.nzind, Ai.nzval)
			push!(terms, MOI.ScalarAffineTerm{Float64}(val, MOI.VariableIndex(ind)));
		end
		MOI.Utilities.normalize_and_add_constraint(model,
		MOI.ScalarAffineFunction(terms, b2[i]), MOI.LessThan(0.0));
	end


	MOI.optimize!(model);
	status = MOI.get(model, MOI.TerminationStatus());

	Xsol = zeros(n);
	s = 1;

	if status == MOI.OPTIMAL
		Xsol = MOI.get(model, MOI.VariablePrimal(), x);
		println(Xsol);
	end

	return(Xsol,s)
end
