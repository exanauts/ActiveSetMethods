struct QpData{T, Tv<:AbstractArray{T}} #, Tm<:AbstractMatrix{T}}
	sense::MOI.OptimizationSense
	#Q::Union{Nothing,Tm}
	Q::Nothing
	c::Tv
	c0::T # objective functiton constant term
	dE::Tv
	b::Tv
	c_lb::Tv
	c_ub::Tv
	v_lb::Tv
	v_ub::Tv
	j_row::Tv
    	j_col::Tv
	constr_v_ub
	constr_v_lb
	constr
end

"""
	sub_optimize!

Solve subproblem

# Arguments
- `model`: MOI abstract optimizer
- `qp`: QP problem data
- `mu`: penalty parameter
- `x_k`: trust region center
- `Δ`: trust region size
"""

function sub_optimize!(
	model::MOI.AbstractOptimizer,
	qp::QpData{T,Tv},
	mu::T,
	x_k::Tv,
	Δ::T,
	tol_error::Float64
) where {T, Tv}

	start_time = time();
	# drop small values to avoid numerical issues
	
	#droptol!(qp.A, tol_error);
	
	qp.c[abs.(qp.c) .<= tol_error] .= zero(eltype(qp.c))
	
	println("-----> LP OPT1 time: $(time()-start_time)"); start_time = time();
	#droptol!(qp.b, tol_error);
	
	# empty optimizer just in case
	#MOI.empty!(model)

	# dimension of LP
	#m, n = size(qp.A)
	#n = length(qp.c); 
	
	n = length(qp.c);
	m = length(qp.c_lb);
	
	@assert n > 0
	@assert m >= 0
	@assert length(qp.c) == n
	@assert length(qp.c_lb) == m
	@assert length(qp.c_ub) == m
	@assert length(qp.v_lb) == n
	@assert length(qp.v_ub) == n
	@assert length(x_k) == n
	
	# variables
	#x = MOI.add_variables(model, n)
	
	# objective function
	#obj_terms = Array{MOI.ScalarAffineTerm{T},1}();
	MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarConstantChange(qp.c0))
	for i in 1:n
		#MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(x[i], qp.c[i]))
		MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(MOI.VariableIndex(i), qp.c[i]))
	end
  println("-----> LP OPT2 time: $(time()-start_time)"); start_time = time();

	# Slacks v and u are added only for constrained problems.
	if m > 0
		for i in 1:m
			#MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(u[i], mu))
			#MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(v[i], mu))
			MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(MOI.VariableIndex(i+n), mu))
			MOI.modify(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarCoefficientChange(MOI.VariableIndex(i+n+m), mu))
		end
		#u = MOI.add_variables(model, m)
		#v = MOI.add_variables(model, m)
		#append!(obj_terms, MOI.ScalarAffineTerm.(mu, u));
		#append!(obj_terms, MOI.ScalarAffineTerm.(mu, v));
	end

	# set constant term to the objective function
	#MOI.set(model,
	#	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
	#	MOI.ScalarAffineFunction(obj_terms, qp.c0))
	#MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	# Add a dummy trust-region to all variables
	#constr_v_ub = MOI.ConstraintIndex[]
	#constr_v_lb = MOI.ConstraintIndex[]
	for i = 1:n
		ub = min(Δ, qp.v_ub[i] - x_k[i])
		lb = max(-Δ, qp.v_lb[i] - x_k[i])
		ub = (abs(ub) <= tol_error) ? 0.0 : ub;
		lb = (abs(lb) <= tol_error) ? 0.0 : lb;
		MOI.set(model, MOI.ConstraintSet(), qp.constr_v_ub[i], MOI.LessThan(ub))
		MOI.set(model, MOI.ConstraintSet(), qp.constr_v_lb[i], MOI.GreaterThan(lb))
		#MOI.set(model, MOI.ConstraintSet(), MOI.ConstraintIndex[2*i-1], MOI.LessThan(ub))
		#MOI.set(model, MOI.ConstraintSet(), MOI.ConstraintIndex(2*i), MOI.GreaterThan(lb))
		#push!(constr_v_ub, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)))
		#push!(constr_v_lb, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)))
	end
  println("-----> LP OPT3 time: $(time()-start_time)"); start_time = time();
	
	# constr is used to retrieve the dual variable of the constraints after solution
	#constr = MOI.ConstraintIndex[]
	
	for i=1:length(qp.dE)
		qp.dE[i] = (abs(qp.dE[i]) <= tol_error) ? 0.0 : qp.dE[i];
		MOI.modify(model, qp.constr[Int(qp.j_row[i])], MOI.ScalarCoefficientChange(MOI.VariableIndex(Int(qp.j_col[i])), qp.dE[i]))
	end
	
	for i=1:m
		#terms = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([qp.A[i,:].nzval;1.0;-1.0], [x[qp.A[i,:].nzind];u[i];v[i]]), 0.0);
		c_ub = qp.c_ub[i]-qp.b[i];
		c_lb = qp.c_lb[i]-qp.b[i];
		
		c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub;
		c_lb = (abs(c_lb) <= tol_error) ? 0.0 : c_lb;
		
		#MOI.modify(model, qp.constr[i], MOI.ScalarCoefficientChange(MOI.VariableIndex(1), 1.0))
		
		#=for (ind, val) in zip(qp.A[i,:].nzind, qp.A[i,:].nzval)
			MOI.modify(model, qp.constr[i], MOI.ScalarCoefficientChange(MOI.VariableIndex(ind), val))
		end=#
		
		if qp.c_lb[i] == qp.c_ub[i] #This means the constraint is equality
			MOI.set(model, MOI.ConstraintSet(), qp.constr[i], MOI.EqualTo(c_lb))
			#push!(constr, MOI.add_constraint(model, terms, MOI.EqualTo(c_lb)))
			#println("----> single constraints Check1")
		elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
			#push!(constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
               	#push!(constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
               	#println("----> single constraints Check2")
		elseif qp.c_lb[i] != -Inf
			MOI.set(model, MOI.ConstraintSet(), qp.constr[i], MOI.GreaterThan(c_lb))
			#push!(constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
			#println("----> single constraints Check3")
		elseif qp.c_ub[i] != Inf
			MOI.set(model, MOI.ConstraintSet(), qp.constr[i], MOI.LessThan(c_ub))
			#push!(constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
			#println("----> single constraints Check4")
		end
		#MOI.add_constraint(model, MOI.SingleVariable(u[i]), MOI.GreaterThan(0.0))
		#MOI.add_constraint(model, MOI.SingleVariable(v[i]), MOI.GreaterThan(0.0))
	end
			
	
   println("-----> LP OPT4 time: $(time()-start_time)"); start_time = time();

	MOI.optimize!(model)
	TerminationStatus = MOI.get(model, MOI.TerminationStatus())
	PrimalStatus = MOI.get(model, MOI.PrimalStatus())
	ResultCount = MOI.get(model, MOI.ResultCount())

	# TODO: These can be part of data.
	Xsol = Tv(undef, n)
	if m > 0
		Usol = Tv(undef, m);
		Vsol = Tv(undef, m);
	end
	lambda = Tv(undef, m)
	mult_x_U = Tv(undef, n)
	mult_x_L = Tv(undef, n)
	infeasibility = 0.0
   println("-----> LP OPT5 time: $(time()-start_time)");
   println("-----> TerminationStatus: $(MOI.get(model, MOI.TerminationStatus()))");
   println("-----> PrimalStatus: $(MOI.get(model, MOI.PrimalStatus()))");
   println("-----> DualStatus: $(MOI.get(model, MOI.DualStatus()))");
   	
	#if ResultCount == 1 && !(PrimalStatus in [MOI.INFEASIBLE_POINT;MOI.NO_SOLUTION;MOI.INFEASIBILITY_CERTIFICATE])
	if PrimalStatus == MOI.FEASIBLE_POINT
		for i=1:n
			Xsol[i] = MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(i));
		end
		if m > 0
			for i=1:m
				Usol[i] = MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(i+n));
				Vsol[i] = MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(i+n+m));
			end
		end

		# extract the multipliers to constraints
		ci = 1
		for i=1:m
			lambda[i] = MOI.get(model, MOI.ConstraintDual(1), qp.constr[ci])
			ci += 1
			# This is for a ranged constraint.
			if qp.c_lb[i] > -Inf && qp.c_ub[i] < Inf && qp.c_lb[i] < qp.c_ub[i]
				lambda[i] += MOI.get(model, MOI.ConstraintDual(1), qp.constr[ci])
				ci += 1
			end
		end

		# extract the multipliers to column bounds
		mult_x_U .= MOI.get(model, MOI.ConstraintDual(1), qp.constr_v_ub)
		mult_x_L .= MOI.get(model, MOI.ConstraintDual(1), qp.constr_v_lb)
		# careful because of the trust region
		for j=1:n
			if Xsol[j] < qp.v_ub[j] - x_k[j]
				mult_x_U[j] = 0.0
			end
			if Xsol[j] > qp.v_lb[j] - x_k[j]
				mult_x_L[j] = 0.0
			end
		end
	elseif TerminationStatus == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected status: $(TerminationStatus)"
	end
	
	#=
	if TerminationStatus == MOI.OPTIMAL
		Xsol .= MOI.get(model, MOI.VariablePrimal(), x);
		if m > 0
			Usol = MOI.get(model, MOI.VariablePrimal(), u);
			Vsol = MOI.get(model, MOI.VariablePrimal(), v);
			infeasibility += max(0.0, sum(Usol) + sum(Vsol))
		end

		# extract the multipliers to constraints
		ci = 1
		for i=1:m
			lambda[i] = MOI.get(model, MOI.ConstraintDual(1), constr[ci])
			ci += 1
			# This is for a ranged constraint.
			if qp.c_lb[i] > -Inf && qp.c_ub[i] < Inf && qp.c_lb[i] < qp.c_ub[i]
				lambda[i] += MOI.get(model, MOI.ConstraintDual(1), constr[ci])
				ci += 1
			end
		end

		# extract the multipliers to column bounds
		mult_x_U .= MOI.get(model, MOI.ConstraintDual(1), constr_v_ub)
		mult_x_L .= MOI.get(model, MOI.ConstraintDual(1), constr_v_lb)
		# careful because of the trust region
		for j=1:n
			if Xsol[j] < qp.v_ub[j] - x_k[j]
				mult_x_U[j] = 0.0
			end
			if Xsol[j] > qp.v_lb[j] - x_k[j]
				mult_x_L[j] = 0.0
			end
		end
	elseif TerminationStatus == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected TerminationStatus: $(TerminationStatus)"
	end=#
	
	
	simplex = try 
			MOI.get(model, MOI.SimplexIterations()); 
		    catch; 
		    	0.0; 
		    end
		    
	barrier = try 
			MOI.get(model, MOI.BarrierIterations()); 
		    catch; 
		    	0.0; 
		    end
		    

	return Xsol, lambda, mult_x_U, mult_x_L, infeasibility, PrimalStatus, simplex, barrier
end


function create_model!(
	model::MOI.AbstractOptimizer,
	qp::QpData{T,Tv},
	mu::T,
	x_k::Tv,
	Δ::T,
	tol_error::Float64
) where {T, Tv}

	start_time = time();
	# drop small values to avoid numerical issues
	
	#droptol!(qp.A, tol_error);
	
	#qp.c[abs.(qp.c) .<= tol_error] .= zero(eltype(qp.c))
	
	println("-----> LP CM1 time: $(time()-start_time)"); start_time = time();
	#droptol!(qp.b, tol_error);
	
	# empty optimizer just in case
	MOI.empty!(model)

	# dimension of LP
	#m, n = size(qp.A)
	
	n = length(qp.c);
	m = length(qp.c_lb);
	
	@show n
	@show m

	@assert n > 0
	@assert m >= 0
	@assert length(qp.c) == n
	@assert length(qp.c_lb) == m
	@assert length(qp.c_ub) == m
	@assert length(qp.v_lb) == n
	@assert length(qp.v_ub) == n
	@assert length(x_k) == n
	
	# variables
	x = MOI.add_variables(model, n)
	
	# objective function
	obj_terms = Array{MOI.ScalarAffineTerm{T},1}();
	for i in 1:n
		#c = qp.c[i];
		#c = (abs(qp.c[i]) <= tol_error) ? 0.0 : qp.c[i];
		push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.c[i], MOI.VariableIndex(i)));
	end
  println("-----> LP OPT2 time: $(time()-start_time)"); start_time = time();

	# Slacks v and u are added only for constrained problems.
	if m > 0
		u = MOI.add_variables(model, m)
		v = MOI.add_variables(model, m)
		append!(obj_terms, MOI.ScalarAffineTerm.(mu, u));
		append!(obj_terms, MOI.ScalarAffineTerm.(mu, v));
	end

	# set constant term to the objective function
	MOI.set(model,
		MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
		MOI.ScalarAffineFunction(obj_terms, qp.c0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	# Add a dummy trust-region to all variables
	#qp.constr_v_ub = MOI.ConstraintIndex[]
	#qp.constr_v_lb = MOI.ConstraintIndex[]
	for i = 1:n
		ub = min(Δ, qp.v_ub[i] - x_k[i])
		lb = max(-Δ, qp.v_lb[i] - x_k[i])
		ub = (abs(ub) <= tol_error) ? 0.0 : ub;
		lb = (abs(lb) <= tol_error) ? 0.0 : lb;
		push!(qp.constr_v_ub, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)))
		push!(qp.constr_v_lb, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)))
	end
  println("-----> LP OPT3 time: $(time()-start_time)"); start_time = time();
	
	# constr is used to retrieve the dual variable of the constraints after solution
	#qp.constr = MOI.ConstraintIndex[]
	
	for i=1:m
		#terms = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([qp.A[i,:].nzval;1.0;-1.0], [x[qp.A[i,:].nzind];u[i];v[i]]), 0.0);
		terms = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0;-1.0], [u[i];v[i]]), 0.0);
		c_ub = qp.c_ub[i]-qp.b[i];
		c_lb = qp.c_lb[i]-qp.b[i];
		
		c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub;
		c_lb = (abs(c_lb) <= tol_error) ? 0.0 : c_lb;
		
		if qp.c_lb[i] == qp.c_ub[i] #This means the constraint is equality
			push!(qp.constr, MOI.add_constraint(model, terms, MOI.EqualTo(c_lb)))
			#println("----> single constraints Check1")
		elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
			push!(qp.constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
               	push!(qp.constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
               	#println("=====> double constraints Check")
		elseif qp.c_lb[i] != -Inf
			push!(qp.constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
			#println("----> single constraints Check3")
		elseif qp.c_ub[i] != Inf
			push!(qp.constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
			#println("----> single constraints Check4")
		end
		MOI.add_constraint(model, MOI.SingleVariable(u[i]), MOI.GreaterThan(0.0))
		MOI.add_constraint(model, MOI.SingleVariable(v[i]), MOI.GreaterThan(0.0))
	end
	
   println("-----> LP OPT4 time: $(time()-start_time)"); start_time = time();
   return qp
end

#=function sub_optimize!(
	model::MOI.AbstractOptimizer,
	qp::QpData{T,Tv,Tm},
	mu::T,
	x_k::Tv,
	Δ::T,
	tol_error::Float64=-Inf
) where {T, Tv, Tm}

	start_time = time();
	# drop small values to avoid numerical issues
	
	droptol!(qp.A, tol_error);
	
	qp.c[abs.(qp.c) .<= tol_error] .= zero(eltype(qp.c))
	
	println("-----> LP OPT1 time: $(time()-start_time)"); start_time = time();
	#droptol!(qp.b, tol_error);
	
	# empty optimizer just in case
	MOI.empty!(model)

	# dimension of LP
	m, n = size(qp.A)
	n = length(qp.c); 
	@assert n > 0
	@assert m >= 0
	@assert length(qp.c) == n
	@assert length(qp.c_lb) == m
	@assert length(qp.c_ub) == m
	@assert length(qp.v_lb) == n
	@assert length(qp.v_ub) == n
	@assert length(x_k) == n
	
	# variables
	x = MOI.add_variables(model, n)
	
	# objective function
	obj_terms = Array{MOI.ScalarAffineTerm{T},1}();
	for i in 1:n
		#c = qp.c[i];
		#c = (abs(qp.c[i]) <= tol_error) ? 0.0 : qp.c[i];
		push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.c[i], MOI.VariableIndex(i)));
	end
  println("-----> LP OPT2 time: $(time()-start_time)"); start_time = time();

	# Slacks v and u are added only for constrained problems.
	if m > 0
		u = MOI.add_variables(model, m)
		v = MOI.add_variables(model, m)
		append!(obj_terms, MOI.ScalarAffineTerm.(mu, u));
		append!(obj_terms, MOI.ScalarAffineTerm.(mu, v));
	end

	# set constant term to the objective function
	MOI.set(model,
		MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
		MOI.ScalarAffineFunction(obj_terms, qp.c0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	# Add a dummy trust-region to all variables
	constr_v_ub = MOI.ConstraintIndex[]
	constr_v_lb = MOI.ConstraintIndex[]
	for i = 1:n
		ub = min(Δ, qp.v_ub[i] - x_k[i])
		lb = max(-Δ, qp.v_lb[i] - x_k[i])
		ub = (abs(ub) <= tol_error) ? 0.0 : ub;
		lb = (abs(lb) <= tol_error) ? 0.0 : lb;
		push!(constr_v_ub, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)))
		push!(constr_v_lb, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)))
	end
  println("-----> LP OPT3 time: $(time()-start_time)"); start_time = time();
	
	# constr is used to retrieve the dual variable of the constraints after solution
	constr = MOI.ConstraintIndex[]
	
	for i=1:m
		terms = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([qp.A[i,:].nzval;1.0;-1.0], [x[qp.A[i,:].nzind];u[i];v[i]]), 0.0);
		c_ub = qp.c_ub[i]-qp.b[i];
		c_lb = qp.c_lb[i]-qp.b[i];
		
		c_ub = (abs(c_ub) <= tol_error) ? 0.0 : c_ub;
		c_lb = (abs(c_lb) <= tol_error) ? 0.0 : c_lb;
		
		if qp.c_lb[i] == qp.c_ub[i] #This means the constraint is equality
			push!(constr, MOI.add_constraint(model, terms, MOI.EqualTo(c_lb)))
			#println("----> single constraints Check1")
		elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
			push!(constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
               	push!(constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
               	println("=====> double constraints Check")
		elseif qp.c_lb[i] != -Inf
			push!(constr, MOI.add_constraint(model, terms, MOI.GreaterThan(c_lb)))
			#println("----> single constraints Check3")
		elseif qp.c_ub[i] != Inf
			push!(constr, MOI.add_constraint(model, terms, MOI.LessThan(c_ub)))
			#println("----> single constraints Check4")
		end
		MOI.add_constraint(model, MOI.SingleVariable(u[i]), MOI.GreaterThan(0.0))
		MOI.add_constraint(model, MOI.SingleVariable(v[i]), MOI.GreaterThan(0.0))
	end
	
   println("m: ", m);
   println("length_constr: ", length(constr));
   println("-----> LP OPT4 time: $(time()-start_time)"); start_time = time();

	MOI.optimize!(model)
	TerminationStatus = MOI.get(model, MOI.TerminationStatus())
	PrimalStatus = MOI.get(model, MOI.PrimalStatus())
	ResultCount = MOI.get(model, MOI.ResultCount())

	# TODO: These can be part of data.
	Xsol = Tv(undef, n)
	lambda = Tv(undef, m)
	mult_x_U = Tv(undef, n)
	mult_x_L = Tv(undef, n)
	infeasibility = 0.0
   println("-----> LP OPT5 time: $(time()-start_time)");
   println("-----> TerminationStatus: $(MOI.get(model, MOI.TerminationStatus()))");
   println("-----> PrimalStatus: $(MOI.get(model, MOI.PrimalStatus()))");
   println("-----> DualStatus: $(MOI.get(model, MOI.DualStatus()))");
   	
	#if ResultCount == 1 && !(PrimalStatus in [MOI.INFEASIBLE_POINT;MOI.NO_SOLUTION;MOI.INFEASIBILITY_CERTIFICATE])
	if PrimalStatus == MOI.FEASIBLE_POINT
		Xsol .= MOI.get(model, MOI.VariablePrimal(), x);
		if m > 0
			Usol = MOI.get(model, MOI.VariablePrimal(), u);
			Vsol = MOI.get(model, MOI.VariablePrimal(), v);
			infeasibility += max(0.0, sum(Usol) + sum(Vsol))
		end

		# extract the multipliers to constraints
		ci = 1
		for i=1:m
			lambda[i] = MOI.get(model, MOI.ConstraintDual(1), constr[ci])
			ci += 1
			# This is for a ranged constraint.
			if qp.c_lb[i] > -Inf && qp.c_ub[i] < Inf && qp.c_lb[i] < qp.c_ub[i]
				lambda[i] += MOI.get(model, MOI.ConstraintDual(1), constr[ci])
				ci += 1
			end
		end

		# extract the multipliers to column bounds
		mult_x_U .= MOI.get(model, MOI.ConstraintDual(1), constr_v_ub)
		mult_x_L .= MOI.get(model, MOI.ConstraintDual(1), constr_v_lb)
		# careful because of the trust region
		for j=1:n
			if Xsol[j] < qp.v_ub[j] - x_k[j]
				mult_x_U[j] = 0.0
			end
			if Xsol[j] > qp.v_lb[j] - x_k[j]
				mult_x_L[j] = 0.0
			end
		end
	elseif TerminationStatus == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected status: $(TerminationStatus)"
	end
	
	#=
	if TerminationStatus == MOI.OPTIMAL
		Xsol .= MOI.get(model, MOI.VariablePrimal(), x);
		if m > 0
			Usol = MOI.get(model, MOI.VariablePrimal(), u);
			Vsol = MOI.get(model, MOI.VariablePrimal(), v);
			infeasibility += max(0.0, sum(Usol) + sum(Vsol))
		end

		# extract the multipliers to constraints
		ci = 1
		for i=1:m
			lambda[i] = MOI.get(model, MOI.ConstraintDual(1), constr[ci])
			ci += 1
			# This is for a ranged constraint.
			if qp.c_lb[i] > -Inf && qp.c_ub[i] < Inf && qp.c_lb[i] < qp.c_ub[i]
				lambda[i] += MOI.get(model, MOI.ConstraintDual(1), constr[ci])
				ci += 1
			end
		end

		# extract the multipliers to column bounds
		mult_x_U .= MOI.get(model, MOI.ConstraintDual(1), constr_v_ub)
		mult_x_L .= MOI.get(model, MOI.ConstraintDual(1), constr_v_lb)
		# careful because of the trust region
		for j=1:n
			if Xsol[j] < qp.v_ub[j] - x_k[j]
				mult_x_U[j] = 0.0
			end
			if Xsol[j] > qp.v_lb[j] - x_k[j]
				mult_x_L[j] = 0.0
			end
		end
	elseif TerminationStatus == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected TerminationStatus: $(TerminationStatus)"
	end=#
	
	
	simplex = try 
			MOI.get(model, MOI.SimplexIterations()); 
		    catch; 
		    	0.0; 
		    end
		    
	barrier = try 
			MOI.get(model, MOI.BarrierIterations()); 
		    catch; 
		    	0.0; 
		    end
		    

	return Xsol, lambda, mult_x_U, mult_x_L, infeasibility, TerminationStatus, simplex, barrier
end =#

"""
	get_moi_constraint_row_terms

Get the array of MOI constraint terms from row `i` of matrix `A`
"""
function get_moi_constraint_row_terms(A::Tm, i::Int) where {T, Tm<:AbstractSparseMatrix{T,Int}}
	Ai = A[i,:]
	terms = Array{MOI.ScalarAffineTerm{T},1}()
	for (ind, val) in zip(Ai.nzind, Ai.nzval)
		push!(terms, MOI.ScalarAffineTerm{T}(val, MOI.VariableIndex(ind)))
	end
	return terms
end

function get_moi_constraint_row_terms(A::Tm, i::Int) where {T, Tm<:DenseArray{T,2}}
	terms = Array{MOI.ScalarAffineTerm{T},1}()
	for j in 1:size(A,2)
		if !isapprox(A[i,j], 0.0)
			push!(terms, MOI.ScalarAffineTerm{T}(A[i,j], MOI.VariableIndex(j)))
		end
	end
	return terms
end
