struct QpData{T, Tv<:AbstractArray{T}, Tm<:AbstractMatrix{T}}
	sense::MOI.OptimizationSense
	Q::Union{Nothing,Tm}
	c::Tv
	c0::T # objective functiton constant term
	A::Tm
	b::Tv
	c_lb::Tv
	c_ub::Tv
	v_lb::Tv
	v_ub::Tv
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
	qp::QpData{T,Tv,Tm},
	mu::T,
	x_k::Tv,
	Δ::T,
) where {T, Tv, Tm}

	# empty optimizer just in case
	MOI.empty!(model)

	# dimension of LP
	println(size(qp.A))
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
		push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.c[i], MOI.VariableIndex(i)));
	end

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
		push!(constr_v_ub, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)))
		push!(constr_v_lb, MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)))
	end
	
	# constr is used to retrieve the dual variable of the constraints after solution
	constr = MOI.ConstraintIndex[]
	for i=1:m
		terms = get_moi_constraint_row_terms(qp.A, i)
		push!(terms, MOI.ScalarAffineTerm{T}(1.0, u[i]));
		push!(terms, MOI.ScalarAffineTerm{T}(-1.0, v[i]));
		if qp.c_lb[i] == qp.c_ub[i] #This means the constraint is equality
			push!(constr, MOI.Utilities.normalize_and_add_constraint(model,
				MOI.ScalarAffineFunction(terms, qp.b[i]), MOI.EqualTo(qp.c_lb[i])
			))
		elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
			push!(constr, MOI.Utilities.normalize_and_add_constraint(model,
				MOI.ScalarAffineFunction(terms, qp.b[i]), MOI.GreaterThan(qp.c_lb[i])
			))
			push!(constr, MOI.Utilities.normalize_and_add_constraint(model,
				MOI.ScalarAffineFunction(terms, qp.b[i]), MOI.LessThan(qp.c_ub[i])
			))
		elseif qp.c_lb[i] != -Inf
			push!(constr, MOI.Utilities.normalize_and_add_constraint(model,
				MOI.ScalarAffineFunction(terms, qp.b[i]), MOI.GreaterThan(qp.c_lb[i])
			))
		elseif qp.c_ub[i] != Inf
			push!(constr, MOI.Utilities.normalize_and_add_constraint(model,
				MOI.ScalarAffineFunction(terms, qp.b[i]), MOI.LessThan(qp.c_ub[i])
			))
		end
		u_term = MOI.ScalarAffineTerm{T}(1.0, u[i]);
		MOI.Utilities.normalize_and_add_constraint(model, 
			MOI.ScalarAffineFunction([u_term], 0.0), MOI.GreaterThan(0.0));
		v_term = MOI.ScalarAffineTerm{T}(1.0, v[i]);
		MOI.Utilities.normalize_and_add_constraint(model,
			MOI.ScalarAffineFunction([v_term], 0.0), MOI.GreaterThan(0.0));
	end

	MOI.optimize!(model)
	status = MOI.get(model, MOI.TerminationStatus())

	# TODO: These can be part of data.
	Xsol = Tv(undef, n)
	lambda = Tv(undef, m)
	mult_x_U = Tv(undef, n)
	mult_x_L = Tv(undef, n)
	infeasibility = 0.0

	if status == MOI.OPTIMAL
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
	elseif status == MOI.DUAL_INFEASIBLE
		@error "Trust region must be employed."
	else
		@error "Unexpected status: $(status)"
	end

	return Xsol, lambda, mult_x_U, mult_x_L, infeasibility, status
end

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
