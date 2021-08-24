struct QpData{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}}
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
- `x_k`: trust region center
- `Δ`: trust region size
- `feasibility`: indicator for feasibility restoration phase 
"""
function sub_optimize!(
    model::MOI.AbstractOptimizer,
    qp::QpData{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
    feasibility = false,
) where {T,Tv,Tm}

    # empty optimizer just in case
    MOI.empty!(model)

    # dimension of LP
    m, n = size(qp.A)
    @assert n > 0
    @assert m >= 0
    @assert length(qp.c) == n
    @assert length(qp.c_lb) == m
    @assert length(qp.c_ub) == m
    @assert length(qp.v_lb) == n
    @assert length(qp.v_ub) == n
    @assert length(x_k) == n

    # Collect primal and dual variable values for warm-start
    has_var_start = false
    has_con_sv_dual_start = false
    has_con_sa_dual_start = false
    var_start = Tv(undef, n)
    con_sv_dual_start = Tv(undef, n)
    con_sa_dual_start = Tv(undef, m)
    if MOI.is_empty(model) == false
        if MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex(1)) == true
            has_var_start = true
            for j = 1:n
                var_start[j] = MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(j))
            end
        end
        if MOI.supports(
            model,
            MOI.ConstraintDualStart(),
            MOI.ConstraintIndex{SingleVariable}(1),
        ) == true
            has_con_sv_dual_start = true
            for i = 1:n
                con_sv_dual_start[i] = MOI.get(
                    model,
                    MOI.ConstraintDual(),
                    MOI.ConstraintIndex{SingleVariable}(i),
                )
            end
        end
        if MOI.supports(
            model,
            MOI.ConstraintDualStart(),
            MOI.ConstraintIndex{ScalarAffineFunction}(1),
        ) == true
            has_con_sa_dual_start = true
            for i = 1:m
                con_sa_dual_start[i] = MOI.get(
                    model,
                    MOI.ConstraintDual(),
                    MOI.ConstraintIndex{ScalarAffineFunction}(i),
                )
            end
        end
    end

    MOI.empty!(model)

    # variables
    x = MOI.add_variables(model, n)

    # Add a dummy trust-region to all variables
    constr_v_ub = MOI.ConstraintIndex[]
    constr_v_lb = MOI.ConstraintIndex[]
    for i = 1:n
        ub = min(Δ, qp.v_ub[i] - x_k[i])
        lb = max(-Δ, qp.v_lb[i] - x_k[i])
        push!(
            constr_v_ub,
            MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.LessThan(ub)),
        )
        push!(
            constr_v_lb,
            MOI.add_constraint(model, MOI.SingleVariable(x[i]), MOI.GreaterThan(lb)),
        )
        # @show i, lb, ub
    end

    # constr is used to retrieve the dual variable of the constraints after solution
    constr = MOI.ConstraintIndex[]

    # do we need slack variables?
    slack_vars = Dict{Int,Vector{MOI.VariableIndex}}()

    # some parameters that may change for feasibility restoration phase
    c0 = qp.c0
    sense = qp.sense

    if feasibility
        sense = MOI.MIN_SENSE
        c0 = 0.0

        # create slack varables
        for i = 1:m
            slack_vars[i] = []
            push!(slack_vars[i], MOI.add_variable(model))
            if qp.c_lb[i] > -Inf && qp.c_ub[i] < Inf
                push!(slack_vars[i], MOI.add_variable(model))
            end
        end

        for i = 1:m
            # constant term in the constraint
            b = qp.b[i]

            # Adjust parameters for feasibility problem
            viol = 0.0
            if qp.b[i] > qp.c_ub[i]
                viol = qp.c_ub[i] - qp.b[i]
            elseif qp.b[i] < qp.c_lb[i]
                viol = qp.c_lb[i] - qp.b[i]
            end
            b -= abs(viol)

            # Add bound constraints
            if length(slack_vars[i]) == 2
                if viol < 0
                    MOI.add_constraint(
                        model,
                        MOI.SingleVariable(slack_vars[i][1]),
                        MOI.GreaterThan(0.0),
                    )
                    MOI.add_constraint(
                        model,
                        MOI.SingleVariable(slack_vars[i][2]),
                        MOI.GreaterThan(viol),
                    )
                else
                    MOI.add_constraint(
                        model,
                        MOI.SingleVariable(slack_vars[i][1]),
                        MOI.GreaterThan(-viol),
                    )
                    MOI.add_constraint(
                        model,
                        MOI.SingleVariable(slack_vars[i][2]),
                        MOI.GreaterThan(0.0),
                    )
                end
            elseif length(slack_vars[i]) == 1
                MOI.add_constraint(
                    model,
                    MOI.SingleVariable(slack_vars[i][1]),
                    MOI.GreaterThan(-abs(viol)),
                )
            else
                @error "unexpected slack_vars"
            end
            # @show viol, slack_vars[i]

            # Add generic linear constraints
            terms = get_moi_constraint_row_terms(qp.A, i)
            if qp.c_lb[i] == qp.c_ub[i]
                push!(terms, MOI.ScalarAffineTerm{T}(+1.0, slack_vars[i][1]))
                push!(terms, MOI.ScalarAffineTerm{T}(-1.0, slack_vars[i][2]))
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, b),
                        MOI.EqualTo(qp.c_lb[i]),
                    ),
                )
            elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(
                            [terms, MOI.ScalarAffineTerm{T}(+1.0, slack_vars[i][1])],
                            b,
                        ),
                        MOI.GreaterThan(qp.c_lb[i]),
                    ),
                )
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(
                            [terms, MOI.ScalarAffineTerm{T}(-1.0, slack_vars[i][2])],
                            b,
                        ),
                        MOI.LessThan(qp.c_ub[i]),
                    ),
                )
            elseif qp.c_lb[i] != -Inf
                push!(terms, MOI.ScalarAffineTerm{T}(+1.0, slack_vars[i][1]))
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, b),
                        MOI.GreaterThan(qp.c_lb[i]),
                    ),
                )
            elseif qp.c_ub[i] != Inf
                push!(terms, MOI.ScalarAffineTerm{T}(-1.0, slack_vars[i][1]))
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, b),
                        MOI.LessThan(qp.c_ub[i]),
                    ),
                )
            end
        end
    else
        for i = 1:m
            terms = get_moi_constraint_row_terms(qp.A, i)
            if qp.c_lb[i] == qp.c_ub[i] #This means the constraint is equality
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, qp.b[i]),
                        MOI.EqualTo(qp.c_lb[i]),
                    ),
                )
            elseif qp.c_lb[i] != -Inf && qp.c_ub[i] != Inf && qp.c_lb[i] < qp.c_ub[i]
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, qp.b[i]),
                        MOI.GreaterThan(qp.c_lb[i]),
                    ),
                )
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, qp.b[i]),
                        MOI.LessThan(qp.c_ub[i]),
                    ),
                )
            elseif qp.c_lb[i] != -Inf
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, qp.b[i]),
                        MOI.GreaterThan(qp.c_lb[i]),
                    ),
                )
            elseif qp.c_ub[i] != Inf
                push!(
                    constr,
                    MOI.Utilities.normalize_and_add_constraint(
                        model,
                        MOI.ScalarAffineFunction(terms, qp.b[i]),
                        MOI.LessThan(qp.c_ub[i]),
                    ),
                )
            end
        end
    end

    # set the objective function
    obj_terms = Array{MOI.ScalarAffineTerm{T},1}()
    if feasibility
        if !isnothing(qp.Q)
            @error "Feasibility restoration for QP is not supported."
        end
        for i = 1:m
            append!(obj_terms, MOI.ScalarAffineTerm.(1.0, slack_vars[i]))
        end
    else
        for i = 1:n
            push!(obj_terms, MOI.ScalarAffineTerm{T}(qp.c[i], MOI.VariableIndex(i)))
        end
    end
    if isnothing(qp.Q)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            MOI.ScalarAffineFunction(obj_terms, c0),
        )
    else
        obj_qp_terms = get_scalar_quadratic_terms(qp.Q)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}(),
            MOI.ScalarQuadraticFunction(obj_terms, obj_qp_terms, c0),
        )
    end
    MOI.set(model, MOI.ObjectiveSense(), sense)

    # assign start variable values
    if has_var_start
        for j = 1:n
            MOI.set(model, MOI.VariablePrimalStart(), MOI.VariableIndex(j), var_start[j])
        end
    end
    if has_con_sv_dual_start
        for i = 1:n
            MOI.set(
                model,
                MOI.ConstraintDualStart(),
                MOI.ConstraintIndex{SingleVariable}(i),
                con_sv_dual_start[i],
            )
        end
    end
    if has_con_sa_dual_start
        for i = 1:m
            MOI.set(
                model,
                MOI.ConstraintDualStart(),
                MOI.ConstraintIndex{ScalarAffineFunction}(i),
                con_sa_dual_start[i],
            )
        end
    end

    MOI.optimize!(model)
    status = MOI.get(model, MOI.TerminationStatus())

    # TODO: These can be part of data.
    Xsol = Tv(undef, n)
    p_slack = Dict{Int,Vector{Float64}}()
    lambda = Tv(undef, m)
    mult_x_U = Tv(undef, n)
    mult_x_L = Tv(undef, n)

    if status == MOI.OPTIMAL
        # @show MOI.get(model, MOI.ObjectiveValue())
        Xsol .= MOI.get(model, MOI.VariablePrimal(), x)
        for (i, slacks) in slack_vars
            p_slack[i] = MOI.get(model, MOI.VariablePrimal(), slacks)
            # @show MOI.get(model, MOI.VariablePrimal(), slacks)
        end

        # extract the multipliers to constraints
        ci = 1
        for i = 1:m
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
        for j = 1:n
            if Xsol[j] < qp.v_ub[j] - x_k[j]
                mult_x_U[j] = 0.0
            end
            if Xsol[j] > qp.v_lb[j] - x_k[j]
                mult_x_L[j] = 0.0
            end
        end
    elseif status == MOI.DUAL_INFEASIBLE
        @error "Trust region must be employed."
    elseif status == MOI.INFEASIBLE
        fill!(Xsol, 0.0)
        fill!(lambda, 0.0)
        fill!(mult_x_U, 0.0)
        fill!(mult_x_L, 0.0)
    else
        @error "Unexpected status: $(status)"
    end

    return Xsol, lambda, mult_x_U, mult_x_L, p_slack, status
end

"""
	get_scalar_quadratic_terms

Collect the quadratic terms of the objective function
"""
function get_scalar_quadratic_terms(Q::Tm) where {T,Tm<:AbstractSparseMatrix{T,Int}}
    terms = Array{MOI.ScalarQuadraticTerm{T},1}()
    rows = rowvals(Q)
    vals = nonzeros(Q)
    m, n = size(Q)
    for j = 1:n
        for i in nzrange(Q, j)
            if i == j
                push!(terms, MOI.ScalarQuadraticTerm{T}(0.5 * vals[i], rows[i], j))
            elseif i > j
                push!(terms, MOI.ScalarQuadraticTerm{T}(vals[i], rows[i], j))
            end
        end
    end
    return terms
end

"""
	get_moi_constraint_row_terms

Get the array of MOI constraint terms from row `i` of matrix `A`
"""
function get_moi_constraint_row_terms(
    A::Tm,
    i::Int,
) where {T,Tm<:AbstractSparseMatrix{T,Int}}
    Ai = A[i, :]
    terms = Array{MOI.ScalarAffineTerm{T},1}()
    for (ind, val) in zip(Ai.nzind, Ai.nzval)
        push!(terms, MOI.ScalarAffineTerm{T}(val, MOI.VariableIndex(ind)))
    end
    return terms
end

function get_moi_constraint_row_terms(A::Tm, i::Int) where {T,Tm<:DenseArray{T,2}}
    terms = Array{MOI.ScalarAffineTerm{T},1}()
    for j = 1:size(A, 2)
        if !isapprox(A[i, j], 0.0)
            push!(terms, MOI.ScalarAffineTerm{T}(A[i, j], MOI.VariableIndex(j)))
        end
    end
    return terms
end
