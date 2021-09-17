mutable struct QpJuMP{T,Tv<:AbstractArray{T},Tm<:AbstractMatrix{T}} <: AbstractSubOptimizer
    model::JuMP.Model
    data::QpData{T,Tv,Tm}
    adj::Vector{Int}
    x::Vector{JuMP.VariableRef}
    constr::Vector{JuMP.ConstraintRef}
    slack_vars::Dict{Int,Vector{JuMP.VariableRef}}

    function QpJuMP(model::JuMP.AbstractModel, data::QpData{T,Tv,Tm}) where {T,Tv,Tm}
        qp = new{T,Tv,Tm}()
        qp.model = model
        qp.data = data
        qp.adj = []
        qp.x = []
        qp.constr = []
        qp.slack_vars = Dict()
        return qp
    end
end

SubOptimizer(model::JuMP.AbstractModel, data::QpData{T,Tv,Tm}) where {T,Tv,Tm} =
    QpJuMP(model, data)

"""
    create_model!

Initialize QP subproblem in JuMP.Model
"""
function create_model!(qp::QpJuMP{T,Tv,Tm}, x_k::Tv, Δ::T) where {T,Tv,Tm}

    qp.adj = []
    qp.constr = []
    empty!(qp.slack_vars)

    n = length(qp.data.c)
    m = length(qp.data.c_lb)

    # create nominal variables
    qp.x = @variable(
        qp.model,
        [i = 1:n],
        base_name = "x",
        lower_bound = max(-Δ, qp.data.v_lb[i] - x_k[i]),
        upper_bound = min(+Δ, qp.data.v_ub[i] - x_k[i]),
    )

    for i = 1:m
        # add slack variables
        qp.slack_vars[i] = []
        push!(qp.slack_vars[i], @variable(qp.model, base_name = "u_$i", lower_bound = 0.0))
        if qp.data.c_lb[i] > -Inf && qp.data.c_ub[i] < Inf
            push!(qp.slack_vars[i], @variable(qp.model, base_name = "v_$i", lower_bound = 0.0))
        end
    end

    # dummy objective function
    @objective(
        qp.model,
        Min,
        sum(qp.data.c[i] * qp.x[i] for i = 1:n) +
        sum(s for (_, slack) in qp.slack_vars, s in slack)
    )

    for i = 1:m
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        c_lb = qp.data.c_lb[i] - qp.data.b[i]

        if qp.data.c_lb[i] == qp.data.c_ub[i] #This means the constraint is equality
            push!(
                qp.constr,
                @constraint(qp.model, qp.slack_vars[i][1] - qp.slack_vars[i][2] == c_lb)
            )
        elseif qp.data.c_lb[i] != -Inf &&
               qp.data.c_ub[i] != Inf &&
               qp.data.c_lb[i] < qp.data.c_ub[i]
            push!(qp.constr, @constraint(qp.model, qp.slack_vars[i][1] >= c_lb))
            push!(qp.adj, i)
        elseif qp.data.c_lb[i] != -Inf
            push!(qp.constr, @constraint(qp.model, qp.slack_vars[i][1] >= c_lb))
        elseif qp.data.c_ub[i] != Inf
            push!(qp.constr, @constraint(qp.model, -qp.slack_vars[i][1] <= c_ub))
        end
    end

    for i in qp.adj
        c_ub = qp.data.c_ub[i] - qp.data.b[i]
        push!(qp.constr, @constraint(qp.model, -qp.slack_vars[i][2] <= c_ub))
    end
end

function sub_optimize!(
    qp::QpJuMP{T,Tv,Tm},
    x_k::Tv,
    Δ::T,
    feasibility = false,
) where {T,Tv,Tm}

    # dimension of LP
    m, n = size(qp.data.A)

    b = deepcopy(qp.data.b)

    if feasibility
        # modify objective function
        @objective(qp.model, Min, sum(s for (_, slacks) in qp.slack_vars, s in slacks))

        # fix slack variables to zeros
        for (_, slacks) in qp.slack_vars, s in slacks
            if JuMP.is_fixed(s)
                JuMP.unfix(s)
            end
        end

        # modify slack variable bounds
        for i = 1:m
            # Adjust parameters for feasibility problem
            viol = 0.0
            if qp.data.b[i] > qp.data.c_ub[i]
                viol = qp.data.c_ub[i] - qp.data.b[i]
            elseif qp.data.b[i] < qp.data.c_lb[i]
                viol = qp.data.c_lb[i] - qp.data.b[i]
            end
            b[i] -= abs(viol)

            if length(qp.slack_vars[i]) == 2
                if viol < 0
                    set_lower_bound(qp.slack_vars[i][1], 0.0)
                    set_lower_bound(qp.slack_vars[i][2], viol)
                else
                    set_lower_bound(qp.slack_vars[i][1], -viol)
                    set_lower_bound(qp.slack_vars[i][2], 0.0)
                end
            elseif length(qp.slack_vars[i]) == 1
                set_lower_bound(qp.slack_vars[i][1], -abs(viol))
            else
                @error "unexpected slack_vars"
            end
        end
    else
        # modify objective function
        if isnothing(qp.data.Q)
            @objective(qp.model, qp.data.sense, sum(qp.data.c[i] * qp.x[i] for i = 1:n))
        else
            obj = QuadExpr(sum(qp.data.c[i] * qp.x[i] for i = 1:n))
            for j = 1:qp.data.Q.n, i in nzrange(qp.data.Q, j)
                add_to_expression!(
                    obj,
                    qp.data.Q.nzval[i],
                    qp.x[qp.data.Q.rowval[i]],
                    qp.x[j],
                )
            end
            @objective(qp.model, qp.data.sense, obj)
        end

        # fix slack variables to zeros
        for (_, slacks) in qp.slack_vars, s in slacks
            if JuMP.has_lower_bound(s)
                JuMP.delete_lower_bound(s)
            end
            JuMP.fix(s, 0.0)
        end
    end

    # set variable bounds
    for i = 1:n
        set_lower_bound(qp.x[i], max(-Δ, qp.data.v_lb[i] - x_k[i]))
        set_upper_bound(qp.x[i], min(+Δ, qp.data.v_ub[i] - x_k[i]))
    end
    # @show Δ, qp.data.v_lb, qp.data.v_ub, x_k

    # modify the constraint coefficients
    for j = 1:qp.data.A.n, i in nzrange(qp.data.A, j)
        set_normalized_coefficient(
            qp.constr[qp.data.A.rowval[i]],
            qp.x[j],
            qp.data.A.nzval[i],
        )
    end
    for (ind, val) in enumerate(qp.adj)
        row_of_A = qp.data.A[val, :]
        for i = 1:row_of_A.n
            j = row_of_A.nzind[i]
            set_normalized_coefficient(qp.constr[m+ind], qp.x[j], row_of_A.nzval[i])
        end
    end

    # modify RHS
    for i = 1:m
        c_ub = qp.data.c_ub[i] - b[i]
        c_lb = qp.data.c_lb[i] - b[i]

        if qp.data.c_lb[i] == qp.data.c_ub[i]
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_lb[i] != -Inf &&
               qp.data.c_ub[i] != Inf &&
               qp.data.c_lb[i] < qp.data.c_ub[i]
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_lb[i] != -Inf
            set_normalized_rhs(qp.constr[i], c_lb)
        elseif qp.data.c_ub[i] != Inf
            set_normalized_rhs(qp.constr[i], c_ub)
        end
    end
    # @show qp.data.c_lb-b, qp.data.c_ub-b, b
    for (i, val) in enumerate(qp.adj)
        c_ub = qp.data.c_ub[val] - b[val]
        set_normalized_rhs(qp.constr[i+m], c_ub)
    end

    # JuMP.write_to_file(qp.model, "debug_jump.lp", format = MOI.FileFormats.FORMAT_LP)
    # @show x_k
    # JuMP.print(qp.model)
    JuMP.optimize!(qp.model)
    status = termination_status(qp.model)

    # TODO: These can be part of data.
    Xsol = Tv(undef, n)
    p_slack = Dict{Int,Vector{Float64}}()
    lambda = Tv(undef, m)
    mult_x_U = Tv(undef, n)
    mult_x_L = Tv(undef, n)

    if status ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        # @show MOI.get(qp.model, MOI.ObjectiveValue())
        Xsol .= JuMP.value.(qp.x)
        for (i, slacks) in qp.slack_vars
            p_slack[i] = JuMP.value.(slacks)
        end
        # @show JuMP.objective_value(qp.model), Xsol
        # @show p_slack

        # extract the multipliers to constraints
        for i = 1:m
            lambda[i] = JuMP.dual(qp.constr[i])
        end
        for (i, val) in enumerate(qp.adj)
            lambda[val] += JuMP.dual(qp.constr[i+m])
        end
        # @show MOI.get(qp.model, MOI.ConstraintDual(1), qp.constr)

        # extract the multipliers to column bounds
        mult_x_U .= JuMP.dual.(UpperBoundRef.(qp.x))
        mult_x_L .= JuMP.dual.(LowerBoundRef.(qp.x))
        # careful because of the trust region
        for j = 1:n
            if Xsol[j] < qp.data.v_ub[j] - x_k[j]
                mult_x_U[j] = 0.0
            end
            if Xsol[j] > qp.data.v_lb[j] - x_k[j]
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