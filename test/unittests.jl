using Test
using ActiveSetMethods
using JuMP

model = Model(ActiveSetMethods.Optimizer);

@variable(model, X <= -1);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@constraint(model, X^2 - X == 2);
@NLconstraint(model, X*Y == 1);
@NLconstraint(model, X*Y >= 0);

MOIU.attach_optimizer(backend(model))
inner = backend(model).optimizer.model

@test length(inner.variable_info) == 2
@test length(inner.linear_le_constraints) == 0
@test length(inner.linear_ge_constraints) == 0
@test length(inner.linear_eq_constraints) == 0
@test length(inner.quadratic_le_constraints) == 0
@test length(inner.quadratic_ge_constraints) == 1
@test length(inner.quadratic_eq_constraints) == 2
@show length(inner.nlp_data.constraint_bounds) == 0