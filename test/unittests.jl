using Test
using ActiveSetMethods
using JuMP

model = Model(ActiveSetMethods.Optimizer);

@variable(model, X);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X*Y == 1);
@NLconstraint(model, X*Y >= 0);
@constraint(model, X >= -2);

asm = backend(model).optimizer.model
